from __future__ import annotations

import os
import traceback
from collections import Counter
from contextlib import ExitStack, nullcontext
from typing import Any, Dict

import numpy as np
import PIL
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchvision.transforms import v2

from sam3.model import box_ops
from sam3.model.data_misc import interpolate
from sam3.model.sam3_image_processor import Sam3Processor


class SafeSam3Processor(Sam3Processor):
    """SAM3 processor wrapper that hard-syncs dtypes and logs failures clearly."""

    _fused_mlp_patch_applied = False

    def __init__(self, model, resolution: int = 1008, device: str = "cuda", confidence_threshold: float = 0.5):
        self.debug = os.environ.get("VOID_SAM3_DEBUG", "0").lower() not in {"0", "false", "no"}
        self.force_dtype = torch.float32
        self.mask_threshold = float(os.environ.get("VOID_SAM3_MASK_THRESHOLD", "0.5"))
        self.top1_fallback_min_score = float(os.environ.get("VOID_SAM3_TOP1_FALLBACK_MIN_SCORE", "0.05"))
        self.device = torch.device(device)
        self._model_sync_dtype: torch.dtype | None = None
        self._patch_sam3_fused_mlp()
        super().__init__(model, resolution=resolution, device=self.device, confidence_threshold=confidence_threshold)
        self._ensure_model_dtype(self._target_dtype(), reason="init", force=True)
        self._log(
            "Initialized processor "
            f"device={self.device} target_dtype={self._target_dtype()} "
            f"model_dtypes={self._summarize_model_dtypes()}"
        )

    def _log(self, message: str):
        if self.debug:
            print(f"         [SAM3] {message}")

    @classmethod
    def _patch_sam3_fused_mlp(cls):
        if cls._fused_mlp_patch_applied:
            return

        import sam3.model.vitdet as sam3_vitdet
        import sam3.perflib.fused as sam3_fused

        def safe_addmm_act(activation, linear, mat1):
            target_dtype = linear.weight.dtype
            x = torch.nn.functional.linear(mat1.to(dtype=target_dtype), linear.weight, linear.bias)

            if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
                return torch.nn.functional.relu(x)
            if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
                return torch.nn.functional.gelu(x)
            raise ValueError(f"Unexpected activation {activation}")

        sam3_fused.addmm_act = safe_addmm_act
        sam3_vitdet.addmm_act = safe_addmm_act
        cls._fused_mlp_patch_applied = True

    def _model_dtype(self) -> torch.dtype:
        for tensor in self.model.parameters():
            if tensor.is_floating_point():
                return tensor.dtype
        for tensor in self.model.buffers():
            if tensor.is_floating_point():
                return tensor.dtype
        return self.force_dtype

    def _target_dtype(self) -> torch.dtype:
        return self.force_dtype

    def _execution_context(self, dtype: torch.dtype):
        stack = ExitStack()
        if self.device.type == "cuda":
            stack.enter_context(torch.autocast(device_type="cuda", enabled=False))
            if dtype == torch.float32:
                # Flash/efficient SDPA may internally downcast to bf16 on CUDA.
                # Force the math backend to keep activations in fp32.
                stack.enter_context(sdpa_kernel([SDPBackend.MATH]))
        else:
            stack.enter_context(nullcontext())
        return stack

    def _coerce_value(self, value: Any, dtype: torch.dtype | None = None, seen: set[int] | None = None) -> Any:
        dtype = dtype or self._target_dtype()
        if seen is None:
            seen = set()

        if torch.is_tensor(value):
            if value.is_floating_point():
                return value.to(device=self.device, dtype=dtype)
            return value.to(device=self.device)

        immutable_scalars = (str, bytes, int, float, bool, type(None))
        if isinstance(value, immutable_scalars):
            return value

        object_id = id(value)
        if object_id in seen:
            return value
        seen.add(object_id)

        if isinstance(value, dict):
            for key, nested in list(value.items()):
                value[key] = self._coerce_value(nested, dtype=dtype, seen=seen)
            return value
        if isinstance(value, list):
            for index, nested in enumerate(list(value)):
                value[index] = self._coerce_value(nested, dtype=dtype, seen=seen)
            return value
        if isinstance(value, tuple):
            return tuple(self._coerce_value(nested, dtype=dtype, seen=seen) for nested in value)
        if isinstance(value, set):
            return {self._coerce_value(nested, dtype=dtype, seen=seen) for nested in value}
        if isinstance(value, torch.nn.Module):
            value.to(device=self.device, dtype=dtype)
            registered = set(value._parameters) | set(value._buffers) | set(value._modules)
            for key, nested in list(vars(value).items()):
                if key in registered:
                    continue
                new_value = self._coerce_value(nested, dtype=dtype, seen=seen)
                if new_value is not nested:
                    setattr(value, key, new_value)
            return value
        if hasattr(value, "__dict__"):
            for key, nested in list(vars(value).items()):
                new_value = self._coerce_value(nested, dtype=dtype, seen=seen)
                if new_value is not nested:
                    setattr(value, key, new_value)
            return value
        return value

    def _normalize_state(self, state: Dict, dtype: torch.dtype | None = None) -> Dict:
        return self._coerce_value(state, dtype=dtype)

    def _prepare_image_tensor(self, image, dtype: torch.dtype) -> tuple[torch.Tensor, int, int]:
        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.transform(image_tensor).to(device=self.device, dtype=dtype).unsqueeze(0)
        return image_tensor, height, width

    def _ensure_model_dtype(self, dtype: torch.dtype, reason: str, force: bool = False):
        if not force and self._model_sync_dtype == dtype:
            return
        self._log(f"Synchronizing model tensors to {dtype} ({reason})")
        self.model = self._coerce_value(self.model, dtype=dtype)
        self._model_sync_dtype = dtype
        self._log(f"Model dtype summary after sync: {self._summarize_model_dtypes()}")

    def _summarize_tensor_dtypes(self, value: Any, max_items: int = 12) -> str:
        samples: list[str] = []
        counts: Counter[str] = Counter()
        seen: set[int] = set()

        def visit(node: Any, path: str):
            if len(samples) >= max_items:
                return
            if torch.is_tensor(node):
                dtype_key = str(node.dtype)
                counts[dtype_key] += 1
                samples.append(f"{path}={node.dtype}{tuple(node.shape)}")
                return

            immutable_scalars = (str, bytes, int, float, bool, type(None))
            if isinstance(node, immutable_scalars):
                return

            object_id = id(node)
            if object_id in seen:
                return
            seen.add(object_id)

            if isinstance(node, dict):
                for key, nested in list(node.items())[:max_items]:
                    visit(nested, f"{path}.{key}")
                return
            if isinstance(node, list):
                for index, nested in enumerate(node[:max_items]):
                    visit(nested, f"{path}[{index}]")
                return
            if isinstance(node, tuple):
                for index, nested in enumerate(node[:max_items]):
                    visit(nested, f"{path}({index})")
                return
            if isinstance(node, torch.nn.Module):
                for name, tensor in list(node.named_parameters(recurse=True))[:max_items]:
                    visit(tensor, f"{path}.param:{name}")
                for name, tensor in list(node.named_buffers(recurse=True))[:max_items]:
                    visit(tensor, f"{path}.buffer:{name}")
                return
            if hasattr(node, "__dict__"):
                for key, nested in list(vars(node).items())[:max_items]:
                    visit(nested, f"{path}.{key}")

        visit(value, "root")
        if not counts:
            return "no tensors found"
        counts_text = ", ".join(f"{dtype}:{count}" for dtype, count in counts.items())
        return f"{counts_text} | samples: {'; '.join(samples)}"

    def _summarize_model_dtypes(self) -> str:
        counts: Counter[str] = Counter()
        samples: list[str] = []
        for name, tensor in self.model.named_parameters():
            if tensor.is_floating_point():
                counts[str(tensor.dtype)] += 1
                if len(samples) < 6:
                    samples.append(f"param:{name}={tensor.dtype}{tuple(tensor.shape)}")
        for name, tensor in self.model.named_buffers():
            if tensor.is_floating_point():
                counts[str(tensor.dtype)] += 1
                if len(samples) < 12:
                    samples.append(f"buffer:{name}={tensor.dtype}{tuple(tensor.shape)}")
        if not counts:
            return "no floating parameters or buffers"
        return f"{dict(counts)} | {'; '.join(samples)}"

    def _log_exception(self, stage: str, exc: Exception, state: Dict | None = None):
        self._log(f"{stage} failed: {exc!r}")
        self._log(traceback.format_exc().rstrip())
        self._log(f"Model dtypes: {self._summarize_model_dtypes()}")
        if state is not None:
            self._log(f"State dtypes: {self._summarize_tensor_dtypes(state)}")

    @torch.inference_mode()
    def set_image(self, image, state=None):
        base_state = {} if state is None else state
        dtype = self._target_dtype()
        working_state = dict(base_state)
        self._ensure_model_dtype(dtype, reason="set_image")
        try:
            image_tensor, height, width = self._prepare_image_tensor(image, dtype=dtype)
            self._log(f"Prepared image tensor dtype={image_tensor.dtype} shape={tuple(image_tensor.shape)}")
            with self._execution_context(dtype):
                working_state["original_height"] = height
                working_state["original_width"] = width
                working_state["backbone_out"] = self.model.backbone.forward_image(image_tensor)

            inst_interactivity_en = self.model.inst_interactive_predictor is not None
            if inst_interactivity_en and "sam2_backbone_out" in working_state["backbone_out"]:
                sam2_backbone_out = working_state["backbone_out"]["sam2_backbone_out"]
                sam2_backbone_out["backbone_fpn"][0] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                        sam2_backbone_out["backbone_fpn"][0]
                    )
                )
                sam2_backbone_out["backbone_fpn"][1] = (
                    self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                        sam2_backbone_out["backbone_fpn"][1]
                    )
                )

            normalized_state = self._normalize_state(working_state, dtype=dtype)
            self._log(f"set_image state summary: {self._summarize_tensor_dtypes(normalized_state)}")
            return normalized_state
        except Exception as exc:
            self._log_exception("set_image", exc, state=working_state)
            raise

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        dtype = self._target_dtype()
        self._ensure_model_dtype(dtype, reason="set_text_prompt")
        working_state = self._normalize_state(state, dtype=dtype)
        try:
            with self._execution_context(dtype):
                text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
            working_state["backbone_out"].update(self._coerce_value(text_outputs, dtype=dtype))
            if "geometric_prompt" not in working_state:
                working_state["geometric_prompt"] = self._coerce_value(self.model._get_dummy_prompt(), dtype=dtype)
            else:
                working_state["geometric_prompt"] = self._coerce_value(working_state["geometric_prompt"], dtype=dtype)
            self._log(
                "Prepared grounding state "
                f"backbone_out={self._summarize_tensor_dtypes(working_state.get('backbone_out', {}), max_items=8)} "
                f"geometric_prompt={self._summarize_tensor_dtypes(working_state.get('geometric_prompt'), max_items=6)}"
            )
            return self._forward_grounding(working_state)
        except Exception as exc:
            self._log_exception("set_text_prompt", exc, state=working_state)
            raise

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        dtype = self._model_dtype()
        normalized_state = self._normalize_state(state, dtype=dtype)
        try:
            with self._execution_context(dtype):
                outputs = self.model.forward_grounding(
                    backbone_out=normalized_state["backbone_out"],
                    find_input=self.find_stage,
                    geometric_prompt=normalized_state["geometric_prompt"],
                    find_target=None,
                )

            out_bbox = outputs["pred_boxes"]
            out_logits = outputs["pred_logits"]
            out_masks = outputs["pred_masks"]
            out_probs = out_logits.sigmoid()
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = (out_probs * presence_score).squeeze(-1)

            # SAM3 returns batched grounding outputs. Stage 3a only processes one image,
            # so flatten candidates across the batch dimension before thresholding.
            out_probs = out_probs.reshape(-1)
            out_bbox = out_bbox.reshape(-1, out_bbox.shape[-1])
            out_masks = out_masks.reshape(-1, *out_masks.shape[-2:])

            top_scores = []
            if out_probs.numel() > 0:
                top_scores = torch.topk(out_probs.flatten(), k=min(5, out_probs.numel())).values.detach().cpu().tolist()
            self._log(
                "Grounding outputs "
                f"candidates={int(out_probs.numel())} "
                f"boxes_shape={tuple(out_bbox.shape)} "
                f"masks_shape={tuple(out_masks.shape)} "
                f"confidence_threshold={self.confidence_threshold} "
                f"top_scores={[round(float(score), 4) for score in top_scores]}"
            )

            keep = out_probs > self.confidence_threshold
            if not keep.any() and out_probs.numel() > 0:
                top_idx = int(torch.argmax(out_probs).item())
                top_score = float(out_probs[top_idx].item())
                if top_score >= self.top1_fallback_min_score:
                    keep = torch.zeros_like(out_probs, dtype=torch.bool)
                    keep[top_idx] = True
                    self._log(
                        "No detections passed the confidence threshold; "
                        f"keeping top-1 candidate score={top_score:.4f} "
                        f"(fallback_min_score={self.top1_fallback_min_score})"
                    )

            out_probs = out_probs[keep]
            out_masks = out_masks[keep]
            out_bbox = out_bbox[keep]

            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            img_h = normalized_state["original_height"]
            img_w = normalized_state["original_width"]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h], device=self.device, dtype=boxes.dtype)
            boxes = boxes * scale_fct[None, :]

            if out_masks.numel() > 0:
                out_masks = interpolate(
                    out_masks.unsqueeze(1),
                    (img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()
            else:
                out_masks = torch.zeros((0, 1, img_h, img_w), device=self.device, dtype=dtype)

            normalized_state["masks_logits"] = out_masks
            normalized_state["masks"] = out_masks > self.mask_threshold
            normalized_state["boxes"] = boxes
            normalized_state["scores"] = out_probs
            self._log(
                "Grounding kept "
                f"count={int(out_probs.numel())} "
                f"mask_threshold={self.mask_threshold} "
                f"mask_pixels={int(normalized_state['masks'].sum().item()) if normalized_state['masks'].numel() > 0 else 0}"
            )
            return normalized_state
        except Exception as exc:
            self._log_exception("_forward_grounding", exc, state=normalized_state)
            raise
