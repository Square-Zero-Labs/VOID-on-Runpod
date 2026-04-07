from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Generator
from urllib.parse import quote

import cv2
import gradio as gr

from void_runpod.app_state import (
    artifact_path,
    build_points_json,
    copy_uploaded_video,
    extract_frames,
    find_pass1_output,
    load_existing_job,
    make_job_paths,
    overlay_points,
    summarize_state,
    write_json,
    write_prompt,
)


APP_ROOT = Path(__file__).resolve().parent
VOID_ROOT = APP_ROOT / "VOID"
PYTHON_BIN = sys.executable
WORKSPACE_DIR = Path(os.environ.get("VOID_WORKSPACE_DIR", str(APP_ROOT / ".void-workspace")))
BASE_MODEL_PATH = os.environ.get("VOID_BASE_MODEL_PATH", str(WORKSPACE_DIR / "checkpoints" / "CogVideoX-Fun-V1.5-5b-InP"))
PASS1_MODEL_PATH = os.environ.get("VOID_PASS1_PATH", str(WORKSPACE_DIR / "checkpoints" / "void_pass1.safetensors"))
PASS2_MODEL_PATH = os.environ.get("VOID_PASS2_PATH", str(WORKSPACE_DIR / "checkpoints" / "void_pass2.safetensors"))
SAM2_CHECKPOINT_PATH = os.environ.get("VOID_SAM2_CHECKPOINT", str(WORKSPACE_DIR / "checkpoints" / "sam2_hiera_large.pt"))
DEFAULT_GRADIO_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GREY_MASK_SEGMENTATION_MODEL = os.environ.get("VOID_GREY_MASK_SEGMENTATION_MODEL", "sam3")


CUSTOM_CSS = """
:root {
  --bg: #0b1020;
  --bg-2: #11182b;
  --panel: rgba(17, 24, 43, 0.92);
  --panel-2: rgba(25, 35, 60, 0.92);
  --border: #2f3d66;
  --ink: #e8eefc;
  --muted: #9fb0d9;
  --accent: #4cc9f0;
  --accent-2: #f4b860;
  --success: #56d39b;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(76, 201, 240, 0.14), transparent 28%),
    radial-gradient(circle at top right, rgba(244, 184, 96, 0.10), transparent 22%),
    linear-gradient(180deg, #09111f 0%, #0f1830 100%);
  color: var(--ink);
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif !important;
}

.gradio-container .prose,
.gradio-container label,
.gradio-container .wrap,
.gradio-container .message {
  color: var(--ink) !important;
}

.block, .gr-panel, .gr-box, .gr-accordion, .step-card {
  background: var(--panel) !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.28);
}

.step-card {
  border-radius: 18px;
  padding: 8px;
}

h1, h2, h3 {
  color: var(--ink) !important;
  font-family: "IBM Plex Serif", Georgia, serif !important;
}

.hero {
  background: linear-gradient(135deg, rgba(76, 201, 240, 0.18), rgba(84, 96, 240, 0.08));
  border: 1px solid var(--border);
  border-radius: 22px;
  padding: 18px 22px;
  margin-bottom: 8px;
}

.hero p,
.step-card p,
.step-card li {
  color: var(--muted) !important;
}

.step-title {
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--ink);
  margin-bottom: 8px;
}

.step-kicker {
  color: var(--accent) !important;
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.callout {
  background: rgba(76, 201, 240, 0.08);
  border: 1px solid rgba(76, 201, 240, 0.3);
  border-radius: 14px;
  padding: 12px 14px;
  margin: 8px 0;
}

.callout strong {
  color: var(--accent-2);
}

button.primary {
  box-shadow: 0 0 0 1px rgba(76, 201, 240, 0.35), 0 12px 24px rgba(76, 201, 240, 0.14);
}

textarea, input, .gr-textbox, .gr-number, .gr-dropdown, .gradio-container .gr-form {
  background: var(--bg-2) !important;
  color: var(--ink) !important;
}

.gradio-container .gr-form, .gradio-container .form {
  border-color: var(--border) !important;
}

.gradio-container .gr-button-secondary {
  background: var(--panel-2) !important;
}

.gradio-container .gr-markdown code,
.gradio-container code {
  background: rgba(255, 255, 255, 0.08) !important;
  color: #d7f4ff !important;
}

.output-panel {
  min-height: 100%;
}

.workflow-shell {
  max-width: 1080px;
  margin: 0 auto;
  width: 100%;
}

.step-actions {
  margin-top: 8px;
}

.compact-note {
  color: var(--muted) !important;
  font-size: 0.94rem;
}

.download-link {
  margin-top: 10px;
}

.download-link a {
  display: inline-block;
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid var(--border);
  background: var(--panel-2);
  color: var(--ink) !important;
  text-decoration: none;
  font-weight: 600;
}

.download-link a:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(76, 201, 240, 0.35);
}

.download-link.muted {
  color: var(--muted);
}
"""


def ensure_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def _active_job_markdown(job_state: dict[str, Any] | None) -> str:
    if not job_state:
        return "**Active job:** `none`"
    return f"**Active job:** `{job_state['job_id']}`"


def _aligned_model_frame_count(frame_count: int) -> int:
    target = max(5, int(frame_count))
    remainder = (target - 5) % 8
    if remainder:
        target += 8 - remainder
    return target


def _effective_inference_settings(
    job_state: dict[str, Any],
    requested_max_video_length: int,
    requested_temporal_window: int,
) -> tuple[int, int, int]:
    source_frames = int(job_state.get("total_frames") or len(job_state.get("frame_paths", [])) or 1)
    capped_frames = max(1, min(source_frames, int(requested_max_video_length)))
    effective_frames = _aligned_model_frame_count(capped_frames)
    effective_window = min(effective_frames, _aligned_model_frame_count(max(1, int(requested_temporal_window))))
    effective_fps = max(1, int(round(float(job_state.get("fps") or 12.0))))
    return effective_frames, effective_window, effective_fps


def _render_frame(job_state: dict[str, Any], frame_index: int) -> Any:
    if not job_state:
        return None
    frame_index = max(0, min(int(frame_index), len(job_state["frame_paths"]) - 1))
    frame_path = job_state["frame_paths"][frame_index]
    return overlay_points(frame_path, job_state["points_by_frame"], frame_index)


def _points_json_preview(job_state: dict[str, Any], instruction: str, min_grid: int, multi_frame_grids: bool) -> dict[str, Any]:
    if not job_state:
        return {"videos": []}
    return build_points_json(job_state, instruction, min_grid, multi_frame_grids)


def _artifacts_markdown(job_state: dict[str, Any] | None) -> str:
    if not job_state:
        return "No artifacts yet."

    black_mask = Path(artifact_path(job_state, "black_mask"))
    quadmask = Path(artifact_path(job_state, "quadmask"))
    pass1 = Path(artifact_path(job_state, "pass1")) if artifact_path(job_state, "pass1") else None
    pass2 = Path(artifact_path(job_state, "pass2"))
    vlm_analysis = Path(artifact_path(job_state, "vlm_analysis"))

    lines = [
        f"- SAM2 mask: `{'ready' if black_mask.exists() else 'missing'}`",
        f"- Quadmask: `{'ready' if quadmask.exists() else 'missing'}`",
        f"- VLM analysis: `{'ready' if vlm_analysis.exists() else 'missing'}`",
        f"- Pass 1 output: `{'ready' if pass1 and pass1.exists() else 'missing'}`",
        f"- Pass 2 output: `{'ready' if pass2.exists() else 'missing'}`",
    ]
    return "\n".join(lines)


def _has_selected_points(job_state: dict[str, Any] | None) -> bool:
    if not job_state:
        return False
    return any(job_state.get("points_by_frame", {}).values())


def _workflow_markdown(job_state: dict[str, Any] | None) -> str:
    stage = _workflow_stage(job_state)
    if stage == "needs_job":
        return (
            "**Next step:** Load a source video or reopen an existing job.\n\n"
            "- Source video: `missing`\n"
            "- Selected points: `missing`\n"
            "- Quadmask: `missing`\n"
            "- Pass 1: `missing`\n"
            "- Pass 2: `missing`"
        )

    has_points = _has_selected_points(job_state)
    has_quadmask = Path(artifact_path(job_state, "quadmask")).exists()
    has_pass1 = bool(find_pass1_output(job_state))
    has_pass2 = Path(artifact_path(job_state, "pass2")).exists()

    if stage == "needs_points":
        next_step = "Add at least one point, or open the optional skip-ahead section if you already have a quadmask."
    elif stage == "ready_quadmask":
        next_step = "Create or regenerate the quadmask."
    elif stage == "ready_pass1":
        next_step = "Run Pass 1."
    elif stage == "ready_pass2":
        next_step = "Run Pass 2."
    else:
        next_step = "Review the outputs below, or reopen this job later using the active job name."

    return (
        f"**Next step:** {next_step}\n\n"
        f"- Source video: `ready`\n"
        f"- Selected points: `{'ready' if has_points else 'missing'}`\n"
        f"- Quadmask: `{'ready' if has_quadmask else 'missing'}`\n"
        f"- Pass 1: `{'ready' if has_pass1 else 'missing'}`\n"
        f"- Pass 2: `{'ready' if has_pass2 else 'missing'}`"
    )


def _frame_points_markdown(job_state: dict[str, Any] | None, frame_index: int) -> str:
    if not job_state:
        return "No frame loaded."

    frame_key = str(int(frame_index))
    frame_points = job_state.get("points_by_frame", {}).get(frame_key, [])
    frames_with_points = sum(1 for value in job_state.get("points_by_frame", {}).values() if value)
    return (
        f"**Current frame:** `{int(frame_index)}`\n\n"
        f"- Points on this frame: `{len(frame_points)}`\n"
        f"- Frames with points: `{frames_with_points}`"
    )


def _workflow_stage(job_state: dict[str, Any] | None) -> str:
    if not job_state:
        return "needs_job"

    has_points = _has_selected_points(job_state)
    has_quadmask = Path(artifact_path(job_state, "quadmask")).exists()
    has_pass1 = bool(find_pass1_output(job_state))
    has_pass2 = Path(artifact_path(job_state, "pass2")).exists()

    if not has_points and not has_quadmask:
        return "needs_points"
    if not has_quadmask:
        return "ready_quadmask"
    if not has_pass1:
        return "ready_pass1"
    if not has_pass2:
        return "ready_pass2"
    return "complete"


def _primary_action_update(job_state: dict[str, Any] | None) -> Any:
    stage = _workflow_stage(job_state)
    if stage == "needs_job":
        return gr.update(value="Load A Job First", interactive=False)
    if stage == "needs_points":
        return gr.update(value="Add Points Before Generating Quadmask", interactive=False)
    if stage == "ready_quadmask":
        return gr.update(value="Create / Regenerate Quadmask", interactive=True)
    if stage == "ready_pass1":
        return gr.update(value="Run Pass 1", interactive=True)
    if stage == "ready_pass2":
        return gr.update(value="Run Pass 2", interactive=True)
    return gr.update(value="Pipeline Complete", interactive=False)


def _action_button_updates(job_state: dict[str, Any] | None) -> tuple[Any, Any, Any, Any]:
    has_job = bool(job_state)
    has_points = _has_selected_points(job_state)
    has_quadmask = has_job and Path(artifact_path(job_state, "quadmask")).exists()
    has_pass1 = has_job and bool(find_pass1_output(job_state))

    return (
        _primary_action_update(job_state),
        gr.update(interactive=has_job and has_points),
        gr.update(interactive=has_quadmask),
        gr.update(interactive=has_quadmask and has_pass1),
    )


def _job_outputs(
    job_state: dict[str, Any],
    import_quadmask_status: str = "",
    import_pass1_status: str = "",
) -> tuple[Any, ...]:
    frame_image = _render_frame(job_state, 0)
    slider_update = gr.update(minimum=0, maximum=max(len(job_state["frame_paths"]) - 1, 0), value=0, step=1)
    points_preview = _points_json_preview(
        job_state,
        job_state.get("removal_instruction", "remove the selected object"),
        job_state.get("min_grid", 8),
        job_state.get("multi_frame_grids", True),
    )
    pass1_output = find_pass1_output(job_state)

    return (
        job_state,
        _active_job_markdown(job_state),
        summarize_state(job_state),
        frame_image,
        slider_update,
        _frame_points_markdown(job_state, 0),
        points_preview,
        job_state["input_video_path"],
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        job_state["config_path"] if Path(job_state["config_path"]).exists() else "",
        job_state.get("removal_instruction", "remove the selected object"),
        job_state.get("background_prompt", ""),
        gr.update(value=job_state.get("min_grid", 8)),
        gr.update(value=job_state.get("multi_frame_grids", True)),
        "",
        import_quadmask_status,
        import_pass1_status,
        artifact_path(job_state, "black_mask") if Path(artifact_path(job_state, "black_mask")).exists() else None,
        _download_link_html(artifact_path(job_state, "black_mask"), "Open SAM2 mask in new tab"),
        artifact_path(job_state, "quadmask") if Path(artifact_path(job_state, "quadmask")).exists() else None,
        _download_link_html(artifact_path(job_state, "quadmask"), "Open quadmask in new tab"),
        pass1_output,
        _download_link_html(pass1_output, "Open Pass 1 video in new tab"),
    )


def _job_outputs_tail(job_state: dict[str, Any]) -> tuple[Any, str, Any, Any, Any, Any]:
    pass2_output = artifact_path(job_state, "pass2")
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    return (
        pass2_output if Path(pass2_output).exists() else None,
        _download_link_html(pass2_output if Path(pass2_output).exists() else None, "Open Pass 2 video in new tab"),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def prepare_job(upload_path: str, job_name: str | None) -> tuple[Any, ...]:
    ensure_workspace()
    if not upload_path:
        raise gr.Error("Upload a video first.")

    try:
        job_state = make_job_paths(WORKSPACE_DIR, upload_path, job_name)
    except FileExistsError as exc:
        raise gr.Error(str(exc)) from exc

    copy_uploaded_video(upload_path, job_state["input_video_path"])
    metadata = extract_frames(job_state["input_video_path"], job_state["frames_dir"])
    job_state.update(metadata)
    job_state["points_by_frame"] = {}
    job_state["removal_instruction"] = "remove the selected object"
    job_state["background_prompt"] = ""
    job_state["min_grid"] = 8
    job_state["multi_frame_grids"] = True

    return _job_outputs(job_state) + _job_outputs_tail(job_state)


def open_existing_job(requested_job_name: str) -> tuple[Any, ...]:
    ensure_workspace()
    try:
        job_state = load_existing_job(WORKSPACE_DIR, requested_job_name)
    except (FileNotFoundError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    return _job_outputs(job_state) + _job_outputs_tail(job_state)


def change_frame(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[Any, dict[str, Any]]:
    return (
        _render_frame(job_state, frame_index),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        _frame_points_markdown(job_state, frame_index),
    )


def add_point(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool, evt: gr.SelectData) -> tuple[Any, ...]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    if evt.index is None:
        raise gr.Error("Click directly on the frame image.")

    x, y = evt.index
    frame_key = str(int(frame_index))
    job_state["points_by_frame"].setdefault(frame_key, []).append([int(x), int(y)])
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
        _frame_points_markdown(job_state, frame_index),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def undo_last_point(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[Any, ...]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    frame_key = str(int(frame_index))
    frame_points = job_state["points_by_frame"].get(frame_key, [])
    if frame_points:
        frame_points.pop()
    if not frame_points and frame_key in job_state["points_by_frame"]:
        job_state["points_by_frame"].pop(frame_key, None)

    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
        _frame_points_markdown(job_state, frame_index),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def clear_frame_points(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[Any, ...]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    job_state["points_by_frame"].pop(str(int(frame_index)), None)
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
        _frame_points_markdown(job_state, frame_index),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def clear_all_points(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[Any, ...]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    job_state["points_by_frame"] = {}
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
        _frame_points_markdown(job_state, frame_index),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def save_points_config(job_state: dict[str, Any], instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[dict[str, Any], str]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    payload = build_points_json(job_state, instruction, min_grid, multi_frame_grids)
    write_json(job_state["config_path"], payload)
    return payload, job_state["config_path"]


def refresh_points_preview(job_state: dict[str, Any], instruction: str, min_grid: int, multi_frame_grids: bool) -> dict[str, Any]:
    return _points_json_preview(job_state, instruction, min_grid, multi_frame_grids)


def _video_match_message(source_video_path: str, candidate_video_path: str, label: str) -> str:
    source_capture = cv2.VideoCapture(source_video_path)
    candidate_capture = cv2.VideoCapture(candidate_video_path)
    if not source_capture.isOpened() or not candidate_capture.isOpened():
        raise gr.Error(f"Failed to open the source video or the uploaded {label}.")

    source_width = int(source_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(source_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(source_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    candidate_width = int(candidate_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    candidate_height = int(candidate_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    candidate_frames = int(candidate_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_capture.release()
    candidate_capture.release()

    warnings: list[str] = []
    if source_width != candidate_width or source_height != candidate_height:
        warnings.append(
            f"Resolution mismatch: source is {source_width}x{source_height}, {label} is {candidate_width}x{candidate_height}."
        )
    if source_frames != candidate_frames:
        warnings.append(
            f"Frame-count mismatch: source has {source_frames} frames, {label} has {candidate_frames} frames."
        )

    if not warnings:
        return f"{label.capitalize()} dimensions match the source video."
    return "WARNING:\n" + "\n".join(f"- {warning}" for warning in warnings)


def import_existing_quadmask(job_state: dict[str, Any], quadmask_upload_path: str | None) -> tuple[Any, ...]:
    if not job_state:
        raise gr.Error("Upload and load a source video first.")
    if not quadmask_upload_path:
        raise gr.Error("Choose a quadmask video to upload.")

    destination = Path(artifact_path(job_state, "quadmask"))
    destination.parent.mkdir(parents=True, exist_ok=True)

    Path(destination).write_bytes(Path(quadmask_upload_path).read_bytes())

    message = f"Imported existing quadmask to `{destination}`."
    message += "\n\n" + _video_match_message(job_state["input_video_path"], quadmask_upload_path, "quadmask")

    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    return (
        message,
        message,
        str(destination),
        _download_link_html(str(destination), "Open quadmask in new tab"),
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def import_existing_pass1(job_state: dict[str, Any], pass1_upload_path: str | None) -> tuple[Any, ...]:
    if not job_state:
        raise gr.Error("Upload or reopen a job first.")
    if not pass1_upload_path:
        raise gr.Error("Choose a Pass 1 video to upload.")

    destination = Path(job_state["pass1_dir"]) / f"{job_state['sequence_name']}-fg=-1-imported.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pass1_upload_path, destination)

    message = f"Imported existing Pass 1 video to `{destination}`."
    message += "\n\n" + _video_match_message(job_state["input_video_path"], pass1_upload_path, "pass 1 video")
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    return (
        message,
        message,
        str(destination),
        _download_link_html(str(destination), "Open Pass 1 video in new tab"),
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def _run_streaming_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path, accumulated: str, step_name: str) -> Generator[str, None, str]:
    with open(log_path, "a", encoding="utf-8") as log_handle:
        log_handle.write(f"\n$ {' '.join(command)}\n")
        log_handle.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            accumulated = (accumulated + line)[-30000:]
            yield accumulated

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"{step_name} failed with exit code {return_code}.\n"
                f"Command: {' '.join(command)}\n"
                f"See log: {log_path}"
            )

    return accumulated


def _quadmask_outputs(job_state: dict[str, Any]) -> tuple[Any, str, Any, str, str]:
    black_mask = artifact_path(job_state, "black_mask")
    quadmask = artifact_path(job_state, "quadmask")
    return (
        black_mask if Path(black_mask).exists() else None,
        _download_link_html(black_mask if Path(black_mask).exists() else None, "Open SAM2 mask in new tab"),
        quadmask if Path(quadmask).exists() else None,
        _download_link_html(quadmask if Path(quadmask).exists() else None, "Open quadmask in new tab"),
        _artifacts_markdown(job_state),
    )


def _format_failure(step_name: str, exc: Exception, log_path: Path | None = None) -> str:
    message = [f"ERROR: {step_name} failed.", str(exc)]
    if log_path is not None:
        message.append(f"Detailed log: {log_path}")
    message.append("Fix the issue above and rerun this step.")
    return "\n".join(message)


def _ensure_segmentation_backend_ready(model_name: str) -> None:
    if model_name != "sam3":
        return
    try:
        import sam3  # noqa: F401
    except ImportError as exc:
        raise gr.Error(
            "SAM3 is selected for grey-mask generation, but the official `sam3` package is not installed. "
            "Restart the app after startup provisioning. The first SAM3 use also needs approved access to the "
            "gated `facebook/sam3` Hugging Face repo via `HF_TOKEN` or prior `hf auth login`."
        ) from exc


def _download_link_html(path: str | None, label: str) -> str:
    if not path:
        return "<div class='download-link muted'>No file available yet.</div>"

    resolved = Path(path).resolve()
    if not resolved.exists():
        return "<div class='download-link muted'>No file available yet.</div>"

    href = f"/gradio_api/file={quote(str(resolved), safe='/')}"
    return (
        "<div class='download-link'>"
        f"<a href=\"{href}\" target=\"_blank\" rel=\"noopener noreferrer\">{label}</a>"
        "</div>"
    )


def _download_pass2_if_needed() -> Generator[str, None, None]:
    checkpoint_path = Path(PASS2_MODEL_PATH)
    if checkpoint_path.exists():
        yield f"Pass 2 checkpoint already cached at `{checkpoint_path}`.\n"
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    yield f"Downloading Pass 2 checkpoint to `{checkpoint_path}`...\n"

    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id="netflix/void-model",
        filename="void_pass2.safetensors",
        local_dir=str(checkpoint_path.parent),
        token=os.environ.get("HF_TOKEN") or None,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    yield "Pass 2 checkpoint download complete.\n"


def _prepare_selected_pass1_dir(job_state: dict[str, Any]) -> str:
    selected_pass1 = find_pass1_output(job_state)
    if not selected_pass1:
        raise gr.Error("Upload or generate a Pass 1 video before running Pass 2.")

    selected_dir = Path(job_state["job_dir"]) / "selected_pass1_input"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for existing_file in selected_dir.glob("*.mp4"):
        if existing_file.name != Path(selected_pass1).name:
            existing_file.unlink()

    target_path = selected_dir / Path(selected_pass1).name
    if Path(selected_pass1).resolve() != target_path.resolve():
        shutil.copy2(selected_pass1, target_path)

    return str(selected_dir)


def run_quadmask_pipeline(
    job_state: dict[str, Any],
    instruction: str,
    background_prompt: str,
    gemini_api_key: str,
    min_grid: int,
    multi_frame_grids: bool,
) -> Generator[tuple[Any, ...], None, None]:
    if not job_state:
        raise gr.Error("Upload a video first.")
    if not gemini_api_key.strip():
        raise gr.Error("Enter a Gemini API key.")
    if not background_prompt.strip():
        raise gr.Error("Enter the clean-background prompt used by VOID inference.")
    if not any(job_state["points_by_frame"].values()):
        raise gr.Error("Add at least one point before running the quadmask pipeline.")

    _ensure_segmentation_backend_ready(GREY_MASK_SEGMENTATION_MODEL)

    payload = build_points_json(job_state, instruction, min_grid, multi_frame_grids)
    write_json(job_state["config_path"], payload)
    write_prompt(job_state["prompt_path"], background_prompt)

    log_path = Path(job_state["logs_dir"]) / "quadmask.log"
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = gemini_api_key.strip()

    commands = [
        ("SAM2 segmentation", [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage1_sam2_segmentation.py"),
            "--config",
            job_state["config_path"],
            "--sam2-checkpoint",
            SAM2_CHECKPOINT_PATH,
            "--device",
            "cuda",
        ]),
        ("Gemini VLM analysis", [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage2_vlm_analysis.py"),
            "--config",
            job_state["config_path"],
        ]),
        ("Grey mask generation", [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage3a_generate_grey_masks_v2.py"),
            "--config",
            job_state["config_path"],
            "--segmentation-model",
            GREY_MASK_SEGMENTATION_MODEL,
        ]),
        ("Quadmask combine", [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage4_combine_masks.py"),
            "--config",
            job_state["config_path"],
        ]),
    ]

    accumulated = ""
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    yield (
        "Preparing quadmask pipeline...\n",
        None,
        None,
        None,
        None,
        job_state["config_path"],
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )
    try:
        for step_name, command in commands:
            accumulated = (accumulated + f"\n=== {step_name} ===\n")[-30000:]
            primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
            yield (
                accumulated,
                None,
                None,
                None,
                None,
                job_state["config_path"],
                _artifacts_markdown(job_state),
                _workflow_markdown(job_state),
                primary_action_update,
                quadmask_button_update,
                pass1_button_update,
                pass2_button_update,
            )
            streamer = _run_streaming_command(command, APP_ROOT, env, log_path, accumulated, step_name)
            try:
                while True:
                    accumulated = next(streamer)
                    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
                    yield (
                        accumulated,
                        None,
                        None,
                        None,
                        None,
                        job_state["config_path"],
                        _artifacts_markdown(job_state),
                        _workflow_markdown(job_state),
                        primary_action_update,
                        quadmask_button_update,
                        pass1_button_update,
                        pass2_button_update,
                    )
            except StopIteration as stop:
                accumulated = stop.value or accumulated
    except Exception as exc:
        failure_log = (accumulated + "\n" + _format_failure("Quadmask pipeline", exc, log_path) + "\n")[-30000:]
        primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
        yield (
            failure_log,
            None,
            None,
            None,
            None,
            job_state["config_path"],
            _artifacts_markdown(job_state),
            _workflow_markdown(job_state),
            primary_action_update,
            quadmask_button_update,
            pass1_button_update,
            pass2_button_update,
        )
        return

    black_mask_video, black_mask_link, quadmask_video, quadmask_link, artifacts_md = _quadmask_outputs(job_state)
    success_log = (accumulated + "\nQuadmask pipeline complete.\n")[-30000:]
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    yield (
        success_log,
        black_mask_video,
        black_mask_link,
        quadmask_video,
        quadmask_link,
        job_state["config_path"],
        artifacts_md,
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def run_pass1(
    job_state: dict[str, Any],
    background_prompt: str,
    sample_height: int,
    sample_width: int,
    max_video_length: int,
    temporal_window: int,
    num_steps: int,
    guidance_scale: float,
) -> Generator[tuple[Any, ...], None, None]:
    if not job_state:
        raise gr.Error("Upload a video first.")
    if not Path(artifact_path(job_state, "quadmask")).exists():
        raise gr.Error("Generate the quadmask first.")
    if not background_prompt.strip():
        raise gr.Error("Enter the VOID background prompt before running Pass 1.")

    write_prompt(job_state["prompt_path"], background_prompt)
    effective_max_frames, effective_window, effective_fps = _effective_inference_settings(
        job_state,
        max_video_length,
        temporal_window,
    )

    log_path = Path(job_state["logs_dir"]) / "pass1.log"
    env = os.environ.copy()
    command = [
        PYTHON_BIN,
        str(VOID_ROOT / "inference" / "cogvideox_fun" / "predict_v2v.py"),
        "--config",
        str(VOID_ROOT / "config" / "quadmask_cogvideox.py"),
        f"--config.data.data_rootdir={job_state['data_root']}",
        f"--config.data.sample_size={sample_height}x{sample_width}",
        f"--config.data.max_video_length={effective_max_frames}",
        f"--config.data.fps={effective_fps}",
        f"--config.video_model.temporal_window_size={effective_window}",
        f"--config.video_model.num_inference_steps={int(num_steps)}",
        f"--config.video_model.guidance_scale={float(guidance_scale)}",
        "--config.experiment.skip_if_exists=false",
        f"--config.experiment.run_seqs={job_state['sequence_name']}",
        f"--config.experiment.save_path={job_state['pass1_dir']}",
        f"--config.video_model.model_name={BASE_MODEL_PATH}",
        f"--config.video_model.transformer_path={PASS1_MODEL_PATH}",
    ]

    accumulated = (
        "Launching Pass 1...\n"
        f"Using source-aligned settings: frames={effective_max_frames}, window={effective_window}, fps={effective_fps}\n"
    )
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    yield (
        accumulated,
        None,
        _download_link_html(None, "Open Pass 1 video in new tab"),
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )
    try:
        streamer = _run_streaming_command(command, APP_ROOT, env, log_path, accumulated, "Pass 1 inference")
        try:
            while True:
                accumulated = next(streamer)
                primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
                yield (
                    accumulated,
                    None,
                    _download_link_html(None, "Open Pass 1 video in new tab"),
                    _artifacts_markdown(job_state),
                    _workflow_markdown(job_state),
                    primary_action_update,
                    quadmask_button_update,
                    pass1_button_update,
                    pass2_button_update,
                )
        except StopIteration as stop:
            accumulated = stop.value or accumulated
    except Exception as exc:
        primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
        yield (
            (accumulated + "\n" + _format_failure("Pass 1", exc, log_path) + "\n")[-30000:],
            None,
            _download_link_html(None, "Open Pass 1 video in new tab"),
            _artifacts_markdown(job_state),
            _workflow_markdown(job_state),
            primary_action_update,
            quadmask_button_update,
            pass1_button_update,
            pass2_button_update,
        )
        return

    pass1_output = find_pass1_output(job_state)
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    yield (
        (accumulated + "\nPass 1 complete.\n")[-30000:],
        pass1_output,
        _download_link_html(pass1_output, "Open Pass 1 video in new tab"),
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def run_pass2(
    job_state: dict[str, Any],
    background_prompt: str,
    sample_height: int,
    sample_width: int,
    max_video_length: int,
    temporal_window: int,
    num_steps: int,
    guidance_scale: float,
) -> Generator[tuple[Any, ...], None, None]:
    if not job_state:
        raise gr.Error("Upload a video first.")
    if not Path(artifact_path(job_state, "quadmask")).exists():
        raise gr.Error("Generate or import the quadmask first.")
    if not find_pass1_output(job_state):
        raise gr.Error("Run Pass 1 first.")
    if not background_prompt.strip():
        raise gr.Error("Enter the VOID background prompt before running Pass 2.")

    write_prompt(job_state["prompt_path"], background_prompt)
    effective_max_frames, effective_window, effective_fps = _effective_inference_settings(
        job_state,
        max_video_length,
        temporal_window,
    )
    selected_pass1_dir = _prepare_selected_pass1_dir(job_state)

    log_path = Path(job_state["logs_dir"]) / "pass2.log"
    env = os.environ.copy()
    accumulated = (
        "Launching Pass 2...\n"
        f"Using source-aligned settings: frames={effective_max_frames}, window={effective_window}, fps={effective_fps}\n"
    )

    try:
        for message in _download_pass2_if_needed():
            accumulated = (accumulated + message)[-30000:]
            primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
            yield (
                accumulated,
                None,
                _download_link_html(None, "Open Pass 2 video in new tab"),
                _artifacts_markdown(job_state),
                _workflow_markdown(job_state),
                primary_action_update,
                quadmask_button_update,
                pass1_button_update,
                pass2_button_update,
            )
    except Exception as exc:
        primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
        yield (
            (accumulated + "\n" + _format_failure("Pass 2 checkpoint download", exc) + "\n")[-30000:],
            None,
            _download_link_html(None, "Open Pass 2 video in new tab"),
            _artifacts_markdown(job_state),
            _workflow_markdown(job_state),
            primary_action_update,
            quadmask_button_update,
            pass1_button_update,
            pass2_button_update,
        )
        return

    command = [
        PYTHON_BIN,
        str(VOID_ROOT / "inference" / "cogvideox_fun" / "inference_with_pass1_warped_noise.py"),
        "--video_name",
        job_state["sequence_name"],
        "--data_rootdir",
        job_state["data_root"],
        "--pass1_dir",
        selected_pass1_dir,
        "--output_dir",
        job_state["pass2_dir"],
        "--model_checkpoint",
        PASS2_MODEL_PATH,
        "--model_name",
        BASE_MODEL_PATH,
        "--height",
        str(int(sample_height)),
        "--width",
        str(int(sample_width)),
        "--max_video_length",
        str(effective_max_frames),
        "--temporal_window_size",
        str(effective_window),
        "--fps",
        str(effective_fps),
        "--guidance_scale",
        str(float(guidance_scale)),
        "--num_inference_steps",
        str(int(num_steps)),
    ]

    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    yield (
        accumulated,
        None,
        _download_link_html(None, "Open Pass 2 video in new tab"),
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )
    try:
        streamer = _run_streaming_command(command, APP_ROOT, env, log_path, accumulated, "Pass 2 inference")
        try:
            while True:
                accumulated = next(streamer)
                primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
                yield (
                    accumulated,
                    None,
                    _download_link_html(None, "Open Pass 2 video in new tab"),
                    _artifacts_markdown(job_state),
                    _workflow_markdown(job_state),
                    primary_action_update,
                    quadmask_button_update,
                    pass1_button_update,
                    pass2_button_update,
                )
        except StopIteration as stop:
            accumulated = stop.value or accumulated
    except Exception as exc:
        primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
        yield (
            (accumulated + "\n" + _format_failure("Pass 2", exc, log_path) + "\n")[-30000:],
            None,
            _download_link_html(None, "Open Pass 2 video in new tab"),
            _artifacts_markdown(job_state),
            _workflow_markdown(job_state),
            primary_action_update,
            quadmask_button_update,
            pass1_button_update,
            pass2_button_update,
        )
        return

    pass2_output = artifact_path(job_state, "pass2")
    primary_action_update, quadmask_button_update, pass1_button_update, pass2_button_update = _action_button_updates(job_state)
    yield (
        (accumulated + "\nPass 2 complete.\n")[-30000:],
        pass2_output if Path(pass2_output).exists() else None,
        _download_link_html(pass2_output if Path(pass2_output).exists() else None, "Open Pass 2 video in new tab"),
        _artifacts_markdown(job_state),
        _workflow_markdown(job_state),
        primary_action_update,
        quadmask_button_update,
        pass1_button_update,
        pass2_button_update,
    )


def run_primary_action(
    job_state: dict[str, Any],
    instruction: str,
    background_prompt: str,
    gemini_api_key: str,
    min_grid: int,
    multi_frame_grids: bool,
    sample_height: int,
    sample_width: int,
    max_video_length: int,
    temporal_window: int,
    pass1_steps: int,
    pass1_guidance: float,
    pass2_steps: int,
    pass2_guidance: float,
) -> Generator[tuple[Any, ...], None, None]:
    stage = _workflow_stage(job_state)

    if stage == "ready_quadmask":
        for (
            activity,
            black_mask_video,
            black_mask_link,
            quadmask_video,
            quadmask_link,
            points_config_path,
            artifacts_md,
            workflow_md,
            primary_action_update,
            quadmask_button_update,
            pass1_button_update,
            pass2_button_update,
        ) in run_quadmask_pipeline(job_state, instruction, background_prompt, gemini_api_key, min_grid, multi_frame_grids):
            yield (
                activity,
                black_mask_video,
                black_mask_link,
                quadmask_video,
                quadmask_link,
                points_config_path,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                artifacts_md,
                workflow_md,
                primary_action_update,
                quadmask_button_update,
                pass1_button_update,
                pass2_button_update,
            )
        return

    if stage == "ready_pass1":
        for (
            activity,
            pass1_video,
            pass1_link,
            artifacts_md,
            workflow_md,
            primary_action_update,
            quadmask_button_update,
            pass1_button_update,
            pass2_button_update,
        ) in run_pass1(
            job_state,
            background_prompt,
            sample_height,
            sample_width,
            max_video_length,
            temporal_window,
            pass1_steps,
            pass1_guidance,
        ):
            yield (
                activity,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                pass1_video,
                pass1_link,
                gr.update(),
                gr.update(),
                artifacts_md,
                workflow_md,
                primary_action_update,
                quadmask_button_update,
                pass1_button_update,
                pass2_button_update,
            )
        return

    if stage == "ready_pass2":
        for (
            activity,
            pass2_video,
            pass2_link,
            artifacts_md,
            workflow_md,
            primary_action_update,
            quadmask_button_update,
            pass1_button_update,
            pass2_button_update,
        ) in run_pass2(
            job_state,
            background_prompt,
            sample_height,
            sample_width,
            max_video_length,
            temporal_window,
            pass2_steps,
            pass2_guidance,
        ):
            yield (
                activity,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                pass2_video,
                pass2_link,
                artifacts_md,
                workflow_md,
                primary_action_update,
                quadmask_button_update,
                pass1_button_update,
                pass2_button_update,
            )
        return

    if stage == "needs_job":
        raise gr.Error("Load a source video or reopen an existing job first.")
    if stage == "needs_points":
        raise gr.Error("Add at least one point before continuing, or import an existing quadmask.")
    raise gr.Error("No further action is required right now.")


with gr.Blocks(title="VOID Runpod") as demo:
    job_state = gr.State({})

    with gr.Column(elem_classes=["workflow-shell"]):
        gr.Markdown(
            """
            <div class="hero">
              <h1>VOID on Runpod</h1>
              <p>Work top to bottom: load or reopen a job, describe the cleanup target, mark the object, generate a quadmask, then run Pass 1 and Pass 2.</p>
              <p class="compact-note">If you already have intermediate files, you can skip ahead by uploading an existing quadmask or Pass 1 result.</p>
            </div>
            """
        )

        with gr.Group(elem_classes=["step-card", "output-panel"]):
            gr.Markdown(
                """
                <div class="step-kicker">Current Job</div>
                <div class="step-title">Track The Active Workspace</div>
                <p>The active job name is the value you can reuse later to reopen the same workspace without reuploading.</p>
                """
            )
            active_job_md = gr.Markdown("**Active job:** `none`")
            job_summary = gr.Markdown("No job loaded.")
            artifacts_md = gr.Markdown("No artifacts yet.")
            workflow_md = gr.Markdown("**Next step:** Load a source video or reopen an existing job.")

        with gr.Group(elem_classes=["step-card"]):
            gr.Markdown(
                """
                <div class="step-kicker">Step 1</div>
                <div class="step-title">Start Or Reopen A Job</div>
                <p>Most users should upload a source video and optionally assign a reusable job name. If you already know the saved job name, reopen it instead.</p>
                """
            )
            with gr.Tabs():
                with gr.Tab("New Job"):
                    upload = gr.File(label="Source video", file_types=["video"], type="filepath")
                    with gr.Row():
                        job_name = gr.Textbox(
                            label="New job name",
                            placeholder="Optional reusable name, for example dog-splash-shot-1",
                            info="If you set this, the exact saved job name will be shown above as Active job.",
                        )
                        load_button = gr.Button("Load Video", variant="primary")
                with gr.Tab("Open Existing"):
                    with gr.Row():
                        existing_job_name = gr.Textbox(
                            label="Existing job name",
                            placeholder="Paste a previously shown active job name",
                        )
                        open_job_button = gr.Button("Open Existing Job")
            input_preview = gr.Video(label="Source video preview")

        with gr.Group(elem_classes=["step-card"]):
            gr.Markdown(
                """
                <div class="step-kicker">Step 2</div>
                <div class="step-title">Describe The Removal And Background</div>
                <p>Set the instruction for Gemini and describe what the scene should look like after the object and its effects are gone.</p>
                """
            )
            removal_instruction = gr.Textbox(
                label="Removal instruction for Gemini",
                value="remove the selected object",
                info="Used by the Gemini reasoning stage. Keep it concrete.",
            )
            background_prompt = gr.Textbox(
                label="VOID background prompt",
                placeholder="Describe what remains after the removed object and its effects are gone.",
                lines=3,
                info="Used by VOID Pass 1 and Pass 2.",
            )
            gemini_api_key = gr.Textbox(
                label="Gemini API key",
                type="password",
                placeholder="Paste your Gemini key here before generating the quadmask",
                info="Only required when you click Create / Regenerate Quadmask.",
            )

        with gr.Group(elem_classes=["step-card"]):
            gr.Markdown(
                """
                <div class="step-kicker">Step 3</div>
                <div class="step-title">Mark The Object Across Frames</div>
                <p>Move the slider, click directly on the object, and add points on as many frames as needed. Save the points JSON if you want an explicit copy of the config.</p>
                """
            )
            frame_image = gr.Image(
                label="Current frame: click to add positive points",
                interactive=True,
                type="numpy",
            )
            frame_slider = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
            frame_points_md = gr.Markdown("No frame loaded.")
            with gr.Row(elem_classes=["step-actions"]):
                undo_button = gr.Button("Undo point")
                clear_frame_button = gr.Button("Clear frame")
                clear_all_button = gr.Button("Clear all points")
                save_points_button = gr.Button("Save points JSON")

            with gr.Accordion("Mask Reasoning Options", open=False):
                min_grid = gr.Number(label="Grid density", value=8, precision=0)
                multi_frame_grids = gr.Checkbox(label="Use multi-frame Gemini grids", value=True)
                points_json = gr.JSON(label="Points config preview")
                points_config_path = gr.Textbox(label="Saved points config path", interactive=False)

        with gr.Accordion("Skip Ahead With Existing Files (Optional)", open=False):
            gr.Markdown(
                """
                <div class="step-kicker">Optional Shortcuts</div>
                <div class="step-title">Reuse Intermediate Files</div>
                <p>Most users should ignore this section and continue with the main workflow below. Open it only if you already generated a quadmask or Pass 1 result elsewhere.</p>
                """
            )
            with gr.Group(elem_classes=["step-card"]):
                gr.Markdown(
                    """
                    <div class="step-title">Use An Existing Quadmask</div>
                    <p>Upload a finished <code>quadmask_0.mp4</code> if you want to skip quadmask generation and go straight to Pass 1.</p>
                    """
                )
                quadmask_upload = gr.File(label="Existing quadmask video", file_types=["video"], type="filepath")
                import_quadmask_button = gr.Button("Use Uploaded Quadmask")
                import_quadmask_status = gr.Markdown("")
            with gr.Group(elem_classes=["step-card"]):
                gr.Markdown(
                    """
                    <div class="step-title">Use An Existing Pass 1 Output</div>
                    <p>Upload a Pass 1 output if you want to skip directly to Pass 2.</p>
                    """
                )
                pass1_upload = gr.File(label="Existing Pass 1 video", file_types=["video"], type="filepath")
                import_pass1_button = gr.Button("Use Uploaded Pass 1")
                import_pass1_status = gr.Markdown("")

        with gr.Group(elem_classes=["step-card"]):
            gr.Markdown(
                """
                <div class="step-kicker">Step 4</div>
                <div class="step-title">Run Quadmask, Pass 1, And Pass 2</div>
                <p>Generate the quadmask first. After that, run Pass 1, then Pass 2. If you uploaded an existing Pass 1 video, you can jump straight to Pass 2.</p>
                """
            )
            with gr.Accordion("Inference Settings", open=False):
                with gr.Row():
                    sample_height = gr.Number(label="Height", value=384, precision=0)
                    sample_width = gr.Number(label="Width", value=672, precision=0)
                    max_video_length = gr.Number(label="Max frames", value=197, precision=0)
                    temporal_window = gr.Number(label="Temporal window", value=85, precision=0)
                with gr.Row():
                    pass1_steps = gr.Number(label="Pass 1 steps", value=50, precision=0)
                    pass1_guidance = gr.Number(label="Pass 1 guidance", value=1.0)
                    pass2_steps = gr.Number(label="Pass 2 steps", value=50, precision=0)
                    pass2_guidance = gr.Number(label="Pass 2 guidance", value=6.0)
            primary_action_button = gr.Button("Load A Job First", variant="primary", interactive=False)
            with gr.Accordion("Manual Stage Actions", open=False):
                with gr.Row(elem_classes=["step-actions"]):
                    quadmask_button = gr.Button("Create / Regenerate Quadmask", interactive=False)
                    pass1_button = gr.Button("Run Pass 1", interactive=False)
                    pass2_button = gr.Button("Run Pass 2", interactive=False)
            activity_log = gr.Textbox(label="Pipeline activity", lines=16, max_lines=22, autoscroll=True)

        with gr.Accordion("Step 5: Review Quadmask Outputs", open=False):
            with gr.Group(elem_classes=["step-card"]):
                gr.Markdown(
                    """
                    <div class="step-title">Quadmask Outputs</div>
                    <p>Use these previews to confirm the quadmask stage before moving into inference.</p>
                    """
                )
                with gr.Row():
                    with gr.Column():
                        black_mask_video = gr.Video(label="SAM2 primary-object mask")
                        black_mask_link = gr.HTML(_download_link_html(None, "Open SAM2 mask in new tab"))
                    with gr.Column():
                        quadmask_video = gr.Video(label="Quadmask preview")
                        quadmask_link = gr.HTML(_download_link_html(None, "Open quadmask in new tab"))

        with gr.Accordion("Step 6: Review Pass 1 Output", open=False):
            with gr.Group(elem_classes=["step-card"]):
                gr.Markdown(
                    """
                    <div class="step-title">Pass 1 Output</div>
                    <p>Check the Pass 1 output before deciding whether to continue to Pass 2.</p>
                    """
                )
                pass1_video = gr.Video(label="Pass 1 output")
                pass1_link = gr.HTML(_download_link_html(None, "Open Pass 1 video in new tab"))

        with gr.Accordion("Step 7: Review Pass 2 Output", open=False):
            with gr.Group(elem_classes=["step-card"]):
                gr.Markdown(
                    """
                    <div class="step-title">Pass 2 Output</div>
                    <p>Pass 2 refines temporal consistency using the selected Pass 1 result.</p>
                    """
                )
                pass2_video = gr.Video(label="Pass 2 output")
                pass2_link = gr.HTML(_download_link_html(None, "Open Pass 2 video in new tab"))

    load_button.click(
        prepare_job,
        inputs=[upload, job_name],
        outputs=[
            job_state,
            active_job_md,
            job_summary,
            frame_image,
            frame_slider,
            frame_points_md,
            points_json,
            input_preview,
            artifacts_md,
            workflow_md,
            points_config_path,
            removal_instruction,
            background_prompt,
            min_grid,
            multi_frame_grids,
            activity_log,
            import_quadmask_status,
            import_pass1_status,
            black_mask_video,
            black_mask_link,
            quadmask_video,
            quadmask_link,
            pass1_video,
            pass1_link,
            pass2_video,
            pass2_link,
            primary_action_button,
            quadmask_button,
            pass1_button,
            pass2_button,
        ],
    )

    open_job_button.click(
        open_existing_job,
        inputs=[existing_job_name],
        outputs=[
            job_state,
            active_job_md,
            job_summary,
            frame_image,
            frame_slider,
            frame_points_md,
            points_json,
            input_preview,
            artifacts_md,
            workflow_md,
            points_config_path,
            removal_instruction,
            background_prompt,
            min_grid,
            multi_frame_grids,
            activity_log,
            import_quadmask_status,
            import_pass1_status,
            black_mask_video,
            black_mask_link,
            quadmask_video,
            quadmask_link,
            pass1_video,
            pass1_link,
            pass2_video,
            pass2_link,
            primary_action_button,
            quadmask_button,
            pass1_button,
            pass2_button,
        ],
    )

    frame_slider.change(
        change_frame,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[frame_image, points_json, frame_points_md],
    )

    frame_image.select(
        add_point,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary, frame_points_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    undo_button.click(
        undo_last_point,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary, frame_points_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    clear_frame_button.click(
        clear_frame_points,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary, frame_points_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    clear_all_button.click(
        clear_all_points,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary, frame_points_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    save_points_button.click(
        save_points_config,
        inputs=[job_state, removal_instruction, min_grid, multi_frame_grids],
        outputs=[points_json, points_config_path],
    )

    import_quadmask_button.click(
        import_existing_quadmask,
        inputs=[job_state, quadmask_upload],
        outputs=[import_quadmask_status, activity_log, quadmask_video, quadmask_link, artifacts_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    import_pass1_button.click(
        import_existing_pass1,
        inputs=[job_state, pass1_upload],
        outputs=[import_pass1_status, activity_log, pass1_video, pass1_link, artifacts_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    primary_action_button.click(
        run_primary_action,
        inputs=[
            job_state,
            removal_instruction,
            background_prompt,
            gemini_api_key,
            min_grid,
            multi_frame_grids,
            sample_height,
            sample_width,
            max_video_length,
            temporal_window,
            pass1_steps,
            pass1_guidance,
            pass2_steps,
            pass2_guidance,
        ],
        outputs=[
            activity_log,
            black_mask_video,
            black_mask_link,
            quadmask_video,
            quadmask_link,
            points_config_path,
            pass1_video,
            pass1_link,
            pass2_video,
            pass2_link,
            artifacts_md,
            workflow_md,
            primary_action_button,
            quadmask_button,
            pass1_button,
            pass2_button,
        ],
    )

    removal_instruction.change(
        refresh_points_preview,
        inputs=[job_state, removal_instruction, min_grid, multi_frame_grids],
        outputs=[points_json],
    )

    min_grid.change(
        refresh_points_preview,
        inputs=[job_state, removal_instruction, min_grid, multi_frame_grids],
        outputs=[points_json],
    )

    multi_frame_grids.change(
        refresh_points_preview,
        inputs=[job_state, removal_instruction, min_grid, multi_frame_grids],
        outputs=[points_json],
    )

    quadmask_button.click(
        run_quadmask_pipeline,
        inputs=[
            job_state,
            removal_instruction,
            background_prompt,
            gemini_api_key,
            min_grid,
            multi_frame_grids,
        ],
        outputs=[activity_log, black_mask_video, black_mask_link, quadmask_video, quadmask_link, points_config_path, artifacts_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    pass1_button.click(
        run_pass1,
        inputs=[job_state, background_prompt, sample_height, sample_width, max_video_length, temporal_window, pass1_steps, pass1_guidance],
        outputs=[activity_log, pass1_video, pass1_link, artifacts_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )

    pass2_button.click(
        run_pass2,
        inputs=[job_state, background_prompt, sample_height, sample_width, max_video_length, temporal_window, pass2_steps, pass2_guidance],
        outputs=[activity_log, pass2_video, pass2_link, artifacts_md, workflow_md, primary_action_button, quadmask_button, pass1_button, pass2_button],
    )


if __name__ == "__main__":
    ensure_workspace()
    demo.queue(default_concurrency_limit=1).launch(
        css=CUSTOM_CSS,
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=DEFAULT_GRADIO_PORT,
        allowed_paths=[str((WORKSPACE_DIR / "jobs").resolve())],
    )
