from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Generator

import gradio as gr

from void_runpod.app_state import (
    artifact_path,
    build_points_json,
    copy_uploaded_video,
    extract_frames,
    find_pass1_output,
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
GREY_MASK_SEGMENTATION_MODEL = os.environ.get("VOID_GREY_MASK_SEGMENTATION_MODEL", "langsam")


CUSTOM_CSS = """
:root {
  --bg: #f1ece2;
  --panel: #f9f6ef;
  --border: #c5b59d;
  --ink: #1f1f1a;
  --accent: #915b2b;
  --accent-2: #5f7f63;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(145, 91, 43, 0.10), transparent 28%),
    radial-gradient(circle at top right, rgba(95, 127, 99, 0.14), transparent 24%),
    linear-gradient(180deg, #f8f3ea 0%, #efe7d7 100%);
  color: var(--ink);
  font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
}

.block, .gr-panel, .gr-box, .gr-accordion {
  background: rgba(249, 246, 239, 0.92) !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 10px 28px rgba(56, 45, 24, 0.08);
}

h1, h2, h3 {
  font-family: "IBM Plex Serif", Georgia, serif;
}
"""


def ensure_workspace() -> None:
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


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

    quadmask = Path(artifact_path(job_state, "quadmask"))
    pass1 = Path(artifact_path(job_state, "pass1")) if artifact_path(job_state, "pass1") else None
    pass2 = Path(artifact_path(job_state, "pass2"))
    vlm_analysis = Path(artifact_path(job_state, "vlm_analysis"))

    lines = [
        f"- Quadmask: `{'ready' if quadmask.exists() else 'missing'}`",
        f"- VLM analysis: `{'ready' if vlm_analysis.exists() else 'missing'}`",
        f"- Pass 1 output: `{'ready' if pass1 and pass1.exists() else 'missing'}`",
        f"- Pass 2 output: `{'ready' if pass2.exists() else 'missing'}`",
    ]
    return "\n".join(lines)


def prepare_job(upload_path: str, job_name: str | None) -> tuple[dict[str, Any], str, Any, Any, Any, Any, Any, str, str, str, str, Any, Any, Any]:
    ensure_workspace()
    if not upload_path:
        raise gr.Error("Upload a video first.")

    job_state = make_job_paths(WORKSPACE_DIR, upload_path, job_name)
    copy_uploaded_video(upload_path, job_state["input_video_path"])
    metadata = extract_frames(job_state["input_video_path"], job_state["frames_dir"])
    job_state.update(metadata)
    job_state["points_by_frame"] = {}

    frame_image = _render_frame(job_state, 0)
    slider_update = gr.update(minimum=0, maximum=max(len(job_state["frame_paths"]) - 1, 0), value=0, step=1)
    points_preview = _points_json_preview(job_state, "remove the selected object", 8, True)

    return (
        job_state,
        summarize_state(job_state),
        frame_image,
        slider_update,
        points_preview,
        job_state["input_video_path"],
        _artifacts_markdown(job_state),
        "",
        "",
        "",
        "",
        None,
        None,
        None,
    )


def change_frame(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[Any, dict[str, Any]]:
    return _render_frame(job_state, frame_index), _points_json_preview(job_state, instruction, min_grid, multi_frame_grids)


def add_point(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool, evt: gr.SelectData) -> tuple[dict[str, Any], Any, dict[str, Any], str]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    if evt.index is None:
        raise gr.Error("Click directly on the frame image.")

    x, y = evt.index
    frame_key = str(int(frame_index))
    job_state["points_by_frame"].setdefault(frame_key, []).append([int(x), int(y)])
    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
    )


def undo_last_point(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[dict[str, Any], Any, dict[str, Any], str]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    frame_key = str(int(frame_index))
    frame_points = job_state["points_by_frame"].get(frame_key, [])
    if frame_points:
        frame_points.pop()
    if not frame_points and frame_key in job_state["points_by_frame"]:
        job_state["points_by_frame"].pop(frame_key, None)

    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
    )


def clear_frame_points(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[dict[str, Any], Any, dict[str, Any], str]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    job_state["points_by_frame"].pop(str(int(frame_index)), None)
    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
    )


def clear_all_points(job_state: dict[str, Any], frame_index: int, instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[dict[str, Any], Any, dict[str, Any], str]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    job_state["points_by_frame"] = {}
    return (
        job_state,
        _render_frame(job_state, int(frame_index)),
        _points_json_preview(job_state, instruction, min_grid, multi_frame_grids),
        summarize_state(job_state),
    )


def save_points_config(job_state: dict[str, Any], instruction: str, min_grid: int, multi_frame_grids: bool) -> tuple[dict[str, Any], str]:
    if not job_state:
        raise gr.Error("Upload a video first.")

    payload = build_points_json(job_state, instruction, min_grid, multi_frame_grids)
    write_json(job_state["config_path"], payload)
    return payload, job_state["config_path"]


def refresh_points_preview(job_state: dict[str, Any], instruction: str, min_grid: int, multi_frame_grids: bool) -> dict[str, Any]:
    return _points_json_preview(job_state, instruction, min_grid, multi_frame_grids)


def _run_streaming_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path, accumulated: str) -> Generator[str, None, str]:
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
            raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")

    return accumulated


def _quadmask_outputs(job_state: dict[str, Any]) -> tuple[Any, str, str]:
    quadmask = artifact_path(job_state, "quadmask")
    vlm_analysis = artifact_path(job_state, "vlm_analysis")
    return (
        quadmask if Path(quadmask).exists() else None,
        vlm_analysis if Path(vlm_analysis).exists() else "",
        _artifacts_markdown(job_state),
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


def run_quadmask_pipeline(
    job_state: dict[str, Any],
    instruction: str,
    background_prompt: str,
    gemini_api_key: str,
    min_grid: int,
    multi_frame_grids: bool,
) -> Generator[tuple[str, Any, str, str], None, None]:
    if not job_state:
        raise gr.Error("Upload a video first.")
    if not gemini_api_key.strip():
        raise gr.Error("Enter a Gemini API key.")
    if not background_prompt.strip():
        raise gr.Error("Enter the clean-background prompt used by VOID inference.")
    if not any(job_state["points_by_frame"].values()):
        raise gr.Error("Add at least one point before running the quadmask pipeline.")

    payload = build_points_json(job_state, instruction, min_grid, multi_frame_grids)
    write_json(job_state["config_path"], payload)
    write_prompt(job_state["prompt_path"], background_prompt)

    log_path = Path(job_state["logs_dir"]) / "quadmask.log"
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = gemini_api_key.strip()

    commands = [
        [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage1_sam2_segmentation.py"),
            "--config",
            job_state["config_path"],
            "--sam2-checkpoint",
            SAM2_CHECKPOINT_PATH,
            "--device",
            "cuda",
        ],
        [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage2_vlm_analysis.py"),
            "--config",
            job_state["config_path"],
        ],
        [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage3a_generate_grey_masks_v2.py"),
            "--config",
            job_state["config_path"],
            "--segmentation-model",
            GREY_MASK_SEGMENTATION_MODEL,
        ],
        [
            PYTHON_BIN,
            str(VOID_ROOT / "VLM-MASK-REASONER" / "stage4_combine_masks.py"),
            "--config",
            job_state["config_path"],
        ],
    ]

    accumulated = ""
    yield "Preparing quadmask pipeline...\n", None, job_state["config_path"], _artifacts_markdown(job_state)
    for command in commands:
        streamer = _run_streaming_command(command, APP_ROOT, env, log_path, accumulated)
        try:
            while True:
                accumulated = next(streamer)
                yield accumulated, None, job_state["config_path"], _artifacts_markdown(job_state)
        except StopIteration as stop:
            accumulated = stop.value or accumulated

    quadmask_video, _, artifacts_md = _quadmask_outputs(job_state)
    success_log = (accumulated + "\nQuadmask pipeline complete.\n")[-30000:]
    yield success_log, quadmask_video, job_state["config_path"], artifacts_md


def run_pass1(
    job_state: dict[str, Any],
    sample_height: int,
    sample_width: int,
    max_video_length: int,
    temporal_window: int,
    num_steps: int,
    guidance_scale: float,
) -> Generator[tuple[str, Any, str], None, None]:
    if not job_state:
        raise gr.Error("Upload a video first.")
    if not Path(artifact_path(job_state, "quadmask")).exists():
        raise gr.Error("Generate the quadmask first.")

    log_path = Path(job_state["logs_dir"]) / "pass1.log"
    env = os.environ.copy()
    command = [
        PYTHON_BIN,
        str(VOID_ROOT / "inference" / "cogvideox_fun" / "predict_v2v.py"),
        "--config",
        str(VOID_ROOT / "config" / "quadmask_cogvideox.py"),
        f"--config.data.data_rootdir={job_state['data_root']}",
        f"--config.data.sample_size={sample_height}x{sample_width}",
        f"--config.data.max_video_length={int(max_video_length)}",
        f"--config.video_model.temporal_window_size={int(temporal_window)}",
        f"--config.video_model.num_inference_steps={int(num_steps)}",
        f"--config.video_model.guidance_scale={float(guidance_scale)}",
        "--config.experiment.skip_if_exists=false",
        f"--config.experiment.run_seqs={job_state['sequence_name']}",
        f"--config.experiment.save_path={job_state['pass1_dir']}",
        f"--config.video_model.model_name={BASE_MODEL_PATH}",
        f"--config.video_model.transformer_path={PASS1_MODEL_PATH}",
    ]

    accumulated = "Launching Pass 1...\n"
    yield accumulated, None, _artifacts_markdown(job_state)
    streamer = _run_streaming_command(command, APP_ROOT, env, log_path, accumulated)
    try:
        while True:
            accumulated = next(streamer)
            yield accumulated, None, _artifacts_markdown(job_state)
    except StopIteration as stop:
        accumulated = stop.value or accumulated

    pass1_output = find_pass1_output(job_state)
    yield (accumulated + "\nPass 1 complete.\n")[-30000:], pass1_output, _artifacts_markdown(job_state)


def run_pass2(
    job_state: dict[str, Any],
    sample_height: int,
    sample_width: int,
    max_video_length: int,
    temporal_window: int,
    num_steps: int,
    guidance_scale: float,
) -> Generator[tuple[str, Any, str], None, None]:
    if not job_state:
        raise gr.Error("Upload a video first.")
    if not find_pass1_output(job_state):
        raise gr.Error("Run Pass 1 first.")

    log_path = Path(job_state["logs_dir"]) / "pass2.log"
    env = os.environ.copy()
    accumulated = "Launching Pass 2...\n"

    for message in _download_pass2_if_needed():
        accumulated = (accumulated + message)[-30000:]
        yield accumulated, None, _artifacts_markdown(job_state)

    command = [
        PYTHON_BIN,
        str(VOID_ROOT / "inference" / "cogvideox_fun" / "inference_with_pass1_warped_noise.py"),
        "--video_name",
        job_state["sequence_name"],
        "--data_rootdir",
        job_state["data_root"],
        "--pass1_dir",
        job_state["pass1_dir"],
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
        str(int(max_video_length)),
        "--temporal_window_size",
        str(int(temporal_window)),
        "--guidance_scale",
        str(float(guidance_scale)),
        "--num_inference_steps",
        str(int(num_steps)),
    ]

    yield accumulated, None, _artifacts_markdown(job_state)
    streamer = _run_streaming_command(command, APP_ROOT, env, log_path, accumulated)
    try:
        while True:
            accumulated = next(streamer)
            yield accumulated, None, _artifacts_markdown(job_state)
    except StopIteration as stop:
        accumulated = stop.value or accumulated

    pass2_output = artifact_path(job_state, "pass2")
    yield (accumulated + "\nPass 2 complete.\n")[-30000:], pass2_output if Path(pass2_output).exists() else None, _artifacts_markdown(job_state)


with gr.Blocks(css=CUSTOM_CSS, title="VOID Runpod") as demo:
    job_state = gr.State({})

    gr.Markdown(
        """
        # VOID on Runpod
        Browser workflow for point selection, quadmask generation, Pass 1 inpainting, and Pass 2 refinement.

        Target profile: A40 / L40 at the upstream default resolution `384x672`.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Input video", file_types=["video"], type="filepath")
            job_name = gr.Textbox(label="Job name", placeholder="Optional label")
            load_button = gr.Button("Load Video", variant="primary")
            input_preview = gr.Video(label="Uploaded video preview")
            job_summary = gr.Markdown("No job loaded.")
            artifacts_md = gr.Markdown("No artifacts yet.")

        with gr.Column(scale=2):
            with gr.Group():
                removal_instruction = gr.Textbox(
                    label="Removal instruction",
                    value="remove the selected object",
                    info="Used by the Gemini reasoning stage. Keep it concrete.",
                )
                background_prompt = gr.Textbox(
                    label="Clean background prompt",
                    placeholder="Describe the scene after the object and its effects are removed.",
                    lines=3,
                )
                gemini_api_key = gr.Textbox(
                    label="Gemini API key",
                    type="password",
                    info="Used only for the VLM mask reasoning stage.",
                )

    with gr.Row():
        with gr.Column(scale=3):
            frame_image = gr.Image(
                label="Click to add points on the current frame",
                interactive=True,
                type="numpy",
            )
            frame_slider = gr.Slider(label="Frame", minimum=0, maximum=0, step=1, value=0)
            with gr.Row():
                undo_button = gr.Button("Undo point")
                clear_frame_button = gr.Button("Clear frame")
                clear_all_button = gr.Button("Clear all points")
                save_points_button = gr.Button("Save points JSON")
        with gr.Column(scale=2):
            min_grid = gr.Number(label="Grid density", value=8, precision=0)
            multi_frame_grids = gr.Checkbox(label="Use multi-frame Gemini grids", value=True)
            points_json = gr.JSON(label="Points config preview")
            points_config_path = gr.Textbox(label="Saved points config", interactive=False)

    with gr.Row():
        quadmask_button = gr.Button("Create / Regenerate Quadmask", variant="primary")
        pass1_button = gr.Button("Run Pass 1")
        pass2_button = gr.Button("Run Pass 2")

    with gr.Accordion("Inference settings", open=False):
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

    with gr.Row():
        quadmask_log = gr.Textbox(label="Quadmask log", lines=18, max_lines=22, autoscroll=True)
        pass1_log = gr.Textbox(label="Pass 1 log", lines=18, max_lines=22, autoscroll=True)
        pass2_log = gr.Textbox(label="Pass 2 log", lines=18, max_lines=22, autoscroll=True)

    with gr.Row():
        quadmask_video = gr.Video(label="Quadmask preview")
        pass1_video = gr.Video(label="Pass 1 output")
        pass2_video = gr.Video(label="Pass 2 output")

    load_button.click(
        prepare_job,
        inputs=[upload, job_name],
        outputs=[
            job_state,
            job_summary,
            frame_image,
            frame_slider,
            points_json,
            input_preview,
            artifacts_md,
            points_config_path,
            quadmask_log,
            pass1_log,
            pass2_log,
            quadmask_video,
            pass1_video,
            pass2_video,
        ],
    )

    frame_slider.change(
        change_frame,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[frame_image, points_json],
    )

    frame_image.select(
        add_point,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary],
    )

    undo_button.click(
        undo_last_point,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary],
    )

    clear_frame_button.click(
        clear_frame_points,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary],
    )

    clear_all_button.click(
        clear_all_points,
        inputs=[job_state, frame_slider, removal_instruction, min_grid, multi_frame_grids],
        outputs=[job_state, frame_image, points_json, job_summary],
    )

    save_points_button.click(
        save_points_config,
        inputs=[job_state, removal_instruction, min_grid, multi_frame_grids],
        outputs=[points_json, points_config_path],
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
        outputs=[quadmask_log, quadmask_video, points_config_path, artifacts_md],
    )

    pass1_button.click(
        run_pass1,
        inputs=[job_state, sample_height, sample_width, max_video_length, temporal_window, pass1_steps, pass1_guidance],
        outputs=[pass1_log, pass1_video, artifacts_md],
    )

    pass2_button.click(
        run_pass2,
        inputs=[job_state, sample_height, sample_width, max_video_length, temporal_window, pass2_steps, pass2_guidance],
        outputs=[pass2_log, pass2_video, artifacts_md],
    )


if __name__ == "__main__":
    ensure_workspace()
    demo.queue(default_concurrency_limit=1).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=DEFAULT_GRADIO_PORT,
        show_api=False,
    )
