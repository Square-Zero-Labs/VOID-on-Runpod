from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def slugify(value: str, default: str = "job") -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return normalized or default


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _job_paths_from_id(workspace_dir: Path, job_id: str, create: bool = True) -> dict[str, str]:
    job_dir = workspace_dir / "jobs" / job_id
    if create:
        ensure_dir(job_dir)
    sequence_name = "sequence"
    data_root = job_dir / "data"
    sequence_dir = data_root / sequence_name

    if create:
        ensure_dir(data_root)
        ensure_dir(sequence_dir)

    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "data_root": str(data_root if not create else ensure_dir(data_root)),
        "sequence_name": sequence_name,
        "sequence_dir": str(sequence_dir if not create else ensure_dir(sequence_dir)),
        "frames_dir": str(job_dir / "frames" if not create else ensure_dir(job_dir / "frames")),
        "logs_dir": str(job_dir / "logs" if not create else ensure_dir(job_dir / "logs")),
        "pass1_dir": str(job_dir / "pass1_outputs" if not create else ensure_dir(job_dir / "pass1_outputs")),
        "pass2_dir": str(job_dir / "pass2_outputs" if not create else ensure_dir(job_dir / "pass2_outputs")),
        "input_video_path": str(sequence_dir / "input_video.mp4"),
        "prompt_path": str(sequence_dir / "prompt.json"),
        "config_path": str(job_dir / "config_points.json"),
    }


def make_job_paths(workspace_dir: Path, upload_name: str, requested_name: str | None = None) -> dict[str, str]:
    upload_stem = Path(upload_name).stem

    if requested_name and requested_name.strip():
        job_id = slugify(requested_name, default="void-job")
        existing_job_dir = workspace_dir / "jobs" / job_id
        if existing_job_dir.exists():
            raise FileExistsError(
                f"Job `{job_id}` already exists. Use Open Existing Job instead of uploading again."
            )
        return _job_paths_from_id(workspace_dir, job_id, create=True)

    base_name = slugify(upload_stem, default="void-job")
    job_id = f"{base_name}-{uuid.uuid4().hex[:8]}"
    return _job_paths_from_id(workspace_dir, job_id, create=True)


def list_existing_jobs(workspace_dir: Path) -> list[str]:
    jobs_root = workspace_dir / "jobs"
    if not jobs_root.exists():
        return []

    job_dirs = [path for path in jobs_root.iterdir() if path.is_dir()]
    job_dirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.name for path in job_dirs]


def _video_metadata(video_path: str) -> dict[str, Any]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    return {
        "fps": float(fps),
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }


def _frame_paths(frames_dir: str) -> list[str]:
    return [str(path) for path in sorted(Path(frames_dir).glob("frame_*.jpg"))]


def copy_uploaded_video(upload_path: str, destination_path: str) -> None:
    shutil.copy2(upload_path, destination_path)


def extract_frames(video_path: str, frames_dir: str) -> dict[str, Any]:
    frames_root = Path(frames_dir)
    ensure_dir(frames_root)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 12.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_paths: list[str] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_path = frames_root / f"frame_{index:05d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(str(frame_path))
        index += 1

    capture.release()

    if not frame_paths:
        raise RuntimeError("No frames could be extracted from the uploaded video.")

    return {
        "frame_paths": frame_paths,
        "fps": float(fps),
        "width": width,
        "height": height,
        "total_frames": len(frame_paths) if total_frames <= 0 else total_frames,
    }


def load_existing_job(workspace_dir: Path, requested_name: str) -> dict[str, Any]:
    if not requested_name or not requested_name.strip():
        raise FileNotFoundError("Enter a job name to reopen an existing job.")

    requested = requested_name.strip()
    candidates = [requested]
    slugified = slugify(requested, default="void-job")
    if slugified not in candidates:
        candidates.append(slugified)

    job_dir: Path | None = None
    for candidate in candidates:
        probe = workspace_dir / "jobs" / candidate
        if probe.exists():
            job_dir = probe
            break

    if job_dir is None:
        raise FileNotFoundError(f"No saved job found for `{requested}`.")

    job_state = _job_paths_from_id(workspace_dir, job_dir.name, create=False)
    input_video_path = Path(job_state["input_video_path"])
    if not input_video_path.exists():
        raise FileNotFoundError(f"Saved job `{job_dir.name}` is missing its source video.")

    frame_paths = _frame_paths(job_state["frames_dir"])
    if frame_paths:
        metadata = _video_metadata(job_state["input_video_path"])
        metadata["frame_paths"] = frame_paths
        if metadata["total_frames"] <= 0:
            metadata["total_frames"] = len(frame_paths)
    else:
        metadata = extract_frames(job_state["input_video_path"], job_state["frames_dir"])

    config_path = Path(job_state["config_path"])
    points_by_frame: dict[str, list[list[int]]] = {}
    removal_instruction = "remove the selected object"
    min_grid = 8
    multi_frame_grids = True
    if config_path.exists():
        config_payload = json.load(open(config_path, "r", encoding="utf-8"))
        video_payload = (config_payload.get("videos") or [{}])[0]
        points_by_frame = video_payload.get("primary_points_by_frame", {}) or {}
        removal_instruction = video_payload.get("instruction") or removal_instruction
        min_grid = int(video_payload.get("min_grid", min_grid))
        multi_frame_grids = bool(video_payload.get("multi_frame_grids", multi_frame_grids))

    background_prompt = ""
    prompt_path = Path(job_state["prompt_path"])
    if prompt_path.exists():
        prompt_payload = json.load(open(prompt_path, "r", encoding="utf-8"))
        background_prompt = prompt_payload.get("bg", "") or ""

    job_state.update(metadata)
    job_state["points_by_frame"] = points_by_frame
    job_state["removal_instruction"] = removal_instruction
    job_state["background_prompt"] = background_prompt
    job_state["min_grid"] = min_grid
    job_state["multi_frame_grids"] = multi_frame_grids
    return job_state


def overlay_points(frame_path: str, points_by_frame: dict[str, list[list[int]]], frame_index: int) -> np.ndarray:
    frame = cv2.imread(frame_path)
    if frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_path}")

    frame_points = points_by_frame.get(str(frame_index), [])
    for idx, (x, y) in enumerate(frame_points, start=1):
        cv2.circle(frame, (int(x), int(y)), 12, (32, 241, 255), thickness=-1)
        cv2.circle(frame, (int(x), int(y)), 16, (0, 0, 0), thickness=2)
        cv2.putText(
            frame,
            str(idx),
            (int(x) + 14, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_points_json(job_state: dict[str, Any], instruction: str, min_grid: int, multi_frame_grids: bool) -> dict[str, Any]:
    points_by_frame = {
        key: value
        for key, value in sorted(job_state.get("points_by_frame", {}).items(), key=lambda item: int(item[0]))
        if value
    }

    return {
        "videos": [
            {
                "video_path": job_state["input_video_path"],
                "output_dir": job_state["sequence_dir"],
                "instruction": instruction.strip(),
                "primary_points_by_frame": points_by_frame,
                "min_grid": int(min_grid),
                "multi_frame_grids": bool(multi_frame_grids),
            }
        ]
    }


def write_json(path: str, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_prompt(path: str, prompt: str) -> None:
    write_json(path, {"bg": prompt.strip()})


def summarize_state(job_state: dict[str, Any] | None) -> str:
    if not job_state:
        return "No job loaded."

    total_points = sum(len(points) for points in job_state.get("points_by_frame", {}).values())
    frames_with_points = sorted(int(key) for key, value in job_state.get("points_by_frame", {}).items() if value)
    frames_summary = ", ".join(str(value) for value in frames_with_points[:12]) or "none"
    if len(frames_with_points) > 12:
        frames_summary += ", ..."

    return (
        f"Job: `{job_state['job_id']}`\n\n"
        f"- Video: `{Path(job_state['input_video_path']).name}`\n"
        f"- Frames extracted: `{len(job_state['frame_paths'])}`\n"
        f"- Resolution: `{job_state['width']}x{job_state['height']}`\n"
        f"- FPS: `{job_state['fps']:.2f}`\n"
        f"- Total selected points: `{total_points}`\n"
        f"- Frames with points: `{frames_summary}`\n"
        f"- Workspace: `{job_state['job_dir']}`"
    )


def artifact_path(job_state: dict[str, Any], name: str) -> str:
    sequence_dir = Path(job_state["sequence_dir"])
    pass1_dir = Path(job_state["pass1_dir"])
    pass2_dir = Path(job_state["pass2_dir"])
    mapping = {
        "quadmask": str(sequence_dir / "quadmask_0.mp4"),
        "grey_mask": str(sequence_dir / "grey_mask.mp4"),
        "black_mask": str(sequence_dir / "black_mask.mp4"),
        "vlm_analysis": str(sequence_dir / "vlm_analysis.json"),
        "pass2": str(pass2_dir / f"{job_state['sequence_name']}_warped_noise_inference.mp4"),
    }
    if name == "pass1":
        return find_pass1_output(job_state) or ""
    return mapping[name]


def find_pass1_output(job_state: dict[str, Any]) -> str | None:
    pass1_dir = Path(job_state["pass1_dir"])
    sequence_name = job_state["sequence_name"]
    candidates = [candidate for candidate in pass1_dir.glob(f"{sequence_name}-fg=-1-*.mp4") if "_tuple" not in candidate.name]
    if not candidates:
        return None

    def candidate_sort_key(path: Path) -> tuple[int, int, str]:
        stem_suffix = path.stem.removeprefix(f"{sequence_name}-fg=-1-")
        if stem_suffix == "imported":
            return (2, 0, path.name)
        if stem_suffix.isdigit():
            return (1, int(stem_suffix), path.name)
        return (0, 0, path.name)

    return str(max(candidates, key=candidate_sort_key))
