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


def make_job_paths(workspace_dir: Path, upload_name: str, requested_name: str | None = None) -> dict[str, str]:
    upload_stem = Path(upload_name).stem
    base_name = slugify(requested_name or upload_stem, default="void-job")
    job_id = f"{base_name}-{uuid.uuid4().hex[:8]}"
    job_dir = ensure_dir(workspace_dir / "jobs" / job_id)
    sequence_name = "sequence"
    data_root = ensure_dir(job_dir / "data")
    sequence_dir = ensure_dir(data_root / sequence_name)

    return {
        "job_id": job_id,
        "job_dir": str(job_dir),
        "data_root": str(data_root),
        "sequence_name": sequence_name,
        "sequence_dir": str(sequence_dir),
        "frames_dir": str(ensure_dir(job_dir / "frames")),
        "logs_dir": str(ensure_dir(job_dir / "logs")),
        "pass1_dir": str(ensure_dir(job_dir / "pass1_outputs")),
        "pass2_dir": str(ensure_dir(job_dir / "pass2_outputs")),
        "input_video_path": str(sequence_dir / "input_video.mp4"),
        "prompt_path": str(sequence_dir / "prompt.json"),
        "config_path": str(job_dir / "config_points.json"),
    }


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
    candidates = sorted(pass1_dir.glob(f"{sequence_name}-fg=-1-*.mp4"))
    for candidate in candidates:
        if "_tuple" not in candidate.name:
            return str(candidate)
    return None
