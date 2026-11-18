"""Sequential OpenCV/FFmpeg decoding; valid decoded frames define indices."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .provenance import digest


def decoder_identity() -> dict:
    return {"backend": "opencv_ffmpeg_sequential", "opencv": cv2.__version__,
            "build_sha256": digest(cv2.getBuildInformation()), "index_policy": "valid_decoded_sequence_v1"}


def midpoint_indices(n: int, count: int = 16) -> np.ndarray:
    if not isinstance(n, (int, np.integer)) or n < count or count != 16:
        raise ValueError("video must contain at least 16 decodable frames")
    return np.floor((np.arange(count) + 0.5) * n / count).astype(np.int64)


def decode_video(path: Path, indices=None) -> tuple[dict, list[np.ndarray]]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    frames, count = [], 0
    wanted = set(map(int, indices)) if indices is not None else set()
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video: {path}")
        metadata_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise ValueError(f"empty decoded frame: {path}")
            if count in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
    finally:
        cap.release()
    if not count:
        raise ValueError(f"no decodable frames: {path}")
    metadata = {"frame_count": count, "metadata_frame_count": metadata_count,
                "frame_count_mismatch": count != metadata_count,
                "duration_seconds": count / fps if np.isfinite(fps) and fps > 0 else None,
                "fps": fps if np.isfinite(fps) and fps > 0 else None}
    if indices is not None and len(frames) != len(wanted):
        raise ValueError(f"missing sampled frames: {path}")
    return metadata, frames


def audit_video(path: Path) -> dict:
    try:
        metadata, _ = decode_video(path)
        status = "ok" if metadata["frame_count"] >= 16 else "too_short"
        return {**metadata, "decode_status": status,
                "exclusion_reason": "" if status == "ok" else "fewer_than_16_frames"}
    except (ValueError, cv2.error) as exc:
        return {"frame_count": None, "duration_seconds": None, "decode_status": "failed",
                "exclusion_reason": f"decode_error: {exc}"}


def sampled_frames(path: Path, expected_count: int) -> tuple[np.ndarray, list[np.ndarray]]:
    indices = midpoint_indices(expected_count)
    metadata, frames = decode_video(path, indices)
    if metadata["frame_count"] != expected_count:
        raise ValueError(f"frozen frame count changed: {path}")
    return indices, frames
