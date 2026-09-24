"""Small deterministic OpenCV video helpers for notebooks and scripts."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def list_videos(directory: str | Path = "./videos") -> list[str]:
    """Return sorted video paths from a directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        str(path)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def get_video_info(path: str | Path) -> dict[str, float | int]:
    """Read basic metadata without leaving an open VideoCapture."""
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        return {
            "frame_count": int(capture.get(cv.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv.CAP_PROP_FPS)),
            "width": int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def read_video_frame(
    path: str | Path,
    frame_index: int,
    loop: bool = True,
):
    """Read one deterministic frame by index.

    Opening/seeking/closing per call is slower than a streaming VideoCapture,
    but makes notebook output a pure function of ``(path, frame_index)``.
    """
    capture = cv.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    try:
        frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Invalid frame count for: {path}")

        index = int(frame_index)
        if loop:
            index %= frame_count
        elif index < 0 or index >= frame_count:
            raise IndexError(f"frame_index={index} outside [0, {frame_count})")

        capture.set(cv.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {index} from {path}")

        return frame, index
    finally:
        capture.release()
