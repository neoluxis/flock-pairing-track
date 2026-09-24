"""OpenCV drawing helpers for detections and tracked targets."""
from __future__ import annotations
from typing import Any
import cv2 as cv
import numpy as np


def draw_detections(image: np.ndarray, detections: list[dict[str, Any]], show_index: bool = True, show_score: bool = False) -> np.ndarray:
    """Draw detector bboxes and centroids on a copy of the input image."""
    vis = image.copy()
    for idx, det in enumerate(detections):
        bbox = det["bbox"]
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
        cx, cy = int(round(det["x"])), int(round(det["y"]))
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
        labels = []
        if show_index:
            labels.append(str(idx))
        if show_score:
            labels.append(f"{det['contrast']:.1f}")
        if labels:
            cv.putText(vis, " ".join(labels), (x, max(y - 5, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv.LINE_AA)
    return vis


def draw_tracks(
    image: np.ndarray,
    tracks: list[dict[str, Any]],
    draw_bbox: bool = True,
    draw_centroid: bool = True,
    draw_trajectory: bool = True,
    trajectory_thickness: int = 1,
    trajectory_length: int | None = 30,
) -> np.ndarray:
    """Draw idx#track_id labels, bboxes, centroids and recent trajectories.

    Parameters
    ----------
    trajectory_length:
        Number of recent trajectory points to draw. ``None`` draws the complete
        stored trajectory.
    """
    vis = image.copy()
    for track in tracks:
        track_id = int(track["track_id"])
        det_idx = track.get("detection_idx")
        cx, cy = int(round(track["x"])), int(round(track["y"]))
        bbox = track["bbox"]
        bx, by, bw, bh = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
        updated = bool(track.get("updated", False))
        if draw_trajectory:
            trajectory = track.get("trajectory", [])
            if trajectory_length is not None:
                trajectory = trajectory[-trajectory_length:]

            for i in range(1, len(trajectory)):
                cv.line(
                    vis,
                    tuple(map(int, trajectory[i - 1])),
                    tuple(map(int, trajectory[i])),
                    (255, 0, 255),
                    trajectory_thickness,
                    cv.LINE_AA,
                )
        if draw_bbox:
            color = (0, 255, 0) if updated else (0, 255, 255)
            cv.rectangle(vis, (bx, by), (bx + bw, by + bh), color, 1)
        if draw_centroid:
            color = (0, 0, 255) if updated else (0, 165, 255)
            cv.circle(vis, (cx, cy), 4, color, -1)
        label = f"{det_idx if det_idx is not None else '-'}#{track_id}"
        cv.putText(vis, label, (bx, max(by - 5, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    return vis


def draw_pair_matches(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_detections: list[dict[str, Any]],
    current_detections: list[dict[str, Any]],
    pairing: dict[str, Any],
) -> np.ndarray:
    """Draw two frames side-by-side with matched detection indices connected."""
    left = previous_image.copy()
    right = current_image.copy()

    # Keep heights equal before horizontal composition.
    if left.shape[0] != right.shape[0]:
        raise ValueError("previous_image and current_image must have equal height")

    for idx, det in enumerate(previous_detections):
        bbox = det["bbox"]
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
        cv.rectangle(left, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.putText(left, f"P{idx}", (x, max(y - 4, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv.LINE_AA)

    for idx, det in enumerate(current_detections):
        bbox = det["bbox"]
        x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
        cv.rectangle(right, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv.putText(right, f"C{idx}", (x, max(y - 4, 0)), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv.LINE_AA)

    canvas = np.hstack([left, right])
    x_offset = left.shape[1]

    # Draw the association links after composition so the line visually joins
    # the previous and current observations.
    for match in pairing.get("matches", []):
        pi = int(match["previous_idx"])
        ci = int(match["current_idx"])
        p = previous_detections[pi]
        c = current_detections[ci]
        p0 = (int(round(p["x"])), int(round(p["y"])))
        p1 = (x_offset + int(round(c["x"])), int(round(c["y"])))
        cv.line(canvas, p0, p1, (255, 0, 255), 1, cv.LINE_AA)

    return canvas
