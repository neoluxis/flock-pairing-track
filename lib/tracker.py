"""Kalman/Hungarian multi-object tracking with hybrid and flock matching.

Association is intentionally split into two stages:

1. Normal matching uses a weighted IoU + center-distance cost.
2. Every tentative match can be checked against translation-invariant flock
   geometry. Suspicious and unmatched observations enter a local re-ID stage
   that uses global image-shift compensation plus local/global flock features.

This design avoids making flock recovery depend on "many tracks failed". A
track can be considered suspicious even when it was successfully assigned by
Hungarian matching, which helps catch ID inheritance during crossings.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TrackerConfig:
    """Configuration for :class:`HybridFlockTracker`."""

    # Kalman model and track life cycle.
    dt: float = 1.0
    process_noise: float = 2.0
    measurement_noise: float = 12.0
    max_age: int = 8
    min_hits: int = 1
    history_size: int = 60

    # Stage-1 normal association: IoU + center distance.
    max_center_distance: float = 80.0
    center_weight: float = 0.70
    iou_weight: float = 0.30
    normal_cost_gate: float = 1.15

    # Flock identity-consistency validation for tentative normal matches.
    local_neighbors: int = 3
    local_weight: float = 0.65
    global_weight: float = 0.35
    flock_suspect_gate: float = 0.55
    min_flock_size: int = 3

    # Stage-2 flock-assisted re-identification.
    reid_center_weight: float = 0.35
    reid_iou_weight: float = 0.15
    reid_local_weight: float = 0.35
    reid_global_weight: float = 0.15
    reid_max_center_distance: float = 120.0
    reid_cost_gate: float = 1.30

    # Global translation estimator. It is estimated from all active tracks and
    # current detections, not only failed tracks, so recovery can help even if
    # only one or two tracks become suspicious.
    translation_inlier_radius: float = 45.0
    translation_min_support: int = 2


def bbox_xyxy(bbox: dict[str, Any]) -> tuple[float, float, float, float]:
    """Convert ``{x, y, w, h}`` to ``(x1, y1, x2, y2)``."""
    x = float(bbox["x"])
    y = float(bbox["y"])
    return x, y, x + float(bbox["w"]), y + float(bbox["h"])


def bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    """Return bbox center coordinates."""
    return (
        float(bbox["x"]) + float(bbox["w"]) * 0.5,
        float(bbox["y"]) + float(bbox["h"]) * 0.5,
    )


def bbox_at_center(
    bbox: dict[str, Any],
    center_x: float,
    center_y: float,
) -> dict[str, float]:
    """Move a bbox to a new center while retaining its size."""
    w = float(bbox["w"])
    h = float(bbox["h"])
    return {
        "x": float(center_x) - 0.5 * w,
        "y": float(center_y) - 0.5 * h,
        "w": w,
        "h": h,
    }


def shift_bbox(
    bbox: dict[str, Any],
    dx: float,
    dy: float,
) -> dict[str, float]:
    """Translate a bbox by an image-space displacement."""
    return {
        "x": float(bbox["x"]) + float(dx),
        "y": float(bbox["y"]) + float(dy),
        "w": float(bbox["w"]),
        "h": float(bbox["h"]),
    }


def bbox_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Compute intersection-over-union for two bboxes."""
    ax1, ay1, ax2, ay2 = bbox_xyxy(a)
    bx1, by1, bx2, by2 = bbox_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0.0 else 0.0


def points_from_tracks(tracks: list["KalmanTrack"]) -> np.ndarray:
    """Convert current track centers to an ``N x 2`` array."""
    if not tracks:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([track.position for track in tracks], dtype=np.float64)


def points_from_detections(detections: list[dict[str, Any]]) -> np.ndarray:
    """Convert detection centers to an ``N x 2`` array."""
    if not detections:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [[float(det["x"]), float(det["y"])] for det in detections],
        dtype=np.float64,
    )


def group_scale(points: np.ndarray) -> float:
    """Return a robust spatial scale for normalizing flock descriptors."""
    if len(points) < 2:
        return 1.0

    center = np.median(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    return max(float(np.median(radii)), 1.0)


def local_signatures(points: np.ndarray, k: int) -> list[np.ndarray]:
    """Build local K-nearest-neighbor distance signatures.

    Sorted neighbor distances are normalized by flock scale. The descriptor is
    translation and rotation invariant and does not require known neighbor IDs.
    """
    n = len(points)
    if n == 0:
        return []

    scale = group_scale(points)
    signatures: list[np.ndarray] = []

    for i in range(n):
        distances = np.linalg.norm(points - points[i], axis=1)
        distances = np.delete(distances, i)
        nearest = np.sort(distances)[: min(int(k), len(distances))] / scale
        signatures.append(nearest.astype(np.float64))

    return signatures


def global_signatures(points: np.ndarray) -> np.ndarray:
    """Build centroid-relative global flock descriptors.

    Each point is represented by normalized ``(dx, dy, radius)`` relative to
    the robust flock center. The feature is translation invariant but retains
    coarse position inside the flock.
    """
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)

    center = np.median(points, axis=0)
    scale = group_scale(points)
    relative = (points - center) / scale
    radius = np.linalg.norm(relative, axis=1, keepdims=True)
    return np.concatenate([relative, radius], axis=1)


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compare two variable-length local signatures."""
    count = min(len(a), len(b))
    if count == 0:
        return 0.0
    return float(np.mean(np.abs(a[:count] - b[:count])))


def estimate_global_translation(
    track_points: np.ndarray,
    detection_points: np.ndarray,
    inlier_radius: float,
) -> tuple[np.ndarray, int, float]:
    """Estimate common image translation by displacement-hypothesis voting.

    Every track/detection pair proposes a translation. For each hypothesis all
    tracks are shifted and a Hungarian assignment is solved. The hypothesis
    with the largest number of inlier assignments wins; ties use lower mean
    inlier distance. Finally, the translation is refined by the median inlier
    displacement.
    """
    if len(track_points) == 0 or len(detection_points) == 0:
        return np.zeros(2, dtype=np.float64), 0, float("inf")

    best_shift = np.zeros(2, dtype=np.float64)
    best_support = 0
    best_error = float("inf")
    best_pairs: list[tuple[int, int]] = []

    for track_point in track_points:
        for detection_point in detection_points:
            shift = detection_point - track_point
            shifted = track_points + shift

            distances = np.linalg.norm(
                shifted[:, None, :] - detection_points[None, :, :],
                axis=2,
            )
            rows, cols = linear_sum_assignment(distances)
            pair_distances = distances[rows, cols]
            inlier_mask = pair_distances <= float(inlier_radius)
            support = int(np.count_nonzero(inlier_mask))
            error = (
                float(pair_distances[inlier_mask].mean())
                if support
                else float("inf")
            )

            if support > best_support or (
                support == best_support and error < best_error
            ):
                best_support = support
                best_error = error
                best_shift = shift.copy()
                best_pairs = [
                    (int(r), int(c))
                    for r, c, keep in zip(rows, cols, inlier_mask)
                    if keep
                ]

    if best_pairs:
        displacements = np.asarray(
            [detection_points[j] - track_points[i] for i, j in best_pairs],
            dtype=np.float64,
        )
        best_shift = np.median(displacements, axis=0)

    return best_shift, best_support, best_error


class KalmanTrack:
    """Single-target constant-velocity Kalman filter."""

    def __init__(
        self,
        track_id: int,
        detection: dict[str, Any],
        detection_idx: int,
        config: TrackerConfig,
    ) -> None:
        self.track_id = int(track_id)
        self.config = config

        self.state = np.array(
            [
                [float(detection["x"])],
                [float(detection["y"])],
                [0.0],
                [0.0],
            ],
            dtype=np.float64,
        )

        dt = float(config.dt)
        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.P = np.diag([25.0, 25.0, 100.0, 100.0]).astype(np.float64)
        self.Q = np.eye(4, dtype=np.float64) * float(config.process_noise)
        self.R = np.eye(2, dtype=np.float64) * float(config.measurement_noise)

        self.bbox = detection["bbox"].copy()
        self.last_detection = detection
        self.detection_idx: int | None = int(detection_idx)
        self.hits = 1
        self.missed = 0
        self.age = 1
        self.updated_this_frame = True
        self.association_mode = "new"
        self.last_cost = 0.0
        self.history = deque(maxlen=int(config.history_size))
        self.history.append(
            (int(round(self.state[0, 0])), int(round(self.state[1, 0])))
        )

    @property
    def position(self) -> tuple[float, float]:
        """Current filtered/predicted image center."""
        return float(self.state[0, 0]), float(self.state[1, 0])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current image-plane velocity estimate."""
        return float(self.state[2, 0]), float(self.state[3, 0])

    @property
    def predicted_bbox(self) -> dict[str, float]:
        """Move the latest bbox to the current predicted center."""
        x, y = self.position
        return bbox_at_center(self.bbox, x, y)

    def predict(self) -> tuple[float, float]:
        """Predict one frame forward and mark the track unmatched."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.missed += 1
        self.updated_this_frame = False
        self.detection_idx = None
        self.association_mode = "predicted"
        return self.position

    def apply_image_shift(self, dx: float, dy: float) -> None:
        """Compensate sudden camera/image translation without changing velocity.

        This shifts only the image-space position state and stored bbox. It
        prevents a gimbal-induced image jump from being learned as bird speed.
        """
        self.state[0, 0] += float(dx)
        self.state[1, 0] += float(dy)
        self.bbox = shift_bbox(self.bbox, dx, dy)

    def update(
        self,
        detection: dict[str, Any],
        detection_idx: int,
        mode: str,
        cost: float,
    ) -> None:
        """Correct the Kalman state with a matched detector observation."""
        measurement = np.array(
            [[float(detection["x"])], [float(detection["y"])]],
            dtype=np.float64,
        )

        innovation = measurement - self.H @ self.state
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        kalman_gain = np.linalg.solve(
            innovation_covariance.T,
            (self.P @ self.H.T).T,
        ).T

        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.P = (identity - kalman_gain @ self.H) @ self.P

        self.bbox = detection["bbox"].copy()
        self.last_detection = detection
        self.detection_idx = int(detection_idx)
        self.hits += 1
        self.missed = 0
        self.updated_this_frame = True
        self.association_mode = str(mode)
        self.last_cost = float(cost)
        self.history.append(
            (int(round(self.state[0, 0])), int(round(self.state[1, 0])))
        )

    def as_dict(self) -> dict[str, Any]:
        """Export current track state for drawing/logging."""
        x, y = self.position
        vx, vy = self.velocity
        return {
            "track_id": self.track_id,
            "detection_idx": self.detection_idx,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "bbox": self.bbox.copy(),
            "hits": self.hits,
            "missed": self.missed,
            "age": self.age,
            "updated": self.updated_this_frame,
            "association_mode": self.association_mode,
            "association_cost": self.last_cost,
            "trajectory": list(self.history),
        }


class HybridFlockTracker:
    """Kalman + Hungarian tracker with hybrid and flock-assisted association."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self.tracks: list[KalmanTrack] = []
        self.next_track_id = 0
        self.frame_index = -1
        # Optional external frame/stream identifiers make the stateful tracker
        # safe to use from reactive notebooks such as marimo.
        self.last_external_frame_index: int | None = None
        self.stream_id: str | None = None
        self.last_global_shift = np.zeros(2, dtype=np.float64)
        self.last_global_shift_support = 0
        self.stats = {
            "normal_matches": 0,
            "suspect_matches": 0,
            "reid_matches": 0,
            "new_tracks": 0,
            "deleted_tracks": 0,
            "translation_recoveries": 0,
        }

    def reset(self) -> None:
        """Clear all temporal state and statistics."""
        self.tracks.clear()
        self.next_track_id = 0
        self.frame_index = -1
        self.last_external_frame_index = None
        self.stream_id = None
        self.last_global_shift = np.zeros(2, dtype=np.float64)
        self.last_global_shift_support = 0
        for key in self.stats:
            self.stats[key] = 0

    def _create_track(
        self,
        detection: dict[str, Any],
        detection_idx: int,
    ) -> None:
        """Create a new identity from an unmatched detection."""
        self.tracks.append(
            KalmanTrack(
                self.next_track_id,
                detection,
                detection_idx,
                self.config,
            )
        )
        self.next_track_id += 1
        self.stats["new_tracks"] += 1

    def _normal_cost(
        self,
        track: KalmanTrack,
        detection: dict[str, Any],
    ) -> tuple[float, float, float]:
        """Return hybrid cost, raw center distance, and IoU."""
        tx, ty = track.position
        dx = float(detection["x"]) - tx
        dy = float(detection["y"]) - ty
        center_distance = float(np.hypot(dx, dy))
        center_term = center_distance / max(self.config.max_center_distance, 1.0)
        iou = bbox_iou(track.predicted_bbox, detection["bbox"])
        iou_term = 1.0 - iou

        cost = (
            self.config.center_weight * center_term
            + self.config.iou_weight * iou_term
        )
        return float(cost), center_distance, iou

    def _flock_cost_matrices(
        self,
        track_indices: list[int],
        detection_indices: list[int],
        detections: list[dict[str, Any]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute local/global flock costs for selected association pools.

        Descriptors are built from all active tracks and all current detections,
        so unaffected neighbors still provide identity context during a local
        repair.
        """
        track_points = points_from_tracks(self.tracks)
        detection_points = points_from_detections(detections)

        track_local = local_signatures(
            track_points,
            self.config.local_neighbors,
        )
        detection_local = local_signatures(
            detection_points,
            self.config.local_neighbors,
        )
        track_global = global_signatures(track_points)
        detection_global = global_signatures(detection_points)

        local_cost = np.zeros(
            (len(track_indices), len(detection_indices)),
            dtype=np.float64,
        )
        global_cost = np.zeros_like(local_cost)

        for row, track_index in enumerate(track_indices):
            for col, detection_index in enumerate(detection_indices):
                local_cost[row, col] = signature_distance(
                    track_local[track_index],
                    detection_local[detection_index],
                )
                global_cost[row, col] = float(
                    np.linalg.norm(
                        track_global[track_index]
                        - detection_global[detection_index]
                    )
                )

        return local_cost, global_cost

    def _estimate_global_shift(
        self,
        detections: list[dict[str, Any]],
    ) -> tuple[np.ndarray, int, float]:
        """Estimate whole-flock image translation from all current points."""
        return estimate_global_translation(
            points_from_tracks(self.tracks),
            points_from_detections(detections),
            self.config.translation_inlier_radius,
        )

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_index: int | None = None,
        stream_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Advance tracker by one frame and return active tracks.

        frame_index and stream_id are optional for ordinary scripts, but useful
        in reactive notebooks. Re-evaluating the same frame does not advance
        Kalman state twice. A stream change or non-sequential seek resets the
        temporal state before processing the requested frame.
        """
        if stream_id is not None and stream_id != self.stream_id:
            self.reset()
            self.stream_id = str(stream_id)

        if frame_index is not None:
            frame_index = int(frame_index)

            if self.last_external_frame_index == frame_index:
                return self.get_tracks()

            if (
                self.last_external_frame_index is not None
                and frame_index != self.last_external_frame_index + 1
            ):
                remembered_stream = self.stream_id
                self.reset()
                self.stream_id = remembered_stream

            self.last_external_frame_index = frame_index

        self.frame_index += 1

        # First predict all existing tracks into the current frame.
        for track in self.tracks:
            track.predict()

        if not self.tracks:
            for detection_idx, detection in enumerate(detections):
                self._create_track(detection, detection_idx)
            return self.get_tracks()

        if not detections:
            self._prune_dead_tracks()
            return self.get_tracks()

        num_tracks = len(self.tracks)
        num_detections = len(detections)

        # Estimate a global translation independently of normal-match failure.
        # It may later be used only by the local re-ID pool.
        shift, shift_support, _ = self._estimate_global_shift(detections)
        self.last_global_shift = shift.copy()
        self.last_global_shift_support = int(shift_support)

        # --------------------------------------------------------------
        # Stage 1: normal IoU + center-distance Hungarian matching.
        # --------------------------------------------------------------
        normal_cost = np.zeros(
            (num_tracks, num_detections),
            dtype=np.float64,
        )
        center_distance = np.zeros_like(normal_cost)

        for track_idx, track in enumerate(self.tracks):
            for detection_idx, detection in enumerate(detections):
                cost, distance, _ = self._normal_cost(track, detection)
                normal_cost[track_idx, detection_idx] = cost
                center_distance[track_idx, detection_idx] = distance

        rows, cols = linear_sum_assignment(normal_cost)

        # Build flock descriptors before committing tentative matches. This is
        # what allows a matched-but-wrong assignment to be marked suspicious.
        all_track_indices = list(range(num_tracks))
        all_detection_indices = list(range(num_detections))
        local_cost, global_cost = self._flock_cost_matrices(
            all_track_indices,
            all_detection_indices,
            detections,
        )

        committed_tracks: set[int] = set()
        committed_detections: set[int] = set()
        flock_available = (
            num_tracks >= self.config.min_flock_size
            and num_detections >= self.config.min_flock_size
        )

        for track_idx, detection_idx in zip(rows, cols):
            track_idx = int(track_idx)
            detection_idx = int(detection_idx)

            if (
                center_distance[track_idx, detection_idx]
                > self.config.max_center_distance
            ):
                continue

            if normal_cost[track_idx, detection_idx] > self.config.normal_cost_gate:
                continue

            flock_cost = 0.0
            if flock_available:
                flock_cost = (
                    self.config.local_weight
                    * local_cost[track_idx, detection_idx]
                    + self.config.global_weight
                    * global_cost[track_idx, detection_idx]
                )

            # A successful Hungarian match can still be an ID inheritance.
            # Defer geometrically inconsistent pairs instead of committing them.
            if (
                flock_available
                and flock_cost > self.config.flock_suspect_gate
            ):
                self.stats["suspect_matches"] += 1
                continue

            self.tracks[track_idx].update(
                detections[detection_idx],
                detection_idx,
                mode="normal",
                cost=float(normal_cost[track_idx, detection_idx]),
            )
            committed_tracks.add(track_idx)
            committed_detections.add(detection_idx)
            self.stats["normal_matches"] += 1

        # --------------------------------------------------------------
        # Stage 2: local flock re-ID for every uncommitted pair.
        # --------------------------------------------------------------
        reid_track_indices = [
            idx for idx in range(num_tracks) if idx not in committed_tracks
        ]
        reid_detection_indices = [
            idx
            for idx in range(num_detections)
            if idx not in committed_detections
        ]

        if reid_track_indices and reid_detection_indices:
            use_shift = shift_support >= self.config.translation_min_support
            recovery_shift = shift if use_shift else np.zeros(2, dtype=np.float64)
            if use_shift and np.linalg.norm(recovery_shift) > 1.0:
                self.stats["translation_recoveries"] += 1

            reid_local, reid_global = self._flock_cost_matrices(
                reid_track_indices,
                reid_detection_indices,
                detections,
            )

            reid_cost = np.zeros(
                (len(reid_track_indices), len(reid_detection_indices)),
                dtype=np.float64,
            )
            reid_center = np.zeros_like(reid_cost)

            for row, track_idx in enumerate(reid_track_indices):
                track = self.tracks[track_idx]
                tx, ty = track.position
                shifted_bbox = shift_bbox(
                    track.predicted_bbox,
                    float(recovery_shift[0]),
                    float(recovery_shift[1]),
                )

                for col, detection_idx in enumerate(reid_detection_indices):
                    detection = detections[detection_idx]
                    dx = float(detection["x"]) - (tx + recovery_shift[0])
                    dy = float(detection["y"]) - (ty + recovery_shift[1])
                    distance = float(np.hypot(dx, dy))
                    reid_center[row, col] = distance

                    center_term = distance / max(
                        self.config.reid_max_center_distance,
                        1.0,
                    )
                    iou_term = 1.0 - bbox_iou(
                        shifted_bbox,
                        detection["bbox"],
                    )

                    reid_cost[row, col] = (
                        self.config.reid_center_weight * center_term
                        + self.config.reid_iou_weight * iou_term
                        + self.config.reid_local_weight * reid_local[row, col]
                        + self.config.reid_global_weight * reid_global[row, col]
                    )

            reid_rows, reid_cols = linear_sum_assignment(reid_cost)

            for row, col in zip(reid_rows, reid_cols):
                row = int(row)
                col = int(col)

                if reid_center[row, col] > self.config.reid_max_center_distance:
                    continue
                if reid_cost[row, col] > self.config.reid_cost_gate:
                    continue

                track_idx = reid_track_indices[row]
                detection_idx = reid_detection_indices[col]
                track = self.tracks[track_idx]

                # Apply common image shift before Kalman correction so a sudden
                # gimbal pan does not become a huge false bird velocity.
                if use_shift:
                    track.apply_image_shift(
                        float(recovery_shift[0]),
                        float(recovery_shift[1]),
                    )

                track.update(
                    detections[detection_idx],
                    detection_idx,
                    mode="flock_reid",
                    cost=float(reid_cost[row, col]),
                )
                committed_tracks.add(track_idx)
                committed_detections.add(detection_idx)
                self.stats["reid_matches"] += 1

        # Remaining observations are considered genuinely new identities.
        for detection_idx, detection in enumerate(detections):
            if detection_idx not in committed_detections:
                self._create_track(detection, detection_idx)

        self._prune_dead_tracks()
        return self.get_tracks()

    def _prune_dead_tracks(self) -> None:
        """Delete tracks that exceeded the missing-frame grace period."""
        before = len(self.tracks)
        self.tracks = [
            track
            for track in self.tracks
            if track.missed <= self.config.max_age
        ]
        self.stats["deleted_tracks"] += before - len(self.tracks)

    def get_tracks(
        self,
        confirmed_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Return current tracks as serializable dictionaries."""
        output: list[dict[str, Any]] = []
        for track in self.tracks:
            if confirmed_only and track.hits < self.config.min_hits:
                continue
            output.append(track.as_dict())
        return output


def pair_detections_hybrid(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    max_center_distance: float = 80.0,
    center_weight: float = 0.70,
    iou_weight: float = 0.30,
    cost_gate: float = 1.15,
) -> dict[str, Any]:
    """Pair two detection sets using center distance + IoU + Hungarian.

    This stateless helper is intended for algorithm validation notebooks. It
    does not use Kalman prediction; the previous detections themselves are the
    reference observations.
    """
    n_prev = len(previous)
    n_cur = len(current)
    if n_prev == 0 or n_cur == 0:
        return {
            "matches": [],
            "unmatched_previous": list(range(n_prev)),
            "unmatched_current": list(range(n_cur)),
            "cost_matrix": np.empty((n_prev, n_cur), dtype=np.float64),
            "center_matrix": np.empty((n_prev, n_cur), dtype=np.float64),
            "iou_matrix": np.empty((n_prev, n_cur), dtype=np.float64),
        }

    cost = np.zeros((n_prev, n_cur), dtype=np.float64)
    center = np.zeros_like(cost)
    iou = np.zeros_like(cost)

    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            distance = float(
                np.hypot(
                    float(new["x"]) - float(old["x"]),
                    float(new["y"]) - float(old["y"]),
                )
            )
            overlap = bbox_iou(old["bbox"], new["bbox"])
            center[i, j] = distance
            iou[i, j] = overlap
            cost[i, j] = (
                float(center_weight) * distance / max(float(max_center_distance), 1.0)
                + float(iou_weight) * (1.0 - overlap)
            )

    rows, cols = linear_sum_assignment(cost)
    matches: list[dict[str, Any]] = []
    matched_prev: set[int] = set()
    matched_cur: set[int] = set()

    for i, j in zip(rows, cols):
        i, j = int(i), int(j)
        if center[i, j] > max_center_distance or cost[i, j] > cost_gate:
            continue
        matches.append(
            {
                "previous_idx": i,
                "current_idx": j,
                "cost": float(cost[i, j]),
                "center_distance": float(center[i, j]),
                "iou": float(iou[i, j]),
            }
        )
        matched_prev.add(i)
        matched_cur.add(j)

    return {
        "matches": matches,
        "unmatched_previous": [i for i in range(n_prev) if i not in matched_prev],
        "unmatched_current": [i for i in range(n_cur) if i not in matched_cur],
        "cost_matrix": cost,
        "center_matrix": center,
        "iou_matrix": iou,
    }


def pair_detections_flock_motion(
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    local_neighbors: int = 3,
    translation_inlier_radius: float = 45.0,
    max_compensated_distance: float = 120.0,
    center_weight: float = 0.35,
    iou_weight: float = 0.15,
    local_weight: float = 0.35,
    global_weight: float = 0.15,
    cost_gate: float = 1.30,
) -> dict[str, Any]:
    """Pair two detection sets using flock motion and geometry.

    A common translation is estimated first. Matching then combines translated
    center distance, translated IoU, local KNN-distance structure, and global
    centroid-relative flock position.
    """
    prev_points = points_from_detections(previous)
    cur_points = points_from_detections(current)
    n_prev, n_cur = len(previous), len(current)

    if n_prev == 0 or n_cur == 0:
        return {
            "matches": [],
            "unmatched_previous": list(range(n_prev)),
            "unmatched_current": list(range(n_cur)),
            "translation": np.zeros(2, dtype=np.float64),
            "translation_support": 0,
            "cost_matrix": np.empty((n_prev, n_cur), dtype=np.float64),
        }

    translation, support, translation_error = estimate_global_translation(
        prev_points,
        cur_points,
        translation_inlier_radius,
    )

    prev_local = local_signatures(prev_points, local_neighbors)
    cur_local = local_signatures(cur_points, local_neighbors)
    prev_global = global_signatures(prev_points)
    cur_global = global_signatures(cur_points)

    cost = np.zeros((n_prev, n_cur), dtype=np.float64)
    compensated_center = np.zeros_like(cost)
    translated_iou = np.zeros_like(cost)
    local_cost = np.zeros_like(cost)
    global_cost = np.zeros_like(cost)

    for i, old in enumerate(previous):
        shifted_bbox = shift_bbox(
            old["bbox"],
            float(translation[0]),
            float(translation[1]),
        )
        shifted_x = float(old["x"]) + float(translation[0])
        shifted_y = float(old["y"]) + float(translation[1])

        for j, new in enumerate(current):
            distance = float(
                np.hypot(
                    float(new["x"]) - shifted_x,
                    float(new["y"]) - shifted_y,
                )
            )
            overlap = bbox_iou(shifted_bbox, new["bbox"])
            local = signature_distance(prev_local[i], cur_local[j])
            global_feature = float(np.linalg.norm(prev_global[i] - cur_global[j]))

            compensated_center[i, j] = distance
            translated_iou[i, j] = overlap
            local_cost[i, j] = local
            global_cost[i, j] = global_feature
            cost[i, j] = (
                center_weight * distance / max(max_compensated_distance, 1.0)
                + iou_weight * (1.0 - overlap)
                + local_weight * local
                + global_weight * global_feature
            )

    rows, cols = linear_sum_assignment(cost)
    matches: list[dict[str, Any]] = []
    matched_prev: set[int] = set()
    matched_cur: set[int] = set()

    for i, j in zip(rows, cols):
        i, j = int(i), int(j)
        if compensated_center[i, j] > max_compensated_distance:
            continue
        if cost[i, j] > cost_gate:
            continue
        matches.append(
            {
                "previous_idx": i,
                "current_idx": j,
                "cost": float(cost[i, j]),
                "compensated_center_distance": float(compensated_center[i, j]),
                "translated_iou": float(translated_iou[i, j]),
                "local_cost": float(local_cost[i, j]),
                "global_cost": float(global_cost[i, j]),
            }
        )
        matched_prev.add(i)
        matched_cur.add(j)

    return {
        "matches": matches,
        "unmatched_previous": [i for i in range(n_prev) if i not in matched_prev],
        "unmatched_current": [i for i in range(n_cur) if i not in matched_cur],
        "translation": translation,
        "translation_support": int(support),
        "translation_error": float(translation_error),
        "cost_matrix": cost,
        "compensated_center_matrix": compensated_center,
        "translated_iou_matrix": translated_iou,
        "local_cost_matrix": local_cost,
        "global_cost_matrix": global_cost,
    }
