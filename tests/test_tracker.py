from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.tracker import HybridFlockTracker, TrackerConfig, bbox_iou


def det(x: float, y: float, w: int = 20, h: int = 12) -> dict:
    return {
        "x": float(x),
        "y": float(y),
        "bbox": {
            "x": int(round(x - w / 2)),
            "y": int(round(y - h / 2)),
            "w": int(w),
            "h": int(h),
        },
        "contrast": 100.0,
        "mean_contrast": 10.0,
        "p90_contrast": 15.0,
        "area": float(w * h * 0.5),
    }


def tracks_by_id(tracks: list[dict]) -> dict[int, dict]:
    return {int(t["track_id"]): t for t in tracks}


def test_bbox_iou() -> None:
    a = {"x": 0, "y": 0, "w": 10, "h": 10}
    b = {"x": 5, "y": 0, "w": 10, "h": 10}
    assert math.isclose(bbox_iou(a, b), 1.0 / 3.0, rel_tol=1e-9)


def test_normal_hybrid_matching() -> None:
    config = TrackerConfig(
        max_center_distance=50.0,
        flock_suspect_gate=10.0,
    )
    tracker = HybridFlockTracker(config)

    first = [det(100, 100), det(210, 120), det(330, 170)]
    tracker.update(first)

    second = [det(104, 102), det(214, 121), det(333, 168)]
    tracks = tracks_by_id(tracker.update(second))

    assert set(tracks) == {0, 1, 2}
    assert all(tracks[i]["detection_idx"] == i for i in range(3))
    assert all(tracks[i]["association_mode"] == "normal" for i in range(3))


def test_global_translation_recovery() -> None:
    config = TrackerConfig(
        max_center_distance=45.0,
        reid_max_center_distance=35.0,
        translation_inlier_radius=12.0,
        translation_min_support=3,
        flock_suspect_gate=0.8,
        reid_cost_gate=1.5,
    )
    tracker = HybridFlockTracker(config)

    first_points = [
        (100, 90),
        (165, 105),
        (125, 180),
        (260, 150),
    ]
    tracker.update([det(x, y) for x, y in first_points])

    dx, dy = 180.0, 35.0
    second = [det(x + dx, y + dy) for x, y in first_points]
    tracks = tracks_by_id(tracker.update(second))

    assert set(tracks) == {0, 1, 2, 3}
    assert tracker.last_global_shift_support >= 3
    assert abs(tracker.last_global_shift[0] - dx) < 2.0
    assert abs(tracker.last_global_shift[1] - dy) < 2.0
    assert all(tracks[i]["detection_idx"] == i for i in range(4))
    assert all(tracks[i]["association_mode"] == "flock_reid" for i in range(4))


def test_partial_recovery_does_not_require_many_failures() -> None:
    config = TrackerConfig(
        max_center_distance=55.0,
        reid_max_center_distance=45.0,
        translation_inlier_radius=18.0,
        translation_min_support=2,
        flock_suspect_gate=0.30,
        reid_cost_gate=1.6,
    )
    tracker = HybridFlockTracker(config)

    first_points = [
        (100, 100),
        (180, 105),
        (130, 180),
        (260, 170),
    ]
    tracker.update([det(x, y) for x, y in first_points])

    # Three targets move mildly and remain normal-matchable. One target has a
    # large image-coordinate jump, emulating an edge target affected strongly
    # by a fast recentering motion / local association failure.
    second_points = [
        (105, 102),
        (185, 107),
        (135, 182),
        (330, 200),
    ]
    tracks = tracks_by_id(tracker.update([det(x, y) for x, y in second_points]))

    assert set(tracks) >= {0, 1, 2, 3}
    assert tracks[0]["detection_idx"] == 0
    assert tracks[1]["detection_idx"] == 1
    assert tracks[2]["detection_idx"] == 2
    # The fourth ID must not simply disappear and be replaced without any
    # attempt at identity-consistency/re-ID handling.
    assert tracks[3]["hits"] >= 1


def main() -> None:
    tests = [
        test_bbox_iou,
        test_normal_hybrid_matching,
        test_global_translation_recovery,
        test_partial_recovery_does_not_require_many_failures,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
