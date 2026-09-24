"""Run contrast detection + hybrid/flock multi-object tracking on a video.

Example:
    uv run python track_hybrid.py videos/00.mp4 --max-frames 300
    uv run python track_hybrid.py videos/00.mp4 --output out/00_tracked.mp4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2 as cv

from lib import (
    ContrastDetectorConfig,
    HybridFlockTracker,
    TrackerConfig,
    detect_contrast_targets,
    draw_tracks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="videos/00.mp4")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)

    # Detector parameters most frequently tuned in the notebook.
    parser.add_argument("--blur-kernel", type=int, default=3)
    parser.add_argument("--threshold-sigma", type=float, default=3.0)
    parser.add_argument("--min-mean-contrast", type=float, default=5.0)
    parser.add_argument("--min-area", type=float, default=10.0)
    parser.add_argument("--max-area", type=float, default=1500.0)

    # Tracker association parameters.
    parser.add_argument("--max-center-distance", type=float, default=80.0)
    parser.add_argument("--reid-max-center-distance", type=float, default=120.0)
    parser.add_argument("--flock-suspect-gate", type=float, default=0.55)
    parser.add_argument("--translation-radius", type=float, default=45.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)

    detector_config = ContrastDetectorConfig(
        blur_kernel=args.blur_kernel,
        threshold_sigma=args.threshold_sigma,
        min_mean_contrast=args.min_mean_contrast,
        min_area=args.min_area,
        max_area=args.max_area,
    )

    tracker_config = TrackerConfig(
        max_center_distance=args.max_center_distance,
        reid_max_center_distance=args.reid_max_center_distance,
        flock_suspect_gate=args.flock_suspect_gate,
        translation_inlier_radius=args.translation_radius,
    )

    tracker = HybridFlockTracker(tracker_config)

    capture = cv.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv.CAP_PROP_FPS)) or 25.0
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    if args.start_frame > 0:
        capture.set(cv.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv.VideoWriter(
            str(output_path),
            cv.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output: {output_path}")

    processed = 0
    total_detections = 0
    total_active_tracks = 0
    detector_seconds = 0.0
    tracker_seconds = 0.0
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if args.max_frames is not None and processed >= args.max_frames:
                break

            t0 = time.perf_counter()
            detections, diagnostics = detect_contrast_targets(
                frame,
                detector_config,
            )
            detector_seconds += time.perf_counter() - t0

            t1 = time.perf_counter()
            tracks = tracker.update(detections)
            tracker_seconds += time.perf_counter() - t1

            total_detections += len(detections)
            total_active_tracks += len(tracks)

            if writer is not None:
                writer.write(draw_tracks(frame, tracks))

            processed += 1

            if args.log_every > 0 and processed % args.log_every == 0:
                print(
                    f"frame={args.start_frame + processed - 1} "
                    f"detections={len(detections)} tracks={len(tracks)} "
                    f"shift=({tracker.last_global_shift[0]:.1f},"
                    f"{tracker.last_global_shift[1]:.1f}) "
                    f"support={tracker.last_global_shift_support}"
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "video": str(video_path),
        "video_total_frames": total_frames,
        "processed_frames": processed,
        "wall_seconds": elapsed,
        "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
        "mean_detector_ms": (
            detector_seconds / processed * 1000.0 if processed else 0.0
        ),
        "mean_tracker_ms": (
            tracker_seconds / processed * 1000.0 if processed else 0.0
        ),
        "mean_detections": total_detections / processed if processed else 0.0,
        "mean_active_tracks": total_active_tracks / processed if processed else 0.0,
        "tracker_stats": tracker.stats,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
