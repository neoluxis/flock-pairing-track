#!/usr/bin/env python3
"""Interactive frame-by-frame tracker viewer.

Keys:
    SPACE : process next frame
    r     : reset tracker
    q     : quit
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2 as cv

from lib import (
    ContrastDetectorConfig,
    HybridFlockTracker,
    TrackerConfig,
    detect_contrast_targets,
    draw_tracks,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?", default="videos/00.mp4")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output path for the processed video; omit to disable saving",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="output video frame rate, defaults to the input video frame rate",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1280,
        help="processing/display width, set 0 to disable resizing",
    )
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be greater than 0")
    return args


def main():
    args = parse_args()

    cap = cv.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")

    cap.set(cv.CAP_PROP_POS_FRAMES, args.start_frame)
    input_fps = float(cap.get(cv.CAP_PROP_FPS)) or 25.0
    writer = None

    detector_cfg = ContrastDetectorConfig(
        blur_kernel=3,
        threshold_sigma=3.0,
        min_mean_contrast=5.0,
        min_area=2 if args.size <= 640 else 10,
        max_area=200 if args.size <= 640 else 1500,
        min_width=2 if args.size <= 640 else 4,
        min_height=1 if args.size <= 640 else 2,
        max_width=40 if args.size <= 640 else 120,
        max_height=30 if args.size <= 640 else 80,
        min_distance=5 if args.size <= 640 else 7,
        border_margin=10 if args.size <= 640 else 20,
        max_auto=100,
    )

    tracker = HybridFlockTracker(
        TrackerConfig(
            max_center_distance=80,
            reid_max_center_distance=120,
            translation_inlier_radius=45,
        )
    )

    frame_index = args.start_frame
    stream_id = args.video

    cv.namedWindow("tracker", cv.WINDOW_NORMAL)

    print("SPACE next frame | r reset | q quit")
    print(f"working resolution: width={args.size} (0 means original)")

    while True:
        key = cv.waitKey(0) & 0xff

        if key == ord("q"):
            break

        if key == ord("r"):
            tracker.reset()
            cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
            print("tracker reset")
            continue

        if key != 32:
            continue

        ok, frame = cap.read()
        if not ok:
            print("end of video")
            break

        # Run detector and tracker in a fixed working resolution.
        # This keeps detector parameters meaningful and greatly reduces the
        # cost of Gaussian filtering / morphology on large surveillance frames.
        if args.size > 0:
            original_h, original_w = frame.shape[:2]
            scale = args.size / original_w
            process_frame = cv.resize(
                frame,
                (args.size, int(original_h * scale)),
                interpolation=cv.INTER_AREA,
            )
        else:
            scale = 1.0
            process_frame = frame

        detections, _ = detect_contrast_targets(
            process_frame,
            detector_cfg,
        )

        # Coordinates returned by detector/tracker remain in process_frame
        # coordinates. Display the same coordinate system.
        frame = process_frame

        cv.putText(
            frame,
            f"size={frame.shape[1]}x{frame.shape[0]}",
            (20, 105),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )

        tracks = tracker.update(
            detections,
            frame_index=frame_index,
            stream_id=stream_id,
        )

        vis = draw_tracks(frame, tracks)

        cv.putText(
            vis,
            f"frame={frame_index} det={len(detections)} tracks={len(tracks)}",
            (20, 35),
            cv.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )

        cv.putText(
            vis,
            (
                f"normal={tracker.stats['normal_matches']} "
                f"reid={tracker.stats['reid_matches']} "
                f"shift_support={tracker.last_global_shift_support}"
            ),
            (20, 70),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv.LINE_AA,
        )

        if args.output and writer is None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv.VideoWriter(
                str(output_path),
                cv.VideoWriter_fourcc(*"mp4v"),
                args.fps or input_fps,
                (vis.shape[1], vis.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output: {output_path}")
            print(f"saving output to {output_path}")

        if writer is not None:
            writer.write(vis)

        cv.imshow("tracker", vis)

        print(
            f"frame={frame_index} "
            f"detections={len(detections)} "
            f"tracks={len(tracks)} "
            f"shift=({tracker.last_global_shift[0]:.1f},"
            f"{tracker.last_global_shift[1]:.1f}) "
            f"support={tracker.last_global_shift_support}"
        )

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
