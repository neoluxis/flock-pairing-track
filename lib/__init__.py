"""Reusable contrast-detection and multi-object-tracking utilities."""

from .detector import ContrastDetectorConfig, detect_contrast_targets
from .tracker import (
    HybridFlockTracker,
    TrackerConfig,
    pair_detections_flock_motion,
    pair_detections_hybrid,
)
from .visualization import draw_detections, draw_pair_matches, draw_tracks
from .video import get_video_info, list_videos, read_video_frame

__all__ = [
    "ContrastDetectorConfig",
    "detect_contrast_targets",
    "HybridFlockTracker",
    "TrackerConfig",
    "pair_detections_hybrid",
    "pair_detections_flock_motion",
    "draw_detections",
    "draw_pair_matches",
    "draw_tracks",
    "get_video_info",
    "list_videos",
    "read_video_frame",
]
