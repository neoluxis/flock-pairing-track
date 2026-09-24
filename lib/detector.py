"""Stateless local-contrast detector extracted from ktop.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class ContrastDetectorConfig:
    """Parameters for dark-target local-contrast detection."""

    blur_kernel: int = 41
    threshold_sigma: float = 3.0
    min_mean_contrast: float = 5.0
    absolute_threshold_floor: float = 3.0

    min_area: float = 10.0
    max_area: float = 1500.0
    min_width: int = 4
    min_height: int = 2
    max_width: int = 120
    max_height: int = 80

    min_distance: int = 7
    border_margin: int = 20
    ktop: int | str = "auto"
    max_auto: int = 100
    close_kernel: int = 3
    open_kernel: int = 3
    open_iterations: int = 0
    close_iterations: int = 1

    # Split merged blobs (e.g. two birds connected by wings) when multiple
    # contrast peaks exist inside one connected component.
    split_touching: bool = False
    peak_min_distance: int = 5
    peak_threshold_ratio: float = 0.60


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV image to grayscale without modifying the input."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy.ndarray")
    if image.size == 0:
        raise ValueError("image must not be empty")

    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0]
    if image.ndim == 3 and image.shape[2] >= 3:
        return cv.cvtColor(image[..., :3], cv.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def compute_local_contrast(
    gray: np.ndarray,
    blur_kernel: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate background and return positive dark-on-bright response."""
    blur_kernel = int(blur_kernel)
    if blur_kernel < 3:
        raise ValueError("blur_kernel must be >= 3")
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    gray_f = gray.astype(np.float32)
    background = cv.GaussianBlur(gray_f, (blur_kernel, blur_kernel), 0)

    # Positive response means the pixel is darker than local background.
    contrast = np.maximum(background - gray_f, 0)
    return contrast, background


def compute_robust_threshold(
    contrast: np.ndarray,
    threshold_sigma: float,
    threshold_floor: float,
) -> float:
    """Compute a robust threshold from median and MAD."""
    values = contrast.ravel()
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad

    return max(
        median + float(threshold_sigma) * robust_sigma,
        float(threshold_floor),
    )


def build_contrast_mask(
    contrast: np.ndarray,
    threshold: float,
    border_margin: int,
    close_kernel: int,
    open_kernel: int = 0,
    open_iterations: int = 0,
    close_iterations: int = 1,
) -> np.ndarray:
    """Threshold response, mask borders, and close tiny internal gaps."""
    mask = (contrast >= threshold).astype(np.uint8) * 255
    h, w = mask.shape

    if border_margin > 0:
        margin = min(int(border_margin), h // 2, w // 2)
        mask[:margin, :] = 0
        mask[-margin:, :] = 0
        mask[:, :margin] = 0
        mask[:, -margin:] = 0

    open_kernel = int(open_kernel)
    if open_kernel >= 2 and open_iterations > 0:
        if open_kernel % 2 == 0:
            open_kernel += 1
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (open_kernel, open_kernel),
        )
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_OPEN,
            kernel,
            iterations=open_iterations,
        )

    close_kernel = int(close_kernel)
    if close_kernel >= 2 and close_iterations > 0:
        if close_kernel % 2 == 0:
            close_kernel += 1
        kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (close_kernel, close_kernel),
        )
        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            kernel,
            iterations=close_iterations,
        )

    return mask


def contour_to_candidate(
    contour: np.ndarray,
    contrast: np.ndarray,
) -> dict[str, Any]:
    """Measure one contour and convert it to a detector observation."""
    area = float(cv.contourArea(contour))
    bx, by, bw, bh = cv.boundingRect(contour)

    moments = cv.moments(contour)
    if moments["m00"] != 0:
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
    else:
        cx = bx + bw // 2
        cy = by + bh // 2

    # Use an ROI-sized mask to avoid a full-frame allocation per contour.
    roi_contrast = contrast[by : by + bh, bx : bx + bw]
    shifted = contour - np.array([[[bx, by]]], dtype=contour.dtype)
    roi_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv.drawContours(roi_mask, [shifted], -1, 255, cv.FILLED)
    values = roi_contrast[roi_mask != 0]

    if values.size:
        mean_contrast = float(values.mean())
        p90_contrast = float(np.percentile(values, 90))
    else:
        mean_contrast = 0.0
        p90_contrast = 0.0

    fill_ratio = area / max(bw * bh, 1)
    score = p90_contrast * np.sqrt(max(area, 1.0))

    return {
        "x": cx,
        "y": cy,
        "contrast": float(score),
        "mean_contrast": mean_contrast,
        "p90_contrast": p90_contrast,
        "area": area,
        "fill_ratio": float(fill_ratio),
        "bbox": {
            "x": int(bx),
            "y": int(by),
            "w": int(bw),
            "h": int(bh),
        },
    }


def validate_candidate(
    candidate: dict[str, Any],
    image_shape: tuple[int, ...],
    config: ContrastDetectorConfig,
) -> tuple[bool, str | None]:
    """Apply geometry and contrast filters to one observation."""
    area = candidate["area"]
    bbox = candidate["bbox"]
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    image_h, image_w = image_shape[:2]

    if area < config.min_area:
        return False, "area_too_small"
    if area > config.max_area:
        return False, "area_too_large"
    if w < config.min_width:
        return False, "width_too_small"
    if h < config.min_height:
        return False, "height_too_small"
    if w > config.max_width:
        return False, "width_too_large"
    if h > config.max_height:
        return False, "height_too_large"
    if candidate["mean_contrast"] < config.min_mean_contrast:
        return False, "contrast_too_low"

    margin = config.border_margin
    if margin > 0 and (
        x <= margin
        or y <= margin
        or x + w >= image_w - margin
        or y + h >= image_h - margin
    ):
        return False, "near_border"

    return True, None


def split_touching_components(
    mask: np.ndarray,
    contrast: np.ndarray,
    min_distance: int,
    threshold_ratio: float,
) -> np.ndarray:
    """Split merged bright/dark blobs using local contrast peaks.

    This handles short occlusions such as two birds whose wings touch.
    Components with multiple strong peaks are separated by watershed.
    """
    distance = cv.distanceTransform(mask, cv.DIST_L2, 5)
    if distance.max() <= 0:
        return mask

    peak_mask = cv.dilate(
        distance,
        np.ones((2 * min_distance + 1, 2 * min_distance + 1), np.uint8),
    )
    peaks = (distance == peak_mask) & (distance > distance.max() * threshold_ratio)
    n_peaks, markers = cv.connectedComponents(peaks.astype(np.uint8))

    if n_peaks <= 2:
        return mask

    markers = markers + 1
    markers[mask == 0] = 0
    markers = cv.watershed(
        cv.cvtColor(contrast.astype(np.uint8), cv.COLOR_GRAY2BGR),
        markers.astype(np.int32),
    )
    return (markers > 1).astype(np.uint8) * 255


def extract_candidates(
    mask: np.ndarray,
    contrast: np.ndarray,
    config: ContrastDetectorConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Extract contours and split them into accepted and rejected observations."""
    if config.split_touching:
        mask = split_touching_components(
            mask,
            contrast,
            config.peak_min_distance,
            config.peak_threshold_ratio,
        )

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: dict[str, int] = {}

    for contour in contours:
        candidate = contour_to_candidate(contour, contrast)
        valid, reason = validate_candidate(candidate, contrast.shape, config)

        if valid:
            accepted.append(candidate)
        else:
            rejected_candidate = candidate.copy()
            rejected_candidate["reject_reason"] = reason
            rejected.append(rejected_candidate)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    return accepted, rejected, rejection_counts


def suppress_close_candidates(
    candidates: list[dict[str, Any]],
    min_distance: int,
) -> list[dict[str, Any]]:
    """Greedily suppress lower-score candidates with very close centroids."""
    ordered = sorted(
        candidates,
        key=lambda p: p["contrast"],
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    min_distance_sq = float(min_distance) ** 2

    for candidate in ordered:
        if all(
            (candidate["x"] - existing["x"]) ** 2
            + (candidate["y"] - existing["y"]) ** 2
            >= min_distance_sq
            for existing in selected
        ):
            selected.append(candidate)

    return selected


def select_top_candidates(
    candidates: list[dict[str, Any]],
    ktop: int | str,
    max_auto: int,
) -> list[dict[str, Any]]:
    """Apply the final top-K cap to already-ranked observations."""
    if ktop == "auto":
        return candidates[: int(max_auto)]

    if not isinstance(ktop, int) or ktop < 1:
        raise ValueError("ktop must be a positive integer or 'auto'")

    return candidates[:ktop]


def detect_contrast_targets(
    image: np.ndarray,
    config: ContrastDetectorConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the complete detector on one frame."""
    if config is None:
        config = ContrastDetectorConfig()

    gray = to_gray(image)

    contrast, background = compute_local_contrast(
        gray,
        config.blur_kernel,
    )

    threshold = compute_robust_threshold(
        contrast,
        config.threshold_sigma,
        config.absolute_threshold_floor,
    )

    mask = build_contrast_mask(
        contrast,
        threshold,
        config.border_margin,
        config.close_kernel,
        config.open_kernel,
        config.open_iterations,
        config.close_iterations,
    )

    candidates, rejected, rejection_counts = extract_candidates(
        mask,
        contrast,
        config,
    )

    candidates = suppress_close_candidates(
        candidates,
        config.min_distance,
    )

    detections = select_top_candidates(
        candidates,
        config.ktop,
        config.max_auto,
    )

    diagnostics = {
        "gray": gray,
        "background": background,
        "contrast": contrast,
        "mask": mask,
        "threshold": threshold,
        "candidates": candidates,
        "rejected": rejected,
        "rejection_counts": rejection_counts,
        "num_detections": len(detections),
    }

    return detections, diagnostics
