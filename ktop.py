import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import time
    from dataclasses import dataclass, replace
    from pathlib import Path
    from typing import Any, Callable

    import cv2 as cv
    import marimo as mo
    import numpy as np

    return Any, Callable, Path, cv, dataclass, mo, np, os, replace, time


@app.cell
def _(Any, Callable, cv, mo, np, time):
    def timer(
        func: Callable,
        params: dict,
    ) -> tuple[Any, float]:
        """Run ``func(**params)`` once and measure wall-clock elapsed time.

        ``time.perf_counter`` is used because it is a high-resolution monotonic
        clock suitable for short algorithm benchmarks.  The function result is
        returned together with elapsed seconds so callers do not need to run the
        target function a second time.
        """
        # Take the timestamp immediately before invoking the target function.
        start = time.perf_counter()
        result = func(**params)
        elapsed = time.perf_counter() - start
        return result, elapsed


    def jshow(image):
        """Display an OpenCV image in marimo."""
        if image is None:
            return mo.md("**Image is None**")

        if image.ndim == 2:
            display_image = image

        elif image.ndim == 3 and image.shape[2] == 3:
            display_image = cv.cvtColor(
                image,
                cv.COLOR_BGR2RGB,
            )

        elif image.ndim == 3 and image.shape[2] == 4:
            display_image = cv.cvtColor(
                image,
                cv.COLOR_BGRA2RGBA,
            )

        else:
            raise ValueError(
                f"Unsupported image shape: {image.shape}"
            )

        return mo.image(display_image)


    def normalize_to_u8(image):
        """Normalize arbitrary numeric image to uint8 for visualization."""
        if image is None:
            return None

        image = np.asarray(image)

        if image.dtype == np.uint8:
            return image

        min_value = float(image.min())
        max_value = float(image.max())

        if max_value <= min_value:
            return np.zeros(
                image.shape,
                dtype=np.uint8,
            )

        output = (
            (image - min_value)
            / (max_value - min_value)
            * 255.0
        )

        return output.astype(np.uint8)


    return jshow, normalize_to_u8


@app.cell
def _(dataclass):
    @dataclass(frozen=True)
    class ContrastDetectorConfig:
        # Local background estimation
        blur_kernel: int = 41

        # Contrast threshold
        threshold_sigma: float = 3.0
        min_mean_contrast: float = 5.0
        absolute_threshold_floor: float = 3.0

        # Region geometry
        min_area: float = 10.0
        max_area: float = 1500.0

        min_width: int = 4
        min_height: int = 2

        max_width: int = 120
        max_height: int = 80

        # Candidate selection
        min_distance: int = 7
        border_margin: int = 20

        ktop: int | str = "auto"
        max_auto: int = 100

        # Morphology
        close_kernel: int = 3


    detector_config = ContrastDetectorConfig()
    return (detector_config,)


@app.cell
def _(cv, np, os):
    def load_image(path):
        """Load an image from disk without changing its channel layout.

        This helper deliberately keeps file I/O outside the detector.  The core
        detector therefore receives only ``numpy.ndarray`` objects and remains
        easy to test with images coming from files, videos, cameras, or ROS.
        """
        # IMREAD_UNCHANGED preserves grayscale / alpha / bit depth when possible.
        image = cv.imread(
            os.fspath(path),
            cv.IMREAD_UNCHANGED,
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {path}"
            )

        return image


    def to_gray(image):
        """Convert a supported OpenCV image array to a 2-D grayscale image.

        Accepted inputs are already-grayscale arrays, HxWx1 arrays, and images
        with at least three channels.  OpenCV BGR ordering is assumed for color
        images because the rest of this notebook uses OpenCV as its I/O layer.
        """
        # Fail early with a clear error instead of letting cvtColor fail later.
        if not isinstance(image, np.ndarray):
            raise TypeError(
                "image must be numpy.ndarray"
            )

        if image.size == 0:
            raise ValueError(
                "image must not be empty"
            )

        if image.ndim == 2:
            return image

        if (
            image.ndim == 3
            and image.shape[2] == 1
        ):
            return image[..., 0]

        if (
            image.ndim == 3
            and image.shape[2] >= 3
        ):
            return cv.cvtColor(
                image[..., :3],
                cv.COLOR_BGR2GRAY,
            )

        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )


    return (to_gray,)


@app.cell
def _(cv, np):
    def compute_local_contrast(
        gray,
        blur_kernel=41,
    ):
        """
        Compute a signed dark-target local-contrast response.

        A Gaussian-blurred image estimates the slowly varying local background
        ``B``.  For a dark target on brighter sky, ``B - I`` becomes positive;
        brighter structures are clamped to zero so they cannot compete with the
        desired dark-target response.

            C(x, y) = max(B(x, y) - I(x, y), 0)
        """

        blur_kernel = int(blur_kernel)

        if blur_kernel < 3:
            raise ValueError(
                "blur_kernel must be >= 3"
            )

        if blur_kernel % 2 == 0:
            blur_kernel += 1

        gray_f = gray.astype(
            np.float32
        )

        background = cv.GaussianBlur(
            gray_f,
            (blur_kernel, blur_kernel),
            0,
        )

        contrast = (
            background
            - gray_f
        )

        # Only retain dark-on-bright responses.
        contrast = np.maximum(
            contrast,
            0,
        )

        return contrast, background


    return (compute_local_contrast,)


@app.cell
def _(cv, np):
    def compute_robust_threshold(
        contrast,
        threshold_sigma=3.0,
        threshold_floor=3.0,
    ):
        """Estimate an adaptive threshold from the contrast-map noise floor.

        Median absolute deviation (MAD) is used instead of standard deviation
        because sparse strong targets should not noticeably increase the noise
        estimate.  ``1.4826 * MAD`` approximates sigma for Gaussian noise.
        """
        # Flatten only as a view; no image geometry is needed for this statistic.
        values = contrast.ravel()

        median = float(
            np.median(values)
        )

        mad = float(
            np.median(
                np.abs(
                    values - median
                )
            )
        )

        robust_sigma = (
            1.4826 * mad
        )

        threshold = (
            median
            + threshold_sigma
            * robust_sigma
        )

        threshold = max(
            threshold,
            threshold_floor,
        )

        return threshold


    def build_contrast_mask(
        contrast,
        threshold,
        border_margin=20,
        close_kernel=3,
    ):
        """Convert the contrast response into a binary candidate mask.

        The border is explicitly suppressed because convolution / compression
        artifacts near image edges frequently become false candidates.  A small
        morphological closing joins tiny gaps in a target without using opening,
        which could erase thin wings or very small distant targets.
        """
        # Threshold every pixel: 255 means it is part of a candidate region.
        mask = (
            contrast >= threshold
        ).astype(np.uint8) * 255

        h, w = mask.shape

        if border_margin > 0:
            margin = min(
                int(border_margin),
                h // 2,
                w // 2,
            )

            mask[:margin, :] = 0
            mask[-margin:, :] = 0

            mask[:, :margin] = 0
            mask[:, -margin:] = 0

        close_kernel = int(
            close_kernel
        )

        if close_kernel >= 2:
            if close_kernel % 2 == 0:
                close_kernel += 1

            kernel = cv.getStructuringElement(
                cv.MORPH_ELLIPSE,
                (
                    close_kernel,
                    close_kernel,
                ),
            )

            mask = cv.morphologyEx(
                mask,
                cv.MORPH_CLOSE,
                kernel,
            )

        return mask


    return build_contrast_mask, compute_robust_threshold


@app.cell
def _(cv, np):
    def contour_to_candidate(
        contour,
        contrast,
    ):
        """Measure geometry and contrast statistics for one contour.

        This function does *not* decide whether the contour is valid.  It only
        converts a raw OpenCV contour into a descriptive candidate dictionary.
        Separating measurement from filtering makes rejected targets inspectable
        and allows filter rules to evolve independently of feature extraction.
        """
        # Geometric area is later used both as a filter and a weak score weight.
        area = float(
            cv.contourArea(contour)
        )

        bx, by, bw, bh = (
            cv.boundingRect(contour)
        )

        moments = cv.moments(
            contour
        )

        if moments["m00"] != 0:
            cx = int(
                round(
                    moments["m10"]
                    / moments["m00"]
                )
            )

            cy = int(
                round(
                    moments["m01"]
                    / moments["m00"]
                )
            )

        else:
            cx = (
                bx + bw // 2
            )
            cy = (
                by + bh // 2
            )

        # Only allocate mask for the candidate bounding box.
        # This is cheaper than allocating a full-size image
        # for every contour.
        roi_contrast = contrast[
            by : by + bh,
            bx : bx + bw,
        ]

        shifted = (
            contour
            - np.array(
                [[[bx, by]]],
                dtype=contour.dtype,
            )
        )

        roi_mask = np.zeros(
            (bh, bw),
            dtype=np.uint8,
        )

        cv.drawContours(
            roi_mask,
            [shifted],
            -1,
            255,
            cv.FILLED,
        )

        values = roi_contrast[
            roi_mask != 0
        ]

        if values.size == 0:
            mean_contrast = 0.0
            p90_contrast = 0.0

        else:
            mean_contrast = float(
                values.mean()
            )

            p90_contrast = float(
                np.percentile(
                    values,
                    90,
                )
            )

        bbox_area = max(
            bw * bh,
            1,
        )

        fill_ratio = (
            area / bbox_area
        )

        score = (
            p90_contrast
            * np.sqrt(
                max(area, 1.0)
            )
        )

        return {
            "x": cx,
            "y": cy,

            "contrast": score,
            "mean_contrast": (
                mean_contrast
            ),
            "p90_contrast": (
                p90_contrast
            ),

            "area": area,
            "fill_ratio": (
                fill_ratio
            ),

            "bbox": {
                "x": int(bx),
                "y": int(by),
                "w": int(bw),
                "h": int(bh),
            },
        }


    return (contour_to_candidate,)


@app.function
def validate_candidate(
    candidate,
    image_shape,
    config,
):
    """Apply hard candidate filters and report why a candidate was rejected.

    Keeping this logic separate from contour measurement is useful during
    detector tuning: diagnostics can count ``area_too_small``,
    ``contrast_too_low`` and similar failure modes independently.

    Returns
    -------
    (valid, reason)
    """

    area = candidate["area"]
    mean_contrast = (
        candidate[
            "mean_contrast"
        ]
    )

    bbox = candidate["bbox"]

    x = bbox["x"]
    y = bbox["y"]
    w = bbox["w"]
    h = bbox["h"]

    image_h, image_w = (
        image_shape[:2]
    )

    if area < config.min_area:
        return (
            False,
            "area_too_small",
        )

    if area > config.max_area:
        return (
            False,
            "area_too_large",
        )

    if w < config.min_width:
        return (
            False,
            "width_too_small",
        )

    if h < config.min_height:
        return (
            False,
            "height_too_small",
        )

    if w > config.max_width:
        return (
            False,
            "width_too_large",
        )

    if h > config.max_height:
        return (
            False,
            "height_too_large",
        )

    if (
        mean_contrast
        < config.min_mean_contrast
    ):
        return (
            False,
            "contrast_too_low",
        )

    margin = (
        config.border_margin
    )

    if margin > 0:
        if (
            x <= margin
            or y <= margin
            or (
                x + w
                >= image_w - margin
            )
            or (
                y + h
                >= image_h - margin
            )
        ):
            return (
                False,
                "near_border",
            )

    return True, None


@app.cell
def _(contour_to_candidate, cv):
    def extract_candidates(
        mask,
        contrast,
        config,
    ):
        """Extract contours, measure them, then split accepted/rejected sets.

        The returned rejection histogram is intentionally part of the API so a
        marimo cell can explain *why* recall or precision changed after tuning.
        """
        # RETR_EXTERNAL is sufficient because each connected response is treated
        # as one candidate; nested holes are irrelevant to this detector.
        contours, _ = (
            cv.findContours(
                mask,
                cv.RETR_EXTERNAL,
                cv.CHAIN_APPROX_SIMPLE,
            )
        )

        accepted = []
        rejected = []

        rejection_counts = {}

        for contour in contours:
            candidate = (
                contour_to_candidate(
                    contour,
                    contrast,
                )
            )

            valid, reason = (
                validate_candidate(
                    candidate,
                    contrast.shape,
                    config,
                )
            )

            if valid:
                accepted.append(
                    candidate
                )

            else:
                candidate = (
                    candidate.copy()
                )

                candidate[
                    "reject_reason"
                ] = reason

                rejected.append(
                    candidate
                )

                rejection_counts[
                    reason
                ] = (
                    rejection_counts.get(
                        reason,
                        0,
                    )
                    + 1
                )

        return (
            accepted,
            rejected,
            rejection_counts,
        )


    return (extract_candidates,)


@app.cell
def _():
    def suppress_close_candidates(
        candidates,
        min_distance,
    ):
        """Greedily suppress lower-scoring candidates with nearby centroids.

        This is a lightweight spatial NMS.  Candidates are sorted strongest first,
        then a point is kept only when it is at least ``min_distance`` pixels from
        every previously selected point.
        """
        # Sorting first ensures that a stronger response wins each local conflict.
        ordered = sorted(
            candidates,
            key=lambda p: p[
                "contrast"
            ],
            reverse=True,
        )

        selected = []

        min_distance_sq = (
            min_distance
            * min_distance
        )

        for candidate in ordered:
            keep = True

            for existing in selected:
                dx = (
                    candidate["x"]
                    - existing["x"]
                )

                dy = (
                    candidate["y"]
                    - existing["y"]
                )

                if (
                    dx * dx
                    + dy * dy
                    < min_distance_sq
                ):
                    keep = False
                    break

            if keep:
                selected.append(
                    candidate
                )

        return selected


    def select_top_candidates(
        candidates,
        ktop="auto",
        max_auto=100,
    ):
        """Apply the final result-count policy after scoring and suppression.

        ``auto`` currently means "return every surviving candidate up to
        ``max_auto``".  Importantly, it does not add a second contrast threshold,
        because that previously caused strong false-negative behavior.
        """
        if ktop == "auto":
            return candidates[
                : int(max_auto)
            ]

        if (
            not isinstance(
                ktop,
                int,
            )
            or ktop < 1
        ):
            raise ValueError(
                "ktop must be positive "
                "integer or 'auto'"
            )

        return candidates[
            :ktop
        ]


    return select_top_candidates, suppress_close_candidates


@app.cell
def _(
    build_contrast_mask,
    compute_local_contrast,
    compute_robust_threshold,
    extract_candidates,
    select_top_candidates,
    suppress_close_candidates,
    to_gray,
):
    def detect_contrast_targets(
        image,
        config,
    ):
        """
        Complete contrast-target detector.

        Pure pipeline:
            image
              -> gray
              -> local contrast
              -> binary mask
              -> contours
              -> candidate measurements
              -> filtering
              -> distance suppression
              -> top-k

        Returns
        -------
        detections, diagnostics
        """

        # Stage 0: normalize input representation.  No filtering happens here.
        gray = to_gray(
            image
        )

        # Stage 1: estimate local background and signed dark-target response.
        (
            contrast,
            background,
        ) = compute_local_contrast(
            gray,
            blur_kernel=(
                config.blur_kernel
            ),
        )

        # Stage 2: estimate a robust global noise threshold on the response map.
        threshold = (
            compute_robust_threshold(
                contrast,
                threshold_sigma=(
                    config.threshold_sigma
                ),
                threshold_floor=(
                    config
                    .absolute_threshold_floor
                ),
            )
        )

        # Stage 3: binarize and lightly connect candidate pixels.
        mask = build_contrast_mask(
            contrast,
            threshold,
            border_margin=(
                config.border_margin
            ),
            close_kernel=(
                config.close_kernel
            ),
        )

        # Stage 4: convert connected regions into measured candidates and filter.
        (
            candidates,
            rejected,
            rejection_counts,
        ) = extract_candidates(
            mask,
            contrast,
            config,
        )

        # Stage 5: suppress duplicate / nearly coincident observations.
        candidates = (
            suppress_close_candidates(
                candidates,
                min_distance=(
                    config.min_distance
                ),
            )
        )

        # Stage 6: enforce the requested output-count policy.
        detections = (
            select_top_candidates(
                candidates,
                ktop=config.ktop,
                max_auto=(
                    config.max_auto
                ),
            )
        )

        diagnostics = {
            "gray": gray,
            "background": (
                background
            ),
            "contrast": contrast,
            "mask": mask,

            "threshold": (
                threshold
            ),

            "candidates": (
                candidates
            ),

            "rejected": (
                rejected
            ),

            "rejection_counts": (
                rejection_counts
            ),

            "num_detections": (
                len(detections)
            ),
        }

        return (
            detections,
            diagnostics,
        )


    return (detect_contrast_targets,)


@app.cell
def _(cv):
    def draw_pnts(
        image,
        pnts,
        show_index=True,
        show_score=False,
    ):
        """Draw bounding boxes, centroids and labels on a copy of the image.

        The input frame is never modified in place.  This matters in a notebook:
        otherwise annotations from one experiment could become high-contrast
        features when the same frame is passed through the detector again.
        """

        # Work on a private visualization buffer; preserve the detector input.
        vis = image.copy()

        for idx, p in enumerate(
            pnts
        ):
            cx = int(p["x"])
            cy = int(p["y"])

            bbox = p["bbox"]

            bx = int(
                bbox["x"]
            )
            by = int(
                bbox["y"]
            )
            bw = int(
                bbox["w"]
            )
            bh = int(
                bbox["h"]
            )

            # BBox
            cv.rectangle(
                vis,
                (bx, by),
                (
                    bx + bw,
                    by + bh,
                ),
                (0, 255, 0),
                1,
            )

            # Centroid
            cv.circle(
                vis,
                (cx, cy),
                4,
                (0, 0, 255),
                -1,
            )

            labels = []

            if show_index:
                labels.append(
                    str(idx)
                )

            if show_score:
                labels.append(
                    f"{p['contrast']:.1f}"
                )

            if labels:
                label = " ".join(
                    labels
                )

                cv.putText(
                    vis,
                    label,
                    (
                        bx,
                        max(
                            by - 5,
                            0,
                        ),
                    ),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv.LINE_AA,
                )

        return vis


    return (draw_pnts,)


@app.cell
def _(detect_contrast_targets, draw_pnts, time):
    def process_frame(
        frame,
        config,
    ):
        """Run detection and visualization for one frame and time the detector.

        This is a convenience orchestration layer for notebook use.  The detector
        itself stays independent from drawing, while callers that only want one
        annotated frame can use this function directly.
        """
        # Time detection + diagnostic construction; drawing is performed afterward.
        start = (
            time.perf_counter()
        )

        (
            pnts,
            diagnostics,
        ) = detect_contrast_targets(
            frame,
            config,
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        vis = draw_pnts(
            frame,
            pnts,
        )

        diagnostics = {
            **diagnostics,
            "elapsed": elapsed,
        }

        return (
            vis,
            pnts,
            diagnostics,
        )


    return (process_frame,)


@app.cell
def _(Path):
    video_dir = Path(
        "./videos"
    )

    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".m4v",
    }

    videos = sorted(
        str(path)
        for path in video_dir.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in video_extensions
        )
    )

    if not videos:
        raise RuntimeError(
            "No videos found in ./videos"
        )
    return (videos,)


@app.cell
def _(mo, videos):
    video_select = mo.ui.dropdown(
        options=videos,
        value=videos[0],
        label="Video",
    )

    video_select
    return (video_select,)


@app.cell
def _(cv, video_select):
    def get_video_info(
        path,
    ):
        """Read stable video metadata and immediately release the capture.

        A short-lived VideoCapture avoids sharing a mutable decode cursor between
        reactive marimo cells.
        """
        cap = cv.VideoCapture(
            path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{path}"
            )

        frame_count = int(
            cap.get(
                cv.CAP_PROP_FRAME_COUNT
            )
        )

        fps = float(
            cap.get(
                cv.CAP_PROP_FPS
            )
        )

        width = int(
            cap.get(
                cv.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv.CAP_PROP_FRAME_HEIGHT
            )
        )

        cap.release()

        return {
            "frame_count": (
                frame_count
            ),
            "fps": fps,
            "width": width,
            "height": height,
        }


    video_info = get_video_info(
        video_select.value
    )
    return (video_info,)


@app.cell
def _(mo):
    next_frame = mo.ui.button(
        value=0,
        on_click=lambda value: (
            value + 1
        ),
        label="Next frame",
    )

    next_frame
    return (next_frame,)


@app.cell
def _(cv):
    def read_video_frame(
        path,
        frame_index,
    ):
        """Read a deterministic frame by index from a video file.

        The function opens, seeks, reads, and closes on every call.  That is slower
        than sequential decoding but gives marimo a pure mapping from
        ``(path, frame_index)`` to a frame, avoiding hidden mutable state.
        """
        # Open a fresh capture so another reactive cell cannot advance our cursor.
        cap = cv.VideoCapture(
            path
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: "
                f"{path}"
            )

        frame_count = int(
            cap.get(
                cv.CAP_PROP_FRAME_COUNT
            )
        )

        if frame_count <= 0:
            cap.release()

            raise RuntimeError(
                "Invalid video frame count"
            )

        frame_index = (
            int(frame_index)
            % frame_count
        )

        cap.set(
            cv.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        ok, frame = cap.read()

        cap.release()

        if not ok:
            raise RuntimeError(
                "Failed to read "
                f"frame {frame_index}"
            )

        return (
            frame,
            frame_index,
        )


    return (read_video_frame,)


@app.cell
def _(detector_config, mo):
    threshold_sigma_ui = (
        mo.ui.slider(
            start=1.0,
            stop=8.0,
            step=0.1,
            value=(
                detector_config
                .threshold_sigma
            ),
            label="Threshold sigma",
        )
    )

    min_contrast_ui = (
        mo.ui.slider(
            start=0.0,
            stop=30.0,
            step=0.5,
            value=(
                detector_config
                .min_mean_contrast
            ),
            label="Min mean contrast",
        )
    )

    min_area_ui = (
        mo.ui.slider(
            start=0,
            stop=200,
            step=1,
            value=int(
                detector_config
                .min_area
            ),
            label="Min area",
        )
    )

    max_area_ui = (
        mo.ui.slider(
            start=50,
            stop=5000,
            step=10,
            value=int(
                detector_config
                .max_area
            ),
            label="Max area",
        )
    )

    blur_kernel_ui = (
        mo.ui.slider(
            start=3,
            stop=101,
            step=2,
            value=(
                detector_config
                .blur_kernel
            ),
            label="Blur kernel",
        )
    )

    mo.vstack(
        [
            blur_kernel_ui,
            threshold_sigma_ui,
            min_contrast_ui,
            min_area_ui,
            max_area_ui,
        ]
    )
    return (
        blur_kernel_ui,
        max_area_ui,
        min_area_ui,
        min_contrast_ui,
        threshold_sigma_ui,
    )


@app.cell
def _(
    blur_kernel_ui,
    detector_config,
    max_area_ui,
    min_area_ui,
    min_contrast_ui,
    replace,
    threshold_sigma_ui,
):
    current_config = replace(
        detector_config,

        blur_kernel=(
            blur_kernel_ui.value
        ),

        threshold_sigma=(
            threshold_sigma_ui.value
        ),

        min_mean_contrast=(
            min_contrast_ui.value
        ),

        min_area=(
            float(
                min_area_ui.value
            )
        ),

        max_area=(
            float(
                max_area_ui.value
            )
        ),
    )
    return (current_config,)


@app.cell
def _(
    current_config,
    next_frame,
    process_frame,
    read_video_frame,
    video_select,
):
    requested_index = (
        next_frame.value
    )

    frame, frame_index = (
        read_video_frame(
            video_select.value,
            requested_index,
        )
    )

    (
        frame_vis,
        frame_pnts,
        frame_diag,
    ) = process_frame(
        frame,
        current_config,
    )
    return frame, frame_diag, frame_index, frame_pnts, frame_vis


@app.cell
def _(frame_diag, frame_index, frame_pnts, mo, video_info):
    _elapsed_ms = (
        frame_diag["elapsed"]
        * 1000.0
    )

    _threshold = frame_diag["threshold"]

    _rejection_counts = frame_diag[
        "rejection_counts"
    ]

    mo.md(
        f"""
    ### Frame information

    - Frame: **{frame_index} / {video_info["frame_count"] - 1}**
    - FPS: **{video_info["fps"]:.2f}**
    - Resolution: **{video_info["width"]} × {video_info["height"]}**
    - Detections: **{len(frame_pnts)}**
    - Detector time: **{_elapsed_ms:.2f} ms**
    - Contrast threshold: **{_threshold:.3f}**
    - Rejected candidates: **{sum(_rejection_counts.values())}**
    """
    )
    return


@app.cell
def _(frame_vis, jshow):
    jshow(
        frame_vis
    )
    return


@app.cell
def _(
    blur_kernel_ui,
    max_area_ui,
    min_area_ui,
    min_contrast_ui,
    mo,
    threshold_sigma_ui,
):
    mo.vstack(
        [
            blur_kernel_ui,
            threshold_sigma_ui,
            min_contrast_ui,
            min_area_ui,
            max_area_ui,
        ]
    )
    return


@app.cell
def _(frame, frame_diag, jshow, mo, normalize_to_u8):
    _contrast_vis = (
        normalize_to_u8(
            frame_diag[
                "contrast"
            ]
        )
    )

    _mask_vis = (
        frame_diag[
            "mask"
        ]
    )

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(
                        "**Original**"
                    ),
                    jshow(frame),
                ]
            ),

            mo.vstack(
                [
                    mo.md(
                        "**Local contrast**"
                    ),
                    jshow(
                        _contrast_vis
                    ),
                ]
            ),

            mo.vstack(
                [
                    mo.md(
                        "**Candidate mask**"
                    ),
                    jshow(
                        _mask_vis
                    ),
                ]
            ),
        ],
        widths=[
            1,
            1,
            1,
        ],
    )
    return


@app.cell
def _(frame_diag, mo):
    _rejection_counts = frame_diag[
        "rejection_counts"
    ]

    if _rejection_counts:
        _lines = [
            f"- `{reason}`: {count}"
            for reason, count
            in sorted(
                _rejection_counts.items()
            )
        ]

        mo.md(
            "### Candidate rejection\n\n"
            + "\n".join(_lines)
        )

    else:
        mo.md(
            "### Candidate rejection\n\n"
            "No candidates were rejected."
        )
    return


@app.cell
def _(frame_pnts):
    frame_pnts[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tracking

    The detector now produces frame-level observations:

    ```mermaid
    flowchart TD
        A[frame] --> B[contrast detector]
        B --> C[observations]
        C --> D[Kalman prediction]
        D --> E[Hungarian association]
        E --> F[track management]
    ```
    """)
    return


@app.cell
def _():
    from collections import deque
    from scipy.optimize import linear_sum_assignment

    return deque, linear_sum_assignment


@app.cell
def _(deque, np):
    class KalmanTrack:
        """Single-target constant-velocity Kalman filter.

        State vector
        ------------
        x = [cx, cy, vx, vy]^T

        Measurement
        -----------
        z = [cx, cy]^T

        The bounding-box size is not estimated by the Kalman filter.
        Instead, the most recently matched detector bbox is retained.
        """

        def __init__(
            self,
            track_id,
            detection,
            detection_idx,
            dt=1.0,
            process_noise=1.0,
            measurement_noise=10.0,
            history_size=50,
        ):
            self.track_id = int(track_id)

            # ----------------------------------------------------
            # Initial state:
            # position comes from the detector; velocity starts
            # from zero because no temporal information exists yet.
            # ----------------------------------------------------
            self.state = np.array(
                [
                    [float(detection["x"])],
                    [float(detection["y"])],
                    [0.0],
                    [0.0],
                ],
                dtype=np.float64,
            )

            # ----------------------------------------------------
            # Constant-velocity state transition:
            #
            # cx' = cx + vx * dt
            # cy' = cy + vy * dt
            # vx' = vx
            # vy' = vy
            # ----------------------------------------------------
            self.F = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

            # Measurement matrix: detector only observes position.
            self.H = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )

            # Initial state uncertainty.
            self.P = np.diag(
                [25.0, 25.0, 100.0, 100.0]
            ).astype(np.float64)

            # Process noise controls how easily velocity can change.
            self.Q = (
                np.eye(4, dtype=np.float64)
                * float(process_noise)
            )

            # Measurement noise controls trust in detector centroids.
            self.R = (
                np.eye(2, dtype=np.float64)
                * float(measurement_noise)
            )

            # Most recently associated detector information.
            self.bbox = detection["bbox"].copy()
            self.detection_idx = int(detection_idx)
            self.last_detection = detection

            # Number of successful detector associations.
            self.hits = 1

            # Number of consecutive frames without a measurement.
            self.missed = 0

            # Number of frames since this track was created.
            self.age = 1

            # True only when a detector observation was associated
            # with this track in the current frame.
            self.updated_this_frame = True

            # Keep recent filtered positions for trajectory drawing.
            self.history = deque(
                maxlen=int(history_size)
            )

            self.history.append(
                (
                    int(round(self.state[0, 0])),
                    int(round(self.state[1, 0])),
                )
            )

        def predict(self):
            """Predict the target state one frame forward."""

            # Standard Kalman prediction:
            #
            # x_k|k-1 = F x_k-1|k-1
            # P_k|k-1 = F P F^T + Q
            self.state = self.F @ self.state

            self.P = (
                self.F
                @ self.P
                @ self.F.T
                + self.Q
            )

            self.age += 1
            self.missed += 1
            self.updated_this_frame = False
            self.detection_idx = None

            return self.position

        def update(
            self,
            detection,
            detection_idx,
        ):
            """Correct the predicted state using a detector observation."""

            measurement = np.array(
                [
                    [float(detection["x"])],
                    [float(detection["y"])],
                ],
                dtype=np.float64,
            )

            # Innovation:
            #
            # y = z - H x
            innovation = (
                measurement
                - self.H @ self.state
            )

            # Innovation covariance:
            #
            # S = H P H^T + R
            S = (
                self.H
                @ self.P
                @ self.H.T
                + self.R
            )

            # Kalman gain:
            #
            # K = P H^T S^-1
            #
            # solve() is preferred to explicitly forming inv(S).
            K = np.linalg.solve(
                S.T,
                (self.P @ self.H.T).T,
            ).T

            # Correct state and covariance.
            self.state = (
                self.state
                + K @ innovation
            )

            identity = np.eye(
                self.P.shape[0],
                dtype=np.float64,
            )

            self.P = (
                identity
                - K @ self.H
            ) @ self.P

            # Save detector-side properties.
            self.bbox = detection["bbox"].copy()
            self.detection_idx = int(detection_idx)
            self.last_detection = detection

            self.hits += 1
            self.missed = 0
            self.updated_this_frame = True

            # Record the filtered location rather than the raw
            # detector centroid, producing a smoother trajectory.
            self.history.append(
                (
                    int(round(self.state[0, 0])),
                    int(round(self.state[1, 0])),
                )
            )

        @property
        def position(self):
            """Current filtered/predicted centroid."""

            return (
                float(self.state[0, 0]),
                float(self.state[1, 0]),
            )

        @property
        def velocity(self):
            """Current estimated image-plane velocity."""

            return (
                float(self.state[2, 0]),
                float(self.state[3, 0]),
            )

        def as_dict(self):
            """Export track information for visualization or later stages."""

            _cx, _cy = self.position
            _vx, _vy = self.velocity

            return {
                "track_id": self.track_id,
                "detection_idx": self.detection_idx,

                "x": _cx,
                "y": _cy,

                "vx": _vx,
                "vy": _vy,

                "bbox": self.bbox.copy(),

                "hits": self.hits,
                "missed": self.missed,
                "age": self.age,

                "updated": self.updated_this_frame,

                "trajectory": list(self.history),
            }

    return (KalmanTrack,)


@app.cell
def _(KalmanTrack, linear_sum_assignment, np):
    class MultiObjectTracker:
        """Multi-object tracker using Kalman filters and Hungarian matching.

        Association cost
        ----------------
        Euclidean distance between:

            predicted track centroid
            and
            detected target centroid

        Tracks are removed after ``max_age`` consecutive missed frames.

        New unmatched detections immediately create new tracks.
        """

        def __init__(
            self,
            max_distance=80.0,
            max_age=8,
            min_hits=1,
            dt=1.0,
            process_noise=1.0,
            measurement_noise=10.0,
            history_size=50,
        ):
            # Maximum allowed centroid distance for a valid
            # Hungarian assignment.
            self.max_distance = float(max_distance)

            # Maximum number of consecutive unmatched frames before
            # a track is deleted.
            self.max_age = int(max_age)

            # Track is considered confirmed after this many hits.
            self.min_hits = int(min_hits)

            self.dt = float(dt)
            self.process_noise = float(process_noise)
            self.measurement_noise = float(measurement_noise)
            self.history_size = int(history_size)

            self.tracks = []
            self.next_track_id = 0

            # Used to protect a stateful tracker from accidental
            # repeated execution of the same marimo frame.
            self.last_frame_index = None

        def reset(self):
            """Delete every active track and restart track IDs."""

            self.tracks.clear()
            self.next_track_id = 0
            self.last_frame_index = None

        def _create_track(
            self,
            detection,
            detection_idx,
        ):
            """Create one new Kalman track from an unmatched detection."""

            _track = KalmanTrack(
                track_id=self.next_track_id,
                detection=detection,
                detection_idx=detection_idx,
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                history_size=self.history_size,
            )

            self.next_track_id += 1
            self.tracks.append(_track)

        def _build_cost_matrix(
            self,
            detections,
        ):
            """Build track-to-detection centroid-distance matrix."""

            _num_tracks = len(self.tracks)
            _num_detections = len(detections)

            _cost = np.zeros(
                (
                    _num_tracks,
                    _num_detections,
                ),
                dtype=np.float64,
            )

            for _track_idx, _track in enumerate(
                self.tracks
            ):
                _tx, _ty = _track.position

                for _det_idx, _det in enumerate(
                    detections
                ):
                    _dx = (
                        float(_det["x"])
                        - _tx
                    )

                    _dy = (
                        float(_det["y"])
                        - _ty
                    )

                    _cost[
                        _track_idx,
                        _det_idx,
                    ] = np.hypot(
                        _dx,
                        _dy,
                    )

            return _cost

        def update(
            self,
            detections,
            frame_index=None,
        ):
            """Process detections belonging to one video frame.

            Parameters
            ----------
            detections
                Output from ``detect_contrast_targets``.

            frame_index
                Optional frame number.

                In a marimo notebook the same cell can be reevaluated
                because a downstream control changed. If the exact same
                frame index arrives twice, the tracker is not advanced
                twice.

                If frame indices jump backwards or skip frames, the
                tracker resets because its temporal state is no longer
                valid.

            Returns
            -------
            list[dict]
                Current active tracks.
            """

            # ----------------------------------------------------
            # Protect the stateful tracker from marimo reevaluating
            # the same frame.
            # ----------------------------------------------------
            if frame_index is not None:
                _frame_index = int(frame_index)

                if (
                    self.last_frame_index
                    == _frame_index
                ):
                    return self.get_tracks()

                # Reset if video position is no longer sequential.
                #
                # This covers:
                # - selecting another video,
                # - wrapping from final frame back to zero,
                # - manually seeking,
                # - skipping frames.
                if (
                    self.last_frame_index
                    is not None
                    and _frame_index
                    != self.last_frame_index + 1
                ):
                    self.reset()

                self.last_frame_index = _frame_index

            # ----------------------------------------------------
            # 1. Predict every existing track before association.
            # ----------------------------------------------------
            for _track in self.tracks:
                _track.predict()

            _num_tracks = len(self.tracks)
            _num_detections = len(detections)

            # ----------------------------------------------------
            # No existing tracks:
            # every observation starts a new track.
            # ----------------------------------------------------
            if _num_tracks == 0:
                for _det_idx, _det in enumerate(
                    detections
                ):
                    self._create_track(
                        _det,
                        _det_idx,
                    )

                return self.get_tracks()

            # ----------------------------------------------------
            # No detections:
            # predictions remain alive until max_age expires.
            # ----------------------------------------------------
            if _num_detections == 0:
                self.tracks = [
                    _track
                    for _track in self.tracks
                    if _track.missed
                    <= self.max_age
                ]

                return self.get_tracks()

            # ----------------------------------------------------
            # 2. Construct association cost matrix.
            # ----------------------------------------------------
            _cost = self._build_cost_matrix(
                detections
            )

            # ----------------------------------------------------
            # 3. Hungarian algorithm finds the globally minimum
            # total assignment cost.
            # ----------------------------------------------------
            (
                _row_indices,
                _col_indices,
            ) = linear_sum_assignment(
                _cost
            )

            _matched_tracks = set()
            _matched_detections = set()

            # ----------------------------------------------------
            # 4. Apply distance gating.
            #
            # Hungarian always produces assignments when possible.
            # A geometrically unreasonable assignment is therefore
            # explicitly rejected if distance > max_distance.
            # ----------------------------------------------------
            for (
                _track_idx,
                _det_idx,
            ) in zip(
                _row_indices,
                _col_indices,
            ):
                if (
                    _cost[
                        _track_idx,
                        _det_idx,
                    ]
                    > self.max_distance
                ):
                    continue

                self.tracks[
                    _track_idx
                ].update(
                    detections[_det_idx],
                    _det_idx,
                )

                _matched_tracks.add(
                    int(_track_idx)
                )

                _matched_detections.add(
                    int(_det_idx)
                )

            # ----------------------------------------------------
            # 5. Unmatched detections become new tracks.
            # ----------------------------------------------------
            for _det_idx, _det in enumerate(
                detections
            ):
                if (
                    _det_idx
                    not in _matched_detections
                ):
                    self._create_track(
                        _det,
                        _det_idx,
                    )

            # ----------------------------------------------------
            # 6. Remove tracks that have disappeared for too long.
            #
            # Unmatched tracks were already predicted above, so
            # their "missed" counters have already increased.
            # ----------------------------------------------------
            self.tracks = [
                _track
                for _track in self.tracks
                if _track.missed
                <= self.max_age
            ]

            return self.get_tracks()

        def get_tracks(
            self,
            confirmed_only=True,
        ):
            """Return serializable dictionaries for active tracks."""

            _output = []

            for _track in self.tracks:
                if (
                    confirmed_only
                    and _track.hits
                    < self.min_hits
                ):
                    continue

                _output.append(
                    _track.as_dict()
                )

            return _output

    return (MultiObjectTracker,)


@app.cell
def _(MultiObjectTracker):
    # Tracker parameters can be tuned independently from detector
    # parameters.
    #
    # max_distance:
    #   maximum allowed motion in pixels between adjacent frames.
    #
    # max_age:
    #   tolerate short detector misses without immediately deleting
    #   the track.
    tracker = MultiObjectTracker(
        max_distance=80.0,
        max_age=8,
        min_hits=1,
        process_noise=2.0,
        measurement_noise=12.0,
        history_size=60,
    )
    return (tracker,)


@app.cell
def _(frame_index, frame_pnts, tracker):
    tracked_targets = tracker.update(
        frame_pnts,
        frame_index=frame_index,
    )
    return (tracked_targets,)


@app.cell
def _(cv):
    def draw_tracks(
        image,
        tracks,
        draw_bbox=True,
        draw_centroid=True,
        draw_trajectory=True,
        trajectory_thickness=1,
    ):
        """Draw tracked targets without modifying the source image.

        Label format
        ------------
        idx#track_id

        Examples
        --------
        3#7
            Detection index 3 in the current detector output
            belongs to persistent track ID 7.

        -#7
            Track 7 has no detector match in the current frame and
            is shown using Kalman prediction only.
        """

        _vis = image.copy()

        for _track in tracks:
            _track_id = int(
                _track["track_id"]
            )

            _det_idx = _track[
                "detection_idx"
            ]

            _cx = int(
                round(_track["x"])
            )

            _cy = int(
                round(_track["y"])
            )

            _bbox = _track["bbox"]

            _bx = int(
                _bbox["x"]
            )

            _by = int(
                _bbox["y"]
            )

            _bw = int(
                _bbox["w"]
            )

            _bh = int(
                _bbox["h"]
            )

            _updated = bool(
                _track["updated"]
            )

            # ----------------------------------------------------
            # Draw trajectory from previous filtered positions.
            # ----------------------------------------------------
            if draw_trajectory:
                _trajectory = _track[
                    "trajectory"
                ]

                if len(_trajectory) >= 2:
                    for _i in range(
                        1,
                        len(_trajectory),
                    ):
                        cv.line(
                            _vis,
                            _trajectory[
                                _i - 1
                            ],
                            _trajectory[_i],
                            (255, 0, 255),
                            trajectory_thickness,
                            cv.LINE_AA,
                        )

            # ----------------------------------------------------
            # Draw the latest detector bbox.
            #
            # Solid green means matched in this frame.
            # Predicted-only tracks remain visible using their last
            # detector bbox.
            # ----------------------------------------------------
            if draw_bbox:
                _bbox_color = (
                    (0, 255, 0)
                    if _updated
                    else (0, 255, 255)
                )

                cv.rectangle(
                    _vis,
                    (_bx, _by),
                    (
                        _bx + _bw,
                        _by + _bh,
                    ),
                    _bbox_color,
                    1,
                )

            # ----------------------------------------------------
            # Filtered/predicted centroid.
            # ----------------------------------------------------
            if draw_centroid:
                _point_color = (
                    (0, 0, 255)
                    if _updated
                    else (0, 165, 255)
                )

                cv.circle(
                    _vis,
                    (_cx, _cy),
                    4,
                    _point_color,
                    -1,
                )

            # ----------------------------------------------------
            # Current detector index + persistent tracking ID.
            #
            # Example:
            #
            #   4#12
            #
            # means detector candidate 4 is associated with
            # persistent track 12.
            # ----------------------------------------------------
            if _det_idx is None:
                _label = (
                    f"-#{_track_id}"
                )
            else:
                _label = (
                    f"{_det_idx}#{_track_id}"
                )

            cv.putText(
                _vis,
                _label,
                (
                    _bx,
                    max(_by - 5, 0),
                ),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )

        return _vis

    return (draw_tracks,)


@app.cell
def _(draw_tracks, frame, tracked_targets):
    tracked_vis = draw_tracks(
        frame,
        tracked_targets,
    )
    return (tracked_vis,)


@app.cell
def _(jshow, tracked_vis):
    jshow(tracked_vis)
    return


@app.cell
def _(next_frame):
    next_frame
    return


@app.cell
def _(
    blur_kernel_ui,
    max_area_ui,
    min_area_ui,
    min_contrast_ui,
    mo,
    threshold_sigma_ui,
):
    mo.vstack(
        [
            blur_kernel_ui,
            threshold_sigma_ui,
            min_contrast_ui,
            min_area_ui,
            max_area_ui,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
