import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import sys
    import cv2 as cv
    import numpy as np

    return cv, np, os


@app.cell
def _(os):
    print(f"{os.listdir() = }")
    print(f"{os.listdir('videos') = }")
    videos = ["./videos/" + x for x in os.listdir('videos')]
    videos
    return (videos,)


@app.cell
def _():
    import time
    from typing import Callable, Any


    def timer(func: Callable, params: dict) -> tuple[Any, float]:
        start = time.perf_counter()
        result = func(**params)
        elapsed = time.perf_counter() - start

        return result, elapsed

    return (timer,)


@app.cell
def _(cv, mo):
    import matplotlib.pyplot as plt


    def jshow(image):
        """在 marimo notebook 中显示 OpenCV 图片。"""
        if image is None:
            return mo.md("**Image is None**")

        if image.ndim == 2:
            # 灰度图
            display_image = image
        elif image.shape[2] == 3:
            # OpenCV BGR -> RGB
            display_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # OpenCV BGRA -> RGBA
            display_image = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return mo.image(display_image)

    return (jshow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Top local-contrast points

    `ktop_contrast_points` computes a local-contrast score for each pixel using the absolute difference from a Gaussian-blurred image, keeps local maxima with non-maximum suppression, and returns points ordered by decreasing contrast. `ktop='auto'` chooses the number of points from the number of strong local maxima detected in the image.
    """)
    return


@app.cell
def _(cv, np, os):
    def ktop_contrast_points(
        picture,
        ktop="auto",
        blur_kernel=41,
        threshold_sigma=3.0,
        min_mean_contrast=5.0,
        min_area=10,
        max_area=1500,
        min_width=4,
        min_height=2,
        max_width=120,
        max_height=80,
        min_distance=7,
        border_margin=20,
        max_auto=100,
    ):
        """
        Detect dark objects against a locally brighter background.

        Returns
        -------
        list[dict]
            {
                "x": centroid x,
                "y": centroid y,
                "contrast": region score,
                "mean_contrast": mean local contrast,
                "p90_contrast": 90th-percentile local contrast,
                "area": contour area,
                "bbox": {
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                },
            }
        """

        # ------------------------------------------------------------
        # Load image
        # ------------------------------------------------------------
        if isinstance(picture, (str, bytes, os.PathLike)):
            image = cv.imread(os.fspath(picture), cv.IMREAD_UNCHANGED)

            if image is None:
                raise ValueError(f"Could not read image: {picture}")

        elif isinstance(picture, np.ndarray):
            image = picture

        else:
            raise TypeError("picture must be numpy.ndarray or image path")

        if image.size == 0:
            raise ValueError("picture must not be empty")

        # ------------------------------------------------------------
        # Gray
        # ------------------------------------------------------------
        if image.ndim == 2:
            gray = image

        elif image.ndim == 3 and image.shape[2] == 1:
            gray = image[..., 0]

        elif image.ndim == 3 and image.shape[2] >= 3:
            gray = cv.cvtColor(
                image[..., :3],
                cv.COLOR_BGR2GRAY,
            )

        else:
            raise ValueError("Unsupported image format")

        # ------------------------------------------------------------
        # Validate
        # ------------------------------------------------------------
        if ktop != "auto":
            if not isinstance(ktop, (int, np.integer)) or ktop < 1:
                raise ValueError("ktop must be positive integer or 'auto'")

        blur_kernel = int(blur_kernel)

        if blur_kernel < 3:
            raise ValueError("blur_kernel must be >= 3")

        if blur_kernel % 2 == 0:
            blur_kernel += 1

        # ------------------------------------------------------------
        # Local background
        #
        # A bird is darker than surrounding sky:
        #
        #     contrast = background - pixel
        #
        # Positive response = dark object.
        # ------------------------------------------------------------
        gray_f = gray.astype(np.float32)

        background = cv.GaussianBlur(
            gray_f,
            (blur_kernel, blur_kernel),
            0,
        )

        contrast = background - gray_f

        # Ignore brighter-than-background structures
        contrast = np.maximum(
            contrast,
            0,
        )

        # ------------------------------------------------------------
        # Robust noise estimation
        # ------------------------------------------------------------
        values = contrast.ravel()

        median = float(
            np.median(values)
        )

        mad = float(
            np.median(
                np.abs(values - median)
            )
        )

        robust_sigma = 1.4826 * mad

        threshold = (
            median
            + threshold_sigma * robust_sigma
        )

        # Useful for smooth 8-bit sky images
        threshold = max(
            threshold,
            3.0,
        )

        mask = (
            contrast >= threshold
        ).astype(np.uint8) * 255

        # ------------------------------------------------------------
        # Remove border responses
        # ------------------------------------------------------------
        h, w = gray.shape

        if border_margin > 0:
            m = min(
                border_margin,
                h // 2,
                w // 2,
            )

            mask[:m, :] = 0
            mask[h - m:, :] = 0
            mask[:, :m] = 0
            mask[:, w - m:] = 0

        # ------------------------------------------------------------
        # Small closing only
        #
        # Do NOT use MORPH_OPEN: thin bird wings may disappear.
        # ------------------------------------------------------------
        close_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (3, 3),
        )

        mask = cv.morphologyEx(
            mask,
            cv.MORPH_CLOSE,
            close_kernel,
        )

        # ------------------------------------------------------------
        # Contours
        # ------------------------------------------------------------
        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:

            area = float(
                cv.contourArea(contour)
            )

            if area < min_area:
                continue

            if max_area is not None and area > max_area:
                continue

            bx, by, bw, bh = cv.boundingRect(
                contour
            )

            # --------------------------------------------------------
            # Bbox size filters
            # --------------------------------------------------------
            if bw < min_width or bh < min_height:
                continue

            if bw > max_width or bh > max_height:
                continue

            # Extra border protection
            if (
                bx <= border_margin
                or by <= border_margin
                or bx + bw >= w - border_margin
                or by + bh >= h - border_margin
            ):
                continue

            # --------------------------------------------------------
            # Region mask
            # --------------------------------------------------------
            region_mask = np.zeros(
                gray.shape,
                dtype=np.uint8,
            )

            cv.drawContours(
                region_mask,
                [contour],
                -1,
                255,
                cv.FILLED,
            )

            region_values = contrast[
                region_mask != 0
            ]

            if region_values.size == 0:
                continue

            mean_contrast = float(
                region_values.mean()
            )

            p90_contrast = float(
                np.percentile(
                    region_values,
                    90,
                )
            )

            # --------------------------------------------------------
            # Reject weak sky texture
            # --------------------------------------------------------
            if mean_contrast < min_mean_contrast:
                continue

            # --------------------------------------------------------
            # Centroid
            # --------------------------------------------------------
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
                cx = bx + bw // 2
                cy = by + bh // 2

            # --------------------------------------------------------
            # Ranking score
            #
            # Contrast dominates; area only weakly contributes.
            # --------------------------------------------------------
            score = (
                p90_contrast
                * np.sqrt(max(area, 1.0))
            )

            candidates.append(
                {
                    "x": cx,
                    "y": cy,
                    "contrast": score,
                    "mean_contrast": mean_contrast,
                    "p90_contrast": p90_contrast,
                    "area": area,
                    "bbox": {
                        "x": int(bx),
                        "y": int(by),
                        "w": int(bw),
                        "h": int(bh),
                    },
                }
            )

        # ------------------------------------------------------------
        # Strongest first
        # ------------------------------------------------------------
        candidates.sort(
            key=lambda p: p["contrast"],
            reverse=True,
        )

        # ------------------------------------------------------------
        # Centroid NMS
        # ------------------------------------------------------------
        selected = []

        min_dist_sq = (
            min_distance
            * min_distance
        )

        for candidate in candidates:

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
                    dx * dx + dy * dy
                    < min_dist_sq
                ):
                    keep = False
                    break

            if keep:
                selected.append(
                    candidate
                )

        # ------------------------------------------------------------
        # ktop
        # ------------------------------------------------------------
        if ktop == "auto":
            return selected[:max_auto]

        return selected[:int(ktop)]

    return (ktop_contrast_points,)


@app.cell
def _(cv, videos):
    video = cv.VideoCapture(videos[0])
    video.isOpened()
    return (video,)


@app.cell
def _(jshow, video):
    _, img = video.read()
    jshow(img)
    return (img,)


@app.cell
def _(img, ktop_contrast_points, timer):
    pnts, el = timer(
        ktop_contrast_points,
        {
            "picture": img,
            "ktop": "auto",

            "blur_kernel": 41,

            "threshold_sigma": 3.0,
            "min_mean_contrast": 5.0,

            "min_area": 10,
            "max_area": 1500,

            "min_width": 4,
            "min_height": 2,
            "max_width": 120,
            "max_height": 80,

            "min_distance": 7,

            "border_margin": 20,

            "max_auto": 100,
        },
    )

    print(
        f"time: {el * 1000:.2f} ms, "
        f"targets: {len(pnts)}"
    )
    return (pnts,)


@app.cell
def _(pnts):
    pnts[0]
    return


@app.cell
def _(cv):
    def draw_pnts(img, pnts):
        vis = img.copy()
    
        for idx, p in enumerate(pnts):
            x, y = int(p["x"]), int(p["y"])
    
            bbox = p["bbox"]
            bx = int(bbox["x"])
            by = int(bbox["y"])
            bw = int(bbox["w"])
            bh = int(bbox["h"])
    
            # bbox
            cv.rectangle(
                vis,
                (bx, by),
                (bx + bw, by + bh),
                (0, 255, 0),
                1,
            )
    
            # centroid point
            cv.circle(
                vis,
                (x, y),
                4,
                (0, 0, 255),
                -1,
            )
    
            # index
            cv.putText(
                vis,
                str(idx),
                (bx, max(by - 5, 0)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv.LINE_AA,
            )
        return vis

    return (draw_pnts,)


@app.cell
def _(draw_pnts, img, jshow, pnts):
    vis = draw_pnts(img, pnts)
    jshow(vis)
    return


@app.cell
def _(mo):
    next_frame = mo.ui.button(
        value=0,
        on_click=lambda value: value + 1,
        label="Next frame",
    )

    next_frame
    return (next_frame,)


@app.cell
def _(cv, draw_pnts, jshow, ktop_contrast_points, next_frame, video):
    _ = next_frame.value

    ok, frame = video.read()

    if not ok:
        video.set(cv.CAP_PROP_POS_FRAMES, 0)
        ok, frame = video.read()

    _pnts = ktop_contrast_points(
        picture=frame,
        ktop="auto",
        blur_kernel=41,
        threshold_sigma=3.0,
        min_mean_contrast=5.0,
        min_area=10,
        max_area=1500,
        min_width=4,
        min_height=2,
        max_width=1000,
        max_height=80,
        min_distance=7,
        border_margin=20,
        max_auto=100,
    )

    _vis = draw_pnts(frame, _pnts)

    jshow(_vis)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kalman et Hungarian for tracking
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
