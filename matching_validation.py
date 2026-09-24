import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")


@app.cell
def _():
    import cv2 as cv
    import marimo as mo

    from lib import (
        ContrastDetectorConfig,
        detect_contrast_targets,
        draw_pair_matches,
        get_video_info,
        list_videos,
        pair_detections_flock_motion,
        pair_detections_hybrid,
        read_video_frame,
    )

    return (
        ContrastDetectorConfig,
        cv,
        detect_contrast_targets,
        draw_pair_matches,
        get_video_info,
        list_videos,
        mo,
        pair_detections_flock_motion,
        pair_detections_hybrid,
        read_video_frame,
    )


@app.cell
def _(cv, mo):
    def jshow(image, width=None):
        """Display an OpenCV BGR/gray image in marimo."""
        if image is None:
            return mo.md("**Image is None**")
        if image.ndim == 2:
            _display = image
        elif image.shape[2] == 3:
            _display = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            _display = cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        return mo.image(_display, width=width)

    return (jshow,)


@app.cell
def _(mo):
    mo.md("""
    # Matching validation

    This notebook validates the two association layers independently of the
    stateful Kalman tracker.

    ```mermaid
    flowchart LR
        A[Frame t detections] --> B[IoU + Center Distance]
        C[Frame t+dt detections] --> B
        B --> D[Hungarian]

        A --> E[Flock Motion Pairing]
        C --> E
        E --> F[Global translation]
        F --> G[Local KNN + global flock feature]
        G --> H[Hungarian]
    ```
    """)
    return


@app.cell
def _(list_videos):
    validation_videos = list_videos("./videos")
    if not validation_videos:
        raise RuntimeError("No videos found in ./videos")
    return (validation_videos,)


@app.cell
def _(mo, validation_videos):
    validation_video_ui = mo.ui.dropdown(
        options=validation_videos,
        value=validation_videos[0],
        label="Video",
    )
    validation_video_ui
    return (validation_video_ui,)


@app.cell
def _(get_video_info, validation_video_ui):
    validation_video_info = get_video_info(validation_video_ui.value)
    return (validation_video_info,)


@app.cell
def _(mo, validation_video_info):
    _max_frame = max(0, int(validation_video_info["frame_count"]) - 2)
    validation_frame_ui = mo.ui.slider(
        start=0,
        stop=_max_frame,
        step=1,
        value=0,
        label="Base frame",
    )
    validation_gap_ui = mo.ui.slider(
        start=1,
        stop=30,
        step=1,
        value=1,
        label="Frame gap",
    )
    mo.hstack([validation_frame_ui, validation_gap_ui])
    return validation_frame_ui, validation_gap_ui


@app.cell
def _(ContrastDetectorConfig):
    validation_detector_config = ContrastDetectorConfig(
        blur_kernel=41,
        threshold_sigma=3.0,
        min_mean_contrast=5.0,
        min_area=10.0,
        max_area=1500.0,
        min_width=4,
        min_height=2,
        max_width=120,
        max_height=80,
        min_distance=7,
        border_margin=20,
        max_auto=100,
    )
    return (validation_detector_config,)


@app.cell
def _(
    read_video_frame,
    validation_frame_ui,
    validation_gap_ui,
    validation_video_info,
    validation_video_ui,
):
    _base_index = int(validation_frame_ui.value)
    _next_index = min(
        _base_index + int(validation_gap_ui.value),
        int(validation_video_info["frame_count"]) - 1,
    )
    validation_prev_frame, validation_prev_index = read_video_frame(
        validation_video_ui.value,
        _base_index,
        loop=False,
    )
    validation_curr_frame, validation_curr_index = read_video_frame(
        validation_video_ui.value,
        _next_index,
        loop=False,
    )
    return (
        validation_curr_frame,
        validation_curr_index,
        validation_prev_frame,
        validation_prev_index,
    )


@app.cell
def _(
    detect_contrast_targets,
    validation_curr_frame,
    validation_detector_config,
    validation_prev_frame,
):
    validation_prev_detections, validation_prev_diag = detect_contrast_targets(
        validation_prev_frame,
        validation_detector_config,
    )
    validation_curr_detections, validation_curr_diag = detect_contrast_targets(
        validation_curr_frame,
        validation_detector_config,
    )
    return validation_curr_detections, validation_prev_detections


@app.cell
def _(mo):
    hybrid_center_ui = mo.ui.slider(
        start=10,
        stop=300,
        step=5,
        value=80,
        label="Hybrid max center distance",
    )
    hybrid_center_weight_ui = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.05,
        value=0.70,
        label="Center weight",
    )
    mo.hstack([hybrid_center_ui, hybrid_center_weight_ui])
    return hybrid_center_ui, hybrid_center_weight_ui


@app.cell
def _(
    hybrid_center_ui,
    hybrid_center_weight_ui,
    pair_detections_hybrid,
    validation_curr_detections,
    validation_prev_detections,
):
    _center_weight = float(hybrid_center_weight_ui.value)
    hybrid_pair_result = pair_detections_hybrid(
        validation_prev_detections,
        validation_curr_detections,
        max_center_distance=float(hybrid_center_ui.value),
        center_weight=_center_weight,
        iou_weight=1.0 - _center_weight,
    )
    return (hybrid_pair_result,)


@app.cell
def _(
    draw_pair_matches,
    hybrid_pair_result,
    jshow,
    mo,
    validation_curr_detections,
    validation_curr_frame,
    validation_prev_detections,
    validation_prev_frame,
):
    _vis = draw_pair_matches(
        validation_prev_frame,
        validation_curr_frame,
        validation_prev_detections,
        validation_curr_detections,
        hybrid_pair_result,
    )
    mo.vstack(
        [
            mo.md(
                f"## IoU + CenterDistance\n"
                f"Matches: **{len(hybrid_pair_result['matches'])}**, "
                f"unmatched prev: **{len(hybrid_pair_result['unmatched_previous'])}**, "
                f"unmatched current: **{len(hybrid_pair_result['unmatched_current'])}**"
            ),
            jshow(_vis),
            mo.ui.table(hybrid_pair_result["matches"]),
        ]
    )
    return


@app.cell
def _(mo):
    flock_radius_ui = mo.ui.slider(
        start=5,
        stop=150,
        step=5,
        value=45,
        label="Translation inlier radius",
    )
    flock_center_gate_ui = mo.ui.slider(
        start=20,
        stop=500,
        step=5,
        value=120,
        label="Compensated center gate",
    )
    flock_neighbors_ui = mo.ui.slider(
        start=1,
        stop=8,
        step=1,
        value=3,
        label="Local neighbors K",
    )
    mo.hstack([flock_radius_ui, flock_center_gate_ui, flock_neighbors_ui])
    return flock_center_gate_ui, flock_neighbors_ui, flock_radius_ui


@app.cell
def _(
    flock_center_gate_ui,
    flock_neighbors_ui,
    flock_radius_ui,
    pair_detections_flock_motion,
    validation_curr_detections,
    validation_prev_detections,
):
    flock_pair_result = pair_detections_flock_motion(
        validation_prev_detections,
        validation_curr_detections,
        local_neighbors=int(flock_neighbors_ui.value),
        translation_inlier_radius=float(flock_radius_ui.value),
        max_compensated_distance=float(flock_center_gate_ui.value),
    )
    return (flock_pair_result,)


@app.cell
def _(
    draw_pair_matches,
    flock_pair_result,
    jshow,
    mo,
    validation_curr_detections,
    validation_curr_frame,
    validation_prev_detections,
    validation_prev_frame,
):
    _translation = flock_pair_result["translation"]
    _vis = draw_pair_matches(
        validation_prev_frame,
        validation_curr_frame,
        validation_prev_detections,
        validation_curr_detections,
        flock_pair_result,
    )
    mo.vstack(
        [
            mo.md(
                f"## Flock Motion Pairing\n"
                f"Estimated translation: **({_translation[0]:.2f}, {_translation[1]:.2f}) px**  \n"
                f"Translation support: **{flock_pair_result['translation_support']}**  \n"
                f"Matches: **{len(flock_pair_result['matches'])}**, "
                f"unmatched prev: **{len(flock_pair_result['unmatched_previous'])}**, "
                f"unmatched current: **{len(flock_pair_result['unmatched_current'])}**"
            ),
            jshow(_vis),
            mo.ui.table(flock_pair_result["matches"]),
        ]
    )
    return


@app.cell
def _(
    hybrid_pair_result,
    mo,
    validation_curr_detections,
    validation_curr_index,
    validation_prev_detections,
    validation_prev_index,
):
    mo.md(
        f"""
    ### Pair summary

    - Previous frame: **{validation_prev_index}** — detections: **{len(validation_prev_detections)}**
    - Current frame: **{validation_curr_index}** — detections: **{len(validation_curr_detections)}**
    - Hybrid pairs: **{len(hybrid_pair_result['matches'])}**
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
