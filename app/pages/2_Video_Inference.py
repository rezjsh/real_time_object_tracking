"""Streamlit page: upload a video and run detection + tracking across frames."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402
from object_tracking_app.deployment.ui import (  # noqa: E402
    available_class_names,
    class_counts_to_dataframe,
    get_video_pipeline,
)
from object_tracking_app.utils.io import ensure_dir  # noqa: E402

settings = get_settings()
deploy_cfg = settings.deploy.streamlit

st.set_page_config(page_title="Video Inference", layout=deploy_cfg.layout)
st.title("🎬 Video Inference")

with st.sidebar:
    st.header("Detection settings")
    conf = st.slider("Confidence threshold", 0.0, 1.0, deploy_cfg.default_conf_threshold, 0.01)
    iou = st.slider("IOU threshold (NMS)", 0.0, 1.0, deploy_cfg.default_iou_threshold, 0.01)
    max_frames = st.number_input(
        "Max frames to process (0 = full video)", min_value=0, value=300, step=50,
        help="Cap runtime for long videos, especially on CPU-only machines.",
    )

    all_classes = available_class_names(settings)
    selected_classes = st.multiselect("Filter to classes (empty = all)", options=all_classes, default=[])

uploaded_file = st.file_uploader(
    "Upload a video", type=deploy_cfg.allowed_video_types, accept_multiple_files=False
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    out_dir = ensure_dir(settings.resolve(deploy_cfg.save_outputs_dir))
    output_path = out_dir / f"video_inference_{Path(uploaded_file.name).stem}.mp4"

    progress_bar = st.progress(0, text="Starting...")

    def _progress(frame_idx: int, total: int) -> None:
        if total > 0:
            pct = min(1.0, frame_idx / total)
            progress_bar.progress(pct, text=f"Processing frame {frame_idx}/{total}")
        else:
            progress_bar.progress(0, text=f"Processing frame {frame_idx}")

    if st.button("▶️ Run detection + tracking", type="primary"):
        pipeline = get_video_pipeline(settings)
        with st.spinner("Processing video... this can take a while on CPU."):
            summary = pipeline.run(
                input_path,
                output_path=output_path,
                conf_threshold=conf,
                iou_threshold=iou,
                class_filter=selected_classes or None,
                max_frames=(max_frames or None),
                progress_callback=_progress,
            )
        progress_bar.progress(1.0, text="Done!")

        st.success(
            f"Processed {summary['frames_processed']} frames, "
            f"tracked {summary['unique_objects_tracked']} unique objects."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Frames processed", summary["frames_processed"])
        m2.metric("Unique objects tracked", summary["unique_objects_tracked"])
        m3.metric("Avg FPS", f"{summary['avg_fps']:.1f}")

        st.subheader("Class totals across all frames")
        df = class_counts_to_dataframe(summary["class_totals"])
        if not df.empty:
            st.bar_chart(df.set_index("class"))

        st.subheader("Tracker stability")
        st.json(summary["tracker_stats"])

        if output_path.exists():
            st.subheader("Annotated video")
            st.video(str(output_path))
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download annotated video", data=f.read(),
                    file_name=output_path.name, mime="video/mp4",
                )
else:
    st.info("Upload a video to get started. Long videos on CPU can be slow -- use the frame cap in the sidebar.")
