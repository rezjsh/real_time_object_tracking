"""Streamlit page: upload an image and run detection (+ optional tracking)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402
from object_tracking_app.deployment.ui import (  # noqa: E402
    available_class_names,
    bgr_to_rgb,
    class_counts_to_dataframe,
    get_image_pipeline,
)
from object_tracking_app.utils.io import save_image  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

settings = get_settings()
deploy_cfg = settings.deploy.streamlit

st.set_page_config(page_title="Image Inference", layout=deploy_cfg.layout)
st.title("🖼️ Image Inference")

with st.sidebar:
    st.header("Detection settings")
    conf = st.slider("Confidence threshold", 0.0, 1.0, deploy_cfg.default_conf_threshold, 0.01)
    iou = st.slider("IOU threshold (NMS)", 0.0, 1.0, deploy_cfg.default_iou_threshold, 0.01)
    use_tracker = st.checkbox("Run tracker on this single frame", value=False)

    all_classes = available_class_names(settings)
    selected_classes = st.multiselect("Filter to classes (empty = all)", options=all_classes, default=[])

uploaded_file = st.file_uploader(
    "Upload an image", type=deploy_cfg.allowed_image_types, accept_multiple_files=False
)

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not decode the uploaded file as an image. Please try a different file.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(bgr_to_rgb(image_bgr), use_container_width=True)

        with st.spinner("Running detection..."):
            pipeline = get_image_pipeline(settings)
            t0 = time.perf_counter()
            result = pipeline.run(
                image_bgr,
                conf_threshold=conf,
                iou_threshold=iou,
                class_filter=selected_classes or None,
                use_tracker=use_tracker,
            )
            elapsed = time.perf_counter() - t0

        with col2:
            st.subheader("Detections")
            st.image(bgr_to_rgb(result["annotated_image"]), use_container_width=True)

        st.success(f"Found {result['summary']['num_detections']} objects in {elapsed*1000:.0f} ms.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Objects detected", result["summary"]["num_detections"])
        m2.metric("Mean confidence", f"{result['summary']['mean_confidence']:.2f}")
        m3.metric("Inference time", f"{elapsed*1000:.0f} ms")

        st.subheader("Class breakdown")
        df = class_counts_to_dataframe(result["summary"]["class_counts"])
        if not df.empty:
            st.bar_chart(df.set_index("class"))
        else:
            st.write("No objects detected above the current confidence threshold.")

        out_dir = settings.resolve(deploy_cfg.save_outputs_dir)
        out_path = out_dir / f"image_inference_{Path(uploaded_file.name).stem}.jpg"
        if st.button("💾 Save annotated image to artifacts/sample_outputs/"):
            save_image(result["annotated_image"], out_path)
            st.success(f"Saved to {out_path}")

        st.download_button(
            "⬇️ Download annotated image",
            data=cv2.imencode(".jpg", result["annotated_image"])[1].tobytes(),
            file_name=f"annotated_{uploaded_file.name}",
            mime="image/jpeg",
        )
else:
    st.info("Upload an image to get started.")
