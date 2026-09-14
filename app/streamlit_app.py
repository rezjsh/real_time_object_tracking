"""Streamlit deployment UI -- main entry page.

Run with:
    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402

settings = get_settings()
deploy_cfg = settings.deploy.streamlit

st.set_page_config(
    page_title=deploy_cfg.title,
    layout=deploy_cfg.layout,
    initial_sidebar_state="expanded",
)

st.title("🎯 " + deploy_cfg.title)

st.markdown(
    """
Welcome! This app runs a fine-tuned YOLO-style detector with a lightweight,
self-contained multi-object tracker (ByteTrack-style, Kalman-filter based)
on images, videos, or a live webcam feed.

**Use the pages in the sidebar to get started:**
- **1 · Image Inference** — upload an image, tune thresholds, see annotated detections.
- **2 · Video Inference** — upload a video, run detection + tracking across frames, download the result.
- **3 · Analytics** — inspect class distributions, confidence histograms, and tracking stability
  from your most recent run.
"""
)

with st.sidebar:
    st.header("⚙️ Model status")
    weights_path = settings.resolve(settings.inference.weights_path)
    if weights_path.exists():
        st.success(f"Fine-tuned weights found:\n`{weights_path.name}`")
    else:
        st.warning(
            f"No fine-tuned checkpoint at `{settings.inference.weights_path}` yet.\n\n"
            f"Falling back to pretrained `{settings.inference.fallback_weights}`.\n\n"
            "Run `uv run python scripts/train.py` to fine-tune on your COCO subset."
        )

    st.header("📚 Curated classes")
    st.write(", ".join(settings.dataset.classes))

    st.header("🔗 Links")
    st.markdown("- [Project README](README.md)\n- [Architecture docs](../docs/architecture.md)")

st.info(
    "Tip: the first inference call lazily loads the model into memory, which can take a few seconds.",
    icon="ℹ️",
)
