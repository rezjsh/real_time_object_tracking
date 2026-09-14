"""Shared helper functions for the Streamlit deployment UI.

Keeping these out of app/streamlit_app.py means the actual page files stay
focused on layout/widgets, while this module owns caching, model loading,
and result-formatting logic that's easy to unit test.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.inference.image_infer import ImageInferencePipeline
from object_tracking_app.inference.video_infer import VideoInferencePipeline
from object_tracking_app.models.detector import Detector
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)

_DETECTOR_CACHE: dict[str, Detector] = {}


def get_cached_detector(settings: Optional[Settings] = None) -> Detector:
    """Reuse a single Detector instance across Streamlit reruns (keyed by weights path)."""
    settings = settings or get_settings()
    key = settings.inference.weights_path
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = Detector(settings=settings)
    return _DETECTOR_CACHE[key]


def get_image_pipeline(settings: Optional[Settings] = None) -> ImageInferencePipeline:
    settings = settings or get_settings()
    return ImageInferencePipeline(detector=get_cached_detector(settings), settings=settings)


def get_video_pipeline(settings: Optional[Settings] = None) -> VideoInferencePipeline:
    settings = settings or get_settings()
    return VideoInferencePipeline(detector=get_cached_detector(settings), settings=settings)


def class_counts_to_dataframe(class_counts: dict) -> pd.DataFrame:
    if not class_counts:
        return pd.DataFrame(columns=["class", "count"])
    df = pd.DataFrame(sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True), columns=["class", "count"])
    return df


def confidence_histogram_data(confidences: list[float], bins: int = 10) -> pd.DataFrame:
    if not confidences:
        return pd.DataFrame(columns=["bin", "count"])
    hist, edges = np.histogram(confidences, bins=bins, range=(0.0, 1.0))
    labels = [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return pd.DataFrame({"bin": labels, "count": hist})


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def available_class_names(settings: Optional[Settings] = None) -> list[str]:
    settings = settings or get_settings()
    try:
        detector = get_cached_detector(settings)
        return sorted(detector.class_names.values())
    except Exception as exc:  # model may not be downloadable in a sandboxed context
        logger.warning("Could not load detector class names (%s); falling back to configured classes.", exc)
        return sorted(settings.dataset.classes)
