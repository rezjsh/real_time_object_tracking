"""Tests for the image/video inference pipelines.

A `FakeDetector` stands in for the real ultralytics-backed `Detector` so
these tests run fast and offline (no pretrained weight download needed).
It implements the same `.predict(image, conf=..., iou=...)` interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from object_tracking_app.config.settings import get_settings
from object_tracking_app.inference.image_infer import ImageInferencePipeline
from object_tracking_app.inference.video_infer import VideoInferencePipeline
from object_tracking_app.models.detector import Detection
from object_tracking_app.utils.io import VideoWriter


class FakeDetector:
    """Deterministic stand-in for Detector: always returns one 'person' box."""

    def __init__(self, *args, **kwargs):
        self.class_names_map = {0: "person", 1: "car"}

    @property
    def class_names(self):
        return self.class_names_map

    def predict(self, image, conf=None, iou=None, classes=None, img_size=None):
        h, w = image.shape[:2]
        box = np.array([w * 0.2, h * 0.2, w * 0.6, h * 0.6], dtype=np.float32)
        return [Detection(xyxy=box, confidence=0.9, class_id=0, class_name="person")]


@pytest.fixture()
def fake_image() -> np.ndarray:
    return np.full((120, 160, 3), 200, dtype=np.uint8)


def test_image_pipeline_runs_and_annotates(fake_image):
    settings = get_settings()
    pipeline = ImageInferencePipeline(detector=FakeDetector(), settings=settings)
    result = pipeline.run(fake_image)

    assert result["summary"]["num_detections"] == 1
    assert result["summary"]["class_counts"] == {"person": 1}
    assert result["annotated_image"].shape == fake_image.shape
    # Annotated image should differ from the original since a box was drawn.
    assert not np.array_equal(result["annotated_image"], fake_image)


def test_image_pipeline_respects_class_filter(fake_image):
    settings = get_settings()
    pipeline = ImageInferencePipeline(detector=FakeDetector(), settings=settings)
    result = pipeline.run(fake_image, class_filter=["car"])
    assert result["summary"]["num_detections"] == 0


def test_image_pipeline_with_tracker(fake_image):
    settings = get_settings()
    pipeline = ImageInferencePipeline(detector=FakeDetector(), settings=settings)
    result = pipeline.run(fake_image, use_tracker=True)
    assert result["track_ids"] is not None
    assert len(result["track_ids"]) == 1


def test_image_pipeline_run_and_save(tmp_path, fake_image):
    settings = get_settings()
    pipeline = ImageInferencePipeline(detector=FakeDetector(), settings=settings)
    out_path = tmp_path / "out.jpg"
    pipeline.run_and_save(fake_image, out_path)
    assert out_path.exists()


def _make_fake_video(path: Path, n_frames: int = 5, size=(160, 120)) -> None:
    writer = VideoWriter(path, fps=10.0, frame_size=size, codec="mp4v")
    for _ in range(n_frames):
        frame = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_video_pipeline_processes_all_frames(tmp_path):
    settings = get_settings()
    video_path = tmp_path / "input.mp4"
    _make_fake_video(video_path, n_frames=5)

    pipeline = VideoInferencePipeline(detector=FakeDetector(), settings=settings)
    output_path = tmp_path / "output.mp4"
    summary = pipeline.run(video_path, output_path=output_path)

    assert summary["frames_processed"] == 5
    assert output_path.exists()
    assert summary["unique_objects_tracked"] >= 1


def test_video_pipeline_respects_max_frames(tmp_path):
    settings = get_settings()
    video_path = tmp_path / "input.mp4"
    _make_fake_video(video_path, n_frames=10)

    pipeline = VideoInferencePipeline(detector=FakeDetector(), settings=settings)
    summary = pipeline.run(video_path, output_path=None, max_frames=3)

    assert summary["frames_processed"] == 3
