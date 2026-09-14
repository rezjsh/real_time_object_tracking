"""Tests for the FastAPI inference service.

Uses FastAPI's TestClient with the pipeline monkeypatched to a fake detector
so no real model weights are downloaded during testing.
"""

from __future__ import annotations


import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from object_tracking_app.deployment import api as api_module
from object_tracking_app.inference.image_infer import ImageInferencePipeline
from object_tracking_app.models.detector import Detection


class FakeDetector:
    def __init__(self, *args, **kwargs):
        self.class_names_map = {0: "person"}

    @property
    def class_names(self):
        return self.class_names_map

    def predict(self, image, conf=None, iou=None, classes=None, img_size=None):
        h, w = image.shape[:2]
        box = np.array([w * 0.1, h * 0.1, w * 0.5, h * 0.5], dtype=np.float32)
        return [Detection(xyxy=box, confidence=0.85, class_id=0, class_name="person")]


@pytest.fixture(autouse=True)
def patch_pipeline(monkeypatch):
    """Force the API's global pipeline to use the fake detector instead of downloading weights."""
    fake_pipeline = ImageInferencePipeline(detector=FakeDetector(), settings=api_module.settings)
    monkeypatch.setattr(api_module, "_pipeline", fake_pipeline)
    yield


@pytest.fixture()
def client() -> TestClient:
    return TestClient(api_module.app)


def _sample_jpeg_bytes() -> bytes:
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_classes_endpoint(client):
    resp = client.get("/classes")
    assert resp.status_code == 200
    assert "person" in resp.json()["classes"].values()


def test_predict_returns_json(client):
    files = {"file": ("test.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_detections"] == 1
    assert data["detections"][0]["class_name"] == "person"


def test_predict_returns_image_when_requested(client):
    files = {"file": ("test.jpg", _sample_jpeg_bytes(), "image/jpeg")}
    resp = client.post("/predict?return_image=true", files=files)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 0


def test_predict_rejects_invalid_file(client):
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400
