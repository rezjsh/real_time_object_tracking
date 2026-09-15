"""Tests for the tracker, post-processing, and visualization utilities.

These tests avoid instantiating the actual ultralytics Detector (which would
require downloading pretrained weights) by constructing `Detection` objects
directly, since Detection is a plain dataclass.
"""

from __future__ import annotations

import numpy as np

from object_tracking_app.config.settings import get_settings
from object_tracking_app.models.detector import Detection
from object_tracking_app.models.postprocess import (
    filter_by_classes,
    filter_by_confidence,
    non_max_suppression,
    postprocess_detections,
)
from object_tracking_app.models.tracker import MultiObjectTracker, iou_matrix
from object_tracking_app.utils.viz import class_count_summary, color_for_id, draw_detections


def _det(x1, y1, x2, y2, conf=0.8, class_id=0, class_name="person"):
    return Detection(xyxy=np.array([x1, y1, x2, y2], dtype=np.float32), confidence=conf, class_id=class_id, class_name=class_name)


# --------------------------------------------------------------------------- #
# IOU matrix
# --------------------------------------------------------------------------- #
def test_iou_matrix_identical_boxes():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    m = iou_matrix(boxes, boxes)
    assert m.shape == (1, 1)
    assert np.isclose(m[0, 0], 1.0)


def test_iou_matrix_no_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[100, 100, 110, 110]], dtype=np.float32)
    m = iou_matrix(a, b)
    assert np.isclose(m[0, 0], 0.0)


def test_iou_matrix_empty_inputs():
    a = np.zeros((0, 4), dtype=np.float32)
    b = np.array([[0, 0, 10, 10]], dtype=np.float32)
    m = iou_matrix(a, b)
    assert m.shape == (0, 1)


# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
def test_filter_by_confidence():
    dets = [_det(0, 0, 10, 10, conf=0.9), _det(0, 0, 10, 10, conf=0.2)]
    filtered = filter_by_confidence(dets, threshold=0.5)
    assert len(filtered) == 1
    assert filtered[0].confidence == 0.9


def test_filter_by_classes():
    dets = [_det(0, 0, 10, 10, class_name="person"), _det(0, 0, 10, 10, class_name="car")]
    filtered = filter_by_classes(dets, ["car"])
    assert len(filtered) == 1
    assert filtered[0].class_name == "car"


def test_filter_by_classes_none_returns_all():
    dets = [_det(0, 0, 10, 10, class_name="person"), _det(0, 0, 10, 10, class_name="car")]
    filtered = filter_by_classes(dets, None)
    assert len(filtered) == 2


def test_non_max_suppression_removes_overlaps():
    dets = [
        _det(0, 0, 10, 10, conf=0.9, class_id=0),
        _det(1, 1, 11, 11, conf=0.8, class_id=0),  # heavily overlaps with the above
        _det(50, 50, 60, 60, conf=0.7, class_id=0),  # far away, should survive
    ]
    kept = non_max_suppression(dets, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0].confidence == 0.9
    kept_boxes = {tuple(d.xyxy.tolist()) for d in kept}
    assert (50.0, 50.0, 60.0, 60.0) in kept_boxes


def test_postprocess_detections_chain():
    dets = [
        _det(0, 0, 10, 10, conf=0.9, class_name="person"),
        _det(0, 0, 10, 10, conf=0.1, class_name="person"),
        _det(0, 0, 10, 10, conf=0.9, class_name="car"),
    ]
    result = postprocess_detections(dets, conf_threshold=0.5, allowed_class_names=["person"])
    assert len(result) == 1
    assert result[0].class_name == "person"


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
def test_tracker_assigns_stable_id_across_frames():
    settings = get_settings()
    tracker = MultiObjectTracker(settings=settings)

    # Object moves slightly to the right each frame; should retain the same ID.
    ids_seen = []
    for i in range(6):
        dets = [_det(10 + i * 2, 10, 40 + i * 2, 40, conf=0.9)]
        tracked = tracker.update(dets)
        if tracked:
            ids_seen.append(tracked[0].track_id)

    assert len(ids_seen) > 0
    assert len(set(ids_seen)) == 1  # same ID maintained throughout


def test_tracker_creates_new_ids_for_distinct_objects():
    settings = get_settings()
    tracker = MultiObjectTracker(settings=settings)

    dets = [_det(0, 0, 20, 20, conf=0.9), _det(100, 100, 120, 120, conf=0.9)]
    for _ in range(4):  # feed several frames so tracks confirm (min_hits)
        tracked = tracker.update(dets)

    assert len(tracked) == 2
    assert tracked[0].track_id != tracked[1].track_id


def test_tracker_drops_tracks_after_max_age():
    settings = get_settings()
    settings.tracker.max_age = 2
    settings.tracker.min_hits = 1
    tracker = MultiObjectTracker(settings=settings)

    dets = [_det(0, 0, 20, 20, conf=0.9)]
    tracker.update(dets)
    assert len(tracker.active_track_ids()) == 1

    # Feed empty frames beyond max_age -- track should be dropped.
    for _ in range(5):
        tracker.update([])

    assert len(tracker.active_track_ids()) == 0


def test_tracker_reset_clears_state():
    settings = get_settings()
    tracker = MultiObjectTracker(settings=settings)
    tracker.update([_det(0, 0, 20, 20, conf=0.9)])
    assert len(tracker.active_track_ids()) >= 0
    tracker.reset()
    assert tracker.active_track_ids() == []
    assert tracker.stats.total_tracks_created == 0


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def test_draw_detections_does_not_crash_with_none_track_ids():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    out = draw_detections(frame, boxes, [0.9], ["person"], track_ids=[None])
    assert out.shape == frame.shape


def test_draw_detections_with_track_ids():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
    out = draw_detections(frame, boxes, [0.9], ["person"], track_ids=[7])
    assert not np.array_equal(out, np.zeros((100, 100, 3), dtype=np.uint8))


def test_color_for_id_is_deterministic():
    assert color_for_id(5) == color_for_id(5)


def test_class_count_summary():
    counts = class_count_summary(["person", "car", "person"])
    assert counts == {"person": 2, "car": 1}
