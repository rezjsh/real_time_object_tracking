"""Multi-object tracker: a self-contained SORT/ByteTrack-style tracker.

Implements the classic SORT pipeline (Kalman-filter motion prediction +
Hungarian/IOU association) extended with ByteTrack's two-stage association:
high-confidence detections are matched first, then unmatched tracks get a
second chance to match against low-confidence detections before being
marked lost. This gives more stable IDs through brief occlusions and
detector confidence dips without requiring an external tracking library.

No network calls, no extra dependencies beyond numpy/scipy -- fully
self-contained so it runs the same in a notebook, a video pipeline, or a
live webcam stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IOU between two sets of xyxy boxes -> (len(a), len(b)) matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    union = area_a + area_b - inter_area

    return np.where(union > 0, inter_area / union, 0.0).astype(np.float32)


class _KalmanBoxTracker:
    """Constant-velocity Kalman filter tracking a single box in [cx, cy, s, r] space.

    State vector: [cx, cy, s, r, vcx, vcy, vs]  where s = area, r = aspect ratio (constant).
    This is the standard SORT state parameterization.
    """

    _next_id = 1

    def __init__(self, xyxy: np.ndarray, class_id: int, class_name: str):
        self.id = _KalmanBoxTracker._next_id
        _KalmanBoxTracker._next_id += 1

        self.class_id = class_id
        self.class_name = class_name

        # State: 7-dim, Measurement: 4-dim (cx, cy, s, r)
        self.ndim = 7
        self.x = np.zeros((self.ndim, 1), dtype=np.float32)
        self.x[:4, 0] = self._xyxy_to_z(xyxy)

        self.P = np.eye(self.ndim, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on initial velocity

        self.F = np.eye(self.ndim, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # constant velocity model

        self.H = np.zeros((4, self.ndim), dtype=np.float32)
        self.H[:4, :4] = np.eye(4, dtype=np.float32)

        self.Q = np.eye(self.ndim, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 1.0

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[Tuple[int, int]] = []

    @staticmethod
    def _xyxy_to_z(xyxy: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = xyxy
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2
        s = max(w * h, 1e-6)
        r = w / max(h, 1e-6)
        return np.array([cx, cy, s, r], dtype=np.float32)

    @staticmethod
    def _z_to_xyxy(z: np.ndarray) -> np.ndarray:
        cx, cy, s, r = z
        s = max(s, 1e-6)
        w = np.sqrt(s * max(r, 1e-6))
        h = s / max(w, 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self._z_to_xyxy(self.x[:4, 0])

    def update(self, xyxy: np.ndarray) -> None:
        z = self._xyxy_to_z(xyxy).reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.ndim, dtype=np.float32) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1

    def current_xyxy(self) -> np.ndarray:
        return self._z_to_xyxy(self.x[:4, 0])


@dataclass
class TrackedObject:
    track_id: int
    xyxy: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    hits: int
    age: int


@dataclass
class TrackerStats:
    """Cumulative stats useful for the 'tracking stability' evaluation metric."""

    total_tracks_created: int = 0
    id_switches: int = 0
    fragmentations: int = 0


class MultiObjectTracker:
    """ByteTrack-style two-stage SORT tracker.

    Call `update(detections)` once per frame with the current frame's
    Detection list (see models/detector.py); returns the list of currently
    confirmed TrackedObject instances.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        cfg = self.settings.tracker
        self.max_age = cfg.max_age
        self.min_hits = cfg.min_hits
        self.iou_threshold = cfg.iou_threshold
        self.high_conf_threshold = cfg.high_conf_threshold
        self.low_conf_threshold = cfg.low_conf_threshold

        self._trackers: List[_KalmanBoxTracker] = []
        self._last_conf_by_track: dict[int, float] = {}
        self.stats = TrackerStats()

    def reset(self) -> None:
        self._trackers = []
        self._last_conf_by_track = {}
        self.stats = TrackerStats()

    def _associate(
        self, predicted_boxes: np.ndarray, det_boxes: np.ndarray, threshold: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Hungarian-algorithm association gated by an IOU threshold.

        Returns (matches, unmatched_track_indices, unmatched_det_indices).
        """
        if len(predicted_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(predicted_boxes))), list(range(len(det_boxes)))

        iou = iou_matrix(predicted_boxes, det_boxes)
        row_idx, col_idx = linear_sum_assignment(-iou)  # maximize IOU

        matches, unmatched_tracks, unmatched_dets = [], [], []
        matched_tracks, matched_dets = set(), set()

        for r, c in zip(row_idx, col_idx):
            if iou[r, c] >= threshold:
                matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        unmatched_tracks = [i for i in range(len(predicted_boxes)) if i not in matched_tracks]
        unmatched_dets = [i for i in range(len(det_boxes)) if i not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def update(self, detections: List) -> List[TrackedObject]:
        """Advance the tracker by one frame.

        Args:
            detections: list of `object_tracking_app.models.detector.Detection`
        Returns:
            list of confirmed TrackedObject (hits >= min_hits or age within grace period)
        """
        # 1. Predict all existing tracks forward.
        predicted_boxes = np.array(
            [t.predict() for t in self._trackers], dtype=np.float32
        ) if self._trackers else np.zeros((0, 4), dtype=np.float32)

        # 2. Split detections into high/low confidence (ByteTrack-style).
        high_dets = [d for d in detections if d.confidence >= self.high_conf_threshold]
        low_dets = [d for d in detections if self.low_conf_threshold <= d.confidence < self.high_conf_threshold]

        high_boxes = np.array([d.xyxy for d in high_dets], dtype=np.float32) if high_dets else np.zeros((0, 4), dtype=np.float32)

        # 3. Stage 1: associate high-confidence detections against all predicted tracks.
        matches, unmatched_track_idx, unmatched_high_idx = self._associate(
            predicted_boxes, high_boxes, self.iou_threshold
        )
        for t_idx, d_idx in matches:
            self._trackers[t_idx].update(high_dets[d_idx].xyxy)
            self._last_conf_by_track[self._trackers[t_idx].id] = high_dets[d_idx].confidence

        # 4. Stage 2: give unmatched tracks a second chance against low-confidence detections.
        if unmatched_track_idx and low_dets:
            remaining_pred_boxes = predicted_boxes[unmatched_track_idx]
            low_boxes = np.array([d.xyxy for d in low_dets], dtype=np.float32)
            stage2_matches, still_unmatched_rel, _ = self._associate(
                remaining_pred_boxes, low_boxes, self.iou_threshold
            )
            matched_relative = set()
            for rel_t, d_idx in stage2_matches:
                t_idx = unmatched_track_idx[rel_t]
                self._trackers[t_idx].update(low_dets[d_idx].xyxy)
                self._last_conf_by_track[self._trackers[t_idx].id] = low_dets[d_idx].confidence
                matched_relative.add(rel_t)
            unmatched_track_idx = [
                unmatched_track_idx[i] for i in range(len(unmatched_track_idx)) if i not in matched_relative
            ]

        # 5. Spawn new tracks from unmatched high-confidence detections.
        for d_idx in unmatched_high_idx:
            det = high_dets[d_idx]
            new_tracker = _KalmanBoxTracker(det.xyxy, det.class_id, det.class_name)
            self._trackers.append(new_tracker)
            self._last_conf_by_track[new_tracker.id] = det.confidence
            self.stats.total_tracks_created += 1

        # 6. Drop tracks that have been lost for too long.
        alive: List[_KalmanBoxTracker] = []
        for t in self._trackers:
            if t.time_since_update <= self.max_age:
                alive.append(t)
            else:
                self.stats.fragmentations += 1
        self._trackers = alive

        # 7. Emit confirmed tracks.
        results: List[TrackedObject] = []
        for t in self._trackers:
            confirmed = t.hits >= self.min_hits or t.age <= self.min_hits
            if confirmed and t.time_since_update == 0:
                results.append(
                    TrackedObject(
                        track_id=t.id,
                        xyxy=t.current_xyxy(),
                        class_id=t.class_id,
                        class_name=t.class_name,
                        confidence=self._last_conf_by_track.get(t.id, 0.0),
                        hits=t.hits,
                        age=t.age,
                    )
                )
        return results

    def active_track_ids(self) -> List[int]:
        return [t.id for t in self._trackers]
