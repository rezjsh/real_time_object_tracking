"""Drawing utilities for boxes, labels, track IDs, and motion trails."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Sequence

import cv2
import numpy as np

_PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (238, 130, 238), (0, 206, 209), (255, 140, 0), (154, 205, 50),
    (219, 112, 147), (30, 144, 255), (255, 69, 0), (147, 112, 219),
]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color for a given track id."""
    return _PALETTE[track_id % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: Sequence[float],
    class_names: Sequence[str],
    track_ids: Sequence[int] | None = None,
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes with class name, confidence, and optional track ID.

    Args:
        frame: BGR image (H, W, 3), modified in place and also returned.
        boxes_xyxy: (N, 4) array of [x1, y1, x2, y2] in pixel coordinates.
        scores: length-N confidence scores.
        class_names: length-N class name strings.
        track_ids: optional length-N track IDs (int). If provided, colors are
            keyed by track id (stable per object) instead of class.
    """
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = [int(v) for v in boxes_xyxy[i]]
        raw_tid = track_ids[i] if track_ids is not None else None
        tid = int(raw_tid) if raw_tid is not None else None
        color = color_for_id(tid) if tid is not None else color_for_id(hash(class_names[i]) % 1000)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_names[i]} {scores[i]:.2f}"
        if tid is not None:
            label = f"ID {tid} | {label}"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        y_label = max(y1, th + 6)
        cv2.rectangle(frame, (x1, y_label - th - 6), (x1 + tw + 4, y_label + baseline - 2), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y_label - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


class TrailDrawer:
    """Keeps a rolling history of track centroids and draws motion trails."""

    def __init__(self, trail_length: int = 30):
        self.trail_length = trail_length
        self._history: Dict[int, Deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.trail_length)
        )

    def update_and_draw(
        self, frame: np.ndarray, boxes_xyxy: np.ndarray, track_ids: Iterable[int]
    ) -> np.ndarray:
        active_ids = set()
        for box, tid in zip(boxes_xyxy, track_ids):
            tid = int(tid)
            active_ids.add(tid)
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
            self._history[tid].append((cx, cy))

        for tid, pts in self._history.items():
            if tid not in active_ids or len(pts) < 2:
                continue
            color = color_for_id(tid)
            pts_list = list(pts)
            for i in range(1, len(pts_list)):
                cv2.line(frame, pts_list[i - 1], pts_list[i], color, 2, cv2.LINE_AA)
        return frame

    def prune(self, active_ids: Iterable[int]) -> None:
        """Drop history for track IDs that are no longer active to bound memory."""
        active = set(int(a) for a in active_ids)
        stale = [tid for tid in self._history if tid not in active]
        for tid in stale:
            del self._history[tid]


def draw_hud(frame: np.ndarray, fps: float, num_tracks: int, extra: str = "") -> np.ndarray:
    """Draw a small heads-up display in the top-left corner (FPS, object count)."""
    text = f"FPS: {fps:.1f} | Objects: {num_tracks}"
    if extra:
        text += f" | {extra}"
    cv2.rectangle(frame, (0, 0), (min(420, frame.shape[1]), 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def class_count_summary(class_names: Sequence[str]) -> dict:
    """Return a {class_name: count} dict for a frame's detections."""
    counts: dict = {}
    for name in class_names:
        counts[name] = counts.get(name, 0) + 1
    return counts
