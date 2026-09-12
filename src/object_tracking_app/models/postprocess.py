"""Post-processing utilities: confidence thresholding, class filtering, and NMS.

Ultralytics already applies NMS internally during `model.predict(...)`, but
these utilities are exposed separately so that:
  1. They can be re-applied to cached/loaded detections (e.g. re-filtering
     by a new confidence threshold in the Streamlit UI without re-running
     the model).
  2. They're independently unit-testable.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from object_tracking_app.models.detector import Detection
from object_tracking_app.models.tracker import iou_matrix


def filter_by_confidence(detections: List[Detection], threshold: float) -> List[Detection]:
    return [d for d in detections if d.confidence >= threshold]


def filter_by_classes(detections: List[Detection], allowed_class_names: Optional[Sequence[str]]) -> List[Detection]:
    if not allowed_class_names:
        return detections
    allowed = set(allowed_class_names)
    return [d for d in detections if d.class_name in allowed]


def non_max_suppression(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """Class-aware greedy NMS. Detections are assumed to already be roughly filtered."""
    if not detections:
        return []

    kept: List[Detection] = []
    by_class: dict[int, List[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    for cls_dets in by_class.values():
        cls_dets = sorted(cls_dets, key=lambda d: d.confidence, reverse=True)
        boxes = np.array([d.xyxy for d in cls_dets], dtype=np.float32)
        suppressed = np.zeros(len(cls_dets), dtype=bool)

        for i in range(len(cls_dets)):
            if suppressed[i]:
                continue
            kept.append(cls_dets[i])
            if i == len(cls_dets) - 1:
                continue
            remaining_idx = [j for j in range(i + 1, len(cls_dets)) if not suppressed[j]]
            if not remaining_idx:
                continue
            ious = iou_matrix(boxes[i:i + 1], boxes[remaining_idx])[0]
            for local_j, iou_val in zip(remaining_idx, ious):
                if iou_val > iou_threshold:
                    suppressed[local_j] = True

    return kept


def postprocess_detections(
    detections: List[Detection],
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    allowed_class_names: Optional[Sequence[str]] = None,
    apply_nms: bool = False,
) -> List[Detection]:
    """Full post-processing chain: confidence -> class filter -> (optional) NMS.

    `apply_nms` defaults to False since ultralytics' predict() already runs
    NMS; set True only when re-processing raw/cached detections.
    """
    result = filter_by_confidence(detections, conf_threshold)
    result = filter_by_classes(result, allowed_class_names)
    if apply_nms:
        result = non_max_suppression(result, iou_threshold)
    return result
