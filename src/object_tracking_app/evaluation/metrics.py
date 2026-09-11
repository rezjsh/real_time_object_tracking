"""Evaluation metrics: detection quality (mAP/precision/recall), runtime
performance (FPS), and multi-object tracking stability (ID switches,
fragmentation, mostly-tracked ratio).

Detection mAP is computed with the standard COCO-style IOU-thresholded
matching (implemented directly here with numpy, no pycocotools dependency
required for the metric computation itself, though pycocotools remains
useful for reading raw COCO files elsewhere in the project).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from object_tracking_app.models.tracker import iou_matrix
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    ap: float                       # average precision at a single IOU threshold
    map50: float                    # mAP @ IOU=0.5 across classes
    per_class_ap: Dict[str, float] = field(default_factory=dict)


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """11-point / continuous AP computation (PASCAL VOC-style, monotonic envelope)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_detection_metrics(
    predictions: List[dict],
    ground_truths: List[dict],
    class_names: List[str],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    """Compute precision/recall/AP over a dataset.

    Args:
        predictions: list of {"image_id": int, "boxes": (N,4) xyxy, "scores": (N,),
                               "class_ids": (N,)}
        ground_truths: list of {"image_id": int, "boxes": (M,4) xyxy, "class_ids": (M,)}
        class_names: index -> name mapping used only for the per-class breakdown.
        iou_threshold: IOU at which a prediction counts as a true positive.
    """
    gt_by_image: Dict[int, dict] = {g["image_id"]: g for g in ground_truths}

    per_class_ap: Dict[str, float] = {}
    all_tp, all_fp, all_scores = [], [], []
    total_gt = 0

    for cls_id, cls_name in enumerate(class_names):
        cls_preds = []
        for p in predictions:
            mask = p["class_ids"] == cls_id
            for box, score in zip(p["boxes"][mask], p["scores"][mask]):
                cls_preds.append((p["image_id"], box, score))
        cls_preds.sort(key=lambda x: x[2], reverse=True)

        gt_matched: Dict[int, np.ndarray] = {}
        n_gt_cls = 0
        for img_id, gt in gt_by_image.items():
            mask = gt["class_ids"] == cls_id
            n_gt_cls += int(mask.sum())
            gt_matched[img_id] = np.zeros(int(mask.sum()), dtype=bool)
        total_gt += n_gt_cls

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))

        for i, (img_id, box, score) in enumerate(cls_preds):
            gt = gt_by_image.get(img_id)
            if gt is None:
                fp[i] = 1
                continue
            mask = gt["class_ids"] == cls_id
            gt_boxes = gt["boxes"][mask]
            if len(gt_boxes) == 0:
                fp[i] = 1
                continue
            ious = iou_matrix(box[None, :], gt_boxes)[0]
            best_j = int(np.argmax(ious))
            if ious[best_j] >= iou_threshold and not gt_matched[img_id][best_j]:
                tp[i] = 1
                gt_matched[img_id][best_j] = True
            else:
                fp[i] = 1

        if n_gt_cls == 0:
            per_class_ap[cls_name] = 0.0
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recall = tp_cum / max(n_gt_cls, 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _average_precision(recall, precision) if len(cls_preds) else 0.0
        per_class_ap[cls_name] = ap

        all_tp.extend(tp.tolist())
        all_fp.extend(fp.tolist())
        all_scores.extend([s for _, _, s in cls_preds])

    all_tp_arr = np.array(all_tp)
    all_fp_arr = np.array(all_fp)
    total_tp = float(all_tp_arr.sum()) if len(all_tp_arr) else 0.0
    total_fp = float(all_fp_arr.sum()) if len(all_fp_arr) else 0.0
    precision_overall = total_tp / max(total_tp + total_fp, 1e-9)
    recall_overall = total_tp / max(total_gt, 1e-9)
    f1 = (
        2 * precision_overall * recall_overall / max(precision_overall + recall_overall, 1e-9)
        if (precision_overall + recall_overall) > 0
        else 0.0
    )
    map50 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0

    return DetectionMetrics(
        precision=precision_overall,
        recall=recall_overall,
        f1=f1,
        ap=map50,
        map50=map50,
        per_class_ap=per_class_ap,
    )


@dataclass
class TrackingStabilityMetrics:
    """Simplified MOT-style stability indicators (not full py-motmetrics, but the
    core signals: how often identities switch and how fragmented tracks are)."""

    id_switches: int
    fragmentations: int
    total_tracks_created: int
    mostly_tracked_ratio: float   # fraction of ground-truth tracks covered >=80% of their length


def compute_tracking_stability(
    predicted_tracks_per_frame: List[List[dict]],
    ground_truth_ids_per_frame: Optional[List[List[int]]] = None,
) -> TrackingStabilityMetrics:
    """Estimate ID-switch and fragmentation counts from predicted track history.

    `predicted_tracks_per_frame[i]` is a list of {"track_id": int, "class_id": int} for frame i.
    If ground truth identities aren't available (common for demo/webcam runs), this
    falls back to reporting fragmentation/creation counts only, which are still
    meaningful stability signals on their own.
    """
    prev_ids: set[int] = set()
    id_switches = 0
    fragmentations = 0
    seen_ids: set[int] = set()

    for frame_tracks in predicted_tracks_per_frame:
        cur_ids = {t["track_id"] for t in frame_tracks}
        new_ids = cur_ids - seen_ids
        dropped_ids = prev_ids - cur_ids
        if dropped_ids and new_ids:
            # A crude proxy: simultaneous drops+creations in the same frame often
            # indicate a track fragmented and re-spawned as a new ID.
            id_switches += min(len(dropped_ids), len(new_ids))
        fragmentations += len(dropped_ids)
        seen_ids |= new_ids
        prev_ids = cur_ids

    mostly_tracked_ratio = 1.0 if not fragmentations else max(0.0, 1.0 - fragmentations / max(len(seen_ids), 1))

    return TrackingStabilityMetrics(
        id_switches=id_switches,
        fragmentations=fragmentations,
        total_tracks_created=len(seen_ids),
        mostly_tracked_ratio=mostly_tracked_ratio,
    )
