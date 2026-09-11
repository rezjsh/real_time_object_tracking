"""Benchmark runner: evaluates the detector (and optionally the tracker) over
a YOLO-format validation split and writes a JSON report to artifacts/metrics/.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.data.datamodule import YoloSubsetDataset
from object_tracking_app.evaluation.metrics import compute_detection_metrics
from object_tracking_app.models.detector import Detector
from object_tracking_app.utils.io import save_json
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)


def run_benchmark(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: list[str],
    detector: Optional[Detector] = None,
    settings: Optional[Settings] = None,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
    max_images: Optional[int] = None,
    output_report_path: Optional[str | Path] = None,
) -> dict:
    """Run the detector over a validation split and compute mAP + FPS."""
    settings = settings or get_settings()
    detector = detector or Detector(settings=settings)
    infer_cfg = settings.inference
    conf = conf_threshold if conf_threshold is not None else infer_cfg.conf_threshold
    iou = iou_threshold if iou_threshold is not None else infer_cfg.iou_threshold

    dataset = YoloSubsetDataset(images_dir, labels_dir, class_names, img_size=infer_cfg.img_size)
    n = len(dataset) if max_images is None else min(max_images, len(dataset))

    if n == 0:
        logger.warning("Benchmark dataset is empty at %s -- skipping evaluation.", images_dir)
        report = {
            "num_images": 0,
            "note": "No images found. Run scripts/download_dataset.py and rebuild the subset first.",
        }
        if output_report_path:
            save_json(report, output_report_path)
        return report

    predictions, ground_truths = [], []
    frame_times = []

    for idx in range(n):
        image, gt_boxes, gt_classes = dataset[idx]
        image_id = idx

        t0 = time.perf_counter()
        detections = detector.predict(image, conf=conf, iou=iou)
        frame_times.append(time.perf_counter() - t0)

        pred_boxes = np.array([d.xyxy for d in detections], dtype=np.float32) if detections else np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.array([d.confidence for d in detections], dtype=np.float32) if detections else np.zeros((0,), dtype=np.float32)
        pred_classes = np.array([d.class_id for d in detections], dtype=np.int64) if detections else np.zeros((0,), dtype=np.int64)

        predictions.append({
            "image_id": image_id, "boxes": pred_boxes, "scores": pred_scores, "class_ids": pred_classes
        })
        ground_truths.append({"image_id": image_id, "boxes": gt_boxes, "class_ids": gt_classes})

        if (idx + 1) % 25 == 0 or idx == n - 1:
            logger.info("Benchmark progress: %d/%d images", idx + 1, n)

    det_metrics = compute_detection_metrics(predictions, ground_truths, class_names, iou_threshold=0.5)

    avg_frame_time = float(np.mean(frame_times)) if frame_times else 0.0
    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    report = {
        "num_images": n,
        "conf_threshold": conf,
        "iou_threshold": iou,
        "precision": det_metrics.precision,
        "recall": det_metrics.recall,
        "f1": det_metrics.f1,
        "map50": det_metrics.map50,
        "per_class_ap": det_metrics.per_class_ap,
        "avg_inference_time_sec": avg_frame_time,
        "fps": fps,
    }

    if output_report_path:
        save_json(report, output_report_path)
        logger.info("Saved benchmark report to %s", output_report_path)

    return report
