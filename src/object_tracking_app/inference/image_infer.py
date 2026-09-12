"""Single-image inference: detect + (optionally) track + annotate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.models.detector import Detection, Detector
from object_tracking_app.models.postprocess import postprocess_detections
from object_tracking_app.models.tracker import MultiObjectTracker
from object_tracking_app.utils.io import read_image, save_image
from object_tracking_app.utils.logger import get_logger
from object_tracking_app.utils.viz import class_count_summary, draw_detections

logger = get_logger(__name__)


class ImageInferencePipeline:
    """Runs detection (and optional single-frame tracking) on a static image."""

    def __init__(self, detector: Optional[Detector] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.detector = detector or Detector(settings=self.settings)

    def run(
        self,
        image: str | Path | np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        class_filter: Optional[list[str]] = None,
        use_tracker: bool = False,
    ) -> dict:
        """Run inference on a single image.

        Returns a dict with the annotated frame plus summary statistics --
        this is the shape both the CLI script and the Streamlit app consume.
        """
        img = read_image(image) if isinstance(image, (str, Path)) else image.copy()

        cfg = self.settings.inference
        conf = conf_threshold if conf_threshold is not None else cfg.conf_threshold
        iou = iou_threshold if iou_threshold is not None else cfg.iou_threshold

        detections = self.detector.predict(img, conf=conf, iou=iou)
        detections = postprocess_detections(
            detections, conf_threshold=conf, iou_threshold=iou, allowed_class_names=class_filter
        )

        track_ids = None
        if use_tracker and detections:
            tracker = MultiObjectTracker(settings=self.settings)
            tracked = tracker.update(detections)
            # Re-align detections/track_ids by nearest box match since the tracker
            # returns its own smoothed boxes.
            detections, track_ids = _align_tracks_to_detections(detections, tracked)

        annotated = img.copy()
        if detections:
            boxes = np.array([d.xyxy for d in detections], dtype=np.float32)
            scores = [d.confidence for d in detections]
            names = [d.class_name for d in detections]
            draw_detections(annotated, boxes, scores, names, track_ids=track_ids)

        summary = {
            "num_detections": len(detections),
            "class_counts": class_count_summary([d.class_name for d in detections]),
            "mean_confidence": float(np.mean([d.confidence for d in detections])) if detections else 0.0,
        }

        return {
            "annotated_image": annotated,
            "detections": detections,
            "track_ids": track_ids,
            "summary": summary,
        }

    def run_and_save(self, image_path: str | Path, output_path: str | Path, **kwargs) -> dict:
        result = self.run(image_path, **kwargs)
        save_image(result["annotated_image"], output_path)
        logger.info("Saved annotated image to %s", output_path)
        return result


def _align_tracks_to_detections(detections: list[Detection], tracked) -> tuple[list[Detection], list[int]]:
    """Match tracker output back to the original detection order via IOU, for drawing."""
    from object_tracking_app.models.tracker import iou_matrix

    if not tracked:
        return detections, [None] * len(detections)

    det_boxes = np.array([d.xyxy for d in detections], dtype=np.float32)
    trk_boxes = np.array([t.xyxy for t in tracked], dtype=np.float32)
    ious = iou_matrix(det_boxes, trk_boxes)

    track_ids = []
    for i in range(len(detections)):
        best_j = int(np.argmax(ious[i])) if ious.shape[1] > 0 else -1
        if best_j >= 0 and ious[i, best_j] > 0:
            track_ids.append(tracked[best_j].track_id)
        else:
            track_ids.append(None)
    return detections, track_ids
