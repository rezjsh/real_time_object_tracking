"""Object detector wrapper.

Wraps an Ultralytics YOLO model (YOLOv8n by default) behind a small,
stable interface so the rest of the codebase never talks to the
`ultralytics` API directly. This keeps training/inference/evaluation code
decoupled from the specific detector backend -- swapping in a different
YOLO variant (or another detector entirely) only requires changes here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """A single detection in absolute pixel coordinates."""

    xyxy: np.ndarray       # shape (4,): x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class Detector:
    """Thin wrapper around an Ultralytics YOLO model for train/predict/export."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or get_settings()
        self.device = device or self.settings.resolved_device()
        self._weights_path = weights_path or self._resolve_weights()
        self._model = None  # lazy-loaded

        logger.info("Detector configured: weights=%s device=%s", self._weights_path, self.device)

    def _resolve_weights(self) -> str:
        cfg = self.settings.inference
        fine_tuned = self.settings.resolve(cfg.weights_path)
        if fine_tuned.exists():
            return str(fine_tuned)
        logger.warning(
            "Fine-tuned weights not found at %s. Falling back to pretrained '%s'. "
            "Run scripts/train.py to produce a fine-tuned checkpoint.",
            fine_tuned, cfg.fallback_weights,
        )
        return cfg.fallback_weights

    @property
    def model(self):
        """Lazily import & construct the ultralytics YOLO model on first use."""
        if self._model is None:
            from ultralytics import YOLO  # local import: keep ultralytics optional at import time

            self._model = YOLO(self._weights_path)
        return self._model

    @property
    def class_names(self) -> dict:
        """Return the {index: name} mapping the underlying model was trained with."""
        return self.model.names

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, data_yaml: str, **overrides) -> dict:
        """Fine-tune on a YOLO-format dataset. `overrides` merges onto configs/train.yaml."""
        train_cfg = self.settings.train
        kwargs = dict(
            data=data_yaml,
            epochs=train_cfg.epochs,
            batch=train_cfg.batch_size,
            imgsz=train_cfg.img_size,
            patience=train_cfg.patience,
            optimizer=train_cfg.optimizer,
            lr0=train_cfg.lr0,
            lrf=train_cfg.lrf,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
            warmup_epochs=train_cfg.warmup_epochs,
            workers=train_cfg.workers,
            seed=train_cfg.seed,
            device=self.device,
            amp=train_cfg.amp,
            project=str(self.settings.resolve(train_cfg.project_dir)),
            name=train_cfg.run_name,
            resume=train_cfg.resume,
            pretrained=train_cfg.pretrained,
            hsv_h=train_cfg.augment.hsv_h,
            hsv_s=train_cfg.augment.hsv_s,
            hsv_v=train_cfg.augment.hsv_v,
            fliplr=train_cfg.augment.fliplr,
            flipud=train_cfg.augment.flipud,
            mosaic=train_cfg.augment.mosaic,
            mixup=train_cfg.augment.mixup,
            degrees=train_cfg.augment.degrees,
            translate=train_cfg.augment.translate,
            scale=train_cfg.augment.scale,
            save_period=train_cfg.checkpoint.save_period,
        )
        kwargs.update(overrides)
        logger.info("Starting training run '%s' for %d epochs...", train_cfg.run_name, kwargs["epochs"])
        results = self.model.train(**kwargs)
        return {"save_dir": str(getattr(results, "save_dir", ""))}

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(
        self,
        image: np.ndarray | str | Path,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[Sequence[int]] = None,
        img_size: Optional[int] = None,
    ) -> List[Detection]:
        """Run detection on a single image (numpy BGR array or file path)."""
        infer_cfg = self.settings.inference
        results = self.model.predict(
            source=image,
            conf=conf if conf is not None else infer_cfg.conf_threshold,
            iou=iou if iou is not None else infer_cfg.iou_threshold,
            imgsz=img_size if img_size is not None else infer_cfg.img_size,
            classes=list(classes) if classes is not None else None,
            max_det=infer_cfg.max_detections,
            device=self.device,
            verbose=False,
        )
        return self._parse_results(results[0])

    def _parse_results(self, result) -> List[Detection]:
        detections: List[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        names = result.names
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        for box, c, k in zip(xyxy, conf, cls):
            detections.append(
                Detection(xyxy=box, confidence=float(c), class_id=int(k), class_name=names.get(int(k), str(k)))
            )
        return detections

    def class_name_to_index(self, name_to_find: str) -> Optional[int]:
        for idx, name in self.class_names.items():
            if name == name_to_find:
                return int(idx)
        return None

    def resolve_class_filter(self, class_names_wanted: Optional[Sequence[str]]) -> Optional[List[int]]:
        """Convert a list of human-readable class names into model class indices."""
        if not class_names_wanted:
            return None
        indices = []
        for name in class_names_wanted:
            idx = self.class_name_to_index(name)
            if idx is not None:
                indices.append(idx)
            else:
                logger.warning("Requested class '%s' not found in model classes.", name)
        return indices or None

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export the current weights to a deployment format (onnx, torchscript, etc.)."""
        path = self.model.export(format=format, **kwargs)
        logger.info("Exported model to %s (format=%s)", path, format)
        return str(path)
