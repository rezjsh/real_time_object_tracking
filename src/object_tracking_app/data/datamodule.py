"""PyTorch Dataset/DataLoader wrapping the YOLO-format COCO subset.

Ultralytics handles its own training-time data loading internally when you
call `model.train(data=...)`, so this datamodule is intentionally focused on
use cases *outside* that training loop: dataset inspection in notebooks,
custom evaluation loops, and unit tests. It reads the same
images/<split>/*.jpg + labels/<split>/*.txt layout produced by
`CocoSubsetBuilder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

from object_tracking_app.data.transforms import letterbox, xywhn_to_xyxy
from object_tracking_app.utils.io import read_image
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class Sample:
    image_path: Path
    label_path: Path


class YoloSubsetDataset(Dataset):
    """Reads a YOLO-format image/label split directory pair.

    Each __getitem__ returns:
        image (np.ndarray, letterboxed BGR),
        boxes_xyxy (N, 4) in the letterboxed image's pixel space,
        class_ids (N,)
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        class_names: List[str],
        img_size: int = 640,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.img_size = img_size
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.samples: List[Sample] = []
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            label_path = self.labels_dir / (img_path.stem + ".txt")
            self.samples.append(Sample(image_path=img_path, label_path=label_path))

        logger.info("YoloSubsetDataset: found %d images in %s", len(self.samples), self.images_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_labels(self, label_path: Path) -> List[Tuple[int, float, float, float, float]]:
        if not label_path.exists():
            return []
        rows = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            rows.append((cls_id, cx, cy, w, h))
        return rows

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = read_image(sample.image_path)
        orig_h, orig_w = image.shape[:2]

        padded, ratio, (pad_w, pad_h) = letterbox(image, self.img_size)

        labels = self._read_labels(sample.label_path)
        boxes_xyxy = []
        class_ids = []
        for cls_id, cx, cy, w, h in labels:
            abs_box = xywhn_to_xyxy(np.array([cx, cy, w, h]), orig_w, orig_h)
            # Map original-image pixel coords into the letterboxed image space.
            abs_box = abs_box * ratio
            abs_box[[0, 2]] += pad_w
            abs_box[[1, 3]] += pad_h
            boxes_xyxy.append(abs_box)
            class_ids.append(cls_id)

        boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32) if boxes_xyxy else np.zeros((0, 4), dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int64) if class_ids else np.zeros((0,), dtype=np.int64)

        if self.transform:
            padded, boxes_xyxy, class_ids = self.transform(padded, boxes_xyxy, class_ids)

        return padded, boxes_xyxy, class_ids

    def class_name(self, cls_id: int) -> str:
        return self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else "unknown"


def _collate_fn(batch):
    """Custom collate: keeps variable-length box/class arrays as a list instead of stacking."""
    images, boxes, classes = zip(*batch)
    images = np.stack(images, axis=0)
    return images, list(boxes), list(classes)


def build_dataloader(
    images_dir: str | Path,
    labels_dir: str | Path,
    class_names: List[str],
    img_size: int = 640,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = YoloSubsetDataset(images_dir, labels_dir, class_names, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
    )
