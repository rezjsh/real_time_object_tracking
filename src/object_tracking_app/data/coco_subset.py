"""COCO 2017 subset builder.

Filters COCO annotations down to a curated set of classes, samples a bounded
number of images per class (for Colab-friendly runtimes), converts the
annotations to YOLO-format label files, and writes train/val split files.

This module operates on an already-downloaded COCO annotation JSON + image
directory (see scripts/download_dataset.py). It does not perform the network
download itself, so it can be unit-tested against small synthetic
COCO-format annotation files.
"""

from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.utils.io import ensure_dir, save_json
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CocoImageRecord:
    image_id: int
    file_name: str
    width: int
    height: int
    annotations: List[dict] = field(default_factory=list)


class CocoSubsetBuilder:
    """Builds a class-filtered, size-bounded subset of COCO in YOLO format."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.dataset_cfg = self.settings.dataset
        self.mode_cfg = self.dataset_cfg.active_mode
        self.classes = list(self.dataset_cfg.classes)
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

    # ------------------------------------------------------------------ #
    # Loading raw COCO annotations
    # ------------------------------------------------------------------ #
    def _load_coco_json(self, annotation_json_path: str | Path) -> dict:
        path = Path(annotation_json_path)
        if not path.exists():
            raise FileNotFoundError(
                f"COCO annotation file not found at {path}. "
                "Run scripts/download_dataset.py first."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_annotations(
        self, annotation_json_path: str | Path
    ) -> tuple[Dict[int, CocoImageRecord], Dict[int, str]]:
        """Load a COCO instances_*.json file, indexed by image_id.

        Returns:
            (records_by_image_id, category_id_to_name)
        """
        coco = self._load_coco_json(annotation_json_path)
        cat_id_to_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
        images_by_id = {img["id"]: img for img in coco.get("images", [])}

        records: Dict[int, CocoImageRecord] = {}
        for ann in coco.get("annotations", []):
            cat_name = cat_id_to_name.get(ann["category_id"])
            if cat_name not in self.class_to_idx:
                continue  # not one of our curated classes
            img_id = ann["image_id"]
            if img_id not in images_by_id:
                continue
            if img_id not in records:
                img_meta = images_by_id[img_id]
                records[img_id] = CocoImageRecord(
                    image_id=img_id,
                    file_name=img_meta["file_name"],
                    width=img_meta["width"],
                    height=img_meta["height"],
                )
            records[img_id].annotations.append(ann)

        logger.info(
            "Loaded %d images containing curated classes out of %d total images.",
            len(records), len(images_by_id),
        )
        return records, cat_id_to_name

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def sample_records(
        self,
        records: Dict[int, CocoImageRecord],
        cat_id_to_name: Dict[int, str],
        seed: Optional[int] = None,
    ) -> List[CocoImageRecord]:
        """Sample up to `images_per_class` images per class, bounded by max_total_images."""
        rng = random.Random(seed if seed is not None else self.settings.project_info.project.seed)

        by_class: Dict[str, List[int]] = defaultdict(list)
        for img_id, rec in records.items():
            names_present = {
                cat_id_to_name.get(ann["category_id"]) for ann in rec.annotations
            }
            names_present.discard(None)
            for name in names_present:
                if name in self.class_to_idx:
                    by_class[name].append(img_id)

        selected_ids: set = set()
        for cls_name in self.classes:
            candidates = list(by_class.get(cls_name, []))
            rng.shuffle(candidates)
            take = candidates[: self.mode_cfg.images_per_class]
            selected_ids.update(take)
            logger.debug("Class '%s': %d candidates, took %d", cls_name, len(candidates), len(take))

        selected_ids_list = list(selected_ids)
        rng.shuffle(selected_ids_list)
        selected_ids_list = selected_ids_list[: self.mode_cfg.max_total_images]

        logger.info(
            "Selected %d images total (mode=%s).", len(selected_ids_list), self.dataset_cfg.mode
        )
        return [records[i] for i in selected_ids_list]

    # ------------------------------------------------------------------ #
    # YOLO export
    # ------------------------------------------------------------------ #
    def to_yolo_label(
        self, ann: dict, img_width: int, img_height: int, cat_id_to_name: Dict[int, str]
    ) -> Optional[str]:
        """Convert a single COCO annotation (xywh, absolute) to a YOLO label line."""
        cat_name = cat_id_to_name.get(ann["category_id"])
        if cat_name not in self.class_to_idx:
            return None
        cls_idx = self.class_to_idx[cat_name]
        x, y, w, h = ann["bbox"]
        if img_width <= 0 or img_height <= 0:
            return None
        cx = (x + w / 2) / img_width
        cy = (y + h / 2) / img_height
        nw = w / img_width
        nh = h / img_height
        cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))
        return f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

    def build(
        self,
        annotation_json_path: str | Path,
        images_dir: str | Path,
        output_root: Optional[str | Path] = None,
    ) -> dict:
        """Full pipeline: load -> sample -> split -> write YOLO images/labels.

        Returns a manifest dict summarizing what was written.
        """
        annotation_json_path = Path(annotation_json_path)
        images_dir = Path(images_dir)
        output_root = Path(output_root) if output_root else self.settings.resolve(
            self.dataset_cfg.yolo_export.output_dir
        )

        records, cat_id_to_name = self.load_annotations(annotation_json_path)
        sampled = self.sample_records(records, cat_id_to_name)

        rng = random.Random(self.settings.project_info.project.seed)
        rng.shuffle(sampled)
        val_fraction = self.mode_cfg.val_fraction
        n_val = max(1, int(len(sampled) * val_fraction)) if sampled else 0
        val_records = sampled[:n_val]
        train_records = sampled[n_val:]

        export_cfg = self.dataset_cfg.yolo_export
        dirs = {
            "train_images": ensure_dir(output_root / export_cfg.train_subdir),
            "val_images": ensure_dir(output_root / export_cfg.val_subdir),
            "train_labels": ensure_dir(output_root / export_cfg.labels_train_subdir),
            "val_labels": ensure_dir(output_root / export_cfg.labels_val_subdir),
        }

        manifest: dict = {"train": [], "val": [], "classes": self.classes, "mode": self.dataset_cfg.mode}

        for split_name, split_records, img_dir_key, lbl_dir_key in (
            ("train", train_records, "train_images", "train_labels"),
            ("val", val_records, "val_images", "val_labels"),
        ):
            for rec in split_records:
                src_img = images_dir / rec.file_name
                if not src_img.exists():
                    logger.warning("Missing image file, skipping: %s", src_img)
                    continue
                dst_img = dirs[img_dir_key] / rec.file_name
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)

                label_lines = []
                for ann in rec.annotations:
                    line = self.to_yolo_label(ann, rec.width, rec.height, cat_id_to_name)
                    if line:
                        label_lines.append(line)

                label_path = dirs[lbl_dir_key] / (Path(rec.file_name).stem + ".txt")
                label_path.write_text("\n".join(label_lines), encoding="utf-8")
                manifest[split_name].append(rec.file_name)

        yolo_data_yaml = self._write_yolo_data_yaml(output_root)
        manifest["yolo_data_yaml"] = str(yolo_data_yaml)

        splits_dir = ensure_dir(self.settings.resolve(self.dataset_cfg.splits_dir))
        save_json(manifest, splits_dir / f"{self.dataset_cfg.mode}_manifest.json")

        logger.info(
            "Built YOLO subset: %d train / %d val images at %s",
            len(manifest["train"]), len(manifest["val"]), output_root,
        )
        return manifest

    def _write_yolo_data_yaml(self, output_root: Path) -> Path:
        import yaml

        export_cfg = self.dataset_cfg.yolo_export
        content = {
            "path": str(output_root),
            "train": export_cfg.train_subdir,
            "val": export_cfg.val_subdir,
            "names": {i: name for i, name in enumerate(self.classes)},
        }
        out_path = output_root / "data.yaml"
        ensure_dir(output_root)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(content, f, sort_keys=False)
        return out_path
