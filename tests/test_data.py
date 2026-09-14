"""Tests for the data pipeline: COCO subset filtering/sampling and YOLO export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.data.coco_subset import CocoSubsetBuilder
from object_tracking_app.data.transforms import letterbox, xywhn_to_xyxy, xyxy_to_xywhn


@pytest.fixture()
def settings() -> Settings:
    return get_settings()


def _make_synthetic_coco(tmp_path: Path, n_images: int = 6) -> tuple[Path, Path]:
    """Build a tiny synthetic COCO-format annotation file + blank images on disk."""
    import cv2

    images_dir = tmp_path / "train2017"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "banana"},  # not in curated class list -- should be filtered out
    ]

    images, annotations = [], []
    ann_id = 1
    for i in range(n_images):
        fname = f"img_{i:03d}.jpg"
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        cv2.imwrite(str(images_dir / fname), img)
        images.append({"id": i, "file_name": fname, "width": 100, "height": 100})

        # Alternate between person/car/banana annotations.
        cat_id = [1, 2, 3][i % 3]
        annotations.append({
            "id": ann_id, "image_id": i, "category_id": cat_id,
            "bbox": [10.0, 10.0, 30.0, 30.0], "area": 900.0, "iscrowd": 0,
        })
        ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    ann_path = tmp_path / "instances_train2017.json"
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    return ann_path, images_dir


def test_load_annotations_filters_curated_classes(tmp_path, settings):
    ann_path, images_dir = _make_synthetic_coco(tmp_path)
    builder = CocoSubsetBuilder(settings=settings)
    records, cat_id_to_name = builder.load_annotations(ann_path)

    # "banana" (category_id=3) should never appear since it's not a curated class.
    for rec in records.values():
        for ann in rec.annotations:
            assert cat_id_to_name[ann["category_id"]] != "banana"


def test_sample_records_respects_max_total(tmp_path, settings):
    ann_path, images_dir = _make_synthetic_coco(tmp_path, n_images=6)
    builder = CocoSubsetBuilder(settings=settings)
    records, cat_id_to_name = builder.load_annotations(ann_path)
    sampled = builder.sample_records(records, cat_id_to_name, seed=0)

    assert len(sampled) <= builder.mode_cfg.max_total_images
    assert len(sampled) > 0


def test_to_yolo_label_normalizes_correctly(settings):
    builder = CocoSubsetBuilder(settings=settings)
    ann = {"category_id": 1, "bbox": [10.0, 20.0, 30.0, 40.0]}
    cat_id_to_name = {1: "person"}
    line = builder.to_yolo_label(ann, img_width=100, img_height=100, cat_id_to_name=cat_id_to_name)

    assert line is not None
    parts = line.split()
    cls_idx, cx, cy = int(parts[0]), float(parts[1]), float(parts[2])
    assert cls_idx == builder.class_to_idx["person"]
    assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    assert abs(cx - 0.25) < 1e-6  # (10 + 15) / 100
    assert abs(cy - 0.40) < 1e-6  # (20 + 20) / 100


def test_to_yolo_label_returns_none_for_uncurated_class(settings):
    builder = CocoSubsetBuilder(settings=settings)
    ann = {"category_id": 3, "bbox": [10.0, 10.0, 5.0, 5.0]}
    cat_id_to_name = {3: "banana"}
    line = builder.to_yolo_label(ann, img_width=100, img_height=100, cat_id_to_name=cat_id_to_name)
    assert line is None


def test_build_writes_yolo_dataset(tmp_path, settings):
    ann_path, images_dir = _make_synthetic_coco(tmp_path, n_images=8)
    output_root = tmp_path / "yolo_out"

    builder = CocoSubsetBuilder(settings=settings)
    manifest = builder.build(annotation_json_path=ann_path, images_dir=images_dir, output_root=output_root)

    assert (output_root / "data.yaml").exists()
    assert len(manifest["train"]) + len(manifest["val"]) > 0

    export_cfg = settings.dataset.yolo_export
    train_images = list((output_root / export_cfg.train_subdir).glob("*.jpg"))
    train_labels = list((output_root / export_cfg.labels_train_subdir).glob("*.txt"))
    assert len(train_images) == len(manifest["train"])
    assert len(train_labels) == len(manifest["train"])


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    padded, ratio, (pad_w, pad_h) = letterbox(img, new_shape=320)
    assert padded.shape[0] == 320 and padded.shape[1] == 320
    assert ratio > 0


def test_box_conversion_round_trip():
    box_xywhn = np.array([0.5, 0.5, 0.2, 0.4], dtype=np.float32)
    xyxy = xywhn_to_xyxy(box_xywhn, img_w=200, img_h=100)
    back = xyxy_to_xywhn(xyxy, img_w=200, img_h=100)
    assert np.allclose(box_xywhn, back, atol=1e-5)
