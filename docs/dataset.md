# Dataset: COCO 2017 Subset

## Source

[COCO 2017](https://cocodataset.org/#download) (Common Objects in Context),
downloaded from `http://images.cocodataset.org`. We use the `train2017`
image set and `instances_train2017.json` annotations.

## Curated Classes

Configured in `configs/data.yaml` under `dataset.classes`:

| Class          | COCO category |
|----------------|----------------|
| person         | ✅ |
| car            | ✅ |
| bicycle        | ✅ |
| bus            | ✅ |
| truck          | ✅ |
| motorcycle     | ✅ |
| traffic light  | ✅ |

These were chosen because they co-occur frequently in real-world traffic /
street-scene footage, which is a natural fit for demoing real-time
multi-object detection and tracking.

## Modes

`configs/data.yaml` → `dataset.modes`:

| Mode          | Images/class | Max total images | Val fraction | Image size |
|----------------|--------------|-------------------|----------------|-------------|
| `demo`         | 40           | 200                | 0.15           | 416         |
| `full_subset`  | 1500         | 8000               | 0.10           | 640         |

Switch modes via `configs/data.yaml` (`dataset.mode: demo`) or
`--mode` on `scripts/download_dataset.py` / `main.py download-data`.

## Build Pipeline

`src/object_tracking_app/data/coco_subset.py::CocoSubsetBuilder`:

1. **`load_annotations`** — parses `instances_train2017.json`, keeping only
   annotations whose category is one of the curated classes, indexed by
   `image_id`.
2. **`sample_records`** — for each curated class, randomly samples up to
   `images_per_class` images containing that class (seeded by
   `project.seed` for reproducibility), then caps the total at
   `max_total_images`.
3. **`build`** — splits the sampled images into train/val
   (`val_fraction`), copies image files into
   `data/processed/yolo/images/{train,val}/`, converts each COCO
   `[x, y, w, h]` box into a normalized YOLO
   `class_id cx cy w h` label line under
   `data/processed/yolo/labels/{train,val}/`, and writes a
   `data/processed/yolo/data.yaml` that Ultralytics can train against
   directly.
4. A manifest (`data/splits/<mode>_manifest.json`) records exactly which
   files ended up in each split, for reproducibility/debugging.

## Colab Usage

For a fast Colab smoke run:

```bash
uv run python main.py download-data --mode demo   # ~200 images, few minutes
uv run python main.py train --mode demo --epochs 10
```

This intentionally avoids downloading the full 18GB `train2017.zip` --
`download_dataset.py` first determines exactly which images are needed for
the curated classes/sample size, then downloads only those individual
files. Pass `--full-zip` if you'd rather download the complete image set
(useful if you plan to later increase `images_per_class` without
re-downloading).

## Re-generating the Subset

Because the manifest + `data.yaml` are deterministic given the same seed
and mode, re-running `download_dataset.py --skip-download` (after the raw
annotations/images already exist locally) will reproduce the same subset.
