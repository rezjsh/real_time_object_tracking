#!/usr/bin/env python3
"""Download COCO 2017 annotations/images and build the curated YOLO-format subset.

Usage:
    uv run python scripts/download_dataset.py --mode demo
    uv run python scripts/download_dataset.py --mode full_subset --skip-download

By default this downloads:
    - annotations_trainval2017.zip (instance annotations only, ~241MB)
    - a *sampled* set of train2017 images (only the ones needed for the
      curated classes/subset size -- NOT the full 18GB train2017.zip)

Set --full-zip to instead download the entire train2017.zip (large, slow;
only recommended if you have fast bandwidth and want the complete dataset).
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402
from object_tracking_app.data.coco_subset import CocoSubsetBuilder  # noqa: E402
from object_tracking_app.utils.io import ensure_dir  # noqa: E402
from object_tracking_app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    if dest.exists():
        logger.info("Already downloaded: %s", dest)
        return dest
    ensure_dir(dest.parent)
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    return dest


def unzip(zip_path: Path, out_dir: Path) -> None:
    logger.info("Extracting %s -> %s", zip_path, out_dir)
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def download_selected_images(image_filenames: list[str], base_url: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    for name in tqdm(image_filenames, desc="Downloading sampled images"):
        dest = out_dir / name
        if dest.exists():
            continue
        url = f"{base_url}/train2017/{name}"
        try:
            download_file(url, dest)
        except requests.RequestException as exc:
            logger.warning("Failed to download %s: %s", name, exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO 2017 and build a curated subset.")
    parser.add_argument("--mode", choices=["demo", "full_subset"], default=None, help="Override configs/data.yaml dataset.mode")
    parser.add_argument("--skip-download", action="store_true", help="Skip network downloads; assume data/raw already populated")
    parser.add_argument("--full-zip", action="store_true", help="Download the entire train2017.zip instead of sampled images")
    args = parser.parse_args()

    settings = get_settings()
    if args.mode:
        # Settings is cached as a singleton; patch the in-memory object directly for this run.
        settings.dataset.mode = args.mode

    ds_cfg = settings.dataset
    raw_dir = settings.resolve(ds_cfg.raw_dir)
    ensure_dir(raw_dir)

    annotations_zip = raw_dir / "annotations_trainval2017.zip"
    annotations_json = raw_dir / "annotations" / "instances_train2017.json"
    images_dir = raw_dir / "train2017"

    if not args.skip_download:
        download_file(ds_cfg.annotations_url, annotations_zip)
        if not annotations_json.exists():
            unzip(annotations_zip, raw_dir)

        if args.full_zip:
            images_zip = raw_dir / "train2017.zip"
            download_file(ds_cfg.train_images_url, images_zip)
            if not images_dir.exists():
                unzip(images_zip, raw_dir)
        else:
            logger.info(
                "Skipping full train2017.zip download (18GB). Building the subset first "
                "to identify only the images we actually need, then fetching those."
            )
            builder = CocoSubsetBuilder(settings=settings)
            records, cat_id_to_name = builder.load_annotations(annotations_json)
            sampled = builder.sample_records(records, cat_id_to_name)
            filenames = [r.file_name for r in sampled]
            download_selected_images(filenames, ds_cfg.base_url, images_dir)
    else:
        logger.info("Skipping downloads (--skip-download). Using existing files in %s", raw_dir)
        if not annotations_json.exists():
            logger.error("Annotations not found at %s. Remove --skip-download or place them manually.", annotations_json)
            sys.exit(1)

    builder = CocoSubsetBuilder(settings=settings)
    manifest = builder.build(annotation_json_path=annotations_json, images_dir=images_dir)

    logger.info(
        "Done. Built subset (mode=%s): %d train / %d val images. YOLO data.yaml at %s",
        ds_cfg.mode, len(manifest["train"]), len(manifest["val"]), manifest["yolo_data_yaml"],
    )


if __name__ == "__main__":
    main()
