#!/usr/bin/env python3
"""Fine-tune the detector on the curated COCO subset.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --epochs 10 --batch-size 8
    uv run python scripts/train.py --mode demo   
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402
from object_tracking_app.models.detector import Detector  # noqa: E402
from object_tracking_app.utils.io import ensure_dir  # noqa: E402
from object_tracking_app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the object detector on the COCO subset.")
    parser.add_argument("--mode", choices=["demo", "full_subset"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base-model", type=str, default=None, help="e.g. yolov8n.pt, yolov8s.pt")
    parser.add_argument(
        "--data-yaml", type=str, default=None,
        help="Path to a YOLO-format data.yaml. Defaults to the one built by download_dataset.py",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.mode:
        settings.dataset.mode = args.mode
    if args.epochs:
        settings.train.epochs = args.epochs
    if args.batch_size:
        settings.train.batch_size = args.batch_size
    if args.base_model:
        settings.train.base_model = args.base_model

    data_yaml = args.data_yaml
    if not data_yaml:
        data_yaml = str(settings.resolve(settings.dataset.yolo_export.output_dir) / "data.yaml")

    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        logger.error(
            "data.yaml not found at %s. Run `uv run python scripts/download_dataset.py --mode %s` first.",
            data_yaml_path, settings.dataset.mode,
        )
        sys.exit(1)

    detector = Detector(weights_path=settings.train.base_model, settings=settings)
    result = detector.train(data_yaml=str(data_yaml_path))

    checkpoint_dest = settings.resolve(settings.train.checkpoint.export_best_to)
    ensure_dir(checkpoint_dest.parent)

    save_dir = Path(result["save_dir"]) if result.get("save_dir") else None
    if save_dir:
        best_pt = save_dir / "weights" / "best.pt"
        if best_pt.exists():
            import shutil

            shutil.copy2(best_pt, checkpoint_dest)
            logger.info("Copied best checkpoint to %s", checkpoint_dest)
        else:
            logger.warning("Could not find best.pt under %s; check the training run directory.", save_dir)

    logger.info("Training complete. Run directory: %s", save_dir)


if __name__ == "__main__":
    main()
