#!/usr/bin/env python3
"""Evaluate the trained detector on the validation split: mAP, precision/recall, FPS.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --max-images 100 --conf 0.4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402
from object_tracking_app.evaluation.benchmark import run_benchmark  # noqa: E402
from object_tracking_app.utils.logger import get_logger  # noqa: E402
from object_tracking_app.utils.io import ensure_dir  # noqa: E402


logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detector performance on the validation split.")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--labels-dir", type=str, default=None)
    args = parser.parse_args()

    settings = get_settings()
    export_cfg = settings.dataset.yolo_export
    output_root = settings.resolve(export_cfg.output_dir)

    images_dir = args.images_dir or str(output_root / export_cfg.val_subdir)
    labels_dir = args.labels_dir or str(output_root / export_cfg.labels_val_subdir)

    report_path = settings.resolve(settings.project_info.paths.metrics_dir) / "evaluation_report.json"
    ensure_dir(report_path.parent)


    report = run_benchmark(
        images_dir=images_dir,
        labels_dir=labels_dir,
        class_names=settings.dataset.classes,
        settings=settings,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_images=args.max_images,
        output_report_path=report_path,
    )

    print(json.dumps(report, indent=2))
    logger.info("Evaluation report saved to %s", report_path)


if __name__ == "__main__":
    main()
