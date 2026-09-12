#!/usr/bin/env python3
"""Export a trained checkpoint to a deployment format (ONNX, TorchScript, etc.).

Usage:
    uv run python scripts/export.py --format onnx
    uv run python scripts/export.py --format torchscript --weights artifacts/exported_models/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402
from object_tracking_app.models.detector import Detector  # noqa: E402
from object_tracking_app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

SUPPORTED_FORMATS = ["onnx", "torchscript", "openvino", "engine", "coreml"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the detector to a deployment format.")
    parser.add_argument("--format", choices=SUPPORTED_FORMATS, default="onnx")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pt checkpoint (defaults to configured weights)")
    parser.add_argument("--img-size", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    detector = Detector(weights_path=args.weights, settings=settings)

    kwargs = {}
    if args.img_size:
        kwargs["imgsz"] = args.img_size

    exported_path = detector.export(format=args.format, **kwargs)
    logger.info("Exported model available at: %s", exported_path)
    print(exported_path)


if __name__ == "__main__":
    main()
