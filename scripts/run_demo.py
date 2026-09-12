#!/usr/bin/env python3
"""Run a quick detect+track demo from the command line.

Usage:
    uv run python scripts/run_demo.py image --source path/to/image.jpg --output artifacts/sample_outputs/out.jpg
    uv run python scripts/run_demo.py video --source path/to/video.mp4 --output artifacts/sample_outputs/out.mp4
    uv run python scripts/run_demo.py webcam --source 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking_app.config.settings import get_settings  # noqa: E402
from object_tracking_app.inference.image_infer import ImageInferencePipeline  # noqa: E402
from object_tracking_app.inference.stream_infer import StreamInferencePipeline  # noqa: E402
from object_tracking_app.inference.video_infer import VideoInferencePipeline  # noqa: E402
from object_tracking_app.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def run_image(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = ImageInferencePipeline(settings=settings)
    output = args.output or "artifacts/sample_outputs/demo_image_output.jpg"
    result = pipeline.run_and_save(args.source, output, use_tracker=args.track)
    print(json.dumps(result["summary"], indent=2))
    logger.info("Annotated image saved to %s", output)


def run_video(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = VideoInferencePipeline(settings=settings)
    output = args.output or "artifacts/sample_outputs/demo_video_output.mp4"
    summary = pipeline.run(args.source, output_path=output, max_frames=args.max_frames)
    print(json.dumps(summary, indent=2, default=str))
    logger.info("Annotated video saved to %s", output)


def run_webcam(args: argparse.Namespace) -> None:
    settings = get_settings()
    pipeline = StreamInferencePipeline(settings=settings)
    source = int(args.source) if str(args.source).isdigit() else args.source
    pipeline.run(source=source, display=not args.no_display)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a detection + tracking demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_image = subparsers.add_parser("image", help="Run inference on a single image")
    p_image.add_argument("--source", required=True)
    p_image.add_argument("--output", default=None)
    p_image.add_argument("--track", action="store_true", help="Also run the tracker on the single frame")
    p_image.set_defaults(func=run_image)

    p_video = subparsers.add_parser("video", help="Run inference on a video file")
    p_video.add_argument("--source", required=True)
    p_video.add_argument("--output", default=None)
    p_video.add_argument("--max-frames", type=int, default=None)
    p_video.set_defaults(func=run_video)

    p_webcam = subparsers.add_parser("webcam", help="Run live inference on a webcam or stream URL")
    p_webcam.add_argument("--source", default="0")
    p_webcam.add_argument("--no-display", action="store_true")
    p_webcam.set_defaults(func=run_webcam)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
