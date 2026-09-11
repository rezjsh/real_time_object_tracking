#!/usr/bin/env python3
"""Unified CLI entrypoint for the object tracking application.

Usage:
    uv run python main.py download-data --mode demo
    uv run python main.py train --mode demo
    uv run python main.py evaluate
    uv run python main.py infer-image --source path/to/image.jpg
    uv run python main.py infer-video --source path/to/video.mp4
    uv run python main.py run-app
    uv run python main.py run-api
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

app = typer.Typer(help="Real-time multi-object detection and tracking application.", no_args_is_help=True)
REPO_ROOT = Path(__file__).resolve().parent


@app.command("download-data")
def download_data(
    mode: str = typer.Option("demo", help="demo | full_subset"),
    skip_download: bool = typer.Option(False, help="Skip network download; use existing data/raw"),
    full_zip: bool = typer.Option(False, help="Download the entire train2017.zip instead of sampled images"),
) -> None:
    """Download COCO 2017 and build the curated YOLO-format subset."""
    cmd = [sys.executable, "scripts/download_dataset.py", "--mode", mode]
    if skip_download:
        cmd.append("--skip-download")
    if full_zip:
        cmd.append("--full-zip")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("train")
def train(
    mode: Optional[str] = typer.Option(None, help="demo | full_subset"),
    epochs: Optional[int] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
    base_model: Optional[str] = typer.Option(None, help="e.g. yolov8n.pt"),
) -> None:
    """Fine-tune the detector on the curated COCO subset."""
    cmd = [sys.executable, "scripts/train.py"]
    if mode:
        cmd += ["--mode", mode]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    if batch_size:
        cmd += ["--batch-size", str(batch_size)]
    if base_model:
        cmd += ["--base-model", base_model]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("evaluate")
def evaluate(
    conf: Optional[float] = typer.Option(None),
    iou: Optional[float] = typer.Option(None),
    max_images: Optional[int] = typer.Option(None),
) -> None:
    """Evaluate the detector on the validation split (mAP, precision, recall, FPS)."""
    cmd = [sys.executable, "scripts/evaluate.py"]
    if conf is not None:
        cmd += ["--conf", str(conf)]
    if iou is not None:
        cmd += ["--iou", str(iou)]
    if max_images is not None:
        cmd += ["--max-images", str(max_images)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("infer-image")
def infer_image(
    source: str = typer.Option(..., help="Path to an image file"),
    output: Optional[str] = typer.Option(None),
    track: bool = typer.Option(False, help="Also run the tracker on this single frame"),
) -> None:
    """Run detection (+ optional tracking) on a single image."""
    cmd = [sys.executable, "scripts/run_demo.py", "image", "--source", source]
    if output:
        cmd += ["--output", output]
    if track:
        cmd.append("--track")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("infer-video")
def infer_video(
    source: str = typer.Option(..., help="Path to a video file"),
    output: Optional[str] = typer.Option(None),
    max_frames: Optional[int] = typer.Option(None),
) -> None:
    """Run detection + tracking on a video file, writing an annotated output."""
    cmd = [sys.executable, "scripts/run_demo.py", "video", "--source", source]
    if output:
        cmd += ["--output", output]
    if max_frames:
        cmd += ["--max-frames", str(max_frames)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("infer-webcam")
def infer_webcam(
    source: str = typer.Option("0", help="Webcam index or stream URL"),
    no_display: bool = typer.Option(False),
) -> None:
    """Run real-time detection + tracking on a webcam or stream."""
    cmd = [sys.executable, "scripts/run_demo.py", "webcam", "--source", source]
    if no_display:
        cmd.append("--no-display")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("export")
def export_model(
    format: str = typer.Option("onnx", help="onnx | torchscript | openvino | engine | coreml"),
    weights: Optional[str] = typer.Option(None),
) -> None:
    """Export a trained checkpoint to a deployment format."""
    cmd = [sys.executable, "scripts/export.py", "--format", format]
    if weights:
        cmd += ["--weights", weights]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("run-app")
def run_app(port: int = typer.Option(8501)) -> None:
    """Launch the Streamlit deployment UI."""
    cmd = ["streamlit", "run", "app/streamlit_app.py", "--server.port", str(port)]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@app.command("run-api")
def run_api(host: str = typer.Option("0.0.0.0"), port: int = typer.Option(8000)) -> None:
    """Launch the FastAPI inference service."""
    cmd = [
        "uvicorn", "object_tracking_app.deployment.api:app",
        "--host", host, "--port", str(port), "--app-dir", "src",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    app()
