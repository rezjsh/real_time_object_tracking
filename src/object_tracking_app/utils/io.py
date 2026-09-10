"""File and media I/O helper utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator
from collections import deque

import cv2
import numpy as np

from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist, return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug("Saved JSON to %s", p)
    return p


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as a BGR numpy array (OpenCV convention)."""
    p = Path(path)
    img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Failed to read image at {p}. File may be corrupt or unsupported.")
    return img


def save_image(image: np.ndarray, path: str | Path) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    ok = cv2.imwrite(str(p), image)
    if not ok:
        raise IOError(f"Failed to write image to {p}")
    return p


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with sane defaults and context-manager support."""

    def __init__(self, source: str | int):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video source: {source}")

    @property
    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def release(self) -> None:
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter for annotated output videos."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ):
        self.path = ensure_dir(Path(path).parent) / Path(path).name

        if len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        fourcc = cv2.VideoWriter_fourcc(*codec)

        if frame_size[0] <= 0 or frame_size[1] <= 0:
            raise ValueError("frame dimensions must be positive")
        
        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        
        self.writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise IOError(f"Could not open VideoWriter for {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer.isOpened():
            self.writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class FPSMeter:
    """Simple rolling FPS counter for real-time inference loops."""

    def __init__(self, window: int = 30):
        if window < 2:
            raise ValueError("window must be at least 2")
        self._timestamps = deque(maxlen=window)


    def tick(self) -> float:
        now = time.perf_counter()
        self._timestamps.append(now)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
