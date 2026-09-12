"""Lightweight image transform utilities used outside of ultralytics' own
training-time augmentation pipeline (e.g. for datamodule preview / evaluation
preprocessing and the Streamlit app).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: int | Tuple[int, int] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize + pad image to a square target size while preserving aspect ratio.

    Mirrors the standard YOLO letterbox preprocessing so that inference-time
    resizing matches what the detector expects.

    Returns:
        (padded_image, scale_ratio, (pad_w, pad_h))
    """
    shape = image.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)


def normalize_bgr_to_rgb_float(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 BGR image to a float32 RGB tensor in [0, 1], HWC."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32)) / 255.0


def xywhn_to_xyxy(box_xywhn: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] to absolute pixel [x1, y1, x2, y2]."""
    cx, cy, w, h = box_xywhn
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def xyxy_to_xywhn(box_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert absolute pixel [x1, y1, x2, y2] to normalized [cx, cy, w, h]."""
    x1, y1, x2, y2 = box_xyxy
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    return np.array([cx, cy, w, h], dtype=np.float32)
