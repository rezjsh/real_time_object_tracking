"""Typed configuration loading for the object tracking application.

All runtime configuration lives in YAML files under `configs/`. This module
loads them into typed dataclasses / Pydantic models so the rest of the
codebase gets autocomplete, validation, and a single source of truth instead
of passing raw dicts around.

Usage:
    from object_tracking_app.config.settings import get_settings
    settings = get_settings()
    print(settings.inference.conf_threshold)
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, Field

# Repo root = three levels up from this file (src/object_tracking_app/config/settings.py)
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(relative_path: str) -> Path:
    """Resolve a path relative to the repo root, returning an absolute Path."""
    p = Path(relative_path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


# --------------------------------------------------------------------------- #
# project.yaml
# --------------------------------------------------------------------------- #
class ProjectPaths(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    external_dir: str = "data/external"
    splits_dir: str = "data/splits"
    samples_dir: str = "data/samples"
    metrics_dir: str = "artifacts/metrics"
    plots_dir: str = "artifacts/plots"
    sample_outputs_dir: str = "artifacts/sample_outputs"
    exported_models_dir: str = "artifacts/exported_models"


class ProjectInfo(BaseModel):
    name: str = "object-tracking-app"
    seed: int = 42
    device: str = "auto"
    artifacts_dir: str = "artifacts"
    data_dir: str = "data"


class ProjectSettings(BaseModel):
    project: ProjectInfo = Field(default_factory=ProjectInfo)
    paths: ProjectPaths = Field(default_factory=ProjectPaths)


# --------------------------------------------------------------------------- #
# data.yaml
# --------------------------------------------------------------------------- #
class ModeConfig(BaseModel):
    images_per_class: int
    max_total_images: int
    val_fraction: float
    img_size: int


class YoloExportConfig(BaseModel):
    output_dir: str = "data/processed/yolo"
    train_subdir: str = "images/train"
    val_subdir: str = "images/val"
    labels_train_subdir: str = "labels/train"
    labels_val_subdir: str = "labels/val"


class DatasetSettings(BaseModel):
    name: str = "coco2017"
    mode: str = "demo"
    base_url: str = "http://images.cocodataset.org"
    annotations_url: str = ""
    train_images_url: str = ""
    val_images_url: str = ""
    classes: List[str] = Field(default_factory=list)
    modes: dict[str, ModeConfig] = Field(default_factory=dict)
    splits_dir: str = "data/splits"
    processed_dir: str = "data/processed"
    raw_dir: str = "data/raw"
    yolo_export: YoloExportConfig = Field(default_factory=YoloExportConfig)

    @property
    def active_mode(self) -> ModeConfig:
        if self.mode not in self.modes:
            raise KeyError(f"Unknown dataset mode '{self.mode}'. Available: {list(self.modes)}")
        return self.modes[self.mode]


# --------------------------------------------------------------------------- #
# train.yaml
# --------------------------------------------------------------------------- #
class AugmentConfig(BaseModel):
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5


class CheckpointConfig(BaseModel):
    save_period: int = 5
    export_best_to: str = "artifacts/exported_models/best.pt"


class TrainSettings(BaseModel):
    base_model: str = "yolov8n.pt"
    epochs: int = 30
    batch_size: int = 16
    img_size: int = 416
    patience: int = 10
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3
    workers: int = 4
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    project_dir: str = "artifacts/exported_models"
    run_name: str = "detector_run"
    resume: bool = False
    pretrained: bool = True
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)


# --------------------------------------------------------------------------- #
# inference.yaml
# --------------------------------------------------------------------------- #
class InferenceSettings(BaseModel):
    weights_path: str = "artifacts/exported_models/best.pt"
    fallback_weights: str = "yolov8n.pt"
    device: str = "auto"
    img_size: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 100
    classes: Optional[List[str]] = None


class TrackerSettings(BaseModel):
    type: str = "sort"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3
    high_conf_threshold: float = 0.5
    low_conf_threshold: float = 0.1


class VideoSettings(BaseModel):
    output_fps: Optional[float] = None
    codec: str = "mp4v"
    draw_trails: bool = True
    trail_length: int = 30


class StreamSettings(BaseModel):
    source: Any = 0
    frame_skip: int = 0
    display: bool = True


# --------------------------------------------------------------------------- #
# deploy.yaml
# --------------------------------------------------------------------------- #
class StreamlitSettings(BaseModel):
    title: str = "Real-Time Multi-Object Detection & Tracking"
    layout: str = "wide"
    default_conf_threshold: float = 0.35
    default_iou_threshold: float = 0.45
    max_upload_size_mb: int = 200
    allowed_image_types: List[str] = Field(default_factory=lambda: ["jpg", "jpeg", "png", "bmp"])
    allowed_video_types: List[str] = Field(default_factory=lambda: ["mp4", "avi", "mov", "mkv"])
    save_outputs_dir: str = "artifacts/sample_outputs"


class ApiSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "Object Tracking Inference API"
    version: str = "0.1.0"
    max_upload_size_mb: int = 200
    cors_allow_origins: List[str] = Field(default_factory=lambda: ["*"])


class RuntimeSettings(BaseModel):
    weights_path: str = "artifacts/exported_models/best.pt"
    fallback_weights: str = "yolov8n.pt"
    device: str = "auto"


class DeploySettings(BaseModel):
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


# --------------------------------------------------------------------------- #
# Aggregate settings object
# --------------------------------------------------------------------------- #
class Settings(BaseModel):
    project_info: ProjectSettings = Field(default_factory=ProjectSettings)
    dataset: DatasetSettings = Field(default_factory=DatasetSettings)
    train: TrainSettings = Field(default_factory=TrainSettings)
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    video: VideoSettings = Field(default_factory=VideoSettings)
    stream: StreamSettings = Field(default_factory=StreamSettings)
    deploy: DeploySettings = Field(default_factory=DeploySettings)

    def resolve(self, relative_path: str) -> Path:
        return resolve_path(relative_path)

    def resolved_device(self, requested: Optional[str] = None) -> str:
        """Resolve 'auto' to an actual torch device string."""
        requested = requested or self.project_info.project.device
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


@functools.lru_cache(maxsize=1)
def get_settings(configs_dir: Optional[str] = None) -> Settings:
    """Load and cache all YAML configs into a single validated Settings object."""
    base = Path(configs_dir) if configs_dir else CONFIGS_DIR

    project_raw = _load_yaml(base / "project.yaml")
    data_raw = _load_yaml(base / "data.yaml")
    train_raw = _load_yaml(base / "train.yaml")
    inference_raw = _load_yaml(base / "inference.yaml")
    deploy_raw = _load_yaml(base / "deploy.yaml")

    dataset_block = data_raw.get("dataset", {})
    modes_raw = dataset_block.get("modes", {})
    dataset_block = {**dataset_block, "modes": {k: ModeConfig(**v) for k, v in modes_raw.items()}}
    if "yolo_export" in dataset_block:
        dataset_block["yolo_export"] = YoloExportConfig(**dataset_block["yolo_export"])

    settings = Settings(
        project_info=ProjectSettings(**project_raw),
        dataset=DatasetSettings(**dataset_block),
        train=TrainSettings(**train_raw.get("train", {})),
        inference=InferenceSettings(**inference_raw.get("inference", {})),
        tracker=TrackerSettings(**inference_raw.get("tracker", {})),
        video=VideoSettings(**inference_raw.get("video", {})),
        stream=StreamSettings(**inference_raw.get("stream", {})),
        deploy=DeploySettings(
            streamlit=StreamlitSettings(**deploy_raw.get("streamlit", {})),
            api=ApiSettings(**deploy_raw.get("api", {})),
            runtime=RuntimeSettings(**deploy_raw.get("runtime", {})),
        ),
    )
    return settings


def reload_settings() -> Settings:
    """Clear the cache and reload settings from disk (useful for tests)."""
    get_settings.cache_clear()
    return get_settings()
