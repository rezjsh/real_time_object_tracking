"""Centralized logging setup for the object tracking application."""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import yaml

_CONFIGURED = False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def setup_logging(config_path: Optional[str] = None, level: Optional[str] = None) -> None:
    """Configure logging from configs/logging.yaml. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = _repo_root()
    cfg_path = Path(config_path) if config_path else root / "configs" / "logging.yaml"

    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Ensure the log directory referenced by the file handler exists.
        for handler in cfg.get("handlers", {}).values():
            filename = handler.get("filename")
            if filename:
                log_path = root / filename if not os.path.isabs(filename) else Path(filename)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handler["filename"] = str(log_path)
        logging.config.dictConfig(cfg)
    else:
        logging.basicConfig(
            level=level or os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )

    if level:
        logging.getLogger("object_tracking_app").setLevel(level)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module logger, configuring logging on first use."""
    setup_logging()
    return logging.getLogger(name)
