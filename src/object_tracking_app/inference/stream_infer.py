"""Webcam / live-stream inference: real-time detect + track loop.

Designed to run as a standalone script (see scripts/run_demo.py) with an
OpenCV display window, or to be driven frame-by-frame from another caller
(e.g. a future websocket-based deployment) via `process_frame`.
"""

from __future__ import annotations

from typing import Iterator, Optional

import cv2
import numpy as np

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.models.detector import Detector
from object_tracking_app.models.postprocess import postprocess_detections
from object_tracking_app.models.tracker import MultiObjectTracker
from object_tracking_app.utils.io import FPSMeter, VideoReader
from object_tracking_app.utils.logger import get_logger
from object_tracking_app.utils.viz import TrailDrawer, draw_detections, draw_hud

logger = get_logger(__name__)


class StreamInferencePipeline:
    """Real-time detection + tracking loop for a webcam or network video stream."""

    def __init__(self, detector: Optional[Detector] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.detector = detector or Detector(settings=self.settings)
        self.tracker = MultiObjectTracker(settings=self.settings)
        self.trail_drawer = TrailDrawer(trail_length=self.settings.video.trail_length)
        self.fps_meter = FPSMeter()

    def process_frame(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        class_filter: Optional[list[str]] = None,
        draw: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Process a single frame: detect, track, optionally annotate. Returns (frame, info)."""
        cfg = self.settings.inference
        conf = conf_threshold if conf_threshold is not None else cfg.conf_threshold
        iou = iou_threshold if iou_threshold is not None else cfg.iou_threshold

        detections = self.detector.predict(frame, conf=conf, iou=iou)
        detections = postprocess_detections(
            detections, conf_threshold=conf, iou_threshold=iou, allowed_class_names=class_filter
        )
        tracked = self.tracker.update(detections)
        cur_fps = self.fps_meter.tick()

        annotated = frame
        if draw:
            annotated = frame.copy()
            if tracked:
                boxes = np.array([t.xyxy for t in tracked], dtype=np.float32)
                scores = [t.confidence for t in tracked]
                names = [t.class_name for t in tracked]
                ids = [t.track_id for t in tracked]
                self.trail_drawer.update_and_draw(annotated, boxes, ids)
                self.trail_drawer.prune(self.tracker.active_track_ids())
                draw_detections(annotated, boxes, scores, names, track_ids=ids)
            draw_hud(annotated, cur_fps, len(tracked))

        info = {"tracked_objects": tracked, "fps": cur_fps}
        return annotated, info

    def run(
        self,
        source: Optional[int | str] = None,
        display: Optional[bool] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        class_filter: Optional[list[str]] = None,
        frame_skip: Optional[int] = None,
    ) -> None:
        """Blocking loop reading frames from a webcam/stream, optionally displaying via cv2.imshow."""
        stream_cfg = self.settings.stream
        source = source if source is not None else stream_cfg.source
        display = display if display is not None else stream_cfg.display
        skip = frame_skip if frame_skip is not None else stream_cfg.frame_skip

        logger.info("Opening stream source=%s", source)
        with VideoReader(source) as reader:
            frame_idx = 0
            try:
                for frame in reader.frames():
                    if skip and frame_idx % (skip + 1) != 0:
                        frame_idx += 1
                        continue

                    annotated, info = self.process_frame(
                        frame, conf_threshold, iou_threshold, class_filter, draw=display
                    )

                    if display:
                        cv2.imshow("Real-Time Object Tracking", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            logger.info("Quit requested by user.")
                            break

                    frame_idx += 1
            finally:
                if display:
                    cv2.destroyAllWindows()

    def frames(self, source: Optional[int | str] = None, **kwargs) -> Iterator[tuple[np.ndarray, dict]]:
        """Generator variant of `run` for callers that want to consume frames themselves
        (e.g. a Streamlit loop) instead of blocking with an OpenCV window."""
        stream_cfg = self.settings.stream
        source = source if source is not None else stream_cfg.source
        with VideoReader(source) as reader:
            for frame in reader.frames():
                yield self.process_frame(frame, **kwargs)
