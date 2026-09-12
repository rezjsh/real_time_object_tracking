"""Video-file inference: detect + track across frames, write annotated output."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from object_tracking_app.config.settings import Settings, get_settings
from object_tracking_app.models.detector import Detector
from object_tracking_app.models.postprocess import postprocess_detections
from object_tracking_app.models.tracker import MultiObjectTracker
from object_tracking_app.utils.io import FPSMeter, VideoReader, VideoWriter
from object_tracking_app.utils.logger import get_logger
from object_tracking_app.utils.viz import TrailDrawer, draw_detections, draw_hud

logger = get_logger(__name__)


class VideoInferencePipeline:
    """Runs detection + multi-object tracking over every frame of a video file."""

    def __init__(self, detector: Optional[Detector] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.detector = detector or Detector(settings=self.settings)

    def run(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        class_filter: Optional[list[str]] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        """Process a video end-to-end, returning aggregate statistics.

        If `output_path` is given, writes an annotated MP4 alongside processing.
        """
        cfg = self.settings.inference
        video_cfg = self.settings.video
        conf = conf_threshold if conf_threshold is not None else cfg.conf_threshold
        iou = iou_threshold if iou_threshold is not None else cfg.iou_threshold

        tracker = MultiObjectTracker(settings=self.settings)
        trail_drawer = TrailDrawer(trail_length=video_cfg.trail_length) if video_cfg.draw_trails else None
        fps_meter = FPSMeter()

        class_totals: dict[str, int] = {}
        unique_track_ids: set[int] = set()
        frame_idx = 0
        confidences: list[float] = []

        with VideoReader(str(video_path)) as reader:
            out_fps = video_cfg.output_fps or reader.fps
            writer = None
            if output_path:
                writer = VideoWriter(
                    output_path, fps=out_fps, frame_size=(reader.width, reader.height), codec=video_cfg.codec
                )

            try:
                for frame in reader.frames():
                    if max_frames is not None and frame_idx >= max_frames:
                        break

                    detections = self.detector.predict(frame, conf=conf, iou=iou)
                    detections = postprocess_detections(
                        detections, conf_threshold=conf, iou_threshold=iou, allowed_class_names=class_filter
                    )
                    tracked = tracker.update(detections)

                    for t in tracked:
                        class_totals[t.class_name] = class_totals.get(t.class_name, 0) + 1
                        unique_track_ids.add(t.track_id)
                        confidences.append(t.confidence)

                    cur_fps = fps_meter.tick()

                    if writer:
                        annotated = frame.copy()
                        if tracked:
                            boxes = np.array([t.xyxy for t in tracked], dtype=np.float32)
                            scores = [t.confidence for t in tracked]
                            names = [t.class_name for t in tracked]
                            ids = [t.track_id for t in tracked]
                            if trail_drawer:
                                trail_drawer.update_and_draw(annotated, boxes, ids)
                                trail_drawer.prune(tracker.active_track_ids())
                            draw_detections(annotated, boxes, scores, names, track_ids=ids)
                        draw_hud(annotated, cur_fps, len(tracked))
                        writer.write(annotated)

                    frame_idx += 1
                    if progress_callback:
                        total = reader.frame_count if reader.frame_count > 0 else -1
                        progress_callback(frame_idx, total)
            finally:
                if writer:
                    writer.release()

        summary = {
            "frames_processed": frame_idx,
            "unique_objects_tracked": len(unique_track_ids),
            "class_totals": class_totals,
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "avg_fps": fps_meter.tick(),
            "output_path": str(output_path) if output_path else None,
            "tracker_stats": {
                "total_tracks_created": tracker.stats.total_tracks_created,
                "fragmentations": tracker.stats.fragmentations,
            },
        }
        logger.info(
            "Video inference complete: %d frames, %d unique objects tracked.",
            frame_idx, len(unique_track_ids),
        )
        return summary
