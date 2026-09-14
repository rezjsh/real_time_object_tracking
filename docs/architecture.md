# Architecture

## Overview

The application is organized into five stages that share a common
configuration layer:

```
configs/*.yaml
      │
      ▼
config/settings.py  (typed Pydantic Settings, loaded once via get_settings())
      │
      ├──► data/            (COCO subset building, YOLO export, PyTorch Dataset)
      ├──► models/           (Detector wrapper, MultiObjectTracker, postprocess)
      ├──► inference/         (image/video/stream pipelines composing data+models)
      ├──► evaluation/        (metrics + benchmark runner)
      └──► deployment/        (Streamlit UI helpers + FastAPI service)
```

Every stage reads its parameters from `configs/*.yaml` through
`get_settings()`, so there is a single source of truth for thresholds,
paths, and hyperparameters -- no magic numbers scattered through the
codebase.

## Data Flow

1. **`scripts/download_dataset.py`** downloads COCO annotations, uses
   `CocoSubsetBuilder.load_annotations` + `.sample_records` to pick a
   bounded, class-curated subset, downloads only those images, and calls
   `.build()` to write a YOLO-format `images/`, `labels/`, `data.yaml`
   layout under `data/processed/yolo/`.

2. **`scripts/train.py`** wraps `Detector.train()`, which forwards
   `configs/train.yaml` hyperparameters into `ultralytics.YOLO.train()`
   against the `data.yaml` produced above. The best checkpoint is copied to
   `artifacts/exported_models/best.pt`.

3. **Inference pipelines** (`inference/image_infer.py`,
   `video_infer.py`, `stream_infer.py`) all follow the same shape:

   ```
   raw frame
     → Detector.predict()            (models/detector.py, wraps ultralytics)
     → postprocess_detections()      (models/postprocess.py: conf/class filter, optional NMS)
     → MultiObjectTracker.update()   (models/tracker.py: Kalman + IOU association)
     → draw_detections() / TrailDrawer  (utils/viz.py)
   ```

   The only difference between image/video/webcam pipelines is the frame
   source and whether results are written to disk, streamed to a display
   window, or returned as a generator (for Streamlit).

4. **`evaluation/benchmark.py`** replays the validation split through the
   detector and calls `evaluation/metrics.py::compute_detection_metrics` for
   COCO-style mAP/precision/recall, and (during video/stream runs) the
   tracker exposes `TrackerStats` (tracks created, fragmentations) that feed
   `compute_tracking_stability`.

5. **Deployment**: `deployment/ui.py` provides cached pipeline
   construction + dataframe helpers consumed by `app/streamlit_app.py` and
   its pages; `deployment/api.py` exposes the same `ImageInferencePipeline`
   behind a FastAPI `/predict` endpoint.

## The Tracker

`models/tracker.py::MultiObjectTracker` is a self-contained SORT/ByteTrack
implementation:

- **Motion model**: a 7-dimensional constant-velocity Kalman filter per
  track (`_KalmanBoxTracker`), state = `[cx, cy, area, aspect_ratio, vcx,
  vcy, varea]`.
- **Association**: Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
  on an IOU cost matrix, gated by `tracker.iou_threshold`.
- **Two-stage matching (ByteTrack-style)**: high-confidence detections are
  associated first; tracks that remain unmatched get a second chance
  against low-confidence detections before being aged out. This meaningfully
  improves ID stability through brief detector confidence dips (e.g.
  partial occlusion) compared to plain single-stage SORT.
- **Track lifecycle**: a track is only "confirmed" (returned by `.update()`)
  once it has accumulated `min_hits` matches; it's dropped once
  `time_since_update > max_age` frames have passed without a match.

No external tracking library (no `motpy`, no `deep_sort_realtime`) is
required -- everything lives in `models/tracker.py` on top of
`numpy`/`scipy`, which keeps the dependency footprint small and the
association logic fully inspectable/testable.

## Why This Split?

- **`Detector` never imports `torch`/`ultralytics` at module import time** --
  the import is deferred into `Detector.model` so that modules which only
  need the `Detection` dataclass (postprocessing, tests) don't pay the cost
  of loading a deep learning framework.
- **Tracker has zero dependency on the detector** -- it operates purely on
  `Detection` objects, so it can be unit-tested with synthetic detections
  and swapped for a different tracking strategy without touching detection
  code.
- **`inference/*` modules are the only place detector + tracker + viz are
  composed together** -- this is the seam where a new deployment surface
  (e.g. a websocket server) would plug in.
