# 🎯 Real-Time Multi-Object Detection & Tracking

A production-quality, end-to-end computer vision application that trains a
YOLO-style object detector on a curated **COCO 2017** subset, tracks multiple
objects across frames with a self-contained **ByteTrack/SORT-style tracker**
(Kalman filter + IOU association, no external tracking library required),
and ships with a **Streamlit deployment UI** plus a **FastAPI inference
service**.

Built with [`uv`](https://docs.astral.sh/uv/) for fast, reproducible Python
environment management.

---

## ✨ Features

- **Detector**: Ultralytics YOLOv8n fine-tuned on a curated, Colab-friendly
  COCO subset (`person`, `car`, `bicycle`, `bus`, `truck`, `motorcycle`,
  `traffic light`).
- **Tracker**: A from-scratch, dependency-light multi-object tracker
  implementing SORT's Kalman-filter motion model with ByteTrack's two-stage
  high/low-confidence association -- stable IDs through brief occlusions.
- **Inference**: single image, video file, and live webcam/stream, all
  sharing the same detection + tracking pipeline.
- **Evaluation**: mAP@0.5, precision, recall, F1, FPS, and tracking
  stability indicators (ID switches, fragmentation).
- **Deployment**: a three-page Streamlit app (image inference, video
  inference, analytics) and a FastAPI `/predict` endpoint for programmatic
  access.
- **Config-driven**: every stage (data, train, inference, deploy, logging)
  is controlled by YAML files under `configs/`, loaded into typed Pydantic
  settings.

---

## 📁 Project Structure

```
real_time_object_tracking/
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .env.example
├── Makefile
├── main.py                  # unified CLI (download-data/train/evaluate/infer/run-app/run-api)
├── configs/                 # YAML configs: project, data, train, inference, deploy, logging
├── data/                    # raw/processed/external/splits/samples (gitignored contents)
├── artifacts/                # metrics/plots/sample_outputs/exported_models (gitignored contents)
├── notebooks/                # EDA, training, error-analysis notebooks
├── docs/                     # architecture, dataset, api, deployment docs
├── scripts/                  # download_dataset.py, train.py, evaluate.py, export.py, run_demo.py
├── app/                       # Streamlit deployment UI (3 pages)
├── src/object_tracking_app/  # installable Python package
│   ├── config/                # settings.py -- typed YAML config loader
│   ├── data/                  # coco_subset.py, transforms.py, datamodule.py
│   ├── models/                # detector.py, tracker.py, postprocess.py
│   ├── inference/              # image_infer.py, video_infer.py, stream_infer.py
│   ├── evaluation/             # metrics.py, benchmark.py
│   ├── deployment/             # api.py (FastAPI), ui.py (Streamlit helpers)
│   └── utils/                  # io.py, logger.py, viz.py
└── tests/                     # pytest suite (34 tests, no GPU/network required)
```

---

## 🚀 Quickstart with `uv`

```bash
# 1. Install uv (if you don't have it): https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the project root, create the environment and install dependencies
uv sync

# (Optional) install dev dependencies too:
uv sync --extra dev

# 3. Run the test suite
uv run pytest -v
```

`uv init` was used to scaffold this project originally; `pyproject.toml` and
`uv.lock` are already committed, so `uv sync` is all you need to reproduce
the exact environment. If you're adding a new dependency yourself:

```bash
uv add <package>            # add a runtime dependency
uv add --dev <package>       # add a dev-only dependency
uv run python main.py --help # run any command inside the managed venv
```

---

## 📦 1. Dataset: COCO Subset for Colab-Friendly Training

This project uses the real **COCO 2017** dataset but never requires
downloading the full 18GB image set. `scripts/download_dataset.py`:

1. Downloads `annotations_trainval2017.zip` (~241MB, always needed).
2. Filters annotations down to the curated classes in `configs/data.yaml`.
3. Samples a bounded number of images per class (`demo` mode: 40/class,
   200 total; `full_subset` mode: 1500/class, 8000 total).
4. Downloads **only the sampled images** (not the full zip) unless you pass
   `--full-zip`.
5. Converts annotations to YOLO format and writes `data/processed/yolo/`.

```bash
# Fast, Colab-friendly smoke-test subset (~200 images)
uv run python main.py download-data --mode demo

# Larger curated subset for real training (~8000 images)
uv run python main.py download-data --mode full_subset
```

Switch modes any time by editing `dataset.mode` in `configs/data.yaml`, or
pass `--mode` on the CLI.

See [`docs/dataset.md`](docs/dataset.md) for the full breakdown.

---

## 🏋️ 2. Training

```bash
uv run python main.py train --mode demo --epochs 10 --batch-size 8
```

This fine-tunes `yolov8n.pt` (configurable in `configs/train.yaml`) on the
subset built above, and copies the best checkpoint to
`artifacts/exported_models/best.pt`. All hyperparameters (epochs, LR,
augmentation, optimizer) live in `configs/train.yaml`.

**Colab tip:** the `demo` dataset mode + `yolov8n` + ~10-15 epochs trains in
well under 30 minutes on a T4 GPU, and is enough to see meaningful
detections on the curated classes. Switch to `full_subset` mode and more
epochs once you've validated the pipeline end-to-end.

---

## 📊 3. Evaluation

```bash
uv run python main.py evaluate --max-images 200
```

Computes mAP@0.5, precision, recall, F1, and FPS over the validation split,
and writes a JSON report to `artifacts/metrics/evaluation_report.json` --
which is also what powers the **Analytics** page in the Streamlit app.

---

## 🎥 4. Inference

```bash
# Single image
uv run python main.py infer-image --source data/samples/example.jpg --track

# Video file (writes an annotated .mp4)
uv run python main.py infer-video --source path/to/video.mp4

# Live webcam (press 'q' to quit the display window)
uv run python main.py infer-webcam --source 0
```

All three paths (image/video/webcam) share the same
`Detector` → `postprocess` → `MultiObjectTracker` → `viz` pipeline defined
under `src/object_tracking_app/`.

---

## 🖥️ 5. Deployment UI

### Streamlit app

```bash
uv run python main.py run-app
# or directly:
uv run streamlit run app/streamlit_app.py
```

Then open `http://localhost:8501`. Three pages:

- **Image Inference** — upload an image, tune confidence/IOU thresholds and
  class filters, view + download annotated results.
- **Video Inference** — upload a video, run detection + tracking with a
  progress bar, view class totals and tracker stability, download the
  annotated video.
- **Analytics** — browse saved `scripts/evaluate.py` reports and previously
  saved inference outputs.

### FastAPI service

```bash
uv run python main.py run-api
# or directly:
uv run uvicorn object_tracking_app.deployment.api:app --app-dir src --reload
```

Then `POST` an image to `http://localhost:8000/predict`:

```bash
curl -X POST "http://localhost:8000/predict?conf_threshold=0.4" \
  -F "file=@data/samples/example.jpg"
```

See [`docs/api.md`](docs/api.md) for the full endpoint reference.

---

## 🧪 Tests

```bash
uv run pytest -v
```

34 tests cover COCO-subset filtering/sampling/YOLO export, the tracker
(ID stability, track creation/expiry, reset), post-processing (confidence
filtering, class filtering, NMS), the image/video inference pipelines, and
the FastAPI endpoints -- all using synthetic data and a `FakeDetector`
stand-in, so the suite runs in ~2 seconds with **no GPU and no model
download required**.

---

## 🔧 Configuration

| File                     | Controls                                              |
|---------------------------|--------------------------------------------------------|
| `configs/project.yaml`    | project name, seed, device, artifact/data paths        |
| `configs/data.yaml`       | COCO classes, dataset modes (demo/full_subset), YOLO export layout |
| `configs/train.yaml`      | base model, epochs, batch size, optimizer, augmentation |
| `configs/inference.yaml`  | confidence/IOU thresholds, tracker params, video/stream settings |
| `configs/deploy.yaml`     | Streamlit UI defaults, FastAPI host/port/CORS           |
| `configs/logging.yaml`    | standard-library `logging.dictConfig`                   |

Loaded centrally via `src/object_tracking_app/config/settings.py::get_settings()`.

---

## 📚 Further Reading

- [`docs/architecture.md`](docs/architecture.md) — system design and data flow
- [`docs/dataset.md`](docs/dataset.md) — COCO subset details
- [`docs/api.md`](docs/api.md) — FastAPI endpoint reference
- [`docs/deployment.md`](docs/deployment.md) — deploying the Streamlit app / API

## 📄 License

MIT — see the `license` field in `pyproject.toml`. Update as needed for your use case.
