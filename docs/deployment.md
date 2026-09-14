# Deployment Guide

## Local Development

```bash
uv sync
uv run python main.py run-app     # Streamlit UI on :8501
uv run python main.py run-api     # FastAPI service on :8000
```

Both read their defaults from `configs/deploy.yaml`.

## Streamlit App

- Entry point: `app/streamlit_app.py`
- Pages: `app/pages/1_Image_Inference.py`, `2_Video_Inference.py`,
  `3_Analytics.py`
- The detector is cached across reruns via
  `deployment/ui.py::get_cached_detector` (keyed by the resolved weights
  path), so repeated inference calls within a session don't reload the
  model each time.
- Uploaded videos are processed to a temp file, annotated, and written to
  `artifacts/sample_outputs/` (configurable via
  `deploy.yaml → streamlit.save_outputs_dir`).

### Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), point to
   `app/streamlit_app.py` as the entry file.
3. Since Streamlit Cloud doesn't use `uv` natively, add a
   `requirements.txt` exported from the lockfile:
   ```bash
   uv export --no-hashes --format requirements-txt > requirements.txt
   ```
4. Make sure a fine-tuned checkpoint is either committed (if small enough)
   or the app should gracefully fall back to the pretrained
   `yolov8n.pt` (it does, automatically, via
   `Detector._resolve_weights`).

## FastAPI Service

- Entry point: `src/object_tracking_app/deployment/api.py`
- Run with `uvicorn` directly for production-style process management:
  ```bash
  uv run uvicorn object_tracking_app.deployment.api:app \
    --app-dir src --host 0.0.0.0 --port 8000 --workers 2
  ```
- For containerized deployment, install dependencies via `uv sync
  --frozen`, then run the `uvicorn` command above as the container
  `CMD`. A minimal Dockerfile:

  ```dockerfile
  FROM python:3.11-slim
  RUN pip install uv
  WORKDIR /app
  COPY pyproject.toml uv.lock ./
  RUN uv sync --frozen --no-dev
  COPY . .
  CMD ["uv", "run", "uvicorn", "object_tracking_app.deployment.api:app", \
       "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
  ```

## Configuration for Production

Before deploying beyond local development:

- Tighten `configs/deploy.yaml → api.cors_allow_origins` from `["*"]` to
  your actual frontend origin(s).
- Set `configs/deploy.yaml → api.max_upload_size_mb` to a sane limit for
  your infrastructure.
- Point `configs/inference.yaml → inference.weights_path` at your
  fine-tuned checkpoint (produced by `scripts/train.py`), and confirm it's
  included in your deployment artifact (e.g. baked into the Docker image or
  fetched from object storage at startup).
- Set `DEVICE=cuda` in `.env` (copied from `.env.example`) if deploying to
  a GPU instance, or leave `auto` to let `Settings.resolved_device()`
  detect it.
