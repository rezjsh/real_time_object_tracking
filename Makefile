.PHONY: help sync dev download-data-demo download-data-full train evaluate \
        infer-image infer-video infer-webcam export run-app run-api test lint format clean

help:
	@echo "Available targets:"
	@echo "  sync                 Install runtime dependencies via uv"
	@echo "  dev                  Install runtime + dev dependencies via uv"
	@echo "  download-data-demo   Build the fast, Colab-friendly COCO subset"
	@echo "  download-data-full   Build the larger curated COCO subset"
	@echo "  train                Fine-tune the detector on the demo subset"
	@echo "  evaluate             Evaluate the detector on the validation split"
	@echo "  infer-image          Run inference on data/samples/example.jpg"
	@echo "  infer-video          Run inference on a video (set VIDEO=path)"
	@echo "  infer-webcam         Run live webcam inference"
	@echo "  export               Export the trained model to ONNX"
	@echo "  run-app              Launch the Streamlit deployment UI"
	@echo "  run-api              Launch the FastAPI inference service"
	@echo "  test                 Run the pytest suite"
	@echo "  lint                 Run ruff lint checks"
	@echo "  format               Auto-format with ruff"
	@echo "  clean                Remove caches and build artifacts"

sync:
	uv sync

dev:
	uv sync --extra dev

download-data-demo:
	uv run python main.py download-data --mode demo

download-data-full:
	uv run python main.py download-data --mode full_subset

train:
	uv run python main.py train --mode demo

evaluate:
	uv run python main.py evaluate

infer-image:
	uv run python main.py infer-image --source data/samples/example.jpg --track

infer-video:
	uv run python main.py infer-video --source $(VIDEO)

infer-webcam:
	uv run python main.py infer-webcam

export:
	uv run python main.py export --format onnx

run-app:
	uv run python main.py run-app

run-api:
	uv run python main.py run-api

test:
	uv run pytest -v

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache
