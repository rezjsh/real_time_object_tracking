"""FastAPI inference service.

Exposes the same detection + tracking pipeline used by the Streamlit app as
an HTTP API, so the model can be integrated into other systems.

Run with:
    uvicorn object_tracking_app.deployment.api:app --host 0.0.0.0 --port 8000
or:
    uv run python main.py run-api
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from object_tracking_app.config.settings import get_settings
from object_tracking_app.inference.image_infer import ImageInferencePipeline
from object_tracking_app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
deploy_cfg = settings.deploy.api

app = FastAPI(title=deploy_cfg.title, version=deploy_cfg.version)

app.add_middleware(
    CORSMiddleware,
    allow_origins=deploy_cfg.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[ImageInferencePipeline] = None


def get_pipeline() -> ImageInferencePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ImageInferencePipeline(settings=settings)
    return _pipeline


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": deploy_cfg.version}


@app.get("/classes")
def classes() -> dict:
    try:
        return {"classes": get_pipeline().detector.class_names}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load model classes: {exc}") from exc


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf_threshold: float = Query(default=None, ge=0.0, le=1.0),
    iou_threshold: float = Query(default=None, ge=0.0, le=1.0),
    return_image: bool = Query(default=False, description="If true, returns annotated JPEG instead of JSON"),
):
    """Run detection on an uploaded image and return structured results (or an annotated JPEG)."""
    contents = await file.read()
    max_bytes = deploy_cfg.max_upload_size_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds max upload size.")

    np_arr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded file as an image.")

    try:
        result = get_pipeline().run(
            image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            use_tracker=False,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    if return_image:
        ok, buf = cv2.imencode(".jpg", result["annotated_image"])
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to encode annotated image.")
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    detections_payload = [
        {
            "box_xyxy": d.xyxy.tolist(),
            "confidence": d.confidence,
            "class_id": d.class_id,
            "class_name": d.class_name,
        }
        for d in result["detections"]
    ]
    return JSONResponse(
        {
            "num_detections": result["summary"]["num_detections"],
            "class_counts": result["summary"]["class_counts"],
            "mean_confidence": result["summary"]["mean_confidence"],
            "detections": detections_payload,
        }
    )
