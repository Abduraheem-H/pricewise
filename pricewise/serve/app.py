"""FastAPI serving app for the trained PriceWise model.

Endpoints:
  GET  /         -> the single-page prediction form
  GET  /health   -> liveness + whether a model is loaded
  GET  /meta     -> model metadata (best model, metrics, params)
  POST /predict  -> {features...} -> {predicted_price, ...}

Run:  python -m pricewise.serve   (or: uvicorn pricewise.serve.app:app --reload)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pricewise import config
from pricewise.predict import predict_one
from pricewise.serve.schemas import HouseFeatures, PredictionResponse

app = FastAPI(
    title="PriceWise API",
    version="0.1.0",
    description="Predict house sale prices with a trained gradient-boosting model.",
)

# Permissive CORS so the form works whether served here or from another origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def _metadata() -> dict:
    if config.METADATA_PATH.exists():
        return json.loads(config.METADATA_PATH.read_text())
    return {}


def _metrics() -> dict:
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text())
    return {}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": config.MODEL_PATH.exists(),
        "best_model": _metadata().get("best_model"),
    }


@app.get("/meta")
def meta() -> dict:
    return {"metadata": _metadata(), "metrics": _metrics()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:
    if not config.MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m pricewise.train` first.",
        )
    try:
        price = predict_one(features.to_features())
    except Exception as exc:  # surface bad inputs as 400 rather than 500
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PredictionResponse(
        predicted_price=round(price, 2),
        model_name=_metadata().get("best_model", "unknown"),
    )
