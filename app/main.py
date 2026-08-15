from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

sys.path.append(str(Path(__file__).resolve().parent.parent))
from app.schemas import (  # noqa: E402
    BatchPredictRequest,
    BatchPredictResponse,
    FeatureSpec,
    HealthResponse,
    HouseFeatures,
    PredictResponse,
    SchemaResponse,
)
from src.schema import FEATURE_COLUMNS  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("house-pricing-api")

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"

state: dict = {"model": None, "model_version": None}


def _load_model() -> None:
    try:
        state["model"] = joblib.load(MODEL_PATH)
        import json

        metadata_path = ARTIFACTS_DIR / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                state["model_version"] = json.load(f).get("model_version")
        logger.info("Modelo cargado desde %s", MODEL_PATH)
    except Exception:
        logger.exception("No se pudo cargar el modelo desde %s", MODEL_PATH)
        state["model"] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    state["model"] = None


app = FastAPI(
    title="House Pricing API",
    description="API de predicción de precios de casas (USA Housing dataset)",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(loc) for loc in e["loc"] if loc != "body"), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"detail": "Entrada inválida", "errors": errors},
    )


def _predict_one(features: HouseFeatures) -> PredictResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado. Intenta más tarde.")

    row = pd.DataFrame(
        [[getattr(features, col) for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS
    )
    try:
        pred = float(state["model"].predict(row)[0])
    except Exception as exc:  # pragma: no cover - defensivo
        logger.exception("Error durante la predicción")
        raise HTTPException(
            status_code=500, detail=f"Error al generar la predicción: {exc}"
        ) from exc

    return PredictResponse(
        predicted_price=round(pred, 2),
        model_version=state["model_version"] or "unknown",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=state["model_version"],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(features: HouseFeatures) -> PredictResponse:
    return _predict_one(features)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    predictions = [_predict_one(item) for item in payload.items]
    return BatchPredictResponse(predictions=predictions, count=len(predictions))


@app.get("/model/schema", response_model=SchemaResponse)
def model_schema() -> SchemaResponse:
    features = [FeatureSpec(name=col, type="float", required=True) for col in FEATURE_COLUMNS]
    return SchemaResponse(features=features, target="price", model_version=state["model_version"])
