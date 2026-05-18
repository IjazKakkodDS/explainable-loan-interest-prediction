"""
FastAPI serving boundary for the Explainable Credit Pricing Intelligence System.

This is a local model-serving layer, not a production lending system.
Outputs are for review and demonstration purposes only.
No financial advice is provided or implied.

Start the server:
    uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

Endpoints:
    GET  /health          System and artifact health check
    POST /predict         Single-record interest rate prediction
    POST /explain         Single-record prediction + SHAP attribution
    POST /batch-predict   Multi-record batch prediction
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.core.config import MODEL_NAME, SYSTEM_NAME
from src.schemas.prediction import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    BorrowerInput,
    ExplanationResponse,
    HealthResponse,
    PredictionResponse,
)
from src.services.explanation_service import explain_single, is_explainer_loaded
from src.services.model_service import is_model_loaded, predict_batch, predict_single

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=SYSTEM_NAME,
    description=(
        "Local model-serving boundary for explainable credit pricing. "
        "Not a production lending system. Not financial advice."
    ),
    version="0.2.0",
)


@app.middleware("http")
async def _log_request_timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("%s %s completed in %.2f ms", request.method, request.url.path, elapsed_ms)
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Return service health and artifact load status."""
    return HealthResponse(
        status="ok",
        system=SYSTEM_NAME,
        model_loaded=is_model_loaded(),
        explainer_loaded=is_explainer_loaded(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(record: BorrowerInput) -> PredictionResponse:
    """
    Predict the interest rate for a single borrower record.

    Returns the predicted interest rate in percentage points.
    This output is not financial advice and is not approved for lending decisions.
    """
    t0 = time.perf_counter()
    try:
        rate = predict_single(record)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("predict: %.4f%% in %.2f ms", rate, elapsed_ms)
    return PredictionResponse(
        predicted_interest_rate=round(rate, 4),
        model_name=MODEL_NAME,
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["explanation"])
def explain(record: BorrowerInput) -> ExplanationResponse:
    """
    Predict the interest rate and return top SHAP feature drivers.

    Returns the predicted rate and the top-8 features by absolute SHAP contribution.
    This output is not financial advice and is not approved for lending decisions.
    """
    t0 = time.perf_counter()
    try:
        result = explain_single(record)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Explanation service unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Explanation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Explanation error: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "explain: %.4f%% | %d drivers | %.2f ms",
        result.predicted_interest_rate,
        len(result.top_drivers),
        elapsed_ms,
    )
    return result


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["prediction"])
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Predict interest rates for a batch of borrower records.

    Returns predictions in the same order as the input records.
    SHAP explanations are not computed for batch requests.
    This output is not financial advice and is not approved for lending decisions.
    """
    t0 = time.perf_counter()
    n = len(request.records)
    try:
        predictions = predict_batch(request.records)
    except Exception as exc:
        logger.error("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {exc}") from exc
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("batch-predict: %d records in %.2f ms (%.3f ms/record)", n, elapsed_ms, elapsed_ms / n)
    return BatchPredictionResponse(
        predictions=predictions,
        record_count=n,
    )
