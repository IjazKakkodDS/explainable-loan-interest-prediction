"""
SHAP explanation service for the Explainable Credit Pricing Intelligence System.

Responsibilities:
- Load the SHAP explainer once (from pre-saved artifact or dynamically from model).
- Generate per-prediction SHAP attribution for a single BorrowerInput.
- Return top-N drivers sorted by absolute SHAP contribution.

Artifact loading strategy:
1. Attempt to load app/shap_explainer.pkl (pre-saved TreeExplainer).
2. If the saved explainer fails to load (version mismatch or missing file),
   fall back to creating shap.Explainer(model) dynamically.
3. Any fallback is logged clearly; behaviour is documented in the explanation response.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import joblib
import shap

from src.core.config import MODEL_PATH, SHAP_EXPLAINER_PATH
from src.schemas.prediction import BorrowerInput, ExplanationResponse, FeatureDriver
from src.services.model_service import _encode_input, _load_model

logger = logging.getLogger(__name__)

_EXPLAINER_SOURCE: str = "unknown"


@lru_cache(maxsize=1)
def _load_explainer():
    global _EXPLAINER_SOURCE

    # Attempt 1: pre-saved artifact
    if SHAP_EXPLAINER_PATH.exists():
        try:
            explainer = joblib.load(SHAP_EXPLAINER_PATH)
            _EXPLAINER_SOURCE = f"pre-saved artifact ({SHAP_EXPLAINER_PATH.name})"
            logger.info("SHAP explainer loaded from saved artifact")
            return explainer
        except Exception as exc:
            logger.warning(
                "Failed to load saved SHAP explainer (%s). Falling back to dynamic build.",
                exc,
            )

    # Attempt 2: build dynamically from model
    try:
        model = _load_model()
        explainer = shap.Explainer(model)
        _EXPLAINER_SOURCE = "dynamically built from model at service startup"
        logger.info("SHAP explainer built dynamically from model")
        return explainer
    except Exception as exc:
        logger.error("SHAP explainer could not be initialised: %s", exc)
        raise RuntimeError(
            f"SHAP explainer unavailable. Cause: {exc}"
        ) from exc


def is_explainer_loaded() -> bool:
    try:
        _load_explainer()
        return True
    except Exception:
        return False


def explain_single(record: BorrowerInput, top_n: int = 8) -> ExplanationResponse:
    """
    Generate SHAP attribution for a single BorrowerInput.

    Returns top_n feature drivers sorted by absolute SHAP value.
    Raises RuntimeError if the SHAP explainer cannot be loaded.
    """
    explainer = _load_explainer()
    encoded = _encode_input(record)

    shap_values = explainer(encoded)
    sv = shap_values[0]

    # sv.values is a 1-D array aligned to features_list
    feature_names: list[str] = list(sv.feature_names) if hasattr(sv, "feature_names") else list(encoded.columns)
    contributions: list[float] = list(sv.values)

    paired = sorted(
        zip(feature_names, contributions),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    drivers = [
        FeatureDriver(
            feature=name,
            shap_value=round(val, 6),
            direction=(
                "increases_rate" if val > 0 else
                "decreases_rate" if val < 0 else
                "neutral"
            ),
        )
        for name, val in paired
    ]

    model = _load_model()
    predicted_rate = float(model.predict(encoded)[0])

    return ExplanationResponse(
        predicted_interest_rate=round(predicted_rate, 4),
        top_drivers=drivers,
        explanation_method=f"SHAP TreeExplainer ({_EXPLAINER_SOURCE})",
    )
