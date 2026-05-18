"""
Model service for the Explainable Credit Pricing Intelligence System.

Responsibilities:
- Load the XGBoost model and feature list once at import time.
- Transform BorrowerInput into the exact encoded feature format the model expects.
- Run single-record and batch predictions.

Preprocessing notes (documented from inspection of app/app.py and app/features_list.pkl):

1. No StandardScaler is applied. The original Streamlit interface passes raw numerical
   values directly to the model. XGBoost does not require feature scaling, and there is
   no saved training-time scaler artifact. Raw values are used here to match the likely
   training pipeline.

2. Categorical one-hot encoding must produce column names that exactly match
   features_list.pkl:
     - term: "36 months" -> "term_36 months", "60 months" -> "term_60 months"
     - purpose: "debt_consolidation" -> "purpose_debt_consolidation" etc.
     - verification_status: "Not Verified" -> "verification_status_Not Verified" etc.

   The original Streamlit app has a preprocessing bug: it converts term to
   "36_months" (underscore) and verification_status to lowercase_underscore before
   pd.get_dummies, producing mismatched column names. After reindex those columns are
   filled with 0. This service corrects that by using the raw original values.

3. After one-hot encoding, the encoded DataFrame is reindexed to features_list to
   ensure column order and presence exactly match what the model expects.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.core.config import (
    CATEGORICAL_FEATURES,
    FEATURES_LIST_PATH,
    MODEL_PATH,
    NUMERICAL_FEATURES,
)
from src.schemas.prediction import BorrowerInput

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    logger.info("Loading model from %s", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded: %d features expected", model.n_features_in_)
    return model


@lru_cache(maxsize=1)
def _load_features() -> list[str]:
    logger.info("Loading feature list from %s", FEATURES_LIST_PATH)
    features = joblib.load(FEATURES_LIST_PATH)
    logger.info("Feature list loaded: %d features", len(features))
    return list(features)


def is_model_loaded() -> bool:
    try:
        _load_model()
        return True
    except Exception:
        return False


def _encode_input(record: BorrowerInput) -> pd.DataFrame:
    """Transform a single BorrowerInput into the encoded feature DataFrame."""
    row = {
        "loan_amnt": record.loan_amnt,
        "installment": record.installment,
        "annual_inc": record.annual_inc,
        "revol_util": record.revol_util,
        "total_rec_int": record.total_rec_int,
        "inq_last_6mths": float(record.inq_last_6mths),
        # Categorical fields kept in their original form so pd.get_dummies
        # produces column names matching features_list.pkl exactly.
        "term": record.term,
        "purpose": record.purpose,
        "verification_status": record.verification_status,
    }
    df = pd.DataFrame([row])
    encoded = pd.get_dummies(df, columns=list(CATEGORICAL_FEATURES))
    features = _load_features()
    encoded = encoded.reindex(columns=features, fill_value=0)
    return encoded


def _encode_batch(records: list[BorrowerInput]) -> pd.DataFrame:
    """Transform a list of BorrowerInput records into the encoded feature DataFrame."""
    rows = [
        {
            "loan_amnt": r.loan_amnt,
            "installment": r.installment,
            "annual_inc": r.annual_inc,
            "revol_util": r.revol_util,
            "total_rec_int": r.total_rec_int,
            "inq_last_6mths": float(r.inq_last_6mths),
            "term": r.term,
            "purpose": r.purpose,
            "verification_status": r.verification_status,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    encoded = pd.get_dummies(df, columns=list(CATEGORICAL_FEATURES))
    features = _load_features()
    encoded = encoded.reindex(columns=features, fill_value=0)
    return encoded


def predict_single(record: BorrowerInput) -> float:
    """Return the predicted interest rate for a single borrower record."""
    model = _load_model()
    encoded = _encode_input(record)
    result = model.predict(encoded)
    return float(result[0])


def predict_batch(records: list[BorrowerInput]) -> list[float]:
    """Return predicted interest rates for a list of borrower records."""
    model = _load_model()
    encoded = _encode_batch(records)
    results = model.predict(encoded)
    return [float(v) for v in results]
