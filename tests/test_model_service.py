"""
Tests for src/services/model_service.py

Verifies:
- Model and feature list load without error
- Single prediction returns a numeric float in a plausible range
- Batch prediction returns the correct number of results
- Encoded feature DataFrame has the correct shape and column order
"""

import pytest

from src.schemas.prediction import BorrowerInput
from src.services.model_service import (
    _encode_input,
    _load_features,
    _load_model,
    predict_batch,
    predict_single,
)


SAMPLE_RECORD = BorrowerInput(
    loan_amnt=15000,
    installment=450.5,
    annual_inc=72000,
    revol_util=55.0,
    total_rec_int=1200.0,
    inq_last_6mths=2,
    term="36 months",
    purpose="debt_consolidation",
    verification_status="Verified",
)

SAMPLE_RECORD_2 = BorrowerInput(
    loan_amnt=8000,
    installment=280.0,
    annual_inc=45000,
    revol_util=30.0,
    total_rec_int=500.0,
    inq_last_6mths=0,
    term="60 months",
    purpose="credit_card",
    verification_status="Not Verified",
)


class TestArtifactLoading:
    def test_model_loads(self):
        model = _load_model()
        assert model is not None
        assert model.n_features_in_ == 25

    def test_features_list_loads(self):
        features = _load_features()
        assert isinstance(features, list)
        assert len(features) == 25

    def test_features_list_contains_expected_columns(self):
        features = _load_features()
        assert "loan_amnt" in features
        assert "term_36 months" in features
        assert "term_60 months" in features
        assert "verification_status_Not Verified" in features
        assert "verification_status_Verified" in features


class TestEncoding:
    def test_encode_single_record_shape(self):
        encoded = _encode_input(SAMPLE_RECORD)
        features = _load_features()
        assert encoded.shape == (1, 25)
        assert list(encoded.columns) == features

    def test_encode_term_36_months(self):
        encoded = _encode_input(SAMPLE_RECORD)
        assert encoded["term_36 months"].iloc[0] == 1
        assert encoded["term_60 months"].iloc[0] == 0

    def test_encode_term_60_months(self):
        encoded = _encode_input(SAMPLE_RECORD_2)
        assert encoded["term_36 months"].iloc[0] == 0
        assert encoded["term_60 months"].iloc[0] == 1

    def test_encode_verification_status_verified(self):
        encoded = _encode_input(SAMPLE_RECORD)
        assert encoded["verification_status_Verified"].iloc[0] == 1
        assert encoded["verification_status_Not Verified"].iloc[0] == 0

    def test_encode_verification_status_not_verified(self):
        encoded = _encode_input(SAMPLE_RECORD_2)
        assert encoded["verification_status_Not Verified"].iloc[0] == 1
        assert encoded["verification_status_Verified"].iloc[0] == 0

    def test_encode_purpose_debt_consolidation(self):
        encoded = _encode_input(SAMPLE_RECORD)
        assert encoded["purpose_debt_consolidation"].iloc[0] == 1

    def test_encode_numerical_passthrough(self):
        encoded = _encode_input(SAMPLE_RECORD)
        assert encoded["loan_amnt"].iloc[0] == 15000
        assert encoded["annual_inc"].iloc[0] == 72000


class TestPrediction:
    def test_single_prediction_returns_float(self):
        result = predict_single(SAMPLE_RECORD)
        assert isinstance(result, float)

    def test_single_prediction_in_plausible_range(self):
        result = predict_single(SAMPLE_RECORD)
        # Interest rates in the LendingClub dataset range from ~5% to ~30%
        assert 0.0 < result < 50.0, f"Predicted rate {result} outside plausible range"

    def test_batch_prediction_length(self):
        results = predict_batch([SAMPLE_RECORD, SAMPLE_RECORD_2])
        assert len(results) == 2

    def test_batch_prediction_single_item(self):
        results = predict_batch([SAMPLE_RECORD])
        assert len(results) == 1
        assert isinstance(results[0], float)

    def test_batch_matches_single(self):
        single = predict_single(SAMPLE_RECORD)
        batch = predict_batch([SAMPLE_RECORD])
        assert abs(single - batch[0]) < 1e-6

    def test_different_inputs_produce_different_rates(self):
        r1 = predict_single(SAMPLE_RECORD)
        r2 = predict_single(SAMPLE_RECORD_2)
        assert r1 != r2
