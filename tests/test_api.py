"""
Tests for src/api/main.py endpoints.

Verifies:
- /health returns status and artifact flags
- /predict returns 200 with a valid rate for a good payload
- /predict returns 422 for a malformed payload
- /batch-predict returns 200 with correct count
- /explain returns 200 with top_drivers list or a 503 with clear error

Uses FastAPI TestClient (backed by httpx).
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

GOOD_PAYLOAD = {
    "loan_amnt": 15000,
    "installment": 450.5,
    "annual_inc": 72000,
    "revol_util": 55.0,
    "total_rec_int": 1200.0,
    "inq_last_6mths": 2,
    "term": "36 months",
    "purpose": "debt_consolidation",
    "verification_status": "Verified",
}

GOOD_PAYLOAD_2 = {
    "loan_amnt": 8000,
    "installment": 280.0,
    "annual_inc": 45000,
    "revol_util": 30.0,
    "total_rec_int": 500.0,
    "inq_last_6mths": 0,
    "term": "60 months",
    "purpose": "credit_card",
    "verification_status": "Not Verified",
}


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_model_loaded(self):
        data = client.get("/health").json()
        assert data["model_loaded"] is True

    def test_health_system_name(self):
        data = client.get("/health").json()
        assert "Credit Pricing" in data["system"]

    def test_health_has_explainer_flag(self):
        data = client.get("/health").json()
        assert "explainer_loaded" in data


class TestPredict:
    def test_predict_200_for_valid_payload(self):
        resp = client.post("/predict", json=GOOD_PAYLOAD)
        assert resp.status_code == 200

    def test_predict_returns_rate(self):
        data = client.post("/predict", json=GOOD_PAYLOAD).json()
        assert "predicted_interest_rate" in data
        assert isinstance(data["predicted_interest_rate"], float)

    def test_predict_rate_plausible_range(self):
        data = client.post("/predict", json=GOOD_PAYLOAD).json()
        rate = data["predicted_interest_rate"]
        assert 0.0 < rate < 50.0, f"Rate {rate} outside expected range"

    def test_predict_contains_evidence_note(self):
        data = client.post("/predict", json=GOOD_PAYLOAD).json()
        assert "evidence_note" in data
        assert len(data["evidence_note"]) > 0

    def test_predict_422_for_invalid_term(self):
        bad = {**GOOD_PAYLOAD, "term": "24 months"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_422_for_missing_field(self):
        bad = {k: v for k, v in GOOD_PAYLOAD.items() if k != "loan_amnt"}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_422_for_invalid_verification_status(self):
        bad = {**GOOD_PAYLOAD, "verification_status": "verified"}  # wrong casing
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_predict_422_for_negative_loan_amount(self):
        bad = {**GOOD_PAYLOAD, "loan_amnt": -1000}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422

    def test_process_time_header_present(self):
        resp = client.post("/predict", json=GOOD_PAYLOAD)
        assert "X-Process-Time-Ms" in resp.headers


class TestBatchPredict:
    def test_batch_200_for_two_records(self):
        resp = client.post("/batch-predict", json={"records": [GOOD_PAYLOAD, GOOD_PAYLOAD_2]})
        assert resp.status_code == 200

    def test_batch_correct_count(self):
        data = client.post(
            "/batch-predict", json={"records": [GOOD_PAYLOAD, GOOD_PAYLOAD_2]}
        ).json()
        assert data["record_count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_single_record(self):
        data = client.post("/batch-predict", json={"records": [GOOD_PAYLOAD]}).json()
        assert data["record_count"] == 1
        assert len(data["predictions"]) == 1

    def test_batch_predictions_are_floats(self):
        data = client.post(
            "/batch-predict", json={"records": [GOOD_PAYLOAD, GOOD_PAYLOAD_2]}
        ).json()
        for rate in data["predictions"]:
            assert isinstance(rate, float)

    def test_batch_matches_single_predict(self):
        single = client.post("/predict", json=GOOD_PAYLOAD).json()["predicted_interest_rate"]
        batch = client.post("/batch-predict", json={"records": [GOOD_PAYLOAD]}).json()["predictions"][0]
        assert abs(single - batch) < 1e-4

    def test_batch_422_for_empty_list(self):
        resp = client.post("/batch-predict", json={"records": []})
        assert resp.status_code == 422


class TestExplain:
    def test_explain_returns_200_or_503(self):
        resp = client.post("/explain", json=GOOD_PAYLOAD)
        # 200 = SHAP working; 503 = documented unavailability
        assert resp.status_code in (200, 503)

    def test_explain_200_has_top_drivers(self):
        resp = client.post("/explain", json=GOOD_PAYLOAD)
        if resp.status_code == 200:
            data = resp.json()
            assert "top_drivers" in data
            assert len(data["top_drivers"]) > 0

    def test_explain_200_drivers_have_required_fields(self):
        resp = client.post("/explain", json=GOOD_PAYLOAD)
        if resp.status_code == 200:
            for driver in resp.json()["top_drivers"]:
                assert "feature" in driver
                assert "shap_value" in driver
                assert "direction" in driver

    def test_explain_200_rate_plausible(self):
        resp = client.post("/explain", json=GOOD_PAYLOAD)
        if resp.status_code == 200:
            rate = resp.json()["predicted_interest_rate"]
            assert 0.0 < rate < 50.0

    def test_explain_503_has_detail(self):
        resp = client.post("/explain", json=GOOD_PAYLOAD)
        if resp.status_code == 503:
            assert "detail" in resp.json()
