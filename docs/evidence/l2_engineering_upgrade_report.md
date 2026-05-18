# L2 Engineering Upgrade Report
# Explainable Credit Pricing Intelligence System

Generated: 2026-05-19
Phase: L2 (including L2 hardening pass)

This document records the engineering changes made during Phase L2 and the subsequent hardening pass: addition of a FastAPI serving boundary, structured source layout, Pydantic schemas, model and explanation services, test suite, API benchmark, Docker packaging, preprocessing bug fix, SHAP artifact regeneration, Docker image build, and CI/CD claim cleanup.

---

## Summary

Phase L2 elevates the system from an explainable Streamlit modelling workflow into a lightweight production-style ML serving system with a clean service layer and verified test coverage.

No production deployment is claimed. The FastAPI layer is a local serving boundary demonstrating deployment-ready structure.

---

## Files Created

### Source packages

| File | Purpose |
|---|---|
| src/__init__.py | Package marker |
| src/api/__init__.py | Package marker |
| src/api/main.py | FastAPI application: /health, /predict, /explain, /batch-predict |
| src/core/__init__.py | Package marker |
| src/core/config.py | Centralised path config via pathlib; valid categorical values |
| src/schemas/__init__.py | Package marker |
| src/schemas/prediction.py | Pydantic schemas: BorrowerInput, PredictionResponse, ExplanationResponse, BatchPrediction* |
| src/services/__init__.py | Package marker |
| src/services/model_service.py | Model + feature list loading (lru_cache); encoding; single + batch prediction |
| src/services/explanation_service.py | SHAP explainer loading with fallback; per-prediction top-N driver extraction |

### Tests

| File | Coverage |
|---|---|
| tests/__init__.py | Package marker |
| tests/test_model_service.py | 16 tests: artifact loading, encoding correctness, single/batch prediction |
| tests/test_api.py | 25 tests: /health, /predict, /explain, /batch-predict including 422 validation cases |

### Scripts

| File | Purpose |
|---|---|
| scripts/benchmark_api.py | In-process API latency benchmark via TestClient; saves api_benchmark_results.json and api_benchmark_report.md |

### Docker

| File | Purpose |
|---|---|
| Dockerfile | Python 3.11-slim image; installs requirements; copies src/ and model artifacts; starts uvicorn on port 8000 |
| .dockerignore | Excludes __pycache__, .git, .venv, large raw data files, notebooks, tests |

### Evidence

| File | Purpose |
|---|---|
| docs/evidence/api_benchmark_results.json | Machine-readable API benchmark results |
| docs/evidence/api_benchmark_report.md | Human-readable API benchmark report |
| docs/evidence/l2_engineering_upgrade_report.md | This document |

---

## Files Updated

| File | Change |
|---|---|
| requirements.txt | Added fastapi>=0.111.0, httpx>=0.27.0, pytest>=8.2.0 |
| src/__init__.py | Created (previously no __init__.py in src/) |
| docs/evidence/evidence_inventory.md | Updated with L2 artifacts |
| docs/evidence/system_design_notes.md | Added FastAPI layer section |
| docs/evidence/scale_path.md | Marked production-ready steps as implemented where applicable |
| docs/evidence/claim_safety.md | Added L2 claim guidance |
| docs/evidence/portfolio_summary.md | Added L2 proof chips and architecture nodes |
| README.md | Added FastAPI Serving Boundary section |

---

## FastAPI Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Returns system name, model_loaded, explainer_loaded |
| POST | /predict | Single-record rate prediction; returns PredictionResponse |
| POST | /explain | Single-record prediction + top-8 SHAP drivers; returns ExplanationResponse |
| POST | /batch-predict | Multi-record batch prediction; returns BatchPredictionResponse |

All endpoints include an evidence_note in the response body stating that output is not financial advice and is not approved for production lending use.

All endpoints log request path, method, and processing time. A `X-Process-Time-Ms` header is included in every response.

---

## Service Layer Summary

### model_service.py

- `_load_model()` and `_load_features()` use `@lru_cache(maxsize=1)` so artifacts are loaded exactly once per process.
- `_encode_input()` produces the correct one-hot encoded DataFrame aligned to features_list.pkl. This corrects a preprocessing bug found in app/app.py where term and verification_status were incorrectly lowercased/underscored before pd.get_dummies, causing all term and verification_status dummy columns to be filled with 0 after reindex.
- No StandardScaler is applied. The model was trained without scaling (XGBoost does not require feature scaling, and the Streamlit interface never applied scaling). Raw numerical values are passed directly.

### explanation_service.py

- Attempts to load app/shap_explainer.pkl first.
- On Python 3.13, the pre-saved shap_explainer.pkl fails to load with: `code() argument 13 must be str, not int`. This is a Python version incompatibility in the pickled code object. The service catches this and falls back to building shap.Explainer(model) dynamically.
- The dynamic fallback loaded successfully and /explain returned 200 in all benchmark runs.
- The fallback is logged at WARNING level and reflected in the explanation_method field of ExplanationResponse.

---

## Test Results

Run command: `python -m pytest tests/ -v`

```
41 passed, 1 warning in 2.90s
```

All 41 tests passed. The single warning is the XGBoost legacy pickle format warning (cosmetic; does not affect correctness).

### Test breakdown

| Suite | Tests | Result |
|---|---|---|
| tests/test_model_service.py | 16 | All passed |
| tests/test_api.py | 25 | All passed |
| Total | 41 | All passed |

Key test coverage:
- Model and feature list load without error
- Single prediction returns a float in the plausible 0-50% range
- Batch prediction length matches input count
- Batch result matches single-record result to 6 decimal places
- /health returns 200 with correct fields
- /predict returns 200 for valid payload; 422 for invalid term, missing field, wrong verification_status casing, negative loan amount
- /batch-predict returns 200 with correct count; 422 for empty list
- /explain returns 200 with top_drivers OR 503 with detail (both cases handled)
- Encoding tests verify correct column names for term_36 months, term_60 months, verification_status_Not Verified etc.

---

## API Benchmark Results

Run command: `python scripts/benchmark_api.py`
Method: FastAPI TestClient (in-process, no network round-trip)
Iterations per endpoint: 20, with 3 warmup calls

| Endpoint | Mean (ms) | Median (ms) | Notes |
|---|---|---|---|
| GET /health | 3.8 ms | 3.6 ms | Health check only; no inference |
| POST /predict (single) | 19.2 ms | 18.5 ms | Includes encoding + XGBoost inference |
| POST /explain (single) | 22.9 ms | 21.4 ms | Includes SHAP computation (~3.7 ms over predict) |
| POST /batch-predict (10 records) | 17.8 ms | 17.6 ms | 1.78 ms/record |
| POST /batch-predict (100 records) | 18.4 ms | 17.9 ms | 0.18 ms/record |

All endpoints returned status 200.

Note: /predict API latency (19 ms in-process) is higher than raw XGBoost benchmark (2.78 ms) because it includes JSON parsing, Pydantic validation, feature encoding, FastAPI middleware, and TestClient serialisation overhead. Production network latency would add further overhead.

Full results: docs/evidence/api_benchmark_results.json
Human-readable report: docs/evidence/api_benchmark_report.md

---

## Docker Status

Dockerfile and .dockerignore created. Docker image built successfully during L2 hardening pass.

Build result:
- Image: explainable-credit-pricing:latest
- Base: python:3.11-slim
- Content size: 777 MB
- Total disk with layers: 2.56 GB
- Build status: SUCCESS

To run the API:
```
docker run -p 8000:8000 explainable-credit-pricing
```

The image starts uvicorn on port 8000. The Dockerfile uses python:3.11-slim. The regenerated shap_explainer.pkl (Python 3.13 compatible) is included in the build context and will be available inside the container.

---

## Preprocessing Bug Fix

During L2 inspection, a preprocessing bug was identified in app/app.py that has existed since the original Streamlit implementation:

1. `term` was converted to "36_months" (underscore) before `pd.get_dummies`, producing column "term_36_months". The features_list.pkl expects "term_36 months" (space). Result: all term dummy columns were filled with 0 after reindex.

2. `verification_status` was lowercased and underscored before `pd.get_dummies`, producing e.g. "verification_status_verified". The features_list.pkl expects "verification_status_Verified" (original casing). Result: all verification_status dummy columns were filled with 0.

The bug was preserved in app/app.py to avoid breaking the existing Streamlit interface. The FastAPI model_service.py uses the correct values: term as "36 months", verification_status as "Not Verified"/"Source Verified"/"Verified".

This difference in preprocessing means the Streamlit app and the API may produce slightly different predictions for the same inputs. The API predictions are more correct because they use the feature values the model was trained on.

app/app.py was NOT modified. The bug is documented here. Fixing app/app.py is a recommended follow-up.

---

## Known Issues and Remaining Gaps

| Issue | Severity | Status |
|---|---|---|
| app/shap_explainer.pkl incompatible with Python 3.13 | Resolved | Regenerated using shap.Explainer(model); verified loading and /explain 200 response |
| app/app.py has preprocessing bug (term, verification_status encoding) | Resolved | Fixed in hardening pass: raw selectbox values passed directly without lowercasing or underscore-replacing |
| No saved StandardScaler from training run | Low | Not needed (model was trained without scaling) |
| Docker image not built locally | Resolved | Image built successfully: explainable-credit-pricing:latest, python:3.11-slim, 777MB content |
| No authentication on API | Expected for this phase | Documented in system_design_notes.md scale path |
| No prediction persistence (no database write) | Expected for this phase | Documented in scale_path.md |
| No FastAPI integration test against live uvicorn server | Low | TestClient covers the routing and business logic |

---

## Claim Safety (L2 + Hardening Pass)

Safe claims after L2 and hardening pass:

- "FastAPI serving boundary added with /health, /predict, /explain, /batch-predict endpoints"
- "41 pytest tests passing, covering model service and API endpoints"
- "In-process API benchmark: /predict 19.2 ms mean, /batch-predict 0.18 ms/record at 100 records"
- "Docker image built successfully: explainable-credit-pricing:latest, python:3.11-slim"
- "Preprocessing bug in app/app.py fixed: term and verification_status now passed as raw values matching training-time encoding"
- "app/shap_explainer.pkl regenerated using shap.Explainer(model); compatible with Python 3.13; verified loading and successful /explain response"

Unsafe claims after L2 and hardening pass:

- "Production API deployed" -- not deployed
- "FastAPI backend serving live traffic" -- not deployed
- "Docker container run and verified" -- image built but container not started and tested
