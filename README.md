# Explainable Credit Pricing Intelligence System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-FF7043?logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Attribution-00C49A)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving_Boundary-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Review_Interface-ff4b4b?logo=streamlit&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-41_passing-brightgreen?logo=pytest&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Image_Built-2496ED?logo=docker&logoColor=white)

An explainable credit pricing workflow that connects LendingClub borrower records, XGBoost interest-rate prediction, SHAP attribution, a FastAPI serving boundary, local inference benchmarks, and a Streamlit review surface into a repeatable, evidence-backed decision-support system.

---

## Business Problem

Interest-rate model outputs are difficult to trust when prediction, feature drivers, evaluation evidence, and reviewer context are separated across notebooks or static reports.

A pricing reviewer who receives a model output without attribution cannot explain it, audit it, or defend it. A model score alone is not a decision. The question is not just "what rate did the model predict?" but "which borrower features drove that rate, by how much, and is the explanation fast enough to be useful at review time?"

---

## Decision Supported

This system supports borrower-level interest rate review by surfacing:

- The predicted interest rate for a given set of borrower inputs
- The top SHAP feature drivers that pushed the rate above or below the model baseline
- Visual attribution evidence (waterfall, bar, beeswarm) attached at the point of prediction
- A structured API boundary that separates model serving from the review interface
- Measured inference and explanation latency confirming the system is fast enough for interactive use

The target user is a reviewer, analyst, or data scientist who needs to understand and defend a credit pricing output, not just receive a number.

---

## System Architecture

```
LendingClub Borrower Records (887,379 raw rows)
    |
    v
Feature Engineering
  Cleaning, imputation, outlier detection
  One-hot encoding: term, purpose, verification_status
  sub_grade excluded (encodes rate tier directly)
  Output: 757,494 processed rows, 25 features
    |
    v
XGBoost Regression
  Training rows: 605,995
  Holdout rows: 151,499
  Test RMSE: 0.98 pp | Test R2: 0.95
    |
    v
FastAPI Serving Boundary (src/api/main.py)
  POST /predict     -- single-record rate prediction
  POST /explain     -- prediction + SHAP attribution
  POST /batch-predict -- batch scoring
  GET  /health      -- artifact load status
    |
    v
SHAP TreeExplainer (src/services/explanation_service.py)
  Per-prediction feature attribution
  Beeswarm, waterfall, and bar plot types
  0.19 ms/record at batch size 100 (local benchmark)
    |
    v
Streamlit Review Interface (app/app.py)
  Single-record interactive review
  SHAP waterfall and bar plots inline
  Batch CSV upload and download
    |
    v
Evidence Artifacts (docs/evidence/)
  benchmark_results.json, api_benchmark_results.json
  model_metrics.json, shap plots
  Reproducible scripts for all measurements
```

---

## Evidence Summary

| Dimension | Value | Source |
|---|---|---|
| Raw dataset | 887,379 LendingClub borrower records | data/loan_data.csv |
| Training records | 605,995 | Notebook output |
| Holdout records | 151,499 | Notebook output |
| Test RMSE | 0.98 percentage points | Notebook output |
| Test R2 | 0.95 | Notebook output |
| Test MAE | ~0.93 pp (2K proxy) | compute_evidence.py |
| Features | 25 after encoding (sub_grade excluded) | app/features_list.pkl |
| Model type | XGBoost regression | app/model.pkl |
| Model families compared | 5 (LR, DT, RF, XGBoost, FNN) | Notebooks |
| SHAP plots | Summary, waterfall, bar | docs/evidence/ |
| FastAPI serving boundary | Implemented: /health, /predict, /explain, /batch-predict | src/api/main.py |
| Tests | 41 passing (pytest) | tests/ |
| Streamlit interface | Implemented | app/app.py |
| Docker packaging | Image built: explainable-credit-pricing:latest (python:3.11-slim) | Dockerfile |

---

## Computational Efficiency

All values are from local benchmark runs on developer hardware (Windows 11, Python 3.13).
Production latency would depend on hosting environment, network, and feature pipeline.

### Model Prediction Latency (raw XGBoost .predict(), scripts/benchmark_inference.py)

| Batch Size | Wall Time | Per Record |
|---|---|---|
| 1 row | 2.78 ms | 2,780 us |
| 10 rows | 2.51 ms | 251 us |
| 100 rows | 2.80 ms | 28.0 us |
| 1,000 rows | Not measured (only 400 test rows in demo sample) | -- |

### API Endpoint Latency (in-process TestClient, scripts/benchmark_api.py)

| Endpoint | Mean (ms) | Per Record |
|---|---|---|
| GET /health | 3.8 ms | -- |
| POST /predict (single) | 19.2 ms | 19.2 ms |
| POST /explain (single) | 22.9 ms | 22.9 ms |
| POST /batch-predict (10 records) | 17.8 ms | 1.78 ms |
| POST /batch-predict (100 records) | 18.4 ms | 0.18 ms |

Note: API endpoint latency includes JSON parsing, Pydantic validation, feature encoding, FastAPI middleware, and TestClient serialisation overhead. Raw XGBoost inference is 2-3 ms of the total.

### SHAP Attribution Latency (TreeExplainer, scripts/benchmark_inference.py)

| Batch Size | Total Time | Per Record |
|---|---|---|
| 1 row | 4.9 ms | 4.9 ms |
| 10 rows | 13.9 ms | 1.4 ms |
| 100 rows | 19.2 ms | 0.19 ms |

### Model Load

| Artifact | Cold Load Time |
|---|---|
| app/model.pkl | 2,405 ms (includes XGBoost legacy pickle overhead) |

Full inference benchmark: [docs/evidence/benchmark_results.json](docs/evidence/benchmark_results.json)
Full API benchmark: [docs/evidence/api_benchmark_results.json](docs/evidence/api_benchmark_results.json)

---

## FastAPI Serving Boundary

A local model-serving layer was added in Phase L2. This is not a production lending system.
Outputs are for review and demonstration purposes only. Not financial advice.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | System name, model_loaded, explainer_loaded |
| POST | /predict | Single-record interest rate prediction |
| POST | /explain | Single-record prediction + top-8 SHAP drivers |
| POST | /batch-predict | Batch prediction for list of borrower records |

All responses include an `evidence_note` field and an `X-Process-Time-Ms` response header.

### Start the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs available at http://127.0.0.1:8000/docs

### Sample request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 15000,
    "installment": 450.5,
    "annual_inc": 72000,
    "revol_util": 55.0,
    "total_rec_int": 1200.0,
    "inq_last_6mths": 2,
    "term": "36 months",
    "purpose": "debt_consolidation",
    "verification_status": "Verified"
  }'
```

### Run the test suite

```bash
python -m pytest tests/ -v
```

41 tests covering model service encoding, prediction, and all API endpoints including 422 validation.

### Run the API benchmark

```bash
python scripts/benchmark_api.py
```

Saves results to docs/evidence/api_benchmark_results.json and docs/evidence/api_benchmark_report.md.

### Docker

```bash
docker build -t explainable-credit-pricing .
docker run -p 8000:8000 explainable-credit-pricing
```

Docker image was built successfully during the L2 hardening pass: explainable-credit-pricing:latest, python:3.11-slim, 777MB content. The container has not been run and tested in a live environment.

---

## System Design Efficiency

The system separates data preparation, training, explanation, serving, interface, and evidence into distinct layers.

| Layer | Artifact | Role |
|---|---|---|
| Data preparation | notebooks/, src/ | Cleaning, imputation, encoding |
| Training | notebooks/ML_XAI_Engineered_Data.ipynb | XGBoost training, evaluation |
| Config | src/core/config.py | Centralised paths, valid categorical values |
| Schemas | src/schemas/prediction.py | Pydantic request/response contracts |
| Model service | src/services/model_service.py | Load once (lru_cache), encode, predict |
| Explanation service | src/services/explanation_service.py | SHAP load with fallback, top-N drivers |
| API | src/api/main.py | FastAPI routes with timing middleware |
| Model artifact | app/model.pkl | Single source of truth for trained model |
| Feature schema | app/features_list.pkl | Schema contract across all prediction paths |
| Review interface | app/app.py | Streamlit UI: prediction + SHAP + context visuals |
| Evidence pipeline | docs/evidence/compute_evidence.py | Reproducible metrics and SHAP generation |
| Benchmark pipeline | scripts/benchmark_inference.py | Raw inference latency measurement |
| API benchmark | scripts/benchmark_api.py | API endpoint latency measurement |

The features_list.pkl acts as a schema contract: every prediction path (Streamlit, FastAPI service, benchmark scripts) loads and aligns to this same ordered feature list.

---

## Scale Path

Step 1 (FastAPI serving boundary) is now implemented locally.
Remaining designed steps are not claimed as implemented.

1. ~~FastAPI serving endpoint~~ -- **Implemented** (src/api/main.py)
2. **Prediction persistence** -- PostgreSQL log of every prediction with input hash, output, model version
3. **Batch scoring worker** -- Celery background worker for bulk pricing jobs
4. **SHAP caching** -- Redis cache on input hash to eliminate redundant explanation computation
5. **Model registry** -- MLflow versioned model artifacts with staging/production promotion
6. **Drift monitoring** -- Feature and prediction distribution monitoring (Evidently or equivalent)
7. **CI/CD pipeline** -- GitHub Actions: test, benchmark, Docker build, deploy to staging on merge
8. **Access control and audit logs** -- JWT auth, role-based permissions, full request audit trail

Full detail at [docs/evidence/scale_path.md](docs/evidence/scale_path.md).

---

## Limitations

- This is an explainable modelling workflow, not a production lending system.
- Predictions are not financial advice and have not been reviewed or approved by any institution.
- No customers, users, revenue, or cost savings are associated with this system.
- All latency values are from local benchmark runs. Production latency is not measured.
- MAE 0.93 pp is from a 2,000-row proxy sample. The full training run did not record MAE.
- The model was saved in legacy XGBoost pickle format. Re-saving in JSON format would improve portability.
- A preprocessing bug in app/app.py (term and verification_status lowercased and underscored before one-hot encoding) was identified and fixed in the L2 hardening pass. Both Streamlit and FastAPI paths now use correct training-time values.
- app/shap_explainer.pkl was regenerated in the L2 hardening pass using shap.Explainer(model); it is now Python 3.13 compatible.
- The FastAPI serving boundary does not persist predictions and does not include authentication.
- Docker image was built locally (explainable-credit-pricing:latest, python:3.11-slim). The container has not been run and tested in a live environment.
- The docs/ folder is currently untracked in git. GitHub evidence claims apply only after docs/ is committed and pushed.

---

## How to Run

### Prerequisites

```bash
git clone https://github.com/IjazKakkodDS/explainable-loan-interest-prediction.git
cd explainable-loan-interest-prediction
pip install -r requirements.txt
```

### Run the FastAPI Serving Boundary

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs: http://127.0.0.1:8000/docs

### Run the Streamlit Review Interface

```bash
streamlit run app/app.py
```

Opens at http://localhost:8501.

### Run Tests

```bash
python -m pytest tests/ -v
```

### Run the Inference Benchmark

```bash
python scripts/benchmark_inference.py
```

### Run the API Benchmark

```bash
python scripts/benchmark_api.py
```

### Regenerate SHAP Evidence Plots

```bash
python scripts/generate_explainability_evidence.py
```

### Regenerate Full Evidence Set

```bash
python docs/evidence/compute_evidence.py
```

---

## Evidence Files

| File | Contents |
|---|---|
| [docs/evidence/benchmark_results.json](docs/evidence/benchmark_results.json) | Raw inference latency at 1/10/100 rows |
| [docs/evidence/benchmark_report.md](docs/evidence/benchmark_report.md) | Human-readable inference benchmark |
| [docs/evidence/api_benchmark_results.json](docs/evidence/api_benchmark_results.json) | API endpoint latency (in-process) |
| [docs/evidence/api_benchmark_report.md](docs/evidence/api_benchmark_report.md) | Human-readable API benchmark |
| [docs/evidence/model_metrics.json](docs/evidence/model_metrics.json) | RMSE, MAE, R2, SHAP latency |
| [docs/evidence/model_metrics.md](docs/evidence/model_metrics.md) | Readable metrics with dataset chain |
| [docs/evidence/shap_summary.png](docs/evidence/shap_summary.png) | SHAP beeswarm (100 test samples) |
| [docs/evidence/shap_waterfall.png](docs/evidence/shap_waterfall.png) | SHAP waterfall (single prediction) |
| [docs/evidence/shap_bar.png](docs/evidence/shap_bar.png) | SHAP mean absolute feature impact |
| [docs/evidence/system_design_notes.md](docs/evidence/system_design_notes.md) | Full architecture and efficiency notes |
| [docs/evidence/scale_path.md](docs/evidence/scale_path.md) | Production extension path |
| [docs/evidence/claim_safety.md](docs/evidence/claim_safety.md) | Safe vs unsafe claims reference |
| [docs/evidence/portfolio_summary.md](docs/evidence/portfolio_summary.md) | Portfolio-ready copy |
| [docs/evidence/evidence_inventory.md](docs/evidence/evidence_inventory.md) | Complete artifact inventory |
| [docs/evidence/l2_engineering_upgrade_report.md](docs/evidence/l2_engineering_upgrade_report.md) | L2 upgrade: files added, tests, benchmarks, findings |

---

## Repository Structure

```
explainable-loan-interest-prediction/
├── app/                         # Streamlit interface + model artifacts
│   ├── app.py
│   ├── model.pkl
│   ├── features_list.pkl
│   └── shap_explainer.pkl
├── data/                        # LendingClub datasets
├── docs/evidence/               # All measured evidence artifacts and documentation
├── models/                      # Additional model artifacts
├── notebooks/                   # EDA, preprocessing, and training notebooks
├── scripts/                     # Reproducible benchmark and evidence scripts
│   ├── benchmark_inference.py
│   ├── benchmark_api.py
│   └── generate_explainability_evidence.py
├── src/                         # Production-style source package
│   ├── api/
│   │   └── main.py              # FastAPI app
│   ├── core/
│   │   └── config.py            # Centralised paths and constants
│   ├── schemas/
│   │   └── prediction.py        # Pydantic request/response models
│   └── services/
│       ├── model_service.py     # Model loading and prediction
│       └── explanation_service.py  # SHAP attribution
├── tests/
│   ├── test_model_service.py    # 16 model service tests
│   └── test_api.py              # 25 API endpoint tests
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

---

## Contact

- **Email:** ijazkakkod@gmail.com
- **LinkedIn:** [linkedin.com/in/ijazkakkod](https://linkedin.com/in/ijazkakkod)
- **GitHub:** [github.com/IjazKakkodDS](https://github.com/IjazKakkodDS)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
