# Explainable Credit Pricing Intelligence System

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regression-FF7043?logo=xgboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Attribution-00C49A)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving_Boundary-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Review_Interface-ff4b4b?logo=streamlit&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-41_passing-brightgreen?logo=pytest&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LassoCV-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Image_Built-2496ED?logo=docker&logoColor=white)

An explainable credit pricing workflow connecting LendingClub borrower records, XGBoost interest-rate prediction, SHAP attribution, a FastAPI serving boundary, measured latency benchmarks, and a Streamlit review interface -- built as a reproducible, evidence-backed decision-support system.

---

## Quantitative Snapshot

| Dimension | Value | Source |
|---|---|---|
| Raw dataset | 887,379 LendingClub borrower records | data/loan_data.csv |
| Processed training records | 757,494 | Feature engineering pipeline |
| Training split | 605,995 rows | 80/20 split |
| Holdout split | 151,499 rows | 80/20 split |
| Test RMSE | 0.9795 percentage points | Notebook output |
| Test R2 | 0.9506 | Notebook output |
| Test MAE | ~0.93 pp | compute_evidence.py (2K proxy) |
| Encoded features | 25 (sub_grade excluded) | app/features_list.pkl |
| SHAP latency | 0.19 ms/record at batch 100 | benchmark_results.json |
| API /predict latency | 19.2 ms mean (in-process) | api_benchmark_results.json |
| pytest coverage | 41/41 passing | tests/ |
| Docker image | explainable-credit-pricing:latest (python:3.11-slim) | Dockerfile |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Model | XGBoost regression |
| Explainability | SHAP TreeExplainer |
| Serving | FastAPI + Pydantic + uvicorn |
| Review interface | Streamlit |
| Testing | pytest + HTTPX TestClient |
| Packaging | Docker (python:3.11-slim) |
| Feature selection | LassoCV + XGBoost importance (training phase) |

---

## Business Problem

Interest-rate model outputs are difficult to trust when prediction, feature drivers, evaluation evidence, and reviewer context are separated across notebooks or static reports.

A pricing reviewer who receives a model output without attribution cannot explain it, audit it, or defend it. A model score alone is not a decision. The question is not just "what rate did the model predict?" but "which borrower features drove that rate, by how much, and is the explanation fast enough to be useful at review time?"

---

## System Objective

This system supports borrower-level interest rate review by surfacing:

- The predicted interest rate for a given set of borrower inputs
- The top SHAP feature drivers that pushed the rate above or below the model baseline
- Visual attribution evidence (waterfall, bar, beeswarm) attached at the point of prediction
- A structured API boundary that separates model serving from the review interface
- Measured inference and explanation latency confirming the system is fast enough for interactive use

The intended user is a reviewer, analyst, or data scientist who needs to understand and defend a credit pricing output, not just receive a number.

---

## Architecture

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
  Holdout rows:  151,499
  Test RMSE: 0.9795 pp | Test R2: 0.9506
    |
    v
FastAPI Serving Boundary (src/api/main.py)
  POST /predict      -- single-record rate prediction
  POST /explain      -- prediction + SHAP attribution
  POST /batch-predict -- batch scoring
  GET  /health       -- artifact load status
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
  model_metrics.json, SHAP plots
  Reproducible scripts for all measurements
```

---

## Model Evidence

XGBoost regression trained on 605,995 LendingClub records. Five model families were evaluated (Linear Regression, Decision Tree, Random Forest, XGBoost, FNN) before selecting XGBoost based on holdout performance. Feature selection used LassoCV and XGBoost importance; sub_grade was excluded because it directly encodes the interest rate tier and would constitute target leakage.

| Metric | Value | Evaluation set |
|---|---|---|
| Test RMSE | 0.9795 percentage points | 151,499 holdout records |
| Test R2 | 0.9506 | 151,499 holdout records |
| Test MAE | ~0.93 pp | 2K proxy sample |
| Training records | 605,995 | 80% of 757,494 |
| Encoded features | 25 | app/features_list.pkl |
| Model families compared | 5 | Training notebooks |

Full metrics: [docs/evidence/model_metrics.md](docs/evidence/model_metrics.md)

---

## Explainability Evidence

SHAP TreeExplainer is applied per-prediction. Three plot types are generated: beeswarm (global importance across 100 test samples), waterfall (single-prediction attribution), and bar (mean absolute SHAP impact across 100 samples).

Top features by mean absolute SHAP value:

1. loan_amnt -- dominant positive driver for higher rates
2. installment -- strong negative driver (collinear with loan_amnt; counterbalances on short terms)
3. term_36 months -- negative (36-month loans predict lower rates than 60-month)
4. total_rec_int -- positive driver
5. annual_inc -- negative (higher income predicts lower rate)
6. revol_util -- positive (higher utilisation predicts higher rate)

SHAP attribution latency (local benchmark, TreeExplainer):

| Batch Size | Total Time | Per Record |
|---|---|---|
| 1 row | 4.9 ms | 4.9 ms |
| 10 rows | 13.9 ms | 1.4 ms |
| 100 rows | 19.2 ms | 0.19 ms |

Evidence plots:
[shap_summary.png](docs/evidence/shap_summary.png) |
[shap_waterfall.png](docs/evidence/shap_waterfall.png) |
[shap_bar.png](docs/evidence/shap_bar.png)

---

## Service Boundary

The serving boundary separates model inference from the review interface, exposing prediction and attribution as structured API calls. Input validation is enforced via Pydantic schemas; invalid categorical values return HTTP 422 without reaching the model. Every response includes a processing-time header for latency observability.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | System name, model_loaded, explainer_loaded |
| POST | /predict | Single-record interest rate prediction |
| POST | /explain | Single-record prediction + top-8 SHAP drivers |
| POST | /batch-predict | Batch prediction for a list of borrower records |

All responses include an `evidence_note` field and an `X-Process-Time-Ms` response header.

### Start the API

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs: http://127.0.0.1:8000/docs

### Sample Request

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

---

## Benchmark Evidence

All values are from local benchmark runs on developer hardware (Windows 11, Python 3.13). Production latency would depend on hosting environment, network topology, and feature pipeline design.

### Model Prediction Latency (raw XGBoost .predict(), scripts/benchmark_inference.py)

| Batch Size | Wall Time | Per Record |
|---|---|---|
| 1 row | 2.78 ms | 2,780 us |
| 10 rows | 2.51 ms | 251 us |
| 100 rows | 2.80 ms | 28.0 us |

### API Endpoint Latency (in-process TestClient, scripts/benchmark_api.py)

| Endpoint | Mean (ms) | Per Record |
|---|---|---|
| GET /health | 3.8 ms | -- |
| POST /predict (single) | 19.2 ms | 19.2 ms |
| POST /explain (single) | 22.9 ms | 22.9 ms |
| POST /batch-predict (10 records) | 17.8 ms | 1.78 ms |
| POST /batch-predict (100 records) | 18.4 ms | 0.18 ms |

API latency includes JSON parsing, Pydantic validation, feature encoding, FastAPI middleware, and TestClient serialisation overhead. Raw XGBoost inference is 2-3 ms of the total.

### Model Cold Load

| Artifact | Cold Load Time |
|---|---|
| app/model.pkl | 2,405 ms (includes XGBoost legacy pickle overhead) |

Full inference benchmark: [docs/evidence/benchmark_results.json](docs/evidence/benchmark_results.json)
Full API benchmark: [docs/evidence/api_benchmark_results.json](docs/evidence/api_benchmark_results.json)

---

## Engineering Decisions

The system separates data preparation, training, explanation, serving, interface, and evidence into distinct layers. Each layer has a single responsibility and a clear artifact boundary.

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
| Inference benchmark | scripts/benchmark_inference.py | Raw inference latency measurement |
| API benchmark | scripts/benchmark_api.py | API endpoint latency measurement |

`features_list.pkl` acts as a schema contract: every prediction path (Streamlit, FastAPI service, benchmark scripts) loads and aligns to the same ordered feature list. This eliminates train/serve column mismatches at the encoding boundary.

---

## Testing Evidence

The test suite covers model service encoding, prediction correctness, and all API endpoints including boundary and validation cases.

| Suite | Tests | Coverage |
|---|---|---|
| tests/test_model_service.py | 16 | Artifact loading, encoding correctness (term and verification_status columns), single and batch prediction |
| tests/test_api.py | 25 | /health, /predict (including 422 for invalid term, wrong verification casing, negative loan_amnt), /batch-predict, /explain |
| Total | 41/41 passing | -- |

```bash
python -m pytest tests/ -v
```

---

## Docker and Packaging

```bash
docker build -t explainable-credit-pricing .
docker run -p 8000:8000 explainable-credit-pricing
```

Image: `explainable-credit-pricing:latest` | Base: `python:3.11-slim` | Content: 777 MB

The Dockerfile copies model artifacts and the `src/` package, then starts uvicorn on port 8000. The image was built locally and verified to build cleanly. It has not been deployed to a container registry or cloud runtime.

---

## Engineering Roadmap

The FastAPI serving boundary (Step 1) is implemented locally. Remaining steps are designed extension paths, not claimed as implemented.

1. FastAPI serving boundary -- Implemented locally in src/api/main.py
2. Prediction persistence -- Future hardening step: PostgreSQL log of each prediction with input hash, output, timestamp, and model version
3. Batch scoring worker -- Future hardening step: background worker for bulk pricing jobs
4. SHAP caching -- Future hardening step: Redis cache on input hash to reduce repeated explanation computation
5. Model registry -- Future hardening step: MLflow or equivalent versioned model artifacts with staging and production promotion
6. Drift monitoring -- Future hardening step: feature and prediction distribution monitoring (Evidently or equivalent)
7. CI/CD pipeline -- Future hardening step: GitHub Actions for automated tests, benchmarks, and Docker build validation
8. Access control and audit logs -- Future hardening step: authentication, role-based permissions, and full request audit trail

Full detail: [docs/evidence/scale_path.md](docs/evidence/scale_path.md)

---

## System Scope and Boundaries

This system is designed as a local explainable ML pricing workflow and engineering evidence system. It demonstrates model training, SHAP-based attribution, FastAPI serving, batch prediction, test coverage, benchmark evidence, Docker packaging, and documented system boundaries. It is not positioned as a production lending platform, loan approval engine, or financial advice system. Detailed claim boundaries are maintained in [docs/evidence/claim_safety.md](docs/evidence/claim_safety.md).

Technical scope notes:

- The XGBoost model was trained on the full 757,494-row dataset. The current `data/interest_rate_df_engineered.csv` is a 2,000-row demo sample used for local review; the model artifact reflects the full training run.
- Test MAE of ~0.93 pp is from the 2K proxy run. MAE was not recorded in the original full training notebook.
- The model is saved in legacy XGBoost pickle format. Re-saving in JSON format would improve long-term portability.
- The FastAPI serving boundary does not include prediction persistence or authentication. Both are documented in the engineering roadmap.
- The Docker image was built locally and has not been pushed to a registry or deployed to a cloud runtime.
- All latency values are local benchmark measurements. Production latency would depend on hosting environment, network topology, and feature pipeline design.

---

## Getting Started

```bash
git clone https://github.com/IjazKakkodDS/explainable-loan-interest-prediction.git
cd explainable-loan-interest-prediction
pip install -r requirements.txt
```

### FastAPI Serving Boundary

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs: http://127.0.0.1:8000/docs

### Streamlit Review Interface

```bash
streamlit run app/app.py
```

Opens at http://localhost:8501.

### Tests

```bash
python -m pytest tests/ -v
```

### Inference Benchmark

```bash
python scripts/benchmark_inference.py
```

### API Benchmark

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

## Evidence Index

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

## Author

- **Email:** ijazkakkod@gmail.com
- **LinkedIn:** [linkedin.com/in/ijazkakkod](https://linkedin.com/in/ijazkakkod)
- **GitHub:** [github.com/IjazKakkodDS](https://github.com/IjazKakkodDS)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
