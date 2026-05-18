# Scale Path
# Explainable Credit Pricing Intelligence System

Generated: 2026-05-18
Phase: L1

No production scale is claimed. This document describes the designed extension path from the current local workflow to a production-grade and enterprise-grade architecture.

---

## 1. Current Implementation Scale

| Dimension | Current State |
|---|---|
| Data input | LendingClub CSV files (887,379 raw rows; 2,000-row demo sample) |
| Model serving | Streamlit app running locally or on Streamlit Cloud |
| Prediction volume | Single-record interactive; batch via CSV upload |
| Persistence | None: predictions are not saved to a database |
| Explainability | Computed at request time; not stored |
| Authentication | None |
| API | None: no REST or FastAPI backend is implemented |
| Monitoring | None |
| CI/CD | No CI/CD pipeline configured. GitHub Actions is a future hardening step. |
| Deployment | Streamlit Cloud or Render (noted in README); not confirmed live at time of this audit |

The current system is a local modelling and review workflow with a Streamlit interface.
It is fully self-contained: model, explainer, and interface run from the same repository.

---

## 2. Production-Ready Next Step

The production-ready next step would require the following changes.
None of these are claimed as implemented.

### 2a. FastAPI model-serving endpoint

Would require:
- Adding a FastAPI app (e.g., api/main.py)
- A POST /predict endpoint that accepts borrower features as JSON
- Returns predicted rate and top-N SHAP attribution values in the response
- Input validation via Pydantic models
- Feature encoding pipeline saved and loaded at startup (not refit at request time)

Scale bottleneck: the current scaler is refit on the demo sample at request time.
This would need to be replaced with a scaler fitted on the full training data and saved to app/scaler.pkl.

### 2b. Prediction persistence

Would require:
- PostgreSQL (or SQLite for small-scale) predictions table
- Schema: prediction_id, timestamp, input_features (JSONB), predicted_rate, shap_values (JSONB), model_version
- Write each prediction response to the database before returning
- This enables audit trails, reviewer lookup, and drift analysis

### 2c. Background batch scoring worker

Could be extended with:
- Celery worker consuming a job queue (Redis or RabbitMQ)
- Accept CSV batch files, score asynchronously, write results and SHAP explanations to database
- Webhook or polling endpoint for job status

### 2d. SHAP explanation caching

Could be extended with:
- A Redis cache keyed on the hash of the input feature vector
- On cache hit, return stored SHAP values without recomputing
- Reduces SHAP latency from ~5 ms to sub-millisecond for repeated inputs

### 2e. Model registry

Would require:
- MLflow Tracking server for experiment logging during training
- MLflow Model Registry for versioned model promotion (staging, production)
- The FastAPI app loads the production-tagged model version on startup
- Rollback path: tag a previous version as production; restart the API

---

## 3. Enterprise-Scale Architecture Path

The enterprise-scale path adds observability, governance, and multi-team operation.
All items below are designed paths only.

### 3a. Feature pipeline reuse

- Move encoding and scaling into a reusable pipeline module (scikit-learn Pipeline object)
- Save the fitted Pipeline to a versioned artifact store (S3 or Azure Blob)
- Both training and serving load the same artifact to eliminate train/serve skew

### 3b. Object storage for artifacts

- model.pkl, shap_explainer.pkl, features_list.pkl, and Pipeline.pkl stored in S3 or Azure Blob
- Versioned by model run ID
- API pulls the correct version on startup without code changes

### 3c. Monitoring and drift reports

- Feature distribution monitoring: compare live request distributions against training distributions
- Prediction distribution monitoring: alert if predicted rate distribution shifts outside expected bounds
- SHAP attribution monitoring: alert if feature importance ranking changes significantly
- Tools: Evidently AI, WhyLogs, or custom scripts writing to a time-series store

### 3d. CI/CD pipeline

- GitHub Actions workflow: on merge to main, run unit tests, run benchmark script, compare benchmark to baseline thresholds, fail if latency regresses beyond threshold
- Docker image build and push to container registry
- Automated deployment to staging environment on successful CI
- Manual promotion gate to production

### 3e. Role-based access control and audit logs

- Authentication layer (OAuth2, JWT) on the API
- Reviewer roles: read-only (see predictions and explanations), admin (trigger retraining, approve model versions)
- Every prediction request logged with user ID, timestamp, input hash, model version
- Audit log retention for regulatory compliance use cases

---

## 4. Scale Bottlenecks to Resolve Before Production

| Bottleneck | Description | Required fix |
|---|---|---|
| Scaler refit at request time | Current Streamlit app refits StandardScaler on the 2K demo sample at startup | Save a scaler trained on the full 757K dataset; load it at API startup |
| Old pickle format | XGBoost model saved in legacy pickle format produces version warning | Re-save model using booster.save_model('model.json') and load with xgboost.XGBRegressor().load_model() |
| No feature pipeline | Encoding and scaling are inline code, not a serialised sklearn Pipeline | Wrap in Pipeline object, save, and load in serving path |
| No prediction persistence | Predictions are lost after each session | Add PostgreSQL write on each prediction |
| No API layer | Streamlit is the only interface; no REST endpoint | Add FastAPI app as a separate service |
| No monitoring | No visibility into prediction or feature drift | Add Evidently or equivalent |

---

## Summary

The current system is a defensible, evidence-backed explainable modelling workflow.
The scale path above describes what would be required to elevate it to a production API service and then to an enterprise-grade credit pricing intelligence platform.
No part of the scale path is claimed as implemented.
