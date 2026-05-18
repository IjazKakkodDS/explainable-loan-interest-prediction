# System Design Notes
# Explainable Credit Pricing Intelligence System

Generated: 2026-05-18
Phase: L1

---

## 1. Current System Architecture

The system connects borrower input data through a modelling and explainability pipeline into a reviewer-facing output surface. Each stage produces reusable artifacts.

```
LendingClub Borrower Records (887,379 raw rows)
    |
    v
Data Cleaning and Feature Engineering
  - Missing value handling (imputation strategies in src/imputation_functions.py)
  - Outlier detection (Tukey fences, Isolation Forest)
  - Categorical encoding: term, purpose, verification_status (one-hot)
  - Numerical scaling: StandardScaler on 6 continuous features
  - sub_grade excluded (encodes interest rate tier directly)
  - Output: 757,494 processed rows, 25 features
    |
    v
XGBoost Regression Model (app/model.pkl)
  - 80/20 train/test split (random_state=42)
  - GridSearchCV hyperparameter tuning
  - LassoCV and XGBoost importance used for feature selection
  - Saved via joblib (~1.4 MB on disk)
    |
    v
SHAP TreeExplainer (app/shap_explainer.pkl)
  - Loaded or recreated from saved model at runtime
  - Produces per-feature attribution values for any prediction
  - Beeswarm (global), waterfall (local), bar (mean |SHAP|) plots
    |
    v
Streamlit Review Interface (app/app.py)
  - Single-record input via sidebar sliders and selectors
  - Predicted interest rate displayed as KPI metric
  - SHAP waterfall and bar plots rendered inline
  - Distribution and correlation visuals for context
  - Batch CSV upload and download supported
    |
    v
Evidence Artifacts (docs/evidence/)
  - model_metrics.json: measured RMSE, MAE, R2, latency
  - benchmark_results.json: fresh benchmark run
  - shap_summary.png, shap_waterfall.png, shap_bar.png
  - benchmark_report.md, model_metrics.md
  - This document and companion evidence files
```

---

## 2. Current Architecture Boundary

This system is explicitly bounded as:

- An explainable modelling and review workflow
- A decision-support tool demonstrating interest rate attribution
- A reproducible evidence system with measured artifacts

This system is NOT:

- A production lending system
- A real-time loan pricing engine in use at any institution
- Financial advice or regulatory guidance
- An enterprise deployment with live users
- A system with any FastAPI or REST backend (uvicorn is in requirements.txt but no API is implemented)

All claims are grounded in measured artifacts or directly observable code.

---

## 3. Computational Efficiency

### Online prediction path

When a single borrower record is submitted through the Streamlit interface:

1. Input dict is constructed from sidebar values
2. pd.get_dummies encodes categorical fields
3. reindex aligns columns to features_list.pkl order
4. model.predict() runs XGBoost inference
5. shap.Explainer() computes SHAP values for the single record
6. Plots are rendered and displayed

The XGBoost .predict() call at 1 row averages 2.78 ms on local hardware.
SHAP attribution at 1 row averages 4.9 ms.
Combined per-record latency is approximately 8 ms on the local machine used for benchmarking.

### Batch prediction path

Users may upload a CSV via the Streamlit interface. The app reads it, applies pd.get_dummies, reindexes to the feature schema, runs model.predict() over all rows, and offers download.
There is no SHAP explanation computed for batch uploads in the current implementation.
At 100 rows, XGBoost prediction takes approximately 2.8 ms total.

### Model loading behaviour

The model is loaded once at Streamlit app startup via joblib.load().
Cold-start load time measured at 2,405 ms on local hardware, which includes XGBoost deserialisation from the older pickle format.
The XGBoost version warning is cosmetic and does not affect prediction correctness.
A re-saved model (Booster.save_model JSON format) would eliminate this warning and reduce load time.

### SHAP computation considerations

The SHAP TreeExplainer uses the exact tree algorithm for XGBoost, which is fast for regression trees.
At 100 rows, SHAP attribution takes 19.2 ms total (0.19 ms per record).
The pre-saved shap_explainer.pkl in app/ accelerates Streamlit startup by avoiding explainer rebuild on every session.
SHAP computation scales roughly linearly with batch size for the TreeExplainer.

### What is measured

- Model load time: measured (benchmark_results.json)
- Prediction latency at 1, 10, 100 rows: measured (benchmark_results.json)
- SHAP latency at 1, 10, 100 rows: measured (benchmark_results.json)
- Inference latency at 1,000 rows: measured in prior compute_evidence.py run (model_metrics.json); not available in current benchmark run because the demo sample provides only 400 test rows

### What is not measured

- Memory RSS at runtime
- Network round-trip latency (no API layer)
- Feature engineering pipeline latency (computed inside the app at request time)
- Streamlit serialisation overhead
- Multi-user concurrency behaviour

---

## 4. System Design Efficiency

### Separation of concerns

| Layer | Artifact | Responsibility |
|---|---|---|
| Data preparation | notebooks/, src/ | Cleaning, imputation, encoding |
| Model training | notebooks/ML_XAI_Engineered_Data.ipynb | XGBoost training and evaluation |
| Model artifact | app/model.pkl, app/features_list.pkl | Serialised model and feature schema |
| Explainability artifact | app/shap_explainer.pkl | Pre-saved SHAP explainer |
| Review interface | app/app.py | Streamlit UI with prediction and SHAP display |
| Evidence pipeline | docs/evidence/compute_evidence.py | Reproducible metric and plot generation |
| Benchmark pipeline | scripts/benchmark_inference.py | Reproducible latency measurement |
| SHAP evidence | scripts/generate_explainability_evidence.py | Reproducible SHAP visualisation |

### Reusable artifacts

- features_list.pkl acts as the schema contract: every prediction path (Streamlit, evidence scripts, benchmark) aligns to this same ordered feature list
- model.pkl is the single source of truth for the trained model; all scripts load it by path
- The 80/20 split with random_state=42 is consistent across all measurement runs

### Reviewer-facing output

The Streamlit interface surfaces prediction, SHAP drivers, and context visuals on a single screen.
A reviewer can input borrower parameters and immediately see the predicted rate alongside the top drivers without access to the underlying model code.

### Explainability attached to prediction

SHAP attribution is computed at prediction time, not as a separate offline report.
This means every prediction comes with a reason: the waterfall plot shows exactly which features pushed the prediction above or below the base rate.

### Current limitations

- The feature engineering pipeline (StandardScaler, one-hot encoding) is applied inside the app at request time using a freshly fitted scaler on the demo sample, not a saved scaler trained on the full dataset. This is a reproducibility gap: predictions may differ slightly from the full-data training run.
- The Streamlit app does not persist predictions or explanations to a database or log file.
- There is no authentication layer on the review interface.
- SHAP explanations for batch uploads are not implemented.
- The model was saved in the older XGBoost pickle format; re-saving in Booster.save_model JSON format would improve portability and eliminate the version warning.

---

## 5. Scale Path Summary

See docs/evidence/scale_path.md for the full designed extension path.

High-level designed steps:
1. Replace Streamlit with a FastAPI model-serving endpoint (GET /predict + SHAP payload)
2. Persist predictions and explanations to PostgreSQL
3. Add a background batch scoring worker (Celery or similar)
4. Cache SHAP explanations for repeated inputs
5. Add model registry (MLflow) and CI/CD pipeline
6. Add drift monitoring and retraining triggers
7. Add role-based access control and audit logs for regulatory traceability
