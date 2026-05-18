# Claim Safety Reference
# Explainable Credit Pricing Intelligence System

Generated: 2026-05-18
Phase: L1

This document defines which claims are safe to make in portfolio copy, presentations, and case-study materials, and which claims must not be made. It is grounded in directly observable evidence from this repository.

Last updated: 2026-05-19 (L2)

---

## Safe Claims

The following claims are backed by directly measurable or directly observable evidence.

| Claim | Evidence source | Type |
|---|---|---|
| 887,379 raw LendingClub borrower records processed | data/loan_data.csv row count | Measured |
| 757,494 processed records used for model training | Notebook cell output (ML_XAI_Engineered_Data.ipynb) | Measured |
| XGBoost regression trained on 605,995 training rows | Notebook output: 80/20 split of 757,494 rows | Measured |
| 151,499 holdout test records evaluated | Notebook output: 20% of 757,494 | Measured |
| Test RMSE 0.98 percentage points on 151,499 test records | Notebook output | Measured |
| Test R2 0.95 on 151,499 test records | Notebook output | Measured |
| Test MAE approximately 0.93 percentage points | compute_evidence.py on 2K proxy sample | Measured (proxy) |
| 25 features after one-hot encoding, sub_grade excluded | app/features_list.pkl | Verified |
| Prediction latency 2.78 ms at 1 row (local benchmark) | benchmark_results.json | Measured |
| Prediction latency 27.97 us/record at 100 rows (local benchmark) | benchmark_results.json | Measured |
| SHAP attribution latency 4.9 ms at 1 row (local benchmark) | benchmark_results.json | Measured |
| SHAP attribution latency 0.19 ms/record at 100 rows (local benchmark) | benchmark_results.json | Measured |
| SHAP summary, waterfall, and bar plots generated | docs/evidence/*.png | Verified |
| Streamlit review interface implemented | app/app.py | Verified |
| FastAPI serving boundary implemented: /health, /predict, /explain, /batch-predict | src/api/main.py | Verified |
| 41 pytest tests passing (16 model service, 25 API) | pytest output | Measured |
| API in-process latency: /predict 19.2 ms mean, /batch-predict 0.18 ms/record at 100 records | docs/evidence/api_benchmark_results.json | Measured |
| Docker image built: explainable-credit-pricing:latest, python:3.11-slim | Docker build output | Verified |
| app/app.py preprocessing bug fixed: term and verification_status passed as raw values | app/app.py | Verified |
| app/shap_explainer.pkl regenerated using shap.Explainer(model); Python 3.13 compatible | app/shap_explainer.pkl reload verified | Verified |
| 5 model families compared (LR, DT, RF, XGBoost, FNN) | Notebooks | Observed |
| Feature selection via LassoCV and XGBoost importance | Notebook code | Observed |
| GitHub repository at github.com/IjazKakkodDS/explainable-loan-interest-prediction | git remote -v confirms correct remote | Verified |
| Local benchmark results available | docs/evidence/benchmark_results.json | Measured |
| SHAP evidence plots available | docs/evidence/shap_summary.png, shap_waterfall.png, shap_bar.png | Verified |

---

## GitHub / Public Evidence Status

git remote -v confirms the local repository is connected to:
  https://github.com/IjazKakkodDS/explainable-loan-interest-prediction.git

The docs/ folder is currently untracked (git status). It has not yet been pushed to GitHub.
Safe to claim GitHub evidence only AFTER docs/ is committed and pushed.
Until then, use: "Evidence artifacts available in local repository."

---

## Unsafe Claims

The following claims must NOT be made. Evidence does not support them.

| Unsafe claim | Reason |
|---|---|
| Production lending deployment | No production system exists |
| Approved loan pricing recommendations | This system does not issue approvals |
| Customer adoption | No users, customers, or adoption of any kind |
| Revenue impact | Not measured, not applicable |
| Cost savings | Not measured, not applicable |
| Enterprise usage | Not applicable |
| Real-time lending system | Not a live system |
| Financial advice | Not applicable; this is a modelling workflow |
| 1 million users served | Not applicable |
| FastAPI deployed to production | FastAPI is implemented locally as a serving boundary; not deployed to any production server |
| Docker container run and verified in production | Docker image was built locally; container was not started and tested |
| Live Streamlit demo confirmed available | Deployment status was not independently verified during this audit; only confirmed locally |
| 1000-row inference benchmark | Only 400 test rows available in current demo sample; 1000-row batch was skipped |
| Memory footprint measured | Runtime RSS was not captured |
| MAE from full training run | MAE was not logged in the full 757K-row training run; the 0.93 value comes from the 2K proxy |

---

## Presentation Guidance

When presenting this system:

1. Lead with the system design, not the R2 score
2. The R2 0.95 result is supporting evidence, not the headline
3. The headline is: prediction + attribution + evidence attached at decision time
4. Qualify latency numbers as local benchmark results
5. Qualify MAE as proxy from 2K sample
6. Do not claim the Streamlit demo is live unless you verify the deployment URL is active
7. The FastAPI layer is a local serving boundary. Do not call it a production API or deployed service
8. API latency numbers are in-process TestClient measurements. Do not present them as network-round-trip production latency
7. Do not claim GitHub public evidence until docs/ is committed and pushed

---

## Evidence Hierarchy for This System

Tier 1 (strongest): Measured from artifacts in this repository
  - benchmark_results.json, model_metrics.json, shap_latency.json

Tier 2 (strong): Observed from notebook outputs
  - Full-run RMSE 0.9795 pp, R2 0.9506, training rows 605,995

Tier 3 (acceptable): Proxy measurement
  - MAE 0.93 pp from 2K sample

Tier 4 (observable): Code inspection
  - 5 model families, LassoCV feature selection, sub_grade exclusion

Do not use Tier 3 or Tier 4 claims without qualifying the source.
