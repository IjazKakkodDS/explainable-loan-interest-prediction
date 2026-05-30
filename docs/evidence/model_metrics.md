# Loan Interest Rate Model — Metrics Summary

Generated: 2026-05-18
Script: docs/evidence/compute_evidence.py
Model: XGBoost (models/xgb_selected_features_model.pkl -- served via app/model.pkl)

---

## Dataset Chain

| Stage | Rows | File |
|---|---|---|
| Raw LendingClub data | 887,379 | data/loan_data.csv |
| After cleaning (non-engineered) | 757,263 | data/interest_rate_non_engineered_df.csv |
| After feature engineering (training) | 757,494 | data/interest_rate_df_engineered.csv (pre-reduction) |
| Current demo sample | 2,000 | data/interest_rate_df_engineered.csv (post sample_reduction.ipynb) |

The model was trained on 757,494 rows. sample_reduction.ipynb later overwrote the engineered CSV to 2,000 rows to support the Streamlit demo deployment.

---

## Model Configuration

- **Algorithm:** XGBoost regression
- **Target:** int_rate (loan interest rate in percentage points)
- **Features:** 25 (after one-hot encoding of term, purpose, verification_status)
- **Raw features:** loan_amnt, installment, annual_inc, revol_util, total_rec_int, inq_last_6mths, term, purpose, verification_status
- **Excluded:** sub_grade (directly encodes interest rate tier; excluded for defensible prediction)
- **Train/test split:** 80/20, random_state=42

---

## Model Metrics — Full Training Run (757,494 rows)

Source: notebook/streamlit_engineered_ml.ipynb cell outputs (train rows 605,995 / test rows 151,499)

| Metric | Value |
|---|---|
| Test RMSE | **0.9795 percentage points** |
| Test R² | **0.9506** |
| Test MAE | Not recorded in that run |
| Training rows | 605,995 |
| Test rows | 151,499 |

---

## Model Metrics — Current 2K Sample (same pipeline, 80/20 split)

Source: compute_evidence.py run against current data/interest_rate_df_engineered.csv (2,000 rows)

| Metric | Value |
|---|---|
| Test RMSE | 1.2848 percentage points |
| Test MAE | **0.9263 percentage points** |
| Test R² | 0.9123 |
| Train RMSE | 1.1803 |
| Train R² | 0.9269 |
| Test rows | 400 |

Note: Slightly lower accuracy on 2K sample vs full run is expected — small holdout set with less distribution coverage.

---

## Inference Latency (XGBoost .predict())

Measured: 3-iteration average, warmed up, on local hardware.

| Batch size | Wall time (ms) | Per-record (µs) |
|---|---|---|
| 1 row | 2.147 ms | 2,146 µs |
| 100 rows | 2.088 ms | 20.9 µs |
| 1,000 rows | 2.242 ms | **2.2 µs** |

---

## SHAP Explanation Latency (TreeExplainer via shap.Explainer)

| Batch size | Total (ms) | Per-record (ms) |
|---|---|---|
| 1 row | 5.8 ms | 5.811 ms |
| 100 rows | 18.2 ms | **0.182 ms** |

---

## Top SHAP Features (from beeswarm plot, 100 test samples)

Ranked by mean absolute SHAP impact:

1. loan_amnt — dominant positive driver for higher rates
2. installment — strong negative correlation (larger installments predict lower rates)
3. term_36 months — negative impact (36-month term predicts lower rate vs 60-month)
4. total_rec_int — positive driver
5. annual_inc — negative driver (higher income predicts lower rate)
6. revol_util — positive driver (higher utilisation predicts higher rate)

---

## Safe Claims for Portfolio

- "887,379 raw LendingClub loan records" — confirmed from loan_data.csv row count
- "757,494 processed records used for model training" — confirmed from notebook output
- "XGBoost regression, R² 0.95 on 151,499 test records" — confirmed from notebook output
- "RMSE 0.98 percentage points" — confirmed from notebook output
- "MAE approximately 0.93 percentage points" — from 2K sample (acceptable proxy; MAE was not logged in full run)
- "2.2 µs per record at batch of 1,000" — measured
- "SHAP explanation in 0.18 ms per record" — measured (100-row batch)
- "25 features after encoding, excluding sub_grade" — confirmed from features_list.pkl

## Unsafe / Do Not Claim

- Any production deployment claim
- FastAPI deployed to production — FastAPI serving boundary is implemented locally; it is not a deployed production service
- Flask backend — not implemented
- Loan recommendation or financial advice framing
- Any specific loss reduction or cost savings estimate
