# Portfolio Summary
# Explainable Credit Pricing Intelligence System

Generated: 2026-05-18
Phase: L1

This document provides portfolio-ready copy for use in case study pages, portfolio sites, recruiter-facing profiles, and presentation decks. All copy is grounded in verified evidence (see claim_safety.md). No production or enterprise claims are included.

---

## Recommended Display Name

Explainable Credit Pricing Intelligence System

---

## Recommended Headline

An explainable credit pricing workflow connecting LendingClub borrower records,
XGBoost interest-rate prediction, SHAP attribution, local inference benchmarks,
and a Streamlit review surface.

---

## Recommended Hero Subtitle

Prediction alone is not enough in credit pricing. This system attaches
SHAP attribution to every prediction, measuring which borrower features
drive the rate and by how much, with reproducible latency evidence and
a reviewer-facing interface built around the explanation.

---

## Recommended Proof Chips

Use as badge/chip labels on a portfolio card or system overview:

- 887K Borrower Records
- XGBoost R2 0.95
- 25 Features
- SHAP Attribution
- 0.19 ms/record (SHAP, 100 rows)
- FastAPI Serving Boundary
- 41 pytest Tests Passing
- Docker Image Built
- Streamlit Review Interface
- Reproducible Evidence Pipeline
- Local Benchmark Verified

Note: qualify "Local Benchmark Verified" as local hardware results.

---

## Recommended ImpactStrip Cells

These are short value statements for a horizontal impact strip on a portfolio page.
Use the format: [Number or qualifier] / [Label].

| Cell | Copy |
|---|---|
| 887,379 | Borrower records processed |
| 757,494 | Training records (XGBoost) |
| R2 0.95 | On 151,499 holdout records |
| RMSE 0.98 pp | Interest rate prediction error |
| 25 | Encoded features (sub_grade excluded) |
| 0.19 ms | SHAP attribution per record at batch 100 |
| 5 | Model families compared |
| 3 | SHAP plot types generated (summary, waterfall, bar) |

---

## Recommended ArchitectureDiagram Nodes

Use these nodes in a left-to-right system diagram:

```
[LendingClub Data]
    --> [Feature Engineering]
        --> [XGBoost Model]
            --> [FastAPI Serving Boundary]
                --> [SHAP TreeExplainer]
                    --> [Streamlit Review Interface]
                        --> [Evidence Artifacts]
```

Node details:

| Node | Label | Sub-label |
|---|---|---|
| LendingClub Data | 887,379 raw rows | CSV input |
| Feature Engineering | 757,494 processed rows | Encoding, one-hot encoding |
| XGBoost Model | R2 0.95, RMSE 0.98 pp | 25 features, 605,995 train rows |
| FastAPI Serving Boundary | /predict, /explain, /batch-predict, /health | 41 pytest tests; 19.2 ms/predict in-process |
| SHAP TreeExplainer | Per-feature attribution | 0.19 ms/record at 100 rows |
| Streamlit Review Interface | Prediction + SHAP display | Batch CSV upload supported |
| Evidence Artifacts | benchmark_results.json, SHAP plots | Reproducible measurement pipeline |

---

## Recommended EvidencePanel Cards

Use these as cards in an evidence panel or proof section.

### Card 1: Model Performance
Title: XGBoost Regression -- Full Training Run
Body: Trained on 605,995 LendingClub borrower records. Evaluated on 151,499 holdout records. RMSE: 0.98 percentage points. R2: 0.95. Prediction error stays within approximately 1 percentage point of the true interest rate.
Source: Notebook output (ML_XAI_Engineered_Data.ipynb)
Evidence file: model_metrics.md

### Card 2: Inference Latency
Title: Local Benchmark Results
Body: XGBoost .predict() measured at 2.78 ms for 1 row and 2.80 ms for 100 rows (27.97 us/record). SHAP attribution measured at 4.9 ms for 1 row and 19.2 ms for 100 rows (0.19 ms/record). Measured on local hardware, Python 3.13, Windows 11. 1,000-row batch not measured (only 400 test rows available in demo sample).
Source: scripts/benchmark_inference.py
Evidence file: benchmark_results.json, benchmark_report.md

### Card 3: SHAP Explainability
Title: Per-Prediction SHAP Attribution
Body: SHAP TreeExplainer applied to XGBoost model. Three plot types generated: beeswarm (global, 100 samples), waterfall (single prediction), bar (mean |SHAP| across 100 samples). Top drivers: loan_amnt (positive), installment (negative), term_36 months (negative). sub_grade excluded from features to avoid target leakage.
Source: scripts/generate_explainability_evidence.py
Evidence file: shap_summary.png, shap_waterfall.png, shap_bar.png

### Card 4: FastAPI Serving Boundary
Title: Production-Style Model Serving Layer
Body: FastAPI app at src/api/main.py exposes /health, /predict, /explain, and /batch-predict endpoints. Pydantic schemas validate inputs. Model and SHAP explainer are loaded once via lru_cache. All responses include an evidence_note and an X-Process-Time-Ms header. 41 pytest tests cover encoding correctness, prediction values, and 422 validation. In-process TestClient benchmark: /predict 19.2 ms mean, /batch-predict 0.18 ms/record at 100 records. Docker image built: explainable-credit-pricing:latest.
Source: src/api/main.py, tests/, scripts/benchmark_api.py
Evidence file: api_benchmark_results.json, api_benchmark_report.md, l2_engineering_upgrade_report.md

### Card 5: System Design
Title: Separation of Concerns
Body: Preprocessing, training, explanation, serving, interface, and evidence generation are separated into distinct layers. The features_list.pkl acts as a schema contract across all prediction paths: Streamlit app, FastAPI service, and benchmark scripts all load and align to the same ordered feature list. The SHAP explainer is pre-saved and loaded at startup, avoiding rebuild latency per session.
Source: Code inspection
Evidence file: system_design_notes.md

---

## Recommended Business Value Wording

Use this framing when explaining why this system matters to non-technical audiences:

"Interest rate pricing models are only useful if reviewers can trust and explain the output.
This system connects the prediction to the evidence: for each borrower record, it shows
the predicted rate and the exact features that drove that rate up or down, with measured
latency evidence confirming the explanation is fast enough for interactive review.
This closes the gap between a model output and a defensible pricing decision."

---

## Recommended ROI Logic

This system does not claim production ROI. The following is the defensible ROI narrative for a portfolio context:

"A pricing reviewer who cannot explain a model output either rejects it or accepts it blindly.
Both outcomes carry cost. SHAP attribution eliminates that binary by surfacing the reason.
The time a reviewer spends second-guessing an unexplained prediction is reduced by
attaching the explanation at the point of output. This system demonstrates that
explanation can be generated in under 5 ms per record, making it a practical
addition to any batch or interactive pricing workflow, not a separate offline report."

Do not claim this ROI has been measured or realised.
This is a designed value argument, not a measured outcome.

---

## Recommended Limitations Copy

Include these limitations in any case study, README, or presentation:

- This is an explainable modelling workflow, not a production lending system.
- Predictions are not financial advice and have not been reviewed or approved by any lending institution.
- No customers, users, or revenue claims are associated with this system.
- Latency results are from local benchmark runs. Production latency would depend on hosting environment, network, and feature pipeline.
- MAE of 0.93 pp is from a 2,000-row proxy sample. The full training run did not log MAE.
- The model was saved in legacy XGBoost pickle format. A re-saved JSON-format model would improve portability.
- The Streamlit interface does not persist predictions or include authentication.
- The FastAPI serving boundary is a local demonstration layer; it is not deployed to a production server.
- The Docker image was built locally; the container has not been run and tested in a live environment.

---

## Recommended Scale Path Wording

Use this in a portfolio "Next Steps" or "Production Path" section:

"The system already includes a FastAPI serving boundary with prediction and SHAP endpoints.
The designed production extension path from here:
persist predictions and SHAP explanations to PostgreSQL,
add a batch scoring worker for volume use cases,
cache SHAP values for repeated inputs,
wrap training in an MLflow pipeline with versioned model artifacts,
and add GitHub Actions CI/CD with automated Docker build and staging deploy.
None of these extension steps are claimed as built."

---

## GitHub / Public Evidence

git remote -v confirms the local repository is linked to:
  https://github.com/IjazKakkodDS/explainable-loan-interest-prediction

The docs/ folder (containing all evidence artifacts) is currently untracked in git.
Until docs/ is committed and pushed to GitHub, use:

  "Evidence artifacts available in local repository at docs/evidence/."

After pushing:

  "Full evidence inventory available in the public GitHub repository at
   github.com/IjazKakkodDS/explainable-loan-interest-prediction"

---

## Portfolio Update Checklist

Before updating the portfolio page for this system, confirm:

- [ ] docs/ has been committed and pushed to the GitHub repository
- [ ] benchmark_results.json reflects a fresh run from the target environment
- [ ] SHAP plots reflect the current model version
- [ ] Streamlit deployment URL (if used) is confirmed live
- [ ] All claims cross-checked against claim_safety.md
- [ ] No production, revenue, or user-count claims included
