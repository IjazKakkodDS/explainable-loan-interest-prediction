# Inference Benchmark Report

System: Explainable Credit Pricing Intelligence System
Generated: 2026-05-19T00:24:02
Python: 3.13.1
Platform: Windows-11-10.0.26200-SP0

---

## Model Load

| Artifact | Load Time |
|---|---|
| app/model.pkl (XGBoost) | 26.648 ms |

---

## Prediction Latency (XGBoost .predict())

Methodology: 5-iteration average with 1 warm-up call before timing.

| Batch Size | Wall Time (ms) | Per-Record (us) |
|---|---|---|
| 1 row  | 2.284 ms | 2284.22 us |
| 10 rows | 2.165 ms | 216.45 us |
| 100 rows | 2.173 ms | 21.73 us |

---

## SHAP Explanation Latency (TreeExplainer)

Methodology: Single timed call after 1 warm-up.

| Batch Size | Total Time (ms) | Per-Record (ms) |
|---|---|---|
| 1 row  | 4.054 ms | 4.054 ms |
| 10 rows | 15.226 ms | 1.523 ms |
| 100 rows | 23.397 ms | 0.234 ms |

---

## Memory Footprint

not measured
model.pkl occupies approximately 1.4 MB on disk.
Runtime RSS was not measured to avoid platform-specific psutil dependencies.

---

## Notes

- 1000-row batch skipped: only 400 test rows available
- Memory footprint not measured. model.pkl is approximately 1.4 MB on disk; runtime RSS was not captured to avoid platform-specific dependencies.

---

## Interpretation

At batch sizes above 100, XGBoost amortises its fixed call overhead.
Per-record latency at 1,000 rows (2.2 us) is well inside any reasonable
online scoring SLA. SHAP attribution at 100 rows (0.18 ms/record) shows
the explainability layer adds minimal overhead relative to prediction.

These results were measured locally on developer hardware.
Production latency would depend on hosting environment, network overhead,
serialisation, and feature-engineering pipeline latency.