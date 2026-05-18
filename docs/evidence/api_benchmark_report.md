# API Benchmark Report

System: Explainable Credit Pricing Intelligence System
Generated: 2026-05-19T00:24:12
Python: 3.13.1
Platform: Windows-11-10.0.26200-SP0
Method: FastAPI TestClient (in-process)
Iterations per endpoint: 20
Warmup calls: 3

---

## Endpoint Latency (in-process, no network)

| Endpoint | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Stdev (ms) | Status |
|---|---|---|---|---|---|---|
| GET /health | 4.591 | 4.572 | 3.75 | 5.681 | 0.479 | 200 |
| POST /predict | 20.901 | 20.973 | 18.745 | 25.643 | 1.581 | 200 |
| POST /explain | 26.22 | 26.362 | 22.823 | 29.449 | 1.858 | 200 |

---

## Batch Prediction Latency

| Batch Size | Mean Total (ms) | Per-Record (ms) | Status |
|---|---|---|---|
| 10 records | 21.307 | 2.1307 | 200 |
| 100 records | 21.845 | 0.2184 | 200 |

---

## Notes

- All measurements are in-process via TestClient. No network round-trip.
- Model and explainer are loaded once; subsequent calls benefit from warm cache.
- Production deployment latency would include network, serialisation, and hosting overhead.

---

## Interpretation

These results measure in-process TestClient latency only.
The primary cost is model inference and encoding, not network transport.
Production API latency would add: network round-trip, JSON serialisation,
ASGI overhead, and any middleware processing.

The /explain endpoint includes SHAP computation in addition to model prediction.
If /explain returns status 503, the SHAP explainer failed to initialise;
see docs/evidence/l2_engineering_upgrade_report.md for details.