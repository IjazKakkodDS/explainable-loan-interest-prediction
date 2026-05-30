# Resource and Cost Profile
# Explainable Credit Pricing Intelligence System

Generated: 2026-05-30
Phase: L2 hardening

This document records what has been measured, what has not been measured, and
the expected scaling pressure points for this system. All values are from
local benchmark runs. No production cost estimates are made.

---

## What Has Been Measured

All measurements are from local hardware: Windows 11, Python 3.13.1, developer
machine. Production values would differ based on hosting environment, network
topology, concurrency, and instance type.

Source files:
- docs/evidence/benchmark_results.json (inference and SHAP latency)
- docs/evidence/api_benchmark_results.json (API endpoint latency, in-process)

### Model Load Time

| Artifact | Cold Load |
|---|---|
| app/model.pkl | 26.6 ms (local) |

The 26.6 ms figure reflects local file I/O and Python environment. A warm
container with the artifact already in OS file cache would load faster. This
is measured once at process startup via lru_cache; it does not repeat per
request.

### Inference Latency -- Raw XGBoost .predict()

Measured via scripts/benchmark_inference.py. Averaged over 5 iterations after warmup.

| Batch Size | Wall Time | Per Record |
|---|---|---|
| 1 row | 2.28 ms | 2,284 us |
| 10 rows | 2.17 ms | 217 us |
| 100 rows | 2.17 ms | 21.7 us |

XGBoost tree traversal is largely batch-insensitive up to 100 rows at this
model size (25 features, standard depth). Per-record cost drops quickly with
batch size.

### SHAP Attribution Latency -- TreeExplainer

Measured via scripts/benchmark_inference.py.

| Batch Size | Total | Per Record |
|---|---|---|
| 1 row | 4.1 ms | 4.1 ms |
| 10 rows | 15.2 ms | 1.52 ms |
| 100 rows | 23.4 ms | 0.23 ms |

SHAP scales better than linearly with batch size for tree models. The per-record
cost at 100 rows (0.23 ms) is approximately 18x lower than at 1 row (4.1 ms).
This makes batched explanation workloads significantly more efficient than
single-record loops.

### API Endpoint Latency -- In-Process TestClient

Measured via scripts/benchmark_api.py. Method: FastAPI TestClient (no network
round-trip). 20 iterations per endpoint, 3 warmup calls.

| Endpoint | Mean | Median | Min | Max |
|---|---|---|---|---|
| GET /health | 4.6 ms | 4.6 ms | 3.8 ms | 5.7 ms |
| POST /predict (1 record) | 20.9 ms | 21.0 ms | 18.7 ms | 25.6 ms |
| POST /explain (1 record) | 26.2 ms | 26.4 ms | 22.8 ms | 29.4 ms |
| POST /batch-predict (10 records) | 21.3 ms | 21.2 ms | 17.3 ms | 28.7 ms |
| POST /batch-predict (100 records) | 21.8 ms | 22.4 ms | 18.9 ms | 24.7 ms |

The gap between raw XGBoost inference (2.2 ms) and the /predict endpoint
(20.9 ms) reflects JSON parsing, Pydantic validation, feature encoding,
FastAPI middleware, and TestClient serialization overhead. The model cost
is approximately 10 percent of the total per-request cost at single-record
throughput.

---

## What Has Not Been Measured

| Item | Status | Note |
|---|---|---|
| Runtime RSS memory | Not measured | model.pkl is 1.4 MB on disk; process RSS during SHAP was not captured |
| Peak memory during SHAP (large batch) | Not measured | Unknown for batches above 400 rows (demo sample limit) |
| API latency under concurrent load | Not measured | All benchmarks are single-threaded in-process; no load test has been run |
| p95 / p99 latency | Not measured | Only mean, median, min, max captured |
| Cold-start latency in container | Not measured | Container startup including uvicorn init was not timed |
| Batch latency above 400 rows | Not feasible with current demo data | Demo sample has 400 test rows; 1000-row batch was skipped |
| Throughput (requests per second) | Not measured | No concurrency or RPS benchmark has been run |
| GPU inference | Not applicable | XGBoost tree model; GPU would not apply |

---

## Scaling Pressure Points

These are the components most likely to limit performance at scale. Listed by
expected impact order.

### 1. SHAP Explanation Cost

SHAP is the dominant per-request cost for /explain. At 1 record, SHAP takes
4.1 ms versus 2.3 ms for the model itself. At scale, if every request calls
/explain rather than /predict, SHAP becomes the throughput bottleneck.

Mitigation options (not yet implemented):
- Cache SHAP values keyed on input hash (Redis or in-memory LRU)
- Use /predict for high-throughput paths; reserve /explain for selective auditing
- Pre-compute SHAP for common borrower profiles

### 2. Single-Record API Overhead

At 20.9 ms per /predict call in-process, the majority of latency is
framework overhead (encoding, validation, middleware), not model inference.
Under concurrent load, this overhead compounds and degrades per-request
throughput faster than the model cost alone would suggest.

Mitigation options (not yet implemented):
- async FastAPI endpoints with a worker pool
- Batch incoming single-record requests (request coalescing)
- Profiling middleware to identify the dominant overhead component

### 3. Model Cold Load

At 26.6 ms locally, model load is fast enough that cold start is not an
issue in a long-running process. In a serverless or scale-to-zero environment,
container cold start (including uvicorn init, model load, and SHAP explainer
load) would dominate per-request latency for the first request after a warm-up
gap.

### 4. Memory Footprint (Unknown)

Runtime RSS has not been measured. At minimum: model.pkl (~1.4 MB) +
SHAP explainer (~1.4 MB) + Python interpreter overhead. Under load with
concurrent SHAP computation, memory pressure would depend on batch size and
worker count.

Measurement needed: capture RSS at startup, after first predict, and after
first batch SHAP call.

---

## Next Measurements Needed

Priority order for a deployment-readiness assessment:

1. Runtime RSS at startup, after warm predict, after batch SHAP (100 rows)
2. p50 and p95 API latency under concurrent load (2, 4, 8 workers)
3. Requests per second throughput for /predict and /batch-predict at sustained load
4. Container cold-start time (uvicorn init + model load + first request)
5. Batch latency at 1K and 10K rows (requires full dataset or synthetic data)
6. Cost per 1K predictions estimate after deployment target is selected

---

## Note on Production Cost Estimation

No cost estimates are included in this document. Compute cost depends on:
- Hosting provider and instance type selected
- Whether model serving is synchronous or async
- Whether SHAP is computed per request or cached
- Traffic volume and concurrency pattern

Cost estimation is a deployment-phase activity. The benchmark evidence in this
document provides the latency inputs needed to estimate cost once a hosting
target is chosen.
