"""
API latency benchmark for the Explainable Credit Pricing Intelligence System.

Uses FastAPI TestClient to benchmark endpoint latency without needing a live server.
All measurements are in-process (no network round-trip).

Results saved to:
  docs/evidence/api_benchmark_results.json
  docs/evidence/api_benchmark_report.md

Run from project root:
    python scripts/benchmark_api.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from typing import Any

from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.api.main import app  # noqa: E402 -- must come after sys.path insert

EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

SAMPLE_RECORD = {
    "loan_amnt": 15000,
    "installment": 450.5,
    "annual_inc": 72000,
    "revol_util": 55.0,
    "total_rec_int": 1200.0,
    "inq_last_6mths": 2,
    "term": "36 months",
    "purpose": "debt_consolidation",
    "verification_status": "Verified",
}

ITERATIONS = 20
WARMUP = 3


def _time_endpoint(
    client: TestClient,
    method: str,
    path: str,
    payload: Any,
    iterations: int = ITERATIONS,
    warmup: int = WARMUP,
) -> dict:
    # Warm-up
    for _ in range(warmup):
        if method == "GET":
            client.get(path)
        else:
            client.post(path, json=payload)

    times_ms: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path, json=payload)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    import statistics
    return {
        "endpoint": path,
        "method": method,
        "iterations": iterations,
        "warmup_calls": warmup,
        "status_code": resp.status_code,
        "mean_ms": round(statistics.mean(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "min_ms": round(min(times_ms), 3),
        "max_ms": round(max(times_ms), 3),
        "stdev_ms": round(statistics.stdev(times_ms), 3) if len(times_ms) > 1 else 0.0,
        "note": "In-process TestClient timing. No network overhead. Production latency would differ.",
    }


def main():
    print("=== Explainable Credit Pricing -- API Benchmark ===\n")
    print("Using FastAPI TestClient (in-process, no network)")
    print(f"Python: {sys.version.split()[0]} | Platform: {platform.platform()}\n")

    client = TestClient(app)
    results: dict[str, Any] = {
        "system": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "method": "FastAPI TestClient (in-process)",
            "iterations_per_endpoint": ITERATIONS,
            "warmup_calls": WARMUP,
        },
        "endpoints": {},
        "batch_endpoints": {},
        "notes": [
            "All measurements are in-process via TestClient. No network round-trip.",
            "Model and explainer are loaded once; subsequent calls benefit from warm cache.",
            "Production deployment latency would include network, serialisation, and hosting overhead.",
        ],
    }

    # /health
    print("Benchmarking GET /health ...")
    r = _time_endpoint(client, "GET", "/health", None)
    results["endpoints"]["health"] = r
    print(f"  mean {r['mean_ms']:.2f} ms | median {r['median_ms']:.2f} ms | status {r['status_code']}")

    # /predict single
    print("Benchmarking POST /predict (single record) ...")
    r = _time_endpoint(client, "POST", "/predict", SAMPLE_RECORD)
    results["endpoints"]["predict_single"] = r
    print(f"  mean {r['mean_ms']:.2f} ms | median {r['median_ms']:.2f} ms | status {r['status_code']}")

    # /explain (may succeed or return 503 if SHAP unavailable)
    print("Benchmarking POST /explain (single record) ...")
    r = _time_endpoint(client, "POST", "/explain", SAMPLE_RECORD)
    results["endpoints"]["explain_single"] = r
    status_note = "ok" if r["status_code"] == 200 else f"status {r['status_code']} (SHAP may be unavailable)"
    print(f"  mean {r['mean_ms']:.2f} ms | median {r['median_ms']:.2f} ms | {status_note}")

    # /batch-predict at 10 records
    batch_10 = {"records": [SAMPLE_RECORD] * 10}
    print("Benchmarking POST /batch-predict (10 records) ...")
    r = _time_endpoint(client, "POST", "/batch-predict", batch_10)
    r["records_per_call"] = 10
    r["per_record_ms"] = round(r["mean_ms"] / 10, 4)
    results["batch_endpoints"]["batch_10"] = r
    print(f"  mean {r['mean_ms']:.2f} ms | {r['per_record_ms']:.4f} ms/record | status {r['status_code']}")

    # /batch-predict at 100 records
    batch_100 = {"records": [SAMPLE_RECORD] * 100}
    print("Benchmarking POST /batch-predict (100 records) ...")
    r = _time_endpoint(client, "POST", "/batch-predict", batch_100)
    r["records_per_call"] = 100
    r["per_record_ms"] = round(r["mean_ms"] / 100, 4)
    results["batch_endpoints"]["batch_100"] = r
    print(f"  mean {r['mean_ms']:.2f} ms | {r['per_record_ms']:.4f} ms/record | status {r['status_code']}")

    # Save JSON
    json_path = os.path.join(EVIDENCE_DIR, "api_benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Save human-readable report
    _write_report(results, EVIDENCE_DIR)
    print("=== API benchmark complete ===")
    return results


def _write_report(results: dict, output_dir: str) -> None:
    sys_info = results["system"]
    ep = results["endpoints"]
    bp = results["batch_endpoints"]

    lines = [
        "# API Benchmark Report",
        "",
        "System: Explainable Credit Pricing Intelligence System",
        f"Generated: {sys_info['timestamp']}",
        f"Python: {sys_info['python_version']}",
        f"Platform: {sys_info['platform']}",
        f"Method: {sys_info['method']}",
        f"Iterations per endpoint: {sys_info['iterations_per_endpoint']}",
        f"Warmup calls: {sys_info['warmup_calls']}",
        "",
        "---",
        "",
        "## Endpoint Latency (in-process, no network)",
        "",
        "| Endpoint | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Stdev (ms) | Status |",
        "|---|---|---|---|---|---|---|",
    ]

    for key, r in ep.items():
        lines.append(
            f"| {r['method']} {r['endpoint']} | {r['mean_ms']} | {r['median_ms']} | "
            f"{r['min_ms']} | {r['max_ms']} | {r['stdev_ms']} | {r['status_code']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Batch Prediction Latency",
        "",
        "| Batch Size | Mean Total (ms) | Per-Record (ms) | Status |",
        "|---|---|---|---|",
    ]

    for key, r in bp.items():
        lines.append(
            f"| {r['records_per_call']} records | {r['mean_ms']} | {r['per_record_ms']} | {r['status_code']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Notes",
        "",
    ]
    for note in results.get("notes", []):
        lines.append(f"- {note}")

    lines += [
        "",
        "---",
        "",
        "## Interpretation",
        "",
        "These results measure in-process TestClient latency only.",
        "The primary cost is model inference and encoding, not network transport.",
        "Production API latency would add: network round-trip, JSON serialisation,",
        "ASGI overhead, and any middleware processing.",
        "",
        "The /explain endpoint includes SHAP computation in addition to model prediction.",
        "If /explain returns status 503, the SHAP explainer failed to initialise;",
        "see docs/evidence/l2_engineering_upgrade_report.md for details.",
    ]

    report_path = os.path.join(output_dir, "api_benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
