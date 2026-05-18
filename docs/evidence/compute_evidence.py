"""
W13C Evidence computation script for Explainable Loan Interest Rate Modelling.
Computes real metrics and generates SHAP visualizations from saved artifacts.
Run from project root: python docs/evidence/compute_evidence.py
"""

import joblib
import json
import time
import os
import sys

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import shap

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

def main():
    print("=== W13C Loan Interest Evidence Computation ===\n")

    # --- Load artifacts ---
    print("Loading model artifacts...")
    model = joblib.load("app/model.pkl")
    features_list = joblib.load("app/features_list.pkl")
    print(f"  model.pkl loaded | n_features: {model.n_features_in_}")
    print(f"  features_list.pkl loaded | {len(features_list)} features")

    # --- Load and prepare data ---
    print("\nLoading data (interest_rate_df_engineered.csv)...")
    df = pd.read_csv("data/interest_rate_df_engineered.csv")
    print(f"  Rows loaded: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Prepare same feature pipeline as training
    numerical_cols = ['loan_amnt', 'installment', 'annual_inc', 'revol_util',
                      'total_rec_int', 'inq_last_6mths']
    one_hot_cols = ['term', 'purpose', 'verification_status']

    X = df.drop(columns=['int_rate'])
    y = df['int_rate']

    # One-hot encode
    X_encoded = pd.get_dummies(X, columns=one_hot_cols)
    X_encoded = X_encoded.reindex(columns=features_list, fill_value=0)

    # Scale numerical cols
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    # Train/test split (same as training: 80/20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # --- Compute metrics ---
    print("\nComputing metrics...")
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_r2 = float(r2_score(y_train, y_train_pred))

    print(f"  Test RMSE:  {rmse:.4f}")
    print(f"  Test MAE:   {mae:.4f}")
    print(f"  Test R²:    {r2:.4f}")
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Train R²:   {train_r2:.4f}")

    # --- Inference latency ---
    print("\nBenchmarking inference latency...")
    latencies = {}
    for n in [1, 100, 1000]:
        if n > len(X_test):
            sample = X_test.iloc[:len(X_test)]
        else:
            sample = X_test.iloc[:n]
        # Warm up
        _ = model.predict(sample)
        # Timed run (3 iterations average)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            model.predict(sample)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        avg_ms = float(np.mean(times) * 1000)
        per_record_us = float((np.mean(times) / n) * 1_000_000)
        latencies[f"{n}_rows"] = {
            "rows": n,
            "avg_wall_time_ms": round(avg_ms, 3),
            "per_record_us": round(per_record_us, 3)
        }
        print(f"  {n:5d} rows: {avg_ms:.2f}ms total | {per_record_us:.1f}us/record")

    # --- SHAP timing and visualization ---
    print("\nGenerating SHAP values and plots...")
    explainer = shap.Explainer(model)

    # SHAP latency
    shap_latencies = {}
    for n in [1, 100]:
        sample = X_test.iloc[:n]
        t0 = time.perf_counter()
        sv = explainer(sample)
        t1 = time.perf_counter()
        elapsed_ms = float((t1 - t0) * 1000)
        per_record_ms = float(elapsed_ms / n)
        shap_latencies[f"{n}_rows"] = {
            "rows": n,
            "total_ms": round(elapsed_ms, 1),
            "per_record_ms": round(per_record_ms, 3)
        }
        print(f"  SHAP {n:3d} rows: {elapsed_ms:.1f}ms total | {per_record_ms:.2f}ms/record")

    # SHAP summary plot (beeswarm)
    print("\n  Generating SHAP summary plot (beeswarm)...")
    shap_values_100 = explainer(X_test.iloc[:100])
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values_100, max_display=12, show=False)
    plt.title("SHAP Feature Impact — Loan Interest Rate (XGBoost, 100 test samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_summary.png")

    # SHAP waterfall plot (single prediction)
    print("  Generating SHAP waterfall plot (single prediction)...")
    shap_values_1 = explainer(X_test.iloc[:1])
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values_1[0], max_display=12, show=False)
    plt.title("SHAP Waterfall — Single Loan Interest Rate Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_waterfall.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_waterfall.png")

    # SHAP bar summary plot
    print("  Generating SHAP bar plot (mean absolute impact)...")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values_100, max_display=12, show=False)
    plt.title("SHAP Mean Feature Impact — Loan Interest Rate (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_bar.png")

    # --- Save metrics JSON ---
    metrics = {
        "model": "XGBoost (xgb_selected_features_model.pkl)",
        "task": "regression",
        "target": "int_rate (loan interest rate, percentage points)",
        "dataset_context": {
            "raw_loan_data_rows": 887379,
            "processed_rows_used_for_training": 757494,
            "current_engineered_sample_rows": len(df),
            "features_after_encoding": int(len(features_list)),
            "train_test_split": "80/20, random_state=42",
            "note": "Model trained on 757,494-row dataset. sample_reduction.ipynb later overwrote engineered CSV to 2,000 rows for Streamlit demo. Metrics reported below use the 2,000-row sample with same 80/20 split."
        },
        "metrics_on_current_sample": {
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
            "test_r2": round(r2, 4),
            "train_rmse": round(train_rmse, 4),
            "train_r2": round(train_r2, 4)
        },
        "metrics_from_full_training_run": {
            "source": "notebook/streamlit_engineered_ml.ipynb output",
            "training_rows": 605995,
            "test_rows": 151499,
            "test_rmse": 0.9795,
            "test_r2": 0.9506,
            "note": "These metrics come directly from the notebook that saved the model pkl. MAE not recorded in that run; computed from current sample above."
        },
        "inference_latency": latencies,
        "shap_latency": shap_latencies
    }

    with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved: model_metrics.json")

    # --- Save SHAP latency JSON ---
    with open(os.path.join(OUTPUT_DIR, "shap_latency.json"), "w") as f:
        json.dump(shap_latencies, f, indent=2)
    print("Saved: shap_latency.json")

    print("\n=== Evidence computation complete ===")
    return metrics

if __name__ == "__main__":
    metrics = main()
    print("\nFinal metrics summary:")
    print(f"  Test RMSE (2K sample): {metrics['metrics_on_current_sample']['test_rmse']}")
    print(f"  Test MAE  (2K sample): {metrics['metrics_on_current_sample']['test_mae']}")
    print(f"  Test R²   (2K sample): {metrics['metrics_on_current_sample']['test_r2']}")
    print(f"  Test RMSE (757K full): {metrics['metrics_from_full_training_run']['test_rmse']}")
    print(f"  Test R²   (757K full): {metrics['metrics_from_full_training_run']['test_r2']}")
