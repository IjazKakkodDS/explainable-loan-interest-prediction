"""
SHAP explainability evidence generator for the
Explainable Credit Pricing Intelligence System.

Loads the saved XGBoost model, builds a SHAP TreeExplainer,
and saves three evidence plots to docs/evidence/:

  shap_summary.png   -- beeswarm showing global feature impact (100 test rows)
  shap_waterfall.png -- single-prediction attribution waterfall
  shap_bar.png       -- mean absolute feature impact bar chart

Run from project root:
    python scripts/generate_explainability_evidence.py
"""

import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVIDENCE_DIR = os.path.join(ROOT, "docs", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)


def load_artifacts():
    model_path = os.path.join(ROOT, "app", "model.pkl")
    features_path = os.path.join(ROOT, "app", "features_list.pkl")
    if not os.path.exists(model_path):
        sys.exit(f"ERROR: model not found at {model_path}")
    if not os.path.exists(features_path):
        sys.exit(f"ERROR: features_list not found at {features_path}")
    model = joblib.load(model_path)
    features_list = joblib.load(features_path)
    print(f"  Loaded model ({model.n_features_in_} features in model)")
    print(f"  Loaded features_list ({len(features_list)} features)")
    return model, features_list


def prepare_test_data(features_list: list[str]) -> pd.DataFrame:
    data_path = os.path.join(ROOT, "data", "interest_rate_df_engineered.csv")
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: data not found at {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows from interest_rate_df_engineered.csv")

    numerical_cols = [
        "loan_amnt", "installment", "annual_inc", "revol_util",
        "total_rec_int", "inq_last_6mths",
    ]
    one_hot_cols = ["term", "purpose", "verification_status"]

    X = df.drop(columns=["int_rate"])
    y = df["int_rate"]
    X_encoded = pd.get_dummies(X, columns=one_hot_cols)
    X_encoded = X_encoded.reindex(columns=features_list, fill_value=0)

    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    _, X_test, _, _ = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    return X_test.reset_index(drop=True)


def build_explainer(model):
    print("  Building SHAP TreeExplainer...")
    explainer = shap.Explainer(model)
    print("  TreeExplainer ready")
    return explainer


def generate_summary_plot(explainer, X_test: pd.DataFrame, n: int = 100) -> str:
    sample = X_test.iloc[:n]
    sv = explainer(sample)
    out_path = os.path.join(EVIDENCE_DIR, "shap_summary.png")
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title(
        "SHAP Feature Impact -- Loan Interest Rate (XGBoost, 100 test samples)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_summary.png ({n} samples, top-12 features)")
    return out_path


def generate_waterfall_plot(explainer, X_test: pd.DataFrame) -> str:
    sample = X_test.iloc[:1]
    sv = explainer(sample)
    out_path = os.path.join(EVIDENCE_DIR, "shap_waterfall.png")
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(sv[0], max_display=12, show=False)
    plt.title(
        "SHAP Waterfall -- Single Loan Interest Rate Prediction (XGBoost)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: shap_waterfall.png (single prediction attribution)")
    return out_path


def generate_bar_plot(explainer, X_test: pd.DataFrame, n: int = 100) -> str:
    sample = X_test.iloc[:n]
    sv = explainer(sample)
    out_path = os.path.join(EVIDENCE_DIR, "shap_bar.png")
    plt.figure(figsize=(10, 6))
    shap.plots.bar(sv, max_display=12, show=False)
    plt.title(
        "SHAP Mean Absolute Feature Impact -- Loan Interest Rate (XGBoost)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_bar.png ({n} samples, mean |SHAP|)")
    return out_path


def main():
    print("=== Explainable Credit Pricing -- SHAP Evidence Generator ===\n")

    print("Loading artifacts...")
    model, features_list = load_artifacts()

    print("\nPreparing test data...")
    X_test = prepare_test_data(features_list)
    print(f"  Test rows available: {len(X_test)}")

    print("\nBuilding SHAP explainer...")
    explainer = build_explainer(model)

    print("\nGenerating SHAP plots...")
    generate_summary_plot(explainer, X_test, n=min(100, len(X_test)))
    generate_waterfall_plot(explainer, X_test)
    generate_bar_plot(explainer, X_test, n=min(100, len(X_test)))

    print("\n=== SHAP evidence generation complete ===")
    print(f"Output directory: {EVIDENCE_DIR}")


if __name__ == "__main__":
    main()
