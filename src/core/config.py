"""
Centralised path configuration for the Explainable Credit Pricing service.
All paths are relative to the repository root so the service works regardless
of where the process is started from.
"""

from pathlib import Path

# Repository root is two levels up from this file (src/core/config.py)
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Model artifacts
MODEL_PATH: Path = REPO_ROOT / "app" / "model.pkl"
FEATURES_LIST_PATH: Path = REPO_ROOT / "app" / "features_list.pkl"
SHAP_EXPLAINER_PATH: Path = REPO_ROOT / "app" / "shap_explainer.pkl"

# Evidence output
EVIDENCE_DIR: Path = REPO_ROOT / "docs" / "evidence"

# Categorical field values that must match the training-time one-hot encoding.
# These are the RAW values passed to pd.get_dummies; the resulting column names
# must match features_list.pkl exactly.
VALID_TERMS: tuple[str, ...] = ("36 months", "60 months")

VALID_PURPOSES: tuple[str, ...] = (
    "car",
    "credit_card",
    "debt_consolidation",
    "educational",
    "home_improvement",
    "house",
    "major_purchase",
    "medical",
    "moving",
    "other",
    "renewable_energy",
    "small_business",
    "vacation",
    "wedding",
)

# Verification status must use the original casing from training data.
# features_list.pkl contains "verification_status_Not Verified" etc.
VALID_VERIFICATION_STATUSES: tuple[str, ...] = (
    "Not Verified",
    "Source Verified",
    "Verified",
)

# Numerical features (passed raw to the model; no scaling applied)
NUMERICAL_FEATURES: tuple[str, ...] = (
    "loan_amnt",
    "installment",
    "annual_inc",
    "revol_util",
    "total_rec_int",
    "inq_last_6mths",
)

# Categorical features to one-hot encode
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "term",
    "purpose",
    "verification_status",
)

SYSTEM_NAME: str = "Explainable Credit Pricing Intelligence System"
MODEL_NAME: str = "XGBoost Regression (xgb_selected_features_model)"
