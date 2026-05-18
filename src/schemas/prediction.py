"""
Pydantic request and response schemas for the credit pricing API.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from src.core.config import (
    VALID_PURPOSES,
    VALID_TERMS,
    VALID_VERIFICATION_STATUSES,
)

_EVIDENCE_NOTE = (
    "This system is an explainable modelling workflow. "
    "Output is not financial advice and is not approved for production lending use."
)


class BorrowerInput(BaseModel):
    """Borrower feature record for interest rate prediction."""

    loan_amnt: Annotated[float, Field(gt=0, le=100_000, description="Loan amount requested (USD)")]
    installment: Annotated[float, Field(gt=0, le=5_000, description="Monthly installment amount (USD)")]
    annual_inc: Annotated[float, Field(gt=0, le=10_000_000, description="Borrower annual income (USD)")]
    revol_util: Annotated[float, Field(ge=0, le=200, description="Revolving credit utilisation (%)")]
    total_rec_int: Annotated[float, Field(ge=0, description="Total interest received to date (USD)")]
    inq_last_6mths: Annotated[int, Field(ge=0, le=50, description="Hard credit inquiries in last 6 months")]
    term: Annotated[
        Literal["36 months", "60 months"],
        Field(description="Loan repayment term"),
    ]
    purpose: Annotated[
        Literal[
            "car", "credit_card", "debt_consolidation", "educational",
            "home_improvement", "house", "major_purchase", "medical",
            "moving", "other", "renewable_energy", "small_business",
            "vacation", "wedding",
        ],
        Field(description="Loan purpose (lowercase, underscored)"),
    ]
    verification_status: Annotated[
        Literal["Not Verified", "Source Verified", "Verified"],
        Field(description="Income verification status (original casing required)"),
    ]

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    predicted_interest_rate: float = Field(
        description="Predicted annual interest rate (percentage points)"
    )
    model_name: str
    evidence_note: str = _EVIDENCE_NOTE


class FeatureDriver(BaseModel):
    feature: str
    shap_value: float = Field(description="SHAP contribution (positive = rate higher, negative = rate lower)")
    direction: Literal["increases_rate", "decreases_rate", "neutral"]


class ExplanationResponse(BaseModel):
    predicted_interest_rate: float
    top_drivers: list[FeatureDriver]
    explanation_method: str
    evidence_note: str = _EVIDENCE_NOTE


class BatchPredictionRequest(BaseModel):
    records: Annotated[
        list[BorrowerInput],
        Field(min_length=1, max_length=10_000, description="List of borrower records to score"),
    ]


class BatchPredictionResponse(BaseModel):
    predictions: list[float] = Field(description="Predicted interest rates in the same order as input records")
    record_count: int
    evidence_note: str = _EVIDENCE_NOTE


class HealthResponse(BaseModel):
    status: str
    system: str
    model_loaded: bool
    explainer_loaded: bool
