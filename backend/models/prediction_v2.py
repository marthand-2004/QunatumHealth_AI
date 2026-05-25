"""Pydantic data models for the v2 prediction response schema.

Defines the structured JSON output returned by ``POST /api/predict/v2``.

Key models
----------
SHAPFeature
    Re-exported from ``xai_service_v2`` for convenience; represents a single
    SHAP feature attribution with direction label.
LatencyBreakdown
    Per-stage wall-clock durations in milliseconds.
PredictionResponse
    The top-level response object returned for each disease.  Includes all
    core fields (Req 10), extended fields (Req 16.4, 19.4, 21.2, 22.1,
    24.1–24.3), and Pydantic validators that:
      - derive ``risk_level`` from ``risk_score`` (Req 10.5)
      - enforce exactly 3 items in ``shap_features`` (Req 18.5)

Requirements: 10.1–10.9, 16.4, 19.4, 21.2, 22.1, 24.1, 24.2, 24.3
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# SHAPFeature is defined in xai_service_v2 and re-exported here so that
# callers only need to import from backend.models.prediction_v2.
from backend.services.xai_service_v2 import SHAPFeature  # noqa: F401

__all__ = [
    "SHAPFeature",
    "LatencyBreakdown",
    "PredictionResponse",
]


# ---------------------------------------------------------------------------
# LatencyBreakdown — Req 21.2
# ---------------------------------------------------------------------------


class LatencyBreakdown(BaseModel):
    """Per-stage wall-clock durations recorded by PerformanceLogger.

    All values are in milliseconds.  ``ocr_ms`` is optional because OCR
    processing is excluded from the 1-second latency budget (Req 9.1, 9.4).

    Attributes
    ----------
    ocr_ms:
        OCR extraction duration (excluded from total budget).
    feature_extraction_ms:
        DataGovernanceModule + FeatureMapper combined duration.
    scaler_normalization_ms:
        StandardScaler.transform() duration per disease.
    classical_ml_ms:
        RF + XGBoost prediction duration.
    vqc_ms:
        VQC circuit inference duration (capped at 800 ms budget, Req 9.3).
    hybrid_fusion_ms:
        HybridFusion.fuse() duration.
    clinical_rules_ms:
        ClinicalRuleEngine.apply_rules() duration.
    xai_layer_ms:
        XAILayer.compute_shap_v2() duration.
    total_ms:
        Sum of all non-OCR stages.  Must be ≤ 1000 ms (Req 9.1).
    """

    ocr_ms: Optional[float] = Field(default=None, description="OCR extraction duration (ms)")
    feature_extraction_ms: float = Field(
        default=0.0, ge=0.0, description="Feature extraction duration (ms)"
    )
    scaler_normalization_ms: float = Field(
        default=0.0, ge=0.0, description="StandardScaler normalization duration (ms)"
    )
    classical_ml_ms: float = Field(
        default=0.0, ge=0.0, description="Classical ML (RF + XGBoost) duration (ms)"
    )
    vqc_ms: float = Field(
        default=0.0, ge=0.0, description="VQC inference duration (ms)"
    )
    hybrid_fusion_ms: float = Field(
        default=0.0, ge=0.0, description="Hybrid fusion duration (ms)"
    )
    clinical_rules_ms: float = Field(
        default=0.0, ge=0.0, description="Clinical rule engine duration (ms)"
    )
    xai_layer_ms: float = Field(
        default=0.0, ge=0.0, description="XAI / SHAP layer duration (ms)"
    )
    total_ms: float = Field(
        default=0.0, ge=0.0, description="Total inference duration excluding OCR (ms)"
    )

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# PredictionResponse — Req 10, 16.4, 19.4, 21.2, 22.1, 24.1–24.3
# ---------------------------------------------------------------------------


class PredictionResponse(BaseModel):
    """Structured JSON response for a single-disease v2 prediction.

    Core fields (Req 10.1–10.8)
    ---------------------------
    disease:
        One of "diabetes", "cvd", "ckd" (Req 10.2).
    risk_score:
        Numeric value in [0, 100] representing predicted disease probability
        as a percentage (Req 10.3).
    confidence:
        Model confidence in [0, 1] (Req 10.4).
    risk_level:
        Categorical label derived from ``risk_score`` (Req 10.5):
          - "Low"      → risk_score ∈ [0, 30]
          - "Moderate" → risk_score ∈ (30, 60]
          - "High"     → risk_score ∈ (60, 80]
          - "Critical" → risk_score ∈ (80, 100]
    explanation:
        Non-empty human-readable SHAP explanation string (Req 10.6).
    shap_features:
        Exactly 3 SHAPFeature objects sorted by descending |shap_value|
        (Req 10.7, 18.5).
    triggered_rules:
        List of clinical rule description strings that were triggered, or an
        empty list (Req 10.8).

    Extended fields
    ---------------
    threshold_used:
        Optimal Youden's Index threshold applied for binary classification
        (Req 16.4).
    missing_flags:
        Dict mapping each disease-specific feature name to a boolean
        indicating whether the value was imputed from the population median
        (Req 24.2).
    robustness_score:
        Stability metric in [0, 1] defined as
        ``1 − (mean_risk_change_at_5pct / 100)`` (Req 19.4, 24.3).
    latency:
        Per-stage latency breakdown in milliseconds (Req 21.2).
    disclaimer:
        Non-dismissible ethics disclaimer string (Req 22.1).
    limitations:
        List of standardised limitation disclaimer strings (Req 24.1).
    """

    # ------------------------------------------------------------------
    # Core fields — Req 10.1–10.8
    # ------------------------------------------------------------------

    disease: Literal["diabetes", "cvd", "ckd"] = Field(
        description="Disease identifier (Req 10.2)"
    )
    risk_score: float = Field(
        ge=0.0,
        le=100.0,
        description="Predicted disease risk as a percentage in [0, 100] (Req 10.3)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence in [0, 1] (Req 10.4)",
    )
    risk_level: Literal["Low", "Moderate", "High", "Critical"] = Field(
        description=(
            "Categorical risk label derived from risk_score: "
            "Low (0–30), Moderate (31–60), High (61–80), Critical (81–100) (Req 10.5)"
        )
    )
    explanation: str = Field(
        min_length=1,
        description="Human-readable SHAP explanation string (Req 10.6)",
    )
    shap_features: list[SHAPFeature] = Field(
        description="Exactly 3 SHAP feature attributions sorted by |shap_value| (Req 10.7, 18.5)"
    )
    triggered_rules: list[str] = Field(
        default_factory=list,
        description="Clinical rules triggered for this disease, or empty list (Req 10.8)",
    )

    # ------------------------------------------------------------------
    # Extended fields — Req 16.4, 19.4, 21.2, 22.1, 24.1–24.3
    # ------------------------------------------------------------------

    threshold_used: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Optimal Youden's Index threshold applied for binary classification (Req 16.4)",
    )
    missing_flags: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Per-feature imputation flags: True when the value was imputed "
            "from the population median (Req 24.2)"
        ),
    )
    robustness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Model stability score in [0, 1]: "
            "1 − (mean_risk_change_at_5pct / 100) (Req 19.4, 24.3)"
        ),
    )
    latency: LatencyBreakdown = Field(
        default_factory=LatencyBreakdown,
        description="Per-stage inference latency in milliseconds (Req 21.2)",
    )
    disclaimer: str = Field(
        default=(
            "This system is NOT a diagnostic tool. Predictions are for clinical "
            "decision support only and must be validated by a licensed medical professional."
        ),
        min_length=1,
        description="Non-dismissible ethics disclaimer (Req 22.1)",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Standardised limitation disclaimer strings (Req 24.1)",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def derive_risk_level(cls, data: dict) -> dict:
        """Derive ``risk_level`` from ``risk_score`` when not explicitly provided.

        Thresholds (Req 10.5):
          - Low      → risk_score ∈ [0, 30]
          - Moderate → risk_score ∈ (30, 60]
          - High     → risk_score ∈ (60, 80]
          - Critical → risk_score ∈ (80, 100]

        If ``risk_level`` is already set in the input data it is left
        unchanged (allows callers to override if needed).
        """
        if "risk_level" not in data or data.get("risk_level") is None:
            score = float(data.get("risk_score", 0))
            if score <= 30.0:
                data["risk_level"] = "Low"
            elif score <= 60.0:
                data["risk_level"] = "Moderate"
            elif score <= 80.0:
                data["risk_level"] = "High"
            else:
                data["risk_level"] = "Critical"
        return data

    @field_validator("shap_features")
    @classmethod
    def must_have_three_shap_features(cls, v: list[SHAPFeature]) -> list[SHAPFeature]:
        """Enforce exactly 3 items in ``shap_features`` (Req 18.5).

        Raises
        ------
        ValueError
            If the list does not contain exactly 3 SHAPFeature objects.
        """
        if len(v) != 3:
            raise ValueError(
                f"shap_features must contain exactly 3 items, got {len(v)}"
            )
        return v

    @field_validator("explanation")
    @classmethod
    def explanation_must_be_non_empty(cls, v: str) -> str:
        """Ensure the explanation string is non-empty (Req 10.6)."""
        if not v or not v.strip():
            raise ValueError("explanation must be a non-empty string")
        return v

    model_config = {
        "populate_by_name": True,
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "disease": "diabetes",
                "risk_score": 72.5,
                "confidence": 0.84,
                "risk_level": "High",
                "explanation": "High Blood Glucose and HbA1c increased Diabetes risk; high BMI also contributed.",
                "shap_features": [
                    {"feature_name": "glucose", "shap_value": 12.3, "direction": "increases risk"},
                    {"feature_name": "hba1c", "shap_value": 8.7, "direction": "increases risk"},
                    {"feature_name": "bmi", "shap_value": -3.1, "direction": "decreases risk"},
                ],
                "triggered_rules": ["Fasting glucose > 126 mg/dL"],
                "threshold_used": 0.42,
                "missing_flags": {"glucose": False, "hba1c": False, "bmi": True, "age": False, "systolic_bp": False, "diastolic_bp": False},
                "robustness_score": 0.91,
                "latency": {
                    "feature_extraction_ms": 12.4,
                    "scaler_normalization_ms": 1.1,
                    "classical_ml_ms": 45.2,
                    "vqc_ms": 320.0,
                    "hybrid_fusion_ms": 0.8,
                    "clinical_rules_ms": 0.3,
                    "xai_layer_ms": 210.5,
                    "total_ms": 590.3,
                },
                "disclaimer": (
                    "This system is NOT a diagnostic tool. Predictions are for clinical "
                    "decision support only and must be validated by a licensed medical professional."
                ),
                "limitations": [
                    "Predictions are based on datasets that may not represent all demographic groups equally.",
                    "Training data is limited in size and may not capture rare clinical presentations.",
                    "Quantum predictions are produced by a classical simulation of a quantum circuit and do not run on physical quantum hardware.",
                ],
            }
        },
    }
