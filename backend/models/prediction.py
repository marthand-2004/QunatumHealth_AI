"""FeatureVector and PredictionResult data models."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend.models.common import PyObjectId

FEATURE_NAMES: list[str] = [
    "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
    "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
    "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level",
]
FEATURE_DIM = 14


class FeatureVector(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    document_id: PyObjectId
    features: list[float]              # exactly 14 dimensions
    feature_names: list[str] = Field(default_factory=lambda: FEATURE_NAMES.copy())
    constructed_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("features")
    @classmethod
    def must_be_14_dimensional(cls, v: list[float]) -> list[float]:
        if len(v) != FEATURE_DIM:
            raise ValueError(f"FeatureVector must have exactly {FEATURE_DIM} dimensions, got {len(v)}")
        return v

    model_config = {"populate_by_name": True}


class PredictionResult(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    feature_vector_id: PyObjectId
    model_used: Literal["quantum", "classical"]
    risk_scores: dict[str, float]      # {"diabetes": 42.3, "cvd": 67.1, "ckd": 18.9}
    quantum_scores: Optional[dict[str, float]] = None
    classical_scores: Optional[dict[str, float]] = None
    shap_values: Optional[dict[str, list[float]]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True, "protected_namespaces": ()}
