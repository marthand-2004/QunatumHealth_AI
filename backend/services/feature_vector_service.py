"""Feature vector construction service.

Builds a 14-dimensional FeatureVector from verified lab values and a
lifestyle profile, applies population-mean imputation for missing features,
and persists the result to MongoDB.

Requirements: 5.1, 6.3, 2.5
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.models.prediction import FEATURE_NAMES, FEATURE_DIM, FeatureVector

# ---------------------------------------------------------------------------
# Smoking status encoding
# ---------------------------------------------------------------------------
SMOKING_ENCODING: dict[str, float] = {
    "never": 0.0,
    "former": 0.5,
    "current": 1.0,
}

# ---------------------------------------------------------------------------
# Population mean defaults for missing features
# Values are clinically reasonable population averages in SI / normalized units
# ---------------------------------------------------------------------------
POPULATION_MEANS: dict[str, float] = {
    "glucose":            5.5,    # mmol/L  (normal fasting)
    "hba1c":              5.4,    # %
    "creatinine":         80.0,   # µmol/L
    "cholesterol":        5.0,    # mmol/L
    "triglycerides":      1.3,    # mmol/L
    "hemoglobin":         14.0,   # g/dL
    "bmi":                25.0,   # kg/m²
    "age":                45.0,   # years
    "systolic_bp":        120.0,  # mmHg
    "diastolic_bp":       80.0,   # mmHg
    "smoking_encoded":    0.0,    # never
    "exercise_frequency": 3.0,    # days/week
    "sleep_hours":        7.0,    # hours/night
    "stress_level":       5.0,    # 1-10 scale
}

# Lab parameter names that map directly to feature names
_LAB_FEATURE_MAP: dict[str, str] = {
    "glucose":          "glucose",
    "hba1c":            "hba1c",
    "creatinine":       "creatinine",
    "total_cholesterol": "cholesterol",
    "cholesterol":      "cholesterol",
    "triglycerides":    "triglycerides",
    "hemoglobin":       "hemoglobin",
}


def _extract_lab_values(lab_parameters: list[dict]) -> dict[str, float]:
    """Map a list of lab parameter dicts to feature-name → value."""
    values: dict[str, float] = {}
    for param in lab_parameters:
        name = param.get("name", "")
        feature = _LAB_FEATURE_MAP.get(name)
        if feature and feature not in values:
            values[feature] = float(param["value"])
    return values


def _encode_smoking(smoking_status: Optional[str]) -> float:
    """Encode smoking status string to numeric value."""
    if smoking_status is None:
        return POPULATION_MEANS["smoking_encoded"]
    return SMOKING_ENCODING.get(smoking_status, POPULATION_MEANS["smoking_encoded"])


def build_feature_array(
    lab_values: dict[str, float],
    lifestyle: Optional[dict],
) -> list[float]:
    """Construct the 14-element feature list from lab values and lifestyle data.

    Missing features are filled with population mean defaults.
    """
    # Start with population means as defaults
    feature_map: dict[str, float] = dict(POPULATION_MEANS)

    # Override with actual lab values
    feature_map.update(lab_values)

    # Override with lifestyle profile fields
    if lifestyle:
        if lifestyle.get("bmi") is not None:
            feature_map["bmi"] = float(lifestyle["bmi"])
        if lifestyle.get("exercise_frequency") is not None:
            feature_map["exercise_frequency"] = float(lifestyle["exercise_frequency"])
        if lifestyle.get("sleep_hours") is not None:
            feature_map["sleep_hours"] = float(lifestyle["sleep_hours"])
        if lifestyle.get("stress_level") is not None:
            feature_map["stress_level"] = float(lifestyle["stress_level"])
        feature_map["smoking_encoded"] = _encode_smoking(lifestyle.get("smoking_status"))

    # Build ordered list matching FEATURE_NAMES
    return [feature_map[name] for name in FEATURE_NAMES]


async def build_feature_vector(
    db: AsyncIOMotorDatabase,
    user_id: str,
    document_id: str,
) -> FeatureVector:
    """Fetch verified lab params and lifestyle profile, build and persist FeatureVector.

    Steps:
    1. Load the verified Document from MongoDB.
    2. Load the LifestyleProfile for the user (if present).
    3. Construct the 14-dim feature array with imputation for missing values.
    4. Persist the FeatureVector to the ``feature_vectors`` collection.
    5. Return the FeatureVector.

    Requirements: 5.1, 6.3, 2.5
    """
    doc_oid = ObjectId(document_id)
    user_oid = ObjectId(user_id)

    # 1. Fetch document
    document = await db["documents"].find_one({"_id": doc_oid, "user_id": user_oid})
    if document is None:
        raise ValueError(f"Document {document_id} not found for user {user_id}")

    lab_parameters: list[dict] = document.get("lab_parameters", [])
    lab_values = _extract_lab_values(lab_parameters)

    # 2. Fetch lifestyle profile (optional — imputation handles absence)
    lifestyle = await db["lifestyle_profiles"].find_one({"user_id": user_oid})

    # 3. Build feature array
    features = build_feature_array(lab_values, lifestyle)

    # 4. Persist to MongoDB
    fv_doc = {
        "user_id": user_oid,
        "document_id": doc_oid,
        "features": features,
        "feature_names": FEATURE_NAMES,
        "constructed_at": datetime.utcnow(),
    }
    result = await db["feature_vectors"].insert_one(fv_doc)
    fv_doc["_id"] = result.inserted_id

    # 5. Return as FeatureVector model
    return FeatureVector(
        id=str(result.inserted_id),
        user_id=str(user_oid),
        document_id=str(doc_oid),
        features=features,
        feature_names=FEATURE_NAMES,
        constructed_at=fv_doc["constructed_at"],
    )
