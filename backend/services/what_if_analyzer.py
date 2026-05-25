"""What-If Analyzer — recompute risk scores with modified patient inputs.

Accepts a reference to an existing feature vector and a set of modified
inputs (e.g. BMI, exercise_frequency, smoking_encoded).  Re-runs the full
v2 inference pipeline with the modified values and returns both the original
and new risk scores along with a human-readable summary.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from backend.services.clinical_rule_engine import apply_rules
from backend.services.classical_ml_v2 import predict_classical_v2
from backend.services.data_governance import DataGovernanceModule
from backend.services.feature_mapper import DISEASE_FEATURE_SUBSETS, FeatureMapper
from backend.services.hybrid_fusion import hybrid_fusion
from backend.services.quantum_engine_v2 import QuantumTimeoutError, predict_quantum_v2
from backend.services.xai_service_v2 import compute_shap_v2_with_explanation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISEASES = ("diabetes", "cvd", "ckd")

# Timeout for the entire what-if recomputation (seconds) — Req 8.6
_WHAT_IF_TIMEOUT_SECONDS = 3.0

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class WhatIfRequest(BaseModel):
    """Request model for a what-if analysis.

    Attributes
    ----------
    original_feature_vector_id:
        MongoDB ObjectId string referencing the stored feature vector document
        in the ``feature_vectors`` collection.  The original risk score is
        derived from this document.
    modified_inputs:
        Dict of feature name → new numeric value.  Accepted keys include any
        feature in the disease-specific subsets:
        BMI, exercise_frequency, smoking_encoded, glucose, hba1c, creatinine,
        hemoglobin, cholesterol, systolic_bp, diastolic_bp, age, etc.
        Only the keys present in this dict are overridden; all other features
        retain their original values from the stored feature vector.

    Requirements: 8.1
    """

    original_feature_vector_id: str
    modified_inputs: dict[str, float]


class WhatIfResponse(BaseModel):
    """Response model for a what-if analysis.

    Attributes
    ----------
    disease:
        One of "diabetes", "cvd", "ckd".
    original_risk_score:
        Risk score computed from the original (unmodified) feature vector.
    new_risk_score:
        Risk score recomputed with the modified inputs applied.
    delta:
        Signed difference: ``new_risk_score − original_risk_score``.
        Positive = risk increased, negative = risk decreased.
    summary:
        Human-readable description of the change, e.g.:
        "Risk reduced from 72 → 58 (−14 points)."

    Requirements: 8.3, 8.4, 8.5
    """

    disease: str
    original_risk_score: float
    new_risk_score: float
    delta: float
    summary: str


# ---------------------------------------------------------------------------
# Module-level singletons (loaded once at import time)
# ---------------------------------------------------------------------------

_governance = DataGovernanceModule()
_feature_mapper = FeatureMapper(medians=_governance.medians)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_summary(
    disease: str,
    original_score: float,
    new_score: float,
    delta: float,
) -> str:
    """Generate a human-readable summary of the risk score change.

    Covers three cases (Req 8.4, 8.5):
    - Risk decreased: "Risk reduced from 72 → 58 (−14 points)."
    - Risk increased: "Risk increased from 58 → 72 (+14 points)."
    - No change:      "No change detected: risk remains at 65 points."

    Parameters
    ----------
    disease:
        One of "diabetes", "cvd", "ckd".
    original_score:
        Original risk score in [0, 100].
    new_score:
        Recomputed risk score in [0, 100].
    delta:
        Signed difference (new − original).

    Returns
    -------
    str
        Non-empty human-readable summary string.
    """
    orig_rounded = round(original_score)
    new_rounded = round(new_score)
    delta_rounded = round(delta)

    # Human-readable disease label
    _disease_labels: dict[str, str] = {
        "diabetes": "Diabetes",
        "cvd": "CVD",
        "ckd": "CKD",
    }
    disease_label = _disease_labels.get(disease, disease.upper())

    if delta_rounded == 0:
        # Req 8.5: explicitly state no change
        return (
            f"No change detected: {disease_label} risk remains at "
            f"{orig_rounded} points."
        )

    if delta < 0:
        sign = "−"
        verb = "reduced"
    else:
        sign = "+"
        verb = "increased"

    abs_delta = abs(delta_rounded)
    return (
        f"Risk {verb} from {orig_rounded} → {new_rounded} "
        f"({sign}{abs_delta} points)."
    )


def _run_inference_pipeline(
    features: list[float],
    disease: str,
    raw_lab_values: dict[str, float],
) -> float:
    """Run the full v2 inference pipeline for a single disease and return risk score.

    Executes all pipeline stages:
    1. Classical ML (RF + XGBoost)
    2. Quantum VQC (with 800ms timeout fallback)
    3. Hybrid fusion
    4. Clinical rule engine

    Parameters
    ----------
    features:
        Disease-specific feature vector (already extracted and ordered).
    disease:
        One of "diabetes", "cvd", "ckd".
    raw_lab_values:
        Raw (pre-unit-conversion) lab values for clinical rule evaluation.

    Returns
    -------
    float
        Adjusted risk score in [0, 100] after clinical rules.
    """
    # Stage 1: Classical ML
    try:
        rf_prob, xgb_prob = predict_classical_v2(features, disease)
    except Exception as exc:
        logger.error(
            "what_if_analyzer: Classical ML failed for '%s': %s. "
            "Using fallback probabilities.",
            disease,
            exc,
        )
        rf_prob, xgb_prob = 0.5, 0.5

    # Stage 2: Quantum VQC (with timeout fallback)
    vqc_prob: Optional[float] = None
    try:
        vqc_prob = predict_quantum_v2(features, disease)
    except QuantumTimeoutError:
        logger.warning(
            "what_if_analyzer: VQC timeout for '%s' — using classical-only fusion.",
            disease,
        )
    except Exception as exc:
        logger.warning(
            "what_if_analyzer: VQC error for '%s' (%s) — using classical-only fusion.",
            disease,
            exc,
        )

    # Stage 3: Hybrid fusion
    risk_score = hybrid_fusion.fuse(
        classical_rf_prob=rf_prob,
        classical_xgb_prob=xgb_prob,
        quantum_prob=vqc_prob,
        disease=disease,
    )

    # Stage 4: Clinical rule engine
    adjusted_scores, _ = apply_rules(
        risk_scores={disease: risk_score},
        raw_lab_values=raw_lab_values,
    )
    return adjusted_scores[disease]


def _extract_features_for_disease(
    feature_names: list[str],
    features: list[float],
    modified_inputs: dict[str, float],
    disease: str,
) -> tuple[list[float], dict[str, float]]:
    """Build a disease-specific feature vector with modified inputs applied.

    Reconstructs the full lab_values + lifestyle dict from the stored feature
    vector, overlays the modified_inputs, then uses FeatureMapper to produce
    the disease-specific ordered vector.

    Parameters
    ----------
    feature_names:
        Ordered list of feature names from the stored feature vector document.
    features:
        Ordered list of feature values from the stored feature vector document.
    modified_inputs:
        Dict of feature name → new value to override.
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    (disease_features, combined_values)
        disease_features: Ordered disease-specific feature vector.
        combined_values:  Full dict of all feature values (original + modified),
                          used for clinical rule evaluation.
    """
    # Reconstruct the full feature dict from the stored vector
    combined: dict[str, float] = {
        name: float(val)
        for name, val in zip(feature_names, features)
        if isinstance(val, (int, float))
    }

    # Apply modified inputs on top (Req 8.1, 8.2)
    combined.update(modified_inputs)

    # Use FeatureMapper to extract the disease-specific ordered subset
    mapper_result = _feature_mapper.map_features(
        lab_values=combined,
        lifestyle=combined,  # lifestyle fields (bmi, age, etc.) may be in combined
        disease=disease,
    )

    return mapper_result.features, combined


def _compute_original_risk_score(
    feature_names: list[str],
    features: list[float],
    disease: str,
) -> float:
    """Compute the original risk score from the stored feature vector.

    Parameters
    ----------
    feature_names:
        Ordered list of feature names from the stored feature vector document.
    features:
        Ordered list of feature values from the stored feature vector document.
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    float
        Original risk score in [0, 100].
    """
    combined: dict[str, float] = {
        name: float(val)
        for name, val in zip(feature_names, features)
        if isinstance(val, (int, float))
    }

    mapper_result = _feature_mapper.map_features(
        lab_values=combined,
        lifestyle=combined,
        disease=disease,
    )

    return _run_inference_pipeline(
        features=mapper_result.features,
        disease=disease,
        raw_lab_values=combined,
    )


def _compute_what_if_sync(
    feature_names: list[str],
    features: list[float],
    modified_inputs: dict[str, float],
    disease: str,
) -> WhatIfResponse:
    """Synchronous core of the what-if computation (runs inside asyncio.wait_for).

    Parameters
    ----------
    feature_names:
        Ordered list of feature names from the stored feature vector document.
    features:
        Ordered list of feature values from the stored feature vector document.
    modified_inputs:
        Dict of feature name → new value to override.
    disease:
        One of "diabetes", "cvd", "ckd".

    Returns
    -------
    WhatIfResponse
        Contains original score, new score, delta, and human-readable summary.
    """
    # Compute original risk score
    original_score = _compute_original_risk_score(feature_names, features, disease)

    # Build modified feature vector and combined values
    modified_features, combined_values = _extract_features_for_disease(
        feature_names=feature_names,
        features=features,
        modified_inputs=modified_inputs,
        disease=disease,
    )

    # Compute new risk score with modified inputs
    new_score = _run_inference_pipeline(
        features=modified_features,
        disease=disease,
        raw_lab_values=combined_values,
    )

    # Compute delta and summary
    delta = new_score - original_score
    summary = _build_summary(disease, original_score, new_score, delta)

    return WhatIfResponse(
        disease=disease,
        original_risk_score=round(original_score, 2),
        new_risk_score=round(new_score, 2),
        delta=round(delta, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def compute_what_if(
    request: WhatIfRequest,
    disease: str,
    db: AsyncIOMotorDatabase,
) -> WhatIfResponse:
    """Recompute the risk score for a disease with modified patient inputs.

    Loads the original feature vector from MongoDB, applies the modified
    inputs on top of the original values, re-runs the full v2 inference
    pipeline (feature extraction → classical ML → VQC → hybrid fusion →
    clinical rules), and returns both the original and new risk scores with
    a human-readable summary.

    Completes within 3 seconds (Req 8.6).  Uses ``asyncio.wait_for`` to
    enforce the timeout.

    Parameters
    ----------
    request:
        ``WhatIfRequest`` containing the feature vector ID and modified inputs.
    disease:
        One of "diabetes", "cvd", "ckd".
    db:
        AsyncIOMotorDatabase instance (from FastAPI dependency injection).

    Returns
    -------
    WhatIfResponse
        Contains ``disease``, ``original_risk_score``, ``new_risk_score``,
        ``delta``, and ``summary``.

    Raises
    ------
    ValueError
        If the feature vector is not found in MongoDB, or if ``disease`` is
        not one of the supported disease identifiers.
    asyncio.TimeoutError
        If the recomputation exceeds 3 seconds (Req 8.6).

    Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6
    """
    if disease not in DISEASE_FEATURE_SUBSETS:
        raise ValueError(
            f"Unsupported disease '{disease}'. "
            f"Must be one of: {sorted(DISEASE_FEATURE_SUBSETS)}"
        )

    # ── Load original feature vector from MongoDB ────────────────────────────
    try:
        fv_oid = ObjectId(request.original_feature_vector_id)
    except Exception:
        raise ValueError(
            f"Invalid feature_vector_id: '{request.original_feature_vector_id}'"
        )

    fv_doc = await db["feature_vectors"].find_one({"_id": fv_oid})
    if fv_doc is None:
        raise ValueError(
            f"Feature vector '{request.original_feature_vector_id}' not found in MongoDB."
        )

    feature_names: list[str] = fv_doc.get("feature_names", [])
    features: list[float] = [float(v) for v in fv_doc.get("features", [])]

    if not feature_names or not features:
        raise ValueError(
            f"Feature vector '{request.original_feature_vector_id}' has no features stored."
        )

    # ── Run recomputation with 3-second timeout (Req 8.6) ───────────────────
    loop = asyncio.get_event_loop()

    async def _run_async() -> WhatIfResponse:
        return await loop.run_in_executor(
            None,
            _compute_what_if_sync,
            feature_names,
            features,
            dict(request.modified_inputs),
            disease,
        )

    try:
        response = await asyncio.wait_for(
            _run_async(),
            timeout=_WHAT_IF_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "what_if_analyzer: compute_what_if for disease '%s' exceeded "
            "%ds timeout.",
            disease,
            _WHAT_IF_TIMEOUT_SECONDS,
        )
        raise asyncio.TimeoutError(
            f"What-if recomputation for '{disease}' exceeded "
            f"{_WHAT_IF_TIMEOUT_SECONDS}s timeout."
        )
    except Exception as exc:
        logger.error(
            "what_if_analyzer: Unexpected error for disease '%s': %s",
            disease,
            exc,
        )
        raise

    logger.info(
        "what_if_analyzer: '%s' — original=%.1f, new=%.1f, delta=%.1f",
        disease,
        response.original_risk_score,
        response.new_risk_score,
        response.delta,
    )

    return response


async def compute_what_if_all_diseases(
    request: WhatIfRequest,
    db: AsyncIOMotorDatabase,
) -> list[WhatIfResponse]:
    """Run what-if analysis for all three diseases concurrently.

    Convenience wrapper that calls ``compute_what_if`` for each disease
    (diabetes, cvd, ckd) and returns a list of ``WhatIfResponse`` objects.

    Parameters
    ----------
    request:
        ``WhatIfRequest`` containing the feature vector ID and modified inputs.
    db:
        AsyncIOMotorDatabase instance.

    Returns
    -------
    list[WhatIfResponse]
        One response per disease, in the order (diabetes, cvd, ckd).

    Raises
    ------
    ValueError
        If the feature vector is not found or has no features.
    asyncio.TimeoutError
        If any disease recomputation exceeds 3 seconds.
    """
    tasks = [
        compute_what_if(request, disease, db)
        for disease in DISEASES
    ]
    return list(await asyncio.gather(*tasks))
