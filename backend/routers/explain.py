"""XAI / SHAP explanation router.

POST /api/explain/{prediction_id}

Returns SHAP values per feature, Chart.js waterfall chart data, and an
LLM-generated natural language summary for a given prediction.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from backend.core.database import get_db
from backend.core.deps import get_current_user
from backend.models.prediction import FEATURE_NAMES
from backend.services.xai_service import (
    build_waterfall_chart_data,
    compute_shap_values,
    generate_llm_explanation,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

EXPLANATION_TIMEOUT = 10.0  # seconds — Requirement 7.4


class DiseaseExplanation(BaseModel):
    disease: str
    shap_values: list[float]
    base_value: float
    waterfall_chart: dict
    model_config = {"protected_namespaces": ()}


class ExplanationResponse(BaseModel):
    prediction_id: str
    model_used: str
    risk_scores: dict[str, float]
    feature_names: list[str]
    explanations: list[DiseaseExplanation]
    llm_summary: str
    computed_at: datetime

    model_config = {"protected_namespaces": ()}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/{prediction_id}", response_model=ExplanationResponse)
async def explain(
    prediction_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Compute SHAP explanation for a stored prediction.

    - Fetches the prediction and its feature vector from MongoDB.
    - Computes SHAP values for each disease (Diabetes, CVD, CKD).
    - Builds Chart.js waterfall chart data per disease.
    - Generates an LLM natural language summary.
    - Persists SHAP values back onto the prediction document.
    - Completes within 10 seconds (Requirement 7.4).

    When the prediction used the classical fallback (model_used="classical"),
    explanations are attributed to the Classical ML model (Requirement 7.5).

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    # ── Validate prediction_id ──────────────────────────────────────────────
    try:
        pred_oid = ObjectId(prediction_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid prediction_id format.",
        )

    # ── Fetch prediction ────────────────────────────────────────────────────
    pred_doc = await db["predictions"].find_one({"_id": pred_oid})
    if pred_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found.",
        )

    # ── Ownership check (patients can only explain their own predictions) ───
    user_oid = ObjectId(str(current_user["_id"]))
    user_role = current_user.get("role", "patient")
    if user_role == "patient" and pred_doc.get("user_id") != user_oid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied.",
        )

    # ── Fetch feature vector ────────────────────────────────────────────────
    fv_oid = pred_doc.get("feature_vector_id")
    if fv_oid is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature vector not found for this prediction.",
        )

    fv_doc = await db["feature_vectors"].find_one({"_id": fv_oid})
    if fv_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"FeatureVector {fv_oid} not found.",
        )

    features: list[float] = fv_doc["features"]
    feature_names: list[str] = fv_doc.get("feature_names", FEATURE_NAMES)
    risk_scores: dict[str, float] = pred_doc.get("risk_scores", {})
    model_used: str = pred_doc.get("model_used", "classical")

    # ── Compute explanations within 10-second timeout ───────────────────────
    try:
        result = await asyncio.wait_for(
            _compute_explanations(features, feature_names, risk_scores, model_used),
            timeout=EXPLANATION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Explanation computation timed out (>10 s).",
        )

    explanations, llm_summary, shap_by_disease = result

    # ── Persist SHAP values back onto the prediction document ───────────────
    await db["predictions"].update_one(
        {"_id": pred_oid},
        {"$set": {"shap_values": shap_by_disease}},
    )

    return ExplanationResponse(
        prediction_id=prediction_id,
        model_used=model_used,
        risk_scores=risk_scores,
        feature_names=feature_names,
        explanations=explanations,
        llm_summary=llm_summary,
        computed_at=datetime.utcnow(),
    )


# ---------------------------------------------------------------------------
# Internal async helper
# ---------------------------------------------------------------------------


async def _compute_explanations(
    features: list[float],
    feature_names: list[str],
    risk_scores: dict[str, float],
    model_used: str,
) -> tuple[list[DiseaseExplanation], str, dict[str, list[float]]]:
    """Run SHAP computation and LLM explanation in a thread pool executor."""
    loop = asyncio.get_event_loop()

    def _sync_compute():
        diseases = list(risk_scores.keys()) or ["diabetes", "cvd", "ckd"]
        explanations: list[DiseaseExplanation] = []
        shap_by_disease: dict[str, list[float]] = {}

        # Use classical attribution when fallback is active (Requirement 7.5)
        effective_model = "classical" if model_used == "classical" else model_used

        # Pick the disease with the highest risk for the primary SHAP values
        # used in the LLM summary
        primary_disease = max(risk_scores, key=lambda d: risk_scores[d]) if risk_scores else "diabetes"
        primary_shap: list[float] = []

        for disease in diseases:
            shap_vals = compute_shap_values(features, disease, effective_model)
            shap_by_disease[disease] = shap_vals

            risk_score = risk_scores.get(disease, 50.0)
            base_value = risk_score - sum(shap_vals)

            waterfall = build_waterfall_chart_data(
                feature_names, shap_vals, base_value, risk_score
            )

            explanations.append(
                DiseaseExplanation(
                    disease=disease,
                    shap_values=shap_vals,
                    base_value=round(base_value, 4),
                    waterfall_chart=waterfall,
                )
            )

            if disease == primary_disease:
                primary_shap = shap_vals

        llm_summary = generate_llm_explanation(
            feature_names, primary_shap, risk_scores, effective_model
        )

        return explanations, llm_summary, shap_by_disease

    return await loop.run_in_executor(None, _sync_compute)
