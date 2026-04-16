"""Recommendation engine router.

GET /api/recommendations/{prediction_id}

Fetches or generates personalized health recommendations for a given
prediction. Recommendations are persisted to MongoDB on first generation
and returned from cache on subsequent requests.

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
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
from backend.models.recommendation import Recommendation
from backend.services.recommendation_service import generate_recommendations

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class RecommendationResponse(BaseModel):
    prediction_id: str
    generated_at: datetime
    recommendations: list[Recommendation]

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/{prediction_id}", response_model=RecommendationResponse)
async def get_recommendations(
    prediction_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get personalized recommendations for a prediction.

    - Returns cached recommendations if already generated.
    - Otherwise runs the full rule-based + LLM pipeline and persists results.
    - Patients can only access their own predictions (HTTP 403 otherwise).

    Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
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

    # ── Ownership check ─────────────────────────────────────────────────────
    user_oid = ObjectId(str(current_user["_id"]))
    user_role = current_user.get("role", "patient")
    if user_role == "patient" and pred_doc.get("user_id") != user_oid:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied.",
        )

    # ── Return cached recommendations if available ──────────────────────────
    existing = await db["recommendations"].find(
        {"prediction_id": prediction_id}
    ).to_list(length=100)

    if existing:
        recs = [
            Recommendation(
                prediction_id=doc["prediction_id"],
                disease=doc["disease"],
                text=doc["text"],
                priority=doc["priority"],
                source=doc["source"],
                requires_physician=doc.get("requires_physician", False),
            )
            for doc in existing
        ]
        recs.sort(key=lambda r: r.priority)
        return RecommendationResponse(
            prediction_id=prediction_id,
            generated_at=existing[0].get("generated_at", datetime.utcnow()),
            recommendations=recs,
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

    feature_vector: list[float] = fv_doc["features"]

    # ── Fetch lifestyle profile (optional) ──────────────────────────────────
    lifestyle_profile = None
    try:
        from backend.models.lifestyle import LifestyleProfile

        lp_doc = await db["lifestyle_profiles"].find_one(
            {"user_id": pred_doc.get("user_id")}
        )
        if lp_doc:
            lifestyle_profile = LifestyleProfile.model_validate(lp_doc)
    except Exception as exc:
        logger.warning("Could not load lifestyle profile: %s", exc)

    # ── Generate recommendations in thread pool ──────────────────────────────
    loop = asyncio.get_event_loop()
    try:
        recs: list[Recommendation] = await loop.run_in_executor(
            None,
            generate_recommendations,
            pred_doc,
            feature_vector,
            lifestyle_profile,
        )
    except Exception as exc:
        logger.error("Recommendation generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations.",
        )

    # ── Persist to MongoDB ───────────────────────────────────────────────────
    generated_at = datetime.utcnow()
    if recs:
        docs = [
            {
                "prediction_id": prediction_id,
                "disease": r.disease,
                "text": r.text,
                "priority": r.priority,
                "source": r.source,
                "requires_physician": r.requires_physician,
                "generated_at": generated_at,
            }
            for r in recs
        ]
        await db["recommendations"].insert_many(docs)

    return RecommendationResponse(
        prediction_id=prediction_id,
        generated_at=generated_at,
        recommendations=recs,
    )
