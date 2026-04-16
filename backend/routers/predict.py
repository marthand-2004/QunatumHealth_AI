"""Prediction router — quantum, classical, combined, and feature vector construction."""
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
from backend.models.prediction import FeatureVector, PredictionResult
from backend.services.feature_vector_service import build_feature_vector
from backend.services.quantum_engine import predict_quantum
from backend.services.classical_ml import predict_classical

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FeatureVectorRequest(BaseModel):
    document_id: str


class QuantumPredictRequest(BaseModel):
    """Accept either a stored feature_vector_id or inline features."""
    feature_vector_id: Optional[str] = None
    features: Optional[list[float]] = None


class ClassicalPredictRequest(BaseModel):
    """Accept either a stored feature_vector_id or inline features."""
    feature_vector_id: Optional[str] = None
    features: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------

@router.post("/feature-vector", response_model=FeatureVector, status_code=status.HTTP_201_CREATED)
async def create_feature_vector(
    body: FeatureVectorRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Build and persist a 14-dimensional FeatureVector for the given document.

    Requirements: 5.1, 6.3, 2.5
    """
    user_id = str(current_user["_id"])
    try:
        fv = await build_feature_vector(db, user_id, body.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return fv


# ---------------------------------------------------------------------------
# Quantum prediction endpoint
# ---------------------------------------------------------------------------

@router.post("/quantum", response_model=PredictionResult, status_code=status.HTTP_201_CREATED)
async def predict_quantum_endpoint(
    body: QuantumPredictRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Run VQC quantum prediction and persist the result.

    Accepts either:
    - ``feature_vector_id``: fetches the stored FeatureVector from MongoDB, or
    - ``features``: inline 14-dimensional list.

    Returns a PredictionResult with risk_scores for Diabetes, CVD, and CKD.
    Completes within 15 seconds (Requirement 5.4).

    Requirements: 5.2, 5.3, 5.4, 5.6, 5.7
    """
    user_id = str(current_user["_id"])
    user_oid = ObjectId(user_id)

    # ── Resolve feature vector ──────────────────────────────────────────────
    if body.feature_vector_id is not None:
        fv_doc = await db["feature_vectors"].find_one(
            {"_id": ObjectId(body.feature_vector_id), "user_id": user_oid}
        )
        if fv_doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"FeatureVector {body.feature_vector_id} not found.",
            )
        features: list[float] = fv_doc["features"]
        fv_oid = fv_doc["_id"]
    elif body.features is not None:
        if len(body.features) != 14:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="features must have exactly 14 dimensions.",
            )
        features = body.features
        # Persist inline features as a new FeatureVector document
        fv_insert = {
            "user_id": user_oid,
            "document_id": None,
            "features": features,
            "feature_names": [
                "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
                "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
                "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level",
            ],
            "constructed_at": datetime.utcnow(),
        }
        fv_result = await db["feature_vectors"].insert_one(fv_insert)
        fv_oid = fv_result.inserted_id
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either feature_vector_id or features.",
        )

    # ── Run quantum prediction with 15-second timeout ───────────────────────
    try:
        risk_scores: dict[str, float] = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, predict_quantum, features),
            timeout=15.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Quantum prediction timed out (>15 s). Use /api/predict/classical.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quantum engine error: {exc}",
        )

    # ── Persist PredictionResult ────────────────────────────────────────────
    now = datetime.utcnow()
    pred_doc = {
        "user_id": user_oid,
        "feature_vector_id": fv_oid,
        "model_used": "quantum",
        "risk_scores": risk_scores,
        "quantum_scores": risk_scores,
        "classical_scores": None,
        "shap_values": None,
        "timestamp": now,
    }
    insert_result = await db["predictions"].insert_one(pred_doc)
    pred_doc["_id"] = insert_result.inserted_id

    # ── High-risk alert creation (Task 14.2, Req 10.6) ─────────────────────
    if any(score > 75 for score in risk_scores.values()):
        await db["alerts"].insert_one({
            "patient_id": user_oid,
            "prediction_id": insert_result.inserted_id,
            "risk_scores": risk_scores,
            "timestamp": now,
        })

    return PredictionResult(
        id=str(insert_result.inserted_id),
        user_id=user_id,
        feature_vector_id=str(fv_oid),
        model_used="quantum",
        risk_scores=risk_scores,
        quantum_scores=risk_scores,
        classical_scores=None,
        shap_values=None,
        timestamp=now,
    )


# ---------------------------------------------------------------------------
# Classical prediction endpoint
# ---------------------------------------------------------------------------

@router.post("/classical", response_model=PredictionResult, status_code=status.HTTP_201_CREATED)
async def predict_classical_endpoint(
    body: ClassicalPredictRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Run RF+XGBoost ensemble prediction and persist the result.

    Accepts either:
    - ``feature_vector_id``: fetches the stored FeatureVector from MongoDB, or
    - ``features``: inline 14-dimensional list.

    Returns a PredictionResult with risk_scores for Diabetes, CVD, and CKD.
    Completes within 2 seconds (Requirement 6.2).

    Requirements: 6.1, 6.2, 6.3
    """
    user_id = str(current_user["_id"])
    user_oid = ObjectId(user_id)

    # ── Resolve feature vector ──────────────────────────────────────────────
    if body.feature_vector_id is not None:
        fv_doc = await db["feature_vectors"].find_one(
            {"_id": ObjectId(body.feature_vector_id), "user_id": user_oid}
        )
        if fv_doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"FeatureVector {body.feature_vector_id} not found.",
            )
        features: list[float] = fv_doc["features"]
        fv_oid = fv_doc["_id"]
    elif body.features is not None:
        if len(body.features) != 14:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="features must have exactly 14 dimensions.",
            )
        features = body.features
        # Persist inline features as a new FeatureVector document
        fv_insert = {
            "user_id": user_oid,
            "document_id": None,
            "features": features,
            "feature_names": [
                "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
                "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
                "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level",
            ],
            "constructed_at": datetime.utcnow(),
        }
        fv_result = await db["feature_vectors"].insert_one(fv_insert)
        fv_oid = fv_result.inserted_id
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either feature_vector_id or features.",
        )

    # ── Run classical prediction with 2-second timeout ──────────────────────
    try:
        risk_scores: dict[str, float] = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, predict_classical, features),
            timeout=2.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Classical prediction timed out (>2 s).",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classical ML engine error: {exc}",
        )

    # ── Persist PredictionResult ────────────────────────────────────────────
    now = datetime.utcnow()
    pred_doc = {
        "user_id": user_oid,
        "feature_vector_id": fv_oid,
        "model_used": "classical",
        "risk_scores": risk_scores,
        "quantum_scores": None,
        "classical_scores": risk_scores,
        "shap_values": None,
        "timestamp": now,
    }
    insert_result = await db["predictions"].insert_one(pred_doc)

    # ── High-risk alert creation (Task 14.2, Req 10.6) ─────────────────────
    if any(score > 75 for score in risk_scores.values()):
        await db["alerts"].insert_one({
            "patient_id": user_oid,
            "prediction_id": insert_result.inserted_id,
            "risk_scores": risk_scores,
            "timestamp": now,
        })

    return PredictionResult(
        id=str(insert_result.inserted_id),
        user_id=user_id,
        feature_vector_id=str(fv_oid),
        model_used="classical",
        risk_scores=risk_scores,
        quantum_scores=None,
        classical_scores=risk_scores,
        shap_values=None,
        timestamp=now,
    )


# ---------------------------------------------------------------------------
# Combined prediction schemas
# ---------------------------------------------------------------------------

class CombinedPredictRequest(BaseModel):
    """Accept either a stored feature_vector_id or inline features."""
    feature_vector_id: Optional[str] = None
    features: Optional[list[float]] = None


class CombinedPredictionResponse(BaseModel):
    id: str
    user_id: str
    feature_vector_id: str
    model_used: str
    risk_scores: dict[str, float]
    quantum_scores: Optional[dict[str, float]]
    classical_scores: dict[str, float]
    fallback_used: bool
    timestamp: datetime

    model_config = {"protected_namespaces": ()}


# ---------------------------------------------------------------------------
# Combined prediction endpoint — Task 9.1
# Requirements: 5.5, 6.4, 6.5, 14.2
# ---------------------------------------------------------------------------

QUANTUM_TIMEOUT = 15.0  # seconds


@router.post("/", response_model=CombinedPredictionResponse, status_code=status.HTTP_201_CREATED)
async def predict_combined(
    body: CombinedPredictRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Run quantum + classical predictions in parallel with quantum fallback.

    - Both models run concurrently via asyncio.gather.
    - If quantum raises an exception or times out (>15 s), falls back to
      classical only: model_used="classical", fallback_used=True.
    - If both succeed: model_used="quantum", fallback_used=False,
      both quantum_scores and classical_scores are present.
    - Persists the combined PredictionResult to MongoDB.
    - If any risk score > 75, inserts an alert document into the `alerts`
      collection.

    Requirements: 5.5, 6.4, 6.5, 14.2
    """
    user_id = str(current_user["_id"])
    user_oid = ObjectId(user_id)

    # ── Resolve feature vector ──────────────────────────────────────────────
    if body.feature_vector_id is not None:
        fv_doc = await db["feature_vectors"].find_one(
            {"_id": ObjectId(body.feature_vector_id), "user_id": user_oid}
        )
        if fv_doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"FeatureVector {body.feature_vector_id} not found.",
            )
        features: list[float] = fv_doc["features"]
        fv_oid = fv_doc["_id"]
    elif body.features is not None:
        if len(body.features) != 14:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="features must have exactly 14 dimensions.",
            )
        features = body.features
        fv_insert = {
            "user_id": user_oid,
            "document_id": None,
            "features": features,
            "feature_names": [
                "glucose", "hba1c", "creatinine", "cholesterol", "triglycerides",
                "hemoglobin", "bmi", "age", "systolic_bp", "diastolic_bp",
                "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level",
            ],
            "constructed_at": datetime.utcnow(),
        }
        fv_result = await db["feature_vectors"].insert_one(fv_insert)
        fv_oid = fv_result.inserted_id
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either feature_vector_id or features.",
        )

    # ── Run quantum + classical concurrently ────────────────────────────────
    loop = asyncio.get_event_loop()

    async def _run_quantum() -> dict[str, float]:
        return await asyncio.wait_for(
            loop.run_in_executor(None, predict_quantum, features),
            timeout=QUANTUM_TIMEOUT,
        )

    async def _run_classical() -> dict[str, float]:
        return await loop.run_in_executor(None, predict_classical, features)

    quantum_result, classical_result = await asyncio.gather(
        _run_quantum(),
        _run_classical(),
        return_exceptions=True,
    )

    # ── Determine classical scores (must always succeed) ────────────────────
    if isinstance(classical_result, BaseException):
        # Classical ML should never fail; propagate as 500
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classical ML engine error: {classical_result}",
        )
    classical_scores: dict[str, float] = classical_result  # type: ignore[assignment]

    # ── Determine quantum outcome and final model_used ───────────────────────
    quantum_failed = isinstance(quantum_result, BaseException)
    if quantum_failed:
        logger.warning("Quantum prediction failed, using classical fallback: %s", quantum_result)
        quantum_scores: Optional[dict[str, float]] = None
        risk_scores = classical_scores
        model_used = "classical"
        fallback_used = True
    else:
        quantum_scores = quantum_result  # type: ignore[assignment]
        risk_scores = quantum_scores
        model_used = "quantum"
        fallback_used = False

    # ── Persist PredictionResult ────────────────────────────────────────────
    now = datetime.utcnow()
    pred_doc = {
        "user_id": user_oid,
        "feature_vector_id": fv_oid,
        "model_used": model_used,
        "risk_scores": risk_scores,
        "quantum_scores": quantum_scores,
        "classical_scores": classical_scores,
        "shap_values": None,
        "timestamp": now,
    }
    insert_result = await db["predictions"].insert_one(pred_doc)
    prediction_id = insert_result.inserted_id

    # ── High-risk alert creation (any score > 75) ───────────────────────────
    if any(score > 75 for score in risk_scores.values()):
        alert_doc = {
            "patient_id": user_oid,
            "prediction_id": prediction_id,
            "risk_scores": risk_scores,
            "timestamp": now,
        }
        await db["alerts"].insert_one(alert_doc)

    return CombinedPredictionResponse(
        id=str(prediction_id),
        user_id=user_id,
        feature_vector_id=str(fv_oid),
        model_used=model_used,
        risk_scores=risk_scores,
        quantum_scores=quantum_scores,
        classical_scores=classical_scores,
        fallback_used=fallback_used,
        timestamp=now,
    )
