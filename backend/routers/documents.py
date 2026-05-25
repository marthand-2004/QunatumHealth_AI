"""Document intelligence router — verify and retrieve.

Requirements: 4.5, 4.6, 4.7
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from backend.core.database import get_db
from backend.core.deps import require_role
from backend.models.document import LabParameter
from backend.services.document_intelligence import flag_abnormal

logger = logging.getLogger(__name__)

router = APIRouter()

_patient_only = require_role(["patient"])


# ── Request / Response schemas ────────────────────────────────────────────────

class LabParameterIn(BaseModel):
    name: str
    value: float
    unit: str = ""
    reference_range: tuple[float, float] = (0.0, 999999.0)
    is_abnormal: bool = False
    raw_text: str = ""


class LabParameterOut(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: tuple[float, float]
    is_abnormal: bool
    raw_text: str


class VerifyRequest(BaseModel):
    doc_id: str
    lab_parameters: list[LabParameterIn]


class VerifyResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    file_hash: str
    file_size_bytes: int
    upload_time: datetime
    ocr_status: str
    lab_parameters: list[LabParameterOut]
    verified: bool
    verified_at: Optional[datetime]
    prediction_id: Optional[str] = None


class DocumentOut(BaseModel):
    id: str
    user_id: str
    filename: str
    file_hash: str
    file_size_bytes: int
    upload_time: datetime
    ocr_status: str
    lab_parameters: list[LabParameterOut]
    verified: bool
    verified_at: Optional[datetime]
    prediction_id: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serialize_doc(doc: dict) -> dict:
    """Convert a MongoDB document dict to a JSON-serialisable dict."""
    return {
        "id": str(doc["_id"]),
        "user_id": str(doc["user_id"]),
        "filename": doc.get("filename", ""),
        "file_hash": doc.get("file_hash", ""),
        "file_size_bytes": doc.get("file_size_bytes", 0),
        "upload_time": doc.get("upload_time"),
        "ocr_status": doc.get("ocr_status", "pending"),
        "lab_parameters": doc.get("lab_parameters", []),
        "verified": doc.get("verified", False),
        "verified_at": doc.get("verified_at"),
        "prediction_id": doc.get("prediction_id"),
    }


async def _run_prediction_pipeline(
    db: AsyncIOMotorDatabase,
    user_id: str,
    document_id: str,
) -> None:
    """Background task: build feature vector + run v2 hybrid prediction after verify."""
    try:
        from backend.services.feature_vector_service import build_feature_vector
        from backend.services.data_governance import DataGovernanceModule
        from backend.services.feature_mapper import FeatureMapper
        from backend.services.classical_ml_v2 import predict_classical_v2
        from backend.services.quantum_engine_v2 import predict_quantum_v2, QuantumTimeoutError
        from backend.services.hybrid_fusion import hybrid_fusion
        from backend.services.clinical_rule_engine import apply_rules
        from backend.services.xai_service_v2 import compute_shap_v2_with_explanation
        from backend.services.limitations_module import get_disclaimer, get_limitations
        from backend.models.prediction_v2 import PredictionResponse

        DISEASES = ("diabetes", "cvd", "ckd")
        loop = asyncio.get_event_loop()

        # ── Step 1: Build 14-dim feature vector from the verified document ──
        fv = await build_feature_vector(db, user_id, document_id)
        user_oid = ObjectId(user_id)

        # Reconstruct lab_values dict from the stored feature vector
        lab_values: dict[str, float] = {
            name: float(val)
            for name, val in zip(fv.feature_names, fv.features)
        }

        # Also pull lifestyle fields from the user's lifestyle profile
        lifestyle_doc = await db["lifestyle_profiles"].find_one({"user_id": user_oid})
        lifestyle: dict[str, float] = {}
        if lifestyle_doc:
            for key in ("bmi", "age", "smoking_encoded", "exercise_frequency",
                        "sleep_hours", "stress_level", "systolic_bp", "diastolic_bp"):
                if key in lifestyle_doc and lifestyle_doc[key] is not None:
                    lifestyle[key] = float(lifestyle_doc[key])

        # Merge lifestyle into lab_values so governance + feature mapper can see them
        merged = {**lab_values, **lifestyle}

        # ── Step 2: Data governance ──────────────────────────────────────────
        governance = DataGovernanceModule()
        feature_mapper = FeatureMapper(medians=governance.medians)
        cleaned_lab_values, _ = governance.validate_and_clean(merged)

        # ── Step 3: Per-disease inference ────────────────────────────────────
        raw_risk_scores: dict[str, float] = {}
        disease_data: dict[str, dict] = {}

        for disease in DISEASES:
            mapper_result = feature_mapper.map_features(
                lab_values=cleaned_lab_values,
                lifestyle=lifestyle,
                disease=disease,
            )
            features = mapper_result.features

            # Classical ML
            rf_prob, xgb_prob = await loop.run_in_executor(
                None, predict_classical_v2, features, disease
            )

            # Quantum VQC (with timeout fallback)
            vqc_prob = None
            try:
                vqc_prob = await asyncio.wait_for(
                    loop.run_in_executor(None, predict_quantum_v2, features, disease),
                    timeout=0.8,
                )
            except Exception as exc:
                logger.warning(
                    "_run_prediction_pipeline: VQC skipped for '%s': %s", disease, exc
                )

            # Hybrid fusion
            risk_score = hybrid_fusion.fuse(
                classical_rf_prob=rf_prob,
                classical_xgb_prob=xgb_prob,
                quantum_prob=vqc_prob,
                disease=disease,
            )
            raw_risk_scores[disease] = risk_score
            disease_data[disease] = {
                "features": features,
                "missing_flags": mapper_result.missing_flags,
                "rf_prob": rf_prob,
                "xgb_prob": xgb_prob,
                "vqc_prob": vqc_prob,
            }

        # ── Step 4: Clinical rules ───────────────────────────────────────────
        adjusted_scores, triggered_rules = apply_rules(
            risk_scores=raw_risk_scores,
            raw_lab_values=merged,
        )

        # ── Step 5: XAI + build PredictionResponse objects ──────────────────
        disclaimer = get_disclaimer()
        limitations = get_limitations()
        responses: list[PredictionResponse] = []

        for disease in DISEASES:
            dd = disease_data[disease]
            shap_features, explanation, _ = compute_shap_v2_with_explanation(
                features=dd["features"],
                disease=disease,
                model_used="rf",
                risk_score=adjusted_scores[disease],
            )

            probs = [dd["rf_prob"], dd["xgb_prob"]]
            if dd["vqc_prob"] is not None:
                probs.append(dd["vqc_prob"])
            avg_prob = sum(probs) / len(probs)
            confidence = float(min(1.0, max(0.0, abs(avg_prob - 0.5) * 2.0)))

            response = PredictionResponse(
                disease=disease,
                risk_score=adjusted_scores[disease],
                confidence=confidence,
                explanation=explanation,
                shap_features=shap_features,
                triggered_rules=triggered_rules.get(disease, []),
                missing_flags=dd["missing_flags"],
                disclaimer=disclaimer,
                limitations=limitations,
            )
            responses.append(response)

        # ── Step 6: Persist v2 snapshot ──────────────────────────────────────
        now = datetime.utcnow()
        v2_snapshot = {
            "user_id": user_oid,
            "document_id": ObjectId(document_id),
            "timestamp": now,
            "predictions": [r.model_dump() for r in responses],
        }
        result = await db["predictions_v2"].insert_one(v2_snapshot)

        await db["documents"].update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"prediction_v2_id": str(result.inserted_id)}},
        )

        logger.info(
            "_run_prediction_pipeline: v2 complete for user %s, doc %s, snapshot %s",
            user_id, document_id, result.inserted_id,
        )

    except Exception as exc:
        logger.error(
            "_run_prediction_pipeline: failed for user %s doc %s: %s",
            user_id, document_id, exc,
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/verify", response_model=VerifyResponse)
async def verify_document(
    payload: VerifyRequest,
    background_tasks: BackgroundTasks,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_only),
):
    """Accept patient-corrected lab values, re-flag abnormals, mark document verified,
    then automatically build feature vector and run prediction in background.

    Requirements: 4.5, 4.6, 4.7
    """
    user_id = str(current_user["_id"])

    try:
        oid = ObjectId(payload.doc_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    doc = await db["documents"].find_one({"_id": oid})
    if not doc or str(doc.get("user_id")) != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    flagged_params = []
    for p in payload.lab_parameters:
        # Normalize name: "Platelet Count" → "platelet_count"
        canonical_name = (
            p.name.lower().strip()
            .replace(" ", "_")
            .replace("-", "_")
        )
        # Normalize value for display units
        value = p.value
        unit = p.unit
        unit_lower = unit.lower().strip()
        if unit_lower in ("lakhs/cumm", "lakh/cumm", "lakhs/µl"):
            value = value * 100000
            unit = "/µL"
        elif unit_lower in ("10^3/µl", "10^3/ul", "thousand/µl"):
            value = value * 1000
            unit = "/µL"

        # Clamp reference_range infinities
        ref_low = p.reference_range[0] if p.reference_range[0] != float("inf") else 0.0
        ref_high = p.reference_range[1] if p.reference_range[1] != float("inf") else 999999.0

        lab_param = LabParameter(
            name=canonical_name,
            value=value,
            unit=unit,
            reference_range=(ref_low, ref_high),
            is_abnormal=p.is_abnormal,
            raw_text=p.raw_text,
        )
        flagged = flag_abnormal(lab_param)
        flagged_params.append(flagged.model_dump())

    now = datetime.utcnow()
    await db["documents"].update_one(
        {"_id": oid},
        {"$set": {
            "lab_parameters": flagged_params,
            "verified": True,
            "verified_at": now,
        }},
    )

    updated = await db["documents"].find_one({"_id": oid})

    # Auto-trigger prediction pipeline in background
    background_tasks.add_task(_run_prediction_pipeline, db, user_id, payload.doc_id)

    return VerifyResponse(**_serialize_doc(updated))


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_only),
):
    """Return document detail with all lab parameters and abnormal flags."""
    user_id = str(current_user["_id"])

    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    doc = await db["documents"].find_one({"_id": oid})
    if not doc or str(doc.get("user_id")) != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    return DocumentOut(**_serialize_doc(doc))
