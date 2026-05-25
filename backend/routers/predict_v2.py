"""Prediction v2 router — full hybrid inference pipeline with per-stage latency logging.

Implements ``POST /api/predict/v2`` which orchestrates the complete v2 inference
pipeline for three diseases (diabetes, cvd, ckd) and returns a list of
PredictionResponse objects.

Pipeline stages (per disease):
  1. feature_extraction  — DataGovernanceModule.validate_and_clean() + FeatureMapper.map_features()
  2. scaler_normalization — StandardScaler.transform() (loaded from training/models/)
  3. classical_ml        — ClassicalMLService.predict_classical_v2()
  4. vqc                 — QuantumEngine.predict_quantum_v2() with 800ms timeout
  5. hybrid_fusion       — HybridFusion.fuse()
  6. clinical_rules      — ClinicalRuleEngine.apply_rules()
  7. xai_layer           — XAILayer.compute_shap_v2()

Requirements: 9.1, 9.2, 9.3, 9.4, 10.10, 20.2, 21.3, 24.5
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, model_validator

from backend.core.database import get_db
from backend.core.deps import require_role
from backend.models.prediction_v2 import LatencyBreakdown, PredictionResponse, SHAPFeature
from backend.services.clinical_rule_engine import apply_rules
from backend.services.classical_ml_v2 import predict_classical_v2
from backend.services.data_governance import DataGovernanceModule
from backend.services.feature_mapper import FeatureMapper
from backend.services.hybrid_fusion import hybrid_fusion
from backend.services.limitations_module import get_disclaimer, get_limitations
from backend.services.performance_logger import PerformanceLogger
from backend.services.quantum_engine_v2 import QuantumTimeoutError, predict_quantum_v2
from backend.services.xai_service_v2 import compute_shap_v2_with_explanation

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISEASES = ("diabetes", "cvd", "ckd")

_REPO_ROOT = Path(__file__).parent.parent.parent
_TRAINING_MODELS_DIR = _REPO_ROOT / "training" / "models"
_TRAINING_RESULTS_DIR = _REPO_ROOT / "training" / "results"

# Latency budget for the full pipeline (excluding OCR), in milliseconds (Req 9.1)
_TOTAL_LATENCY_BUDGET_MS = 1000.0

# ---------------------------------------------------------------------------
# Startup artifact loading
# ---------------------------------------------------------------------------


def _load_json_artifact(path: Path, name: str) -> dict:
    """Load a JSON artifact from disk; return empty dict on failure."""
    if not path.exists():
        logger.error("predict_v2: %s not found at %s", name, path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("predict_v2: Failed to load %s from %s: %s", name, path, exc)
        return {}


# Load all artifacts at module level (once at import time)
_thresholds: dict = _load_json_artifact(
    _TRAINING_MODELS_DIR / "thresholds.json", "thresholds.json"
)
_robustness_report: dict = _load_json_artifact(
    _TRAINING_RESULTS_DIR / "robustness_report.json", "robustness_report.json"
)
_fusion_weights: dict = _load_json_artifact(
    _TRAINING_MODELS_DIR / "fusion_weights_v2.json", "fusion_weights_v2.json"
)
_data_governance_config: dict = _load_json_artifact(
    _TRAINING_MODELS_DIR / "data_governance.json", "data_governance.json"
)

# Initialise DataGovernanceModule (loads medians + IQR fences from data_governance.json)
_governance = DataGovernanceModule()

# Initialise FeatureMapper with medians from governance config
_feature_mapper = FeatureMapper(medians=_governance.medians)

# Check for critical missing artifacts and log errors
_ARTIFACTS_OK = True
if not _thresholds:
    logger.error("predict_v2: thresholds.json missing — threshold_used will default to 0.5")
if not _robustness_report:
    logger.error("predict_v2: robustness_report.json missing — robustness_score will default to 1.0")


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class PredictV2Request(BaseModel):
    """Request body for POST /api/predict/v2.

    Either ``feature_vector_id`` OR (``raw_lab_values`` + ``lifestyle``) must
    be provided.

    Attributes
    ----------
    feature_vector_id:
        Reference to a stored feature vector document in MongoDB.
    raw_lab_values:
        Direct lab values input (e.g. glucose, hba1c, creatinine, etc.).
    lifestyle:
        Lifestyle data (e.g. exercise_frequency, smoking_encoded, bmi, age).
    """

    feature_vector_id: Optional[str] = None
    raw_lab_values: Optional[dict[str, float]] = None
    lifestyle: Optional[dict] = None

    @model_validator(mode="after")
    def validate_input_combination(self) -> "PredictV2Request":
        """Ensure either feature_vector_id OR (raw_lab_values + lifestyle) is provided."""
        has_fv_id = self.feature_vector_id is not None
        has_raw = self.raw_lab_values is not None

        if not has_fv_id and not has_raw:
            raise ValueError(
                "Provide either 'feature_vector_id' or 'raw_lab_values' (with optional 'lifestyle')."
            )
        if has_fv_id and has_raw:
            raise ValueError(
                "Provide either 'feature_vector_id' or 'raw_lab_values', not both."
            )
        return self


# ---------------------------------------------------------------------------
# Helper: resolve raw lab values from request
# ---------------------------------------------------------------------------


async def _resolve_lab_values(
    body: PredictV2Request,
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> tuple[dict[str, float], dict]:
    """Resolve raw_lab_values and lifestyle from the request.

    If ``feature_vector_id`` is provided, loads the stored feature vector from
    MongoDB and reconstructs raw_lab_values from it.

    Returns
    -------
    (raw_lab_values, lifestyle)
    """
    if body.feature_vector_id is not None:
        user_oid = ObjectId(user_id)
        try:
            fv_oid = ObjectId(body.feature_vector_id)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid feature_vector_id: '{body.feature_vector_id}'",
            )

        fv_doc = await db["feature_vectors"].find_one(
            {"_id": fv_oid, "user_id": user_oid}
        )
        if fv_doc is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"FeatureVector '{body.feature_vector_id}' not found.",
            )

        # Reconstruct raw_lab_values from stored feature vector
        feature_names: list[str] = fv_doc.get("feature_names", [])
        features: list[float] = fv_doc.get("features", [])
        raw_lab_values: dict[str, float] = {
            name: float(val)
            for name, val in zip(feature_names, features)
            if isinstance(val, (int, float))
        }
        # Lifestyle fields that may be embedded in the feature vector
        lifestyle_keys = {"bmi", "age", "smoking_encoded", "exercise_frequency", "sleep_hours", "stress_level"}
        lifestyle: dict = {k: v for k, v in raw_lab_values.items() if k in lifestyle_keys}
        return raw_lab_values, lifestyle

    # Direct input path
    raw_lab_values = dict(body.raw_lab_values or {})
    lifestyle = dict(body.lifestyle or {})
    return raw_lab_values, lifestyle


# ---------------------------------------------------------------------------
# Helper: get threshold for a disease
# ---------------------------------------------------------------------------


def _get_threshold(disease: str) -> float:
    """Return the RF threshold for a disease from thresholds.json."""
    disease_thresholds = _thresholds.get(disease, {})
    # Prefer RF threshold (Youden's Index); fall back to 0.5
    return float(disease_thresholds.get("rf", disease_thresholds.get("xgb", 0.5)))


# ---------------------------------------------------------------------------
# Helper: get robustness score for a disease
# ---------------------------------------------------------------------------


def _get_robustness_score(disease: str) -> float:
    """Return the robustness score for a disease from robustness_report.json."""
    disease_report = _robustness_report.get(disease, {})
    return float(disease_report.get("robustness_score", 1.0))


# ---------------------------------------------------------------------------
# Core inference pipeline for a single disease
# ---------------------------------------------------------------------------


def _run_disease_pipeline(
    disease: str,
    cleaned_lab_values: dict[str, float],
    raw_lab_values: dict[str, float],
    lifestyle: dict,
    perf: PerformanceLogger,
) -> tuple[dict, float]:
    """Run the full inference pipeline for a single disease.

    Returns
    -------
    (pipeline_data, risk_score)
        pipeline_data contains all intermediate results needed to build PredictionResponse.
    """
    # ── Stage 1: feature_extraction ─────────────────────────────────────────
    with perf.stage("feature_extraction"):
        mapper_result = _feature_mapper.map_features(
            lab_values=cleaned_lab_values,
            lifestyle=lifestyle,
            disease=disease,
        )
    features = mapper_result.features
    missing_flags = mapper_result.missing_flags

    # ── Stage 2: scaler_normalization ───────────────────────────────────────
    # The scaler is applied inside predict_classical_v2 and predict_quantum_v2.
    # We record the timing here as a lightweight pass-through stage.
    with perf.stage("scaler_normalization"):
        # Scaler transform is handled internally by classical_ml_v2 and quantum_engine_v2.
        # This stage records the overhead of the scaler lookup.
        pass

    # ── Stage 3: classical_ml ───────────────────────────────────────────────
    with perf.stage("classical_ml"):
        rf_prob, xgb_prob = predict_classical_v2(features, disease)

    # ── Stage 4: vqc (with 800ms timeout) ───────────────────────────────────
    vqc_prob: Optional[float] = None
    with perf.stage("vqc"):
        try:
            vqc_prob = predict_quantum_v2(features, disease)
        except (QuantumTimeoutError, Exception) as exc:
            if isinstance(exc, QuantumTimeoutError):
                logger.warning(
                    "predict_v2: VQC timeout for '%s' — falling back to classical-only fusion.",
                    disease,
                )
            else:
                logger.warning(
                    "predict_v2: VQC error for '%s' (%s) — falling back to classical-only fusion.",
                    disease,
                    exc,
                )
            vqc_prob = None

    # ── Stage 5: hybrid_fusion ──────────────────────────────────────────────
    with perf.stage("hybrid_fusion"):
        risk_score = hybrid_fusion.fuse(
            classical_rf_prob=rf_prob,
            classical_xgb_prob=xgb_prob,
            quantum_prob=vqc_prob,
            disease=disease,
        )

    return {
        "features": features,
        "missing_flags": missing_flags,
        "rf_prob": rf_prob,
        "xgb_prob": xgb_prob,
        "vqc_prob": vqc_prob,
        "risk_score": risk_score,
    }, risk_score


# ---------------------------------------------------------------------------
# POST /api/predict/v2
# ---------------------------------------------------------------------------


@router.post(
    "/v2",
    response_model=list[PredictionResponse],
    status_code=status.HTTP_200_OK,
    summary="Full hybrid inference pipeline (v2)",
    description=(
        "Orchestrates the complete v2 inference pipeline for diabetes, CVD, and CKD. "
        "Requires JWT Bearer token with 'patient' or 'doctor' role."
    ),
)
async def predict_v2(
    body: PredictV2Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(require_role(["patient", "doctor", "admin"])),
) -> list[PredictionResponse]:
    """Run the full v2 hybrid inference pipeline and return predictions for all 3 diseases.

    Requirements: 9.1, 9.2, 9.3, 9.4, 10.10, 20.2, 21.3, 24.5
    """
    user_id = str(current_user["_id"])
    pipeline_start = time.perf_counter()

    # ── Resolve lab values ──────────────────────────────────────────────────
    raw_lab_values, lifestyle = await _resolve_lab_values(body, db, user_id)

    # ── Stage 1 (shared): DataGovernanceModule.validate_and_clean() ─────────
    governance_perf = PerformanceLogger()
    with governance_perf.stage("feature_extraction"):
        try:
            cleaned_lab_values, governance_warnings = _governance.validate_and_clean(raw_lab_values)
        except Exception as exc:
            logger.error("predict_v2: DataGovernanceModule.validate_and_clean() failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Data governance validation failed: {exc}",
            )

    if governance_warnings:
        logger.warning(
            "predict_v2: DataGovernance warnings for user %s: %s",
            user_id,
            governance_warnings,
        )

    # ── Per-disease pipeline ─────────────────────────────────────────────────
    # Collect per-disease results; clinical rules and XAI run after all diseases
    disease_pipeline_data: dict[str, dict] = {}
    disease_risk_scores: dict[str, float] = {}

    # We use a single PerformanceLogger per disease for stage timing
    disease_perf_loggers: dict[str, PerformanceLogger] = {}

    loop = asyncio.get_event_loop()

    for disease in DISEASES:
        perf = PerformanceLogger()
        disease_perf_loggers[disease] = perf

        try:
            pipeline_data, risk_score = await loop.run_in_executor(
                None,
                _run_disease_pipeline,
                disease,
                cleaned_lab_values,
                raw_lab_values,
                lifestyle,
                perf,
            )
        except Exception as exc:
            logger.error(
                "predict_v2: Pipeline error for disease '%s': %s", disease, exc
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference pipeline error for '{disease}': {exc}",
            )

        disease_pipeline_data[disease] = pipeline_data
        disease_risk_scores[disease] = risk_score

    # ── Stage 6: clinical_rules (applied across all diseases at once) ────────
    clinical_perf = PerformanceLogger()
    with clinical_perf.stage("clinical_rules"):
        adjusted_scores, triggered_rules_per_disease = apply_rules(
            risk_scores=disease_risk_scores,
            raw_lab_values=raw_lab_values,
        )

    # Update risk scores with clinical rule adjustments
    for disease in DISEASES:
        disease_pipeline_data[disease]["risk_score"] = adjusted_scores[disease]
        disease_pipeline_data[disease]["triggered_rules"] = triggered_rules_per_disease.get(disease, [])

    # ── Stage 7: xai_layer (per disease) ────────────────────────────────────
    for disease in DISEASES:
        pd = disease_pipeline_data[disease]
        xai_perf = PerformanceLogger()
        with xai_perf.stage("xai_layer"):
            try:
                shap_features, explanation, base_value = compute_shap_v2_with_explanation(
                    features=pd["features"],
                    disease=disease,
                    model_used="rf",
                    risk_score=pd["risk_score"],
                )
            except Exception as exc:
                logger.error(
                    "predict_v2: XAI computation failed for '%s': %s", disease, exc
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"XAI layer error for '{disease}': {exc}",
                )

        pd["shap_features"] = shap_features
        pd["explanation"] = explanation
        pd["base_value"] = base_value
        pd["xai_timing_ms"] = xai_perf.timings.get("xai_layer", 0.0)

    # ── Build PredictionResponse list ────────────────────────────────────────
    responses: list[PredictionResponse] = []
    disclaimer = get_disclaimer()
    limitations = get_limitations()

    for disease in DISEASES:
        pd = disease_pipeline_data[disease]
        perf = disease_perf_loggers[disease]
        timings = perf.timings

        # Compute confidence from RF probability (primary classical model)
        rf_prob = pd["rf_prob"]
        xgb_prob = pd["xgb_prob"]
        vqc_prob = pd.get("vqc_prob")
        # Confidence = average of available model probabilities, distance from 0.5
        probs = [rf_prob, xgb_prob]
        if vqc_prob is not None:
            probs.append(vqc_prob)
        avg_prob = sum(probs) / len(probs)
        confidence = float(min(1.0, max(0.0, abs(avg_prob - 0.5) * 2.0)))

        # Build LatencyBreakdown
        clinical_rules_ms = clinical_perf.timings.get("clinical_rules", 0.0)
        xai_layer_ms = pd.get("xai_timing_ms", 0.0)
        feature_extraction_ms = timings.get("feature_extraction", 0.0)
        scaler_normalization_ms = timings.get("scaler_normalization", 0.0)
        classical_ml_ms = timings.get("classical_ml", 0.0)
        vqc_ms = timings.get("vqc", 0.0)
        hybrid_fusion_ms = timings.get("hybrid_fusion", 0.0)

        total_ms = (
            feature_extraction_ms
            + scaler_normalization_ms
            + classical_ml_ms
            + vqc_ms
            + hybrid_fusion_ms
            + clinical_rules_ms
            + xai_layer_ms
        )

        latency = LatencyBreakdown(
            feature_extraction_ms=feature_extraction_ms,
            scaler_normalization_ms=scaler_normalization_ms,
            classical_ml_ms=classical_ml_ms,
            vqc_ms=vqc_ms,
            hybrid_fusion_ms=hybrid_fusion_ms,
            clinical_rules_ms=clinical_rules_ms,
            xai_layer_ms=xai_layer_ms,
            total_ms=total_ms,
        )

        try:
            response = PredictionResponse(
                disease=disease,
                risk_score=pd["risk_score"],
                confidence=confidence,
                explanation=pd["explanation"],
                shap_features=pd["shap_features"],
                triggered_rules=pd.get("triggered_rules", []),
                threshold_used=_get_threshold(disease),
                missing_flags=pd["missing_flags"],
                robustness_score=_get_robustness_score(disease),
                latency=latency,
                disclaimer=disclaimer,
                limitations=limitations,
            )
        except Exception as exc:
            logger.error(
                "predict_v2: PredictionResponse validation failed for '%s': %s",
                disease,
                exc,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Response validation error for '{disease}': {exc}",
            )

        responses.append(response)

    # ── Total latency check (Req 9.1, 9.4) ──────────────────────────────────
    total_pipeline_ms = (time.perf_counter() - pipeline_start) * 1000.0
    if total_pipeline_ms > _TOTAL_LATENCY_BUDGET_MS:
        # Build per-stage breakdown for the warning log
        stage_breakdown = {}
        for disease in DISEASES:
            perf = disease_perf_loggers[disease]
            for stage_name, ms in perf.timings.items():
                key = f"{disease}.{stage_name}"
                stage_breakdown[key] = round(ms, 2)
        stage_breakdown["clinical_rules"] = round(
            clinical_perf.timings.get("clinical_rules", 0.0), 2
        )
        logger.warning(
            "predict_v2: Total inference latency %.1fms exceeds %dms budget. "
            "Per-stage breakdown: %s",
            total_pipeline_ms,
            int(_TOTAL_LATENCY_BUDGET_MS),
            stage_breakdown,
        )

    # ── Persist v2 prediction snapshot + latency records ────────────────────
    now = datetime.utcnow()
    user_oid = ObjectId(user_id)

    # Serialise all three disease responses into one MongoDB document so that
    # GET /v2/latest can reconstruct them without re-running inference.
    v2_snapshot = {
        "user_id": user_oid,
        "timestamp": now,
        "predictions": [r.model_dump() for r in responses],
    }
    await db["predictions_v2"].insert_one(v2_snapshot)

    for response in responses:
        perf_logger = PerformanceLogger()
        await perf_logger.persist({
            "disease": response.disease,
            "user_id": user_id,
            "timestamp": now.isoformat(),
            "latency": response.latency.model_dump(),
            "risk_score": response.risk_score,
            "risk_level": response.risk_level,
        })

    return responses


# ---------------------------------------------------------------------------
# GET /api/predict/v2/latest
# ---------------------------------------------------------------------------


@router.get(
    "/v2/latest",
    response_model=list[PredictionResponse],
    status_code=status.HTTP_200_OK,
    summary="Latest v2 prediction for the authenticated user",
    description=(
        "Returns the most recent v2 PredictionResponse list for the logged-in user. "
        "Pass ?document_id=<id> to fetch the prediction for a specific document. "
        "Returns HTTP 404 if no v2 predictions exist yet."
    ),
)
async def get_latest_v2_prediction(
    document_id: Optional[str] = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(require_role(["patient", "doctor", "admin"])),
) -> list[PredictionResponse]:
    """Fetch the most recent v2 prediction snapshot from MongoDB.

    If ``document_id`` is provided, returns the prediction for that specific
    document. Otherwise returns the most recent prediction for the user.
    """
    user_id = str(current_user["_id"])
    user_oid = ObjectId(user_id)

    query: dict = {"$or": [{"user_id": user_oid}, {"user_id": user_id}]}

    # If a specific document_id is requested, filter by it
    if document_id:
        try:
            doc_oid = ObjectId(document_id)
            query["document_id"] = doc_oid
        except Exception:
            pass  # invalid ObjectId — fall back to latest

    doc = await db["predictions_v2"].find_one(
        query,
        sort=[("timestamp", -1)],
    )

    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No v2 predictions found. Upload a medical report to get started.",
        )

    try:
        predictions = [PredictionResponse(**p) for p in doc.get("predictions", [])]
    except Exception as exc:
        logger.error("predict_v2: Failed to deserialise latest v2 snapshot: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading latest prediction: {exc}",
        )

    if not predictions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No v2 predictions found in the latest snapshot.",
        )

    return predictions
