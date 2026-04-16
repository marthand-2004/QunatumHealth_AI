"""Clinical dashboard router — doctor-facing endpoints.

All endpoints require Doctor or Admin role (Requirement 10.7).
Tasks 14.1, 14.2, 14.3.
"""
from datetime import datetime
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.database import get_db
from backend.core.deps import get_current_user, require_role
from backend.models.audit import AuditLog

router = APIRouter()

_doctor_or_admin = require_role(["doctor", "admin"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _str_id(doc: dict) -> dict:
    """Convert _id ObjectId to string in-place and return doc."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def _write_audit_log(
    db: AsyncIOMotorDatabase,
    doctor_id: str,
    patient_id: str,
    endpoint: str,
) -> None:
    """Persist an AuditLog entry for doctor access to a patient record (Req 13.5)."""
    entry = AuditLog(
        doctor_id=doctor_id,
        patient_id=patient_id,
        accessed_at=datetime.utcnow(),
        endpoint=endpoint,
    )
    await db["audit_log"].insert_one(entry.model_dump(exclude={"id"}))


# ── GET /api/clinical/high-risk ───────────────────────────────────────────────

@router.get("/high-risk", summary="List high-risk patients (risk > 75)")
async def high_risk(
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_doctor_or_admin),
) -> list[dict]:
    """Return patients whose latest prediction has any risk score > 75.

    Joins predictions → users → alerts.
    Requirements: 10.1, 10.6
    """
    # 1. Find all predictions where any risk score value > 75
    #    MongoDB: $gt on a field inside a sub-document requires $where or
    #    we use an aggregation pipeline with $filter / $anyElementTrue.
    pipeline = [
        # Unwind risk_scores map into key/value pairs so we can filter
        {
            "$addFields": {
                "risk_values": {"$objectToArray": "$risk_scores"}
            }
        },
        {
            "$match": {
                "risk_values.v": {"$gt": 75}
            }
        },
        # Keep only the latest prediction per patient
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": "$user_id",
                "prediction_id": {"$first": "$_id"},
                "risk_scores": {"$first": "$risk_scores"},
                "model_used": {"$first": "$model_used"},
                "timestamp": {"$first": "$timestamp"},
            }
        },
    ]

    high_risk_docs: list[dict] = []
    async for doc in db["predictions"].aggregate(pipeline):
        high_risk_docs.append(doc)

    if not high_risk_docs:
        return []

    # 2. Collect patient user IDs
    patient_oids = []
    for doc in high_risk_docs:
        uid = doc["_id"]
        if isinstance(uid, str) and ObjectId.is_valid(uid):
            patient_oids.append(ObjectId(uid))
        elif isinstance(uid, ObjectId):
            patient_oids.append(uid)

    # 3. Fetch user info for those patients
    users_by_id: dict[str, dict] = {}
    async for user in db["users"].find({"_id": {"$in": patient_oids}}):
        users_by_id[str(user["_id"])] = user

    # 4. Fetch active alerts for those patients (Task 14.2)
    alerts_by_patient: dict[str, list[dict]] = {}
    async for alert in db["alerts"].find(
        {"patient_id": {"$in": [str(oid) for oid in patient_oids]}, "resolved": {"$ne": True}}
    ):
        pid = str(alert.get("patient_id", ""))
        alerts_by_patient.setdefault(pid, []).append(_str_id(alert))

    # 5. Build response
    result = []
    for doc in high_risk_docs:
        patient_id_str = str(doc["_id"])
        user = users_by_id.get(patient_id_str, {})
        result.append({
            "patient_id": patient_id_str,
            "email": user.get("email", ""),
            "role": user.get("role", ""),
            "prediction_id": str(doc.get("prediction_id", "")),
            "risk_scores": doc.get("risk_scores", {}),
            "model_used": doc.get("model_used", ""),
            "prediction_timestamp": doc.get("timestamp"),
            "alerts": alerts_by_patient.get(patient_id_str, []),
            "alert_count": len(alerts_by_patient.get(patient_id_str, [])),
        })

    return result


# ── GET /api/clinical/patient/{id} ────────────────────────────────────────────

@router.get("/patient/{patient_id}", summary="Full patient detail with prediction history")
async def patient_detail(
    patient_id: str,
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_doctor_or_admin),
) -> dict:
    """Return full patient detail: profile, prediction history, latest lab values, SHAP.

    Also writes an AuditLog entry (Task 14.3, Req 13.5).
    Requirements: 10.2, 10.3, 13.5
    """
    if not ObjectId.is_valid(patient_id):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid patient ID.")

    # 1. Fetch patient user record
    patient = await db["users"].find_one({"_id": ObjectId(patient_id)})
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")
    if patient.get("role") not in ("patient",):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    # 2. Write audit log entry (Task 14.3)
    doctor_id = str(current_user.get("_id", ""))
    await _write_audit_log(
        db=db,
        doctor_id=doctor_id,
        patient_id=patient_id,
        endpoint=str(request.url.path),
    )

    # 3. Fetch all predictions for this patient (history), sorted newest first
    predictions: list[dict] = []
    async for pred in db["predictions"].find({"user_id": patient_id}).sort("timestamp", -1):
        _str_id(pred)
        predictions.append(pred)

    # 4. Latest lab values — from the most recent completed document
    latest_doc = await db["documents"].find_one(
        {"user_id": patient_id, "ocr_status": "complete"},
        sort=[("upload_time", -1)],
    )
    lab_parameters: list[dict] = []
    if latest_doc:
        lab_parameters = latest_doc.get("lab_parameters", [])

    # 5. SHAP values from the latest prediction
    shap_values: dict[str, Any] = {}
    if predictions:
        shap_values = predictions[0].get("shap_values") or {}

    # 6. Fetch active alerts for this patient
    alerts: list[dict] = []
    async for alert in db["alerts"].find({"patient_id": patient_id, "resolved": {"$ne": True}}):
        alerts.append(_str_id(alert))

    return {
        "patient": {
            "id": str(patient["_id"]),
            "email": patient.get("email", ""),
            "role": patient.get("role", ""),
            "created_at": patient.get("created_at"),
            "is_active": patient.get("is_active", True),
        },
        "prediction_history": predictions,
        "lab_parameters": lab_parameters,
        "shap_values": shap_values,
        "alerts": alerts,
    }


# ── POST /api/clinical/export/bulk ────────────────────────────────────────────

@router.post("/export/bulk", summary="Bulk export patient records")
async def bulk_export(
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_doctor_or_admin),
) -> dict:
    """Bulk export patient records.

    Accepts a JSON body with a list of patient IDs and returns a summary of
    the exported records.  Requirements: 10.5
    """
    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    patient_ids: list[str] = body.get("patient_ids", [])
    if not patient_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="patient_ids list is required.",
        )

    valid_oids = [ObjectId(pid) for pid in patient_ids if ObjectId.is_valid(pid)]
    if not valid_oids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid patient IDs provided.",
        )

    exported: list[dict] = []
    async for patient in db["users"].find({"_id": {"$in": valid_oids}, "role": "patient"}):
        pid_str = str(patient["_id"])
        # Latest prediction
        latest_pred = await db["predictions"].find_one(
            {"user_id": pid_str}, sort=[("timestamp", -1)]
        )
        exported.append({
            "patient_id": pid_str,
            "email": patient.get("email", ""),
            "latest_risk_scores": latest_pred.get("risk_scores", {}) if latest_pred else {},
            "prediction_timestamp": latest_pred.get("timestamp") if latest_pred else None,
        })

    return {
        "exported_count": len(exported),
        "patients": exported,
        "exported_by": str(current_user.get("_id", "")),
        "exported_at": datetime.utcnow().isoformat(),
    }


# ── PUT /api/clinical/patient/{id}/status ─────────────────────────────────────

@router.put("/patient/{patient_id}/status", summary="Update patient status")
async def update_status(
    patient_id: str,
    request: Request,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_doctor_or_admin),
) -> dict:
    """Update a patient's active status.

    Accepts JSON body: {"is_active": bool, "notes": str (optional)}.
    Requirements: 10.5
    """
    if not ObjectId.is_valid(patient_id):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid patient ID.")

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    if "is_active" not in body:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="is_active field is required.",
        )

    result = await db["users"].update_one(
        {"_id": ObjectId(patient_id), "role": "patient"},
        {"$set": {"is_active": bool(body["is_active"])}},
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    return {
        "patient_id": patient_id,
        "is_active": bool(body["is_active"]),
        "updated_by": str(current_user.get("_id", "")),
        "updated_at": datetime.utcnow().isoformat(),
    }
