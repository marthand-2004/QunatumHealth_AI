"""Report generation and download router.

POST /api/reports/generate/{patient_id}  — background task, returns report_id
GET  /api/reports/{report_id}/download   — stream PDF as file download
POST /api/reports/{report_id}/share      — generate 72-hour shareable link
GET  /api/reports/shared/{token}         — return PDF or HTTP 410 if expired

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
Tasks: 15.1, 15.2
"""
from __future__ import annotations

import logging
import os
from datetime import datetime

from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, Response
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.database import get_db
from backend.core.deps import get_current_user, require_role
from backend.services.report_service import (
    create_report,
    generate_pdf,
    generate_share_token,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_patient_or_above = require_role(["patient", "doctor", "admin"])
_doctor_or_admin = require_role(["doctor", "admin"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _str_id(doc: dict) -> dict:
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


async def _run_generate(
    db: AsyncIOMotorDatabase,
    patient_id: str,
    generated_by: str,
    prediction_id: str | None,
) -> str:
    """Fetch data, build PDF, persist report, return report_id.

    This is called both directly (for immediate response) and as a background
    task.  Returns the report_id string.
    """
    # ── Fetch patient ─────────────────────────────────────────────────────
    if not ObjectId.is_valid(patient_id):
        raise ValueError(f"Invalid patient_id: {patient_id}")
    patient = await db["users"].find_one({"_id": ObjectId(patient_id)})
    if not patient:
        raise ValueError(f"Patient {patient_id} not found.")

    # ── Resolve prediction ────────────────────────────────────────────────
    if prediction_id and ObjectId.is_valid(prediction_id):
        prediction = await db["predictions"].find_one({"_id": ObjectId(prediction_id)})
    else:
        prediction = await db["predictions"].find_one(
            {"user_id": patient_id}, sort=[("timestamp", -1)]
        )
    if not prediction:
        prediction = {
            "model_used": "N/A",
            "risk_scores": {},
            "timestamp": datetime.utcnow(),
            "shap_values": None,
        }

    # ── Fetch lab parameters from latest verified document ────────────────
    lab_parameters: list[dict] = []
    latest_doc = await db["documents"].find_one(
        {"user_id": patient_id, "ocr_status": "complete"},
        sort=[("upload_time", -1)],
    )
    if latest_doc:
        lab_parameters = latest_doc.get("lab_parameters", [])

    # ── Fetch recommendations ─────────────────────────────────────────────
    recommendations: list[dict] = []
    if prediction.get("_id"):
        pred_id_str = str(prediction["_id"])
        async for rec in db["recommendations"].find({"prediction_id": pred_id_str}):
            recommendations.append(rec)

    # ── Generate PDF ──────────────────────────────────────────────────────
    pdf_bytes = generate_pdf(
        patient=patient,
        prediction=prediction,
        lab_parameters=lab_parameters,
        recommendations=recommendations,
    )

    # ── Persist report ────────────────────────────────────────────────────
    pred_id_for_report = str(prediction.get("_id", "")) if prediction.get("_id") else ""
    report_id = await create_report(
        db=db,
        patient_id=patient_id,
        generated_by=generated_by,
        prediction_id=pred_id_for_report,
        pdf_bytes=pdf_bytes,
    )
    return report_id


# ── POST /api/reports/generate/{patient_id} ───────────────────────────────────

@router.post("/generate/{patient_id}", status_code=202)
async def generate_report(
    patient_id: str,
    background_tasks: BackgroundTasks,
    prediction_id: str | None = None,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_or_above),
) -> dict:
    """Trigger PDF report generation for *patient_id*.

    - Patients may only generate their own report.
    - Doctors/Admins may generate for any patient.
    - Runs as a background task; returns report_id immediately.

    Requirements: 11.1, 11.2, 11.4
    """
    user_role = current_user.get("role", "patient")
    user_id = str(current_user.get("_id", ""))

    # Patients can only generate their own report
    if user_role == "patient" and patient_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Patients may only generate their own reports.",
        )

    if not ObjectId.is_valid(patient_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid patient_id.",
        )

    # Run synchronously to return report_id in the 202 response.
    # For very large reports this could be moved to a true background task,
    # but the spec requires completion within 10 seconds.
    try:
        report_id = await _run_generate(
            db=db,
            patient_id=patient_id,
            generated_by=user_id,
            prediction_id=prediction_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("PDF generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PDF generation failed. Please retry.",
        )

    return {"report_id": report_id, "status": "generated"}


# ── GET /api/reports/{report_id}/download ─────────────────────────────────────

@router.get("/{report_id}/download")
async def download_report(
    report_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(get_current_user),
) -> Response:
    """Stream the PDF for *report_id* as a file download.

    - Patients may only download their own reports.
    - Doctors/Admins may download any report.

    Requirements: 11.1, 11.4
    """
    if not ObjectId.is_valid(report_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid report_id.",
        )

    report = await db["reports"].find_one({"_id": ObjectId(report_id)})
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    # Ownership check for patients
    user_role = current_user.get("role", "patient")
    user_id = str(current_user.get("_id", ""))
    if user_role == "patient" and report.get("patient_id") != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    file_path: str = report.get("file_path", "")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found on disk.",
        )

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"report_{report_id}.pdf",
    )


# ── POST /api/reports/{report_id}/share ──────────────────────────────────────

@router.post("/{report_id}/share")
async def share_report(
    report_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_or_above),
) -> dict:
    """Generate a 72-hour shareable link for *report_id*.

    Requirements: 11.3, 11.5
    """
    if not ObjectId.is_valid(report_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid report_id.",
        )

    report = await db["reports"].find_one({"_id": ObjectId(report_id)})
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found.")

    # Ownership check for patients
    user_role = current_user.get("role", "patient")
    user_id = str(current_user.get("_id", ""))
    if user_role == "patient" and report.get("patient_id") != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    try:
        result = await generate_share_token(report_id=report_id, db=db)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return {
        "report_id": report_id,
        "share_token": result["share_token"],
        "share_expires_at": result["share_expires_at"].isoformat(),
        "share_url": f"/api/reports/shared/{result['share_token']}",
    }


# ── GET /api/reports/shared/{token} ──────────────────────────────────────────

@router.get("/shared/{token}")
async def shared_report(
    token: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> Response:
    """Return the PDF for a shared token, or HTTP 410 if expired.

    No authentication required — the token itself is the credential.

    Requirements: 11.3, 11.5
    """
    report = await db["reports"].find_one({"share_token": token})
    if not report:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Shared link is invalid or has expired.",
        )

    expires_at: datetime | None = report.get("share_expires_at")
    if expires_at is None or datetime.utcnow() > expires_at:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Shared link has expired.",
        )

    file_path: str = report.get("file_path", "")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found.",
        )

    report_id = str(report.get("_id", ""))
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=f"report_{report_id}.pdf",
    )
