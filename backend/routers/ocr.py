"""OCR pipeline router — upload, status, result.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""
import os

from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.deps import require_role
from backend.services.ocr_service import (
    ALLOWED_CONTENT_TYPES,
    get_document_status,
    get_ocr_result,
    run_ocr_background,
    upload_document,
)

router = APIRouter()

_patient_or_above = require_role(["patient", "doctor", "admin"])


@router.post("/upload", status_code=202)
async def upload_document_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_or_above),
):
    """Accept a medical document upload (PDF, JPG, PNG <= 20 MB).

    - Rejects files > 20 MB with HTTP 413.
    - Rejects unsupported formats with HTTP 415.
    - Encrypts the file with AES-256 (Fernet) and stores it in data/uploads/.
    - Creates a Document record with ocr_status="pending".
    - Enqueues background OCR processing.
    - Returns the job ID within 2 seconds.
    """
    job_id = await upload_document(db, file, str(current_user["_id"]))

    # Derive the stored file path from the job_id lookup isn't needed —
    # upload_document stores the file as {sha256}{ext}.enc; we need the path.
    # Re-derive it: fetch the document to get file_hash + original content_type.
    from bson import ObjectId
    doc = await db["documents"].find_one({"_id": ObjectId(job_id)})
    if doc:
        content_type = (file.content_type or "").split(";")[0].strip().lower()
        ext = ALLOWED_CONTENT_TYPES.get(content_type, "")
        stored_filename = f"{doc['file_hash']}{ext}.enc"
        file_path = os.path.join(settings.UPLOAD_DIR, stored_filename)

        background_tasks.add_task(
            run_ocr_background,
            db,
            job_id,
            file_path,
            str(current_user["_id"]),
        )

    return {"job_id": job_id, "ocr_status": "pending"}


@router.get("/status/{job_id}")
async def get_status(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_or_above),
):
    """Return the current ocr_status for the given job ID."""
    return await get_document_status(db, job_id, str(current_user["_id"]))


@router.get("/result/{job_id}")
async def get_result(
    job_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_or_above),
):
    """Return the OCR result: extracted text and raw lab parameter candidates.

    Returns HTTP 202 if OCR is still in progress.
    Returns HTTP 404 if the document does not exist or belongs to another user.
    """
    return await get_ocr_result(db, job_id, str(current_user["_id"]))
