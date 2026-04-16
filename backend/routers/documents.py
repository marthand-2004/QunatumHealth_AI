"""Document intelligence router — verify and retrieve.

Requirements: 4.5, 4.6, 4.7
"""
from datetime import datetime
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel

from backend.core.database import get_db
from backend.core.deps import require_role
from backend.models.document import LabParameter
from backend.services.document_intelligence import flag_abnormal

router = APIRouter()

_patient_only = require_role(["patient"])


# ── Request / Response schemas ────────────────────────────────────────────────

class LabParameterIn(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: tuple[float, float]
    is_abnormal: bool = False
    raw_text: str = ""


class VerifyRequest(BaseModel):
    doc_id: str
    lab_parameters: list[LabParameterIn]


class LabParameterOut(BaseModel):
    name: str
    value: float
    unit: str
    reference_range: tuple[float, float]
    is_abnormal: bool
    raw_text: str


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
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/verify", response_model=DocumentOut)
async def verify_document(
    payload: VerifyRequest,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_only),
):
    """Accept patient-corrected lab values, re-flag abnormals, and mark document verified.

    - Accepts a list of corrected LabParameter objects.
    - Re-runs flag_abnormal on each corrected parameter.
    - Updates the Document: sets lab_parameters, verified=True, verified_at=now.
    - Returns the updated document.

    Requirements: 4.5, 4.6, 4.7
    """
    user_id = str(current_user["_id"])

    # Validate doc_id format
    try:
        oid = ObjectId(payload.doc_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # Fetch and verify ownership
    doc = await db["documents"].find_one({"_id": oid})
    if not doc or str(doc.get("user_id")) != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    # Re-run flag_abnormal on each corrected parameter
    flagged_params = []
    for p in payload.lab_parameters:
        lab_param = LabParameter(
            name=p.name,
            value=p.value,
            unit=p.unit,
            reference_range=p.reference_range,
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
    return _serialize_doc(updated)


@router.get("/{doc_id}", response_model=DocumentOut)
async def get_document(
    doc_id: str,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(_patient_only),
):
    """Return document detail with all lab parameters and abnormal flags.

    Requirements: 4.6, 4.7
    """
    user_id = str(current_user["_id"])

    try:
        oid = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    doc = await db["documents"].find_one({"_id": oid})
    if not doc or str(doc.get("user_id")) != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    return _serialize_doc(doc)
