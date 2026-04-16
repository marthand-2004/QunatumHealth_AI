"""Onboarding / lifestyle profile router.

Requirements: 2.1, 2.2, 2.3, 2.4
"""
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import BaseModel, Field

from backend.core.database import get_db
from backend.core.deps import get_current_user, require_role
from backend.services.onboarding_service import create_profile, get_profile, update_profile

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class LifestyleProfileIn(BaseModel):
    bmi: float = Field(..., gt=0, description="Body Mass Index (must be > 0)")
    family_history: dict[str, bool] = Field(
        ...,
        description='Family history of target diseases, e.g. {"diabetes": true, "cvd": false, "ckd": false}',
    )
    smoking_status: Literal["never", "former", "current"]
    alcohol_frequency: Literal["never", "occasional", "regular"]
    exercise_frequency: int = Field(..., ge=0, le=7, description="Days per week (0–7)")
    diet_type: str = Field(..., min_length=1)
    sleep_hours: float = Field(..., gt=0, le=24, description="Average hours of sleep per night")
    stress_level: int = Field(..., ge=1, le=10, description="Stress level on a 1–10 scale")
    medications: list[str] = Field(..., description="List of current medications (may be empty)")


class LifestyleProfileOut(LifestyleProfileIn):
    user_id: str
    updated_at: datetime

    model_config = {"populate_by_name": True}


def _serialize(doc: dict) -> dict:
    """Convert MongoDB document to a JSON-serialisable dict."""
    out = {k: v for k, v in doc.items() if k != "_id"}
    out["user_id"] = str(out["user_id"])
    return out


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/profile", status_code=status.HTTP_201_CREATED, response_model=LifestyleProfileOut)
async def create_lifestyle_profile(
    payload: LifestyleProfileIn,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(require_role(["patient"])),
):
    """Create a lifestyle profile for the authenticated Patient.

    Returns HTTP 409 if a profile already exists (use PUT to update).
    """
    user_id = str(current_user["_id"])
    existing = await get_profile(db, user_id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Lifestyle profile already exists. Use PUT /api/onboarding/profile to update it.",
        )
    doc = await create_profile(db, user_id, payload.model_dump())
    return _serialize(doc)


@router.put("/profile", status_code=status.HTTP_200_OK, response_model=LifestyleProfileOut)
async def update_lifestyle_profile(
    payload: LifestyleProfileIn,
    db: AsyncIOMotorDatabase = Depends(get_db),
    current_user: dict = Depends(require_role(["patient"])),
):
    """Overwrite the lifestyle profile for the authenticated Patient.

    Records an ``updated_at`` timestamp on every update.
    """
    user_id = str(current_user["_id"])
    doc = await update_profile(db, user_id, payload.model_dump())
    return _serialize(doc)
