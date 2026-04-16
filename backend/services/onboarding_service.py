"""Onboarding service — lifestyle profile persistence.

Requirements: 2.1, 2.2, 2.3, 2.4
"""
from datetime import datetime
from typing import Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase


async def get_profile(db: AsyncIOMotorDatabase, user_id: str) -> Optional[dict]:
    """Return the lifestyle profile for *user_id*, or None if not found."""
    return await db["lifestyle_profiles"].find_one({"user_id": ObjectId(user_id)})


async def create_profile(db: AsyncIOMotorDatabase, user_id: str, data: dict) -> dict:
    """Insert a new lifestyle profile linked to *user_id*.

    Returns the inserted document (with ``_id``).
    """
    doc = {
        **data,
        "user_id": ObjectId(user_id),
        "updated_at": datetime.utcnow(),
    }
    result = await db["lifestyle_profiles"].insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


async def update_profile(db: AsyncIOMotorDatabase, user_id: str, data: dict) -> dict:
    """Overwrite the lifestyle profile for *user_id* and record ``updated_at``.

    Creates the document if it does not yet exist (upsert).
    Returns the updated document.
    """
    update_doc = {
        **data,
        "user_id": ObjectId(user_id),
        "updated_at": datetime.utcnow(),
    }
    await db["lifestyle_profiles"].replace_one(
        {"user_id": ObjectId(user_id)},
        update_doc,
        upsert=True,
    )
    return await db["lifestyle_profiles"].find_one({"user_id": ObjectId(user_id)})
