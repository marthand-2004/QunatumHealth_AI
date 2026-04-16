"""Auth service — registration, login, token management."""
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.core.security import hash_password, verify_password, create_access_token, create_refresh_token, decode_token
from backend.models.user import User


async def get_user_by_email(db: AsyncIOMotorDatabase, email: str) -> Optional[dict]:
    return await db["users"].find_one({"email": email})


async def get_user_by_id(db: AsyncIOMotorDatabase, user_id: str) -> Optional[dict]:
    from bson import ObjectId
    return await db["users"].find_one({"_id": ObjectId(user_id)})


async def create_user(db: AsyncIOMotorDatabase, email: str, password: str, role: str) -> dict:
    doc = {
        "email": email,
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.utcnow(),
        "is_active": True,
    }
    result = await db["users"].insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


def issue_token_pair(user_doc: dict) -> dict:
    payload = {"sub": str(user_doc["_id"]), "role": user_doc["role"]}
    return {
        "access_token": create_access_token(payload),
        "refresh_token": create_refresh_token(payload),
        "token_type": "bearer",
    }
