"""User data model."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field

from backend.models.common import PyObjectId


class User(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    email: EmailStr
    password_hash: str
    role: Literal["patient", "doctor", "admin"]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    model_config = {"populate_by_name": True}
