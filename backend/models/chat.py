"""ChatSession and ChatMessage data models."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.models.common import PyObjectId


class ChatSession(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}


class ChatMessage(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    session_id: PyObjectId
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}
