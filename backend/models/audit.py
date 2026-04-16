"""AuditLog data model — records doctor access to patient records."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.models.common import PyObjectId


class AuditLog(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    doctor_id: PyObjectId
    patient_id: PyObjectId
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    endpoint: str = ""

    model_config = {"populate_by_name": True}
