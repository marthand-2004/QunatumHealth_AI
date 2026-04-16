"""Report data model."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from backend.models.common import PyObjectId


class Report(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    patient_id: PyObjectId
    generated_by: PyObjectId           # user who requested
    prediction_id: PyObjectId
    file_path: str
    share_token: Optional[str] = None
    share_expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}
