"""Document and LabParameter data models."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.models.common import PyObjectId


class LabParameter(BaseModel):
    name: str                          # e.g. "glucose"
    value: float
    unit: str                          # normalized SI unit
    reference_range: tuple[float, float]
    is_abnormal: bool
    raw_text: str                      # original OCR text


class Document(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    filename: str
    file_hash: str
    file_size_bytes: int
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    ocr_status: Literal["pending", "processing", "complete", "failed"] = "pending"
    lab_parameters: list[LabParameter] = Field(default_factory=list)
    verified: bool = False
    verified_at: Optional[datetime] = None

    model_config = {"populate_by_name": True}
