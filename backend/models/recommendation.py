"""Recommendation data model."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.models.common import PyObjectId


class Recommendation(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    prediction_id: PyObjectId
    disease: str
    text: str
    priority: int                      # derived from SHAP magnitude
    source: Literal["rule", "llm"]
    requires_physician: bool = False

    model_config = {"populate_by_name": True}
