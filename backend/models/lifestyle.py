"""LifestyleProfile data model."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.models.common import PyObjectId


class LifestyleProfile(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    user_id: PyObjectId
    bmi: float
    family_history: dict[str, bool]   # {"diabetes": True, "cvd": False, "ckd": False}
    smoking_status: Literal["never", "former", "current"]
    alcohol_frequency: Literal["never", "occasional", "regular"]
    exercise_frequency: int            # days/week 0-7
    diet_type: str
    sleep_hours: float                 # hours/night
    stress_level: int                  # 1-10
    medications: list[str]
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = {"populate_by_name": True}
