from datetime import datetime

from pydantic import BaseModel


class ValidationResponse(BaseModel):
    valid: bool
    message: str
    confidence: float | None = None


class PredictionData(BaseModel):
    id: str
    disease: str
    confidence: float
    is_healthy: bool
    created_at: datetime
    image_url: str | None = None
    image_path: str | None = None
    validation_status: str


class PredictionResponse(BaseModel):
    success: bool = True
    prediction: PredictionData
