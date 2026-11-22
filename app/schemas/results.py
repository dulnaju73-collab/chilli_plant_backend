from datetime import datetime

from pydantic import BaseModel


class ScanResult(BaseModel):
    id: str
    disease: str
    confidence: float
    is_healthy: bool
    validation_status: str
    created_at: datetime | None = None
    image_url: str | None = None
    image_path: str | None = None
    prediction_model_version: str


class ResultsPage(BaseModel):
    items: list[ScanResult]
    total: int
    page: int
    page_size: int
    has_more: bool
