from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.dependencies.authentication import current_user
from app.schemas.prediction import PredictionData, PredictionResponse, ValidationResponse
from app.services.firebase_service import FirebaseService
from app.services.image_validation_service import ImageValidationService
from app.services.prediction_service import PredictionService
from app.utils.exceptions import InvalidImage, ValidatorUnavailable
from app.utils.image_processing import decode_image

router = APIRouter(prefix="/predictions", tags=["predictions"])
service: PredictionService | None = None


def configure_prediction_service(prediction_service: PredictionService) -> None:
    global service
    service = prediction_service


async def _read_image(file: UploadFile) -> tuple[bytes, Any, str]:
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    data = await file.read(settings.max_upload_size_bytes + 1)
    try:
        decoded = await run_in_threadpool(decode_image, data)
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    expected_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[decoded.image_format]
    if file.content_type != expected_mime:
        raise HTTPException(status_code=415, detail="The image file signature does not match its declared content type.")
    return data, decoded.image, decoded.image_format


def _service() -> PredictionService:
    if service is None:
        raise HTTPException(status_code=503, detail="Prediction services are not initialized.")
    return service


@router.get("/classes")
def get_classes() -> dict[str, Any]:
    return {"classes": ["Bacterial Spot", "Cercospora Leaf Spot", "Curl Virus", "Healthy Leaf", "Nutrition Deficiency", "White Spot"], "model": settings.model_version}


@router.post("/validate", response_model=ValidationResponse)
async def validate_image(file: UploadFile = File(...), claims: dict[str, Any] = Depends(current_user)) -> ValidationResponse:
    _, image, _ = await _read_image(file)
    try:
        prediction_service = _service()
        prediction_service.check_rate_limit(claims["uid"], "validate")
        return await run_in_threadpool(prediction_service.validate, image)
    except ValidatorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/analyze", response_model=PredictionResponse)
async def analyze_image(file: UploadFile = File(...), claims: dict[str, Any] = Depends(current_user)) -> dict[str, Any]:
    data, image, image_format = await _read_image(file)
    try:
        result = await run_in_threadpool(_service().analyze, claims["uid"], data, image, image_format)
    except ValidatorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"success": True, "prediction": result}
