from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app.core.config import settings
from app.core.firebase import FirebaseUnavailable
from app.models.disease_classifier import DiseaseClassifier
from app.schemas.prediction import ValidationResponse
from app.services.firebase_service import FirebaseService
from app.services.image_validation_service import ImageValidationService
from app.utils.exceptions import ValidatorUnavailable


class PredictionService:
    def __init__(self, classifier: DiseaseClassifier | None, validator: ImageValidationService, firebase_service: FirebaseService) -> None:
        self.classifier = classifier
        self.validator = validator
        self.firebase_service = firebase_service
        self._scans: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check_rate_limit(self, uid: str, operation: str = "analyze") -> None:
        now = time.monotonic()
        with self._lock:
            scans = self._scans[(uid, operation)]
            while scans and now - scans[0] >= 60:
                scans.popleft()
            if len(scans) >= settings.max_scans_per_minute:
                from fastapi import HTTPException
                raise HTTPException(status_code=429, detail="Scan limit reached. Please wait a minute and try again.")
            scans.append(now)

    def validate(self, image: Any) -> ValidationResponse:
        try:
            valid, confidence = self.validator.validate(image)
        except ValidatorUnavailable:
            raise
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="The chilli image validator failed. Check its model configuration and try again.") from exc
        return ValidationResponse(valid=valid, confidence=round(confidence * 100, 2), message="Chilli plant image successfully validated." if valid else "The uploaded image does not appear to contain a chilli plant. Please upload a clear chilli leaf image.")

    def analyze(self, uid: str, image_data: bytes, image: Any, image_format: str) -> dict[str, Any]:
        self.check_rate_limit(uid, "analyze")
        try:
            valid, _ = self.validator.validate(image)
        except ValidatorUnavailable:
            raise
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="The chilli image validator failed. Disease classification was not run.") from exc
        if not valid:
            from fastapi import HTTPException
            raise HTTPException(status_code=422, detail={
                "code": "INVALID_PLANT_IMAGE",
                "message": "The uploaded image does not appear to contain a chilli plant. Please upload a clear chilli leaf image.",
            })
        if self.classifier is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="Disease model is unavailable. Configure MODEL_PATH and install the backend model dependencies.")
        try:
            disease, confidence = self.classifier.predict(image)
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Disease classification failed. Please try a different clear image.") from exc
        try:
            result = self.firebase_service.save_scan(uid, disease, confidence, image_data, image_format, settings.model_version)
        except FirebaseUnavailable as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail="Prediction completed, but the scan could not be saved. Please retry.") from exc
        result["created_at"] = result.get("created_at") or datetime.now(timezone.utc)
        return result
