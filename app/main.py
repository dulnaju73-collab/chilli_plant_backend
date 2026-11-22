from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import predictions
from app.api.routes.predictions import router as predictions_router
from app.api.routes.results import router as results_router
from app.api.routes.users import router as users_router
from app.core.config import settings
from app.core.firebase import firebase
from app.models.chilli_validator import ChilliImageValidator
from app.models.disease_classifier import DiseaseClassifier
from app.services.firebase_service import FirebaseService
from app.services.image_validation_service import ImageValidationService
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    firebase.initialize()
    classifier = None
    validator = None
    try:
        classifier = DiseaseClassifier(settings.model_path, settings.labels_path)
        logger.info("Disease classifier loaded: %s", settings.model_path)
    except Exception:
        logger.exception("Disease classifier could not be loaded")
    if settings.validator_model_path:
        try:
            validator = ChilliImageValidator(settings.validator_model_path, settings.validator_positive_class_index, settings.validator_threshold)
            logger.info("Chilli validation model loaded: %s", settings.validator_model_path)
        except Exception:
            logger.exception("Chilli image validator could not be loaded")
    app.state.classifier = classifier
    app.state.validator = validator
    app.state.prediction_service = PredictionService(classifier, ImageValidationService(validator), FirebaseService())
    predictions.configure_prediction_service(app.state.prediction_service)
    yield


app = FastAPI(
    title="Chilli Plant Health API",
    description="Authenticated API for validating chilli images, classifying leaf health, and managing private scan history.",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins), allow_credentials=True, allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], allow_headers=["Authorization", "Content-Type"])
api = APIRouter(prefix="/api/v1")
api.include_router(users_router)
api.include_router(predictions_router)
api.include_router(results_router)
app.include_router(api)


@app.get("/", tags=["health"])
def root() -> dict[str, str]:
    return {"message": "Chilli Plant Health API", "docs": "/docs"}


@app.get("/health", tags=["health"])
@app.get("/api/v1/health", tags=["health"])
def health() -> dict[str, Any]:
    classifier = getattr(app.state, "classifier", None)
    validator = getattr(app.state, "validator", None)
    return {
        "status": "ready" if classifier and validator and firebase.ready else "degraded",
        "disease_model_ready": classifier is not None,
        "validator_ready": validator is not None,
        "firebase_ready": firebase.ready,
        "disease_model": settings.model_path.name,
        "model_version": settings.model_version,
        "classes": ["Bacterial Spot", "Cercospora Leaf Spot", "Curl Virus", "Healthy Leaf", "Nutrition Deficiency", "White Spot"],
    }
