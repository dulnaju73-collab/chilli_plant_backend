from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parents[2]


def _csv_env(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    firebase_project_id: str | None = os.getenv("FIREBASE_PROJECT_ID") or None
    firebase_storage_bucket: str | None = os.getenv("FIREBASE_STORAGE_BUCKET") or None
    firebase_service_account_path: str | None = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH") or None
    firebase_service_account_json: str | None = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON") or None
    model_path: Path = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models/chilli_leaf_disease_final.tflite")))
    labels_path: Path = Path(os.getenv("LABELS_PATH", str(BASE_DIR / "models/class_names.json")))
    validator_model_path: Path | None = Path(os.environ["IMAGE_VALIDATION_MODEL_PATH"]) if os.getenv("IMAGE_VALIDATION_MODEL_PATH") else None
    validator_positive_class_index: int = int(os.getenv("IMAGE_VALIDATION_POSITIVE_CLASS_INDEX", "1"))
    validator_threshold: float = float(os.getenv("IMAGE_VALIDATION_THRESHOLD", "0.70"))
    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10")) * 1024 * 1024
    max_image_width: int = int(os.getenv("MAX_IMAGE_WIDTH", "8000"))
    max_image_height: int = int(os.getenv("MAX_IMAGE_HEIGHT", "8000"))
    allowed_origins: tuple[str, ...] = _csv_env("ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1")
    max_scans_per_minute: int = int(os.getenv("MAX_SCANS_PER_MINUTE", "10"))
    model_version: str = os.getenv("MODEL_VERSION", "chilli-leaf-disease-v1")

    def __post_init__(self) -> None:
        # Resolve relative paths from the backend root, independent of the
        # process working directory (Render starts the image from /app).
        for field_name in ("model_path", "labels_path", "validator_model_path"):
            value = getattr(self, field_name)
            if value is not None and not value.is_absolute():
                object.__setattr__(self, field_name, BASE_DIR / value)


settings = Settings()
