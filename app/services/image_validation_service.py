from __future__ import annotations

from typing import Any

from app.models.chilli_validator import ChilliImageValidator
from app.utils.exceptions import ValidatorUnavailable


class ImageValidationService:
    def __init__(self, validator: ChilliImageValidator | None) -> None:
        self.validator = validator

    @property
    def ready(self) -> bool:
        return self.validator is not None

    def validate(self, image: Any) -> tuple[bool, float]:
        if self.validator is None:
            raise ValidatorUnavailable("Chilli image validation is not configured. Set IMAGE_VALIDATION_MODEL_PATH to a separately trained chilli-versus-other-plant model.")
        return self.validator.validate(image)
