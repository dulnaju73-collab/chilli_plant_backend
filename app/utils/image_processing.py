from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.utils.exceptions import InvalidImage

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image
    image_format: str


def decode_image(data: bytes) -> DecodedImage:
    if not data:
        raise InvalidImage("The uploaded image is empty.")
    if len(data) > settings.max_upload_size_bytes:
        raise InvalidImage(f"Image exceeds the {settings.max_upload_size_bytes // (1024 * 1024)} MB upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                image_format = (source.format or "").upper()
                if image_format not in ALLOWED_FORMATS:
                    raise InvalidImage("Upload a JPEG, PNG, or WebP image.")
                width, height = source.size
                if width < 32 or height < 32:
                    raise InvalidImage("Image dimensions must be at least 32 × 32 pixels.")
                if width > settings.max_image_width or height > settings.max_image_height:
                    raise InvalidImage(f"Image dimensions may not exceed {settings.max_image_width} × {settings.max_image_height} pixels.")
                source.verify()
            with Image.open(io.BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
                return DecodedImage(image, image_format)
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning, ValueError) as exc:
        raise InvalidImage("The uploaded image is unreadable or corrupted. Please choose a clear JPEG, PNG, or WebP image.") from exc
