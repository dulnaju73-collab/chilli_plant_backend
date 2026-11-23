from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.models.model_loader import load_tflite_interpreter

SUPPORTED_INPUT_SCALES = frozenset({"0_1", "raw_255", "neg1_1"})


class ChilliImageValidator:
    """Dedicated chilli-vs-other image classifier; never substitutes disease labels."""

    def __init__(
        self,
        model_path: Path,
        positive_class_index: int,
        threshold: float,
        *,
        invert_score: bool = False,
        input_scale: str = "0_1",
    ) -> None:
        normalized_scale = input_scale.strip().lower()
        if normalized_scale not in SUPPORTED_INPUT_SCALES:
            raise ValueError(
                f"Unsupported IMAGE_VALIDATION_INPUT_SCALE: {input_scale!r}. "
                f"Expected one of {sorted(SUPPORTED_INPUT_SCALES)}."
            )
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("IMAGE_VALIDATION_THRESHOLD must be between 0.0 and 1.0")
        self.interpreter: Any = load_tflite_interpreter(model_path)
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.positive_class_index = 0 if int(self.output["shape"][-1]) == 1 else positive_class_index
        self.threshold = threshold
        self.invert_score = invert_score
        self.input_scale = normalized_scale
        shape = self.input["shape"]
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[-1]) != 3:
            raise ValueError(f"Unsupported validator input shape: {shape}")
        self.height, self.width = int(shape[1]), int(shape[2])
        output_count = int(self.output["shape"][-1])
        if output_count < 1 or self.positive_class_index < 0 or self.positive_class_index >= output_count:
            raise ValueError("IMAGE_VALIDATION_POSITIVE_CLASS_INDEX does not match validator model output")

    def _preprocess(self, rgb_image: Any) -> np.ndarray:
        resized = rgb_image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32)
        input_scale = getattr(self, "input_scale", "0_1")
        if input_scale == "0_1":
            pixels = pixels / 255.0
        elif input_scale == "neg1_1":
            pixels = pixels / 127.5 - 1.0
        elif input_scale != "raw_255":
            raise ValueError(f"Unsupported validator input_scale: {input_scale!r}")
        tensor = np.expand_dims(pixels, 0)
        if self.input["dtype"] != np.float32:
            scale, zero_point = self.input.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError("Validator model quantization metadata is unsupported")
            limits = np.iinfo(self.input["dtype"])
            tensor = np.clip(np.round(tensor / scale + zero_point), limits.min, limits.max).astype(self.input["dtype"])
        return tensor

    def validate(self, rgb_image: Any) -> tuple[bool, float]:
        tensor = self._preprocess(rgb_image)
        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output["index"]).astype(np.float32).reshape(-1)
        if self.output["dtype"] != np.float32:
            scale, zero_point = self.output.get("quantization", (0.0, 0))
            if scale:
                raw = (raw - zero_point) * scale
        if not np.all(np.isfinite(raw)):
            raise ValueError("Validator model returned invalid output")
        if len(raw) == 1:
            score = float(raw[0])
            if score < 0 or score > 1:
                score = float(1 / (1 + np.exp(-score)))
            if getattr(self, "invert_score", False):
                score = 1.0 - score
        else:
            if np.all(raw >= 0) and np.all(raw <= 1) and abs(float(raw.sum()) - 1) < 0.01:
                scores = raw
            else:
                shifted = raw - np.max(raw)
                exp_values = np.exp(shifted)
                scores = exp_values / exp_values.sum()
            score = float(scores[self.positive_class_index])
        return score >= self.threshold, score
