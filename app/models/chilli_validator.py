from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.models.model_loader import load_tflite_interpreter


class ChilliImageValidator:
    """Dedicated chilli-vs-other image classifier; never substitutes disease labels."""

    def __init__(self, model_path: Path, positive_class_index: int, threshold: float) -> None:
        self.interpreter: Any = load_tflite_interpreter(model_path)
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.positive_class_index = 0 if int(self.output["shape"][-1]) == 1 else positive_class_index
        self.threshold = threshold
        shape = self.input["shape"]
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[-1]) != 3:
            raise ValueError(f"Unsupported validator input shape: {shape}")
        self.height, self.width = int(shape[1]), int(shape[2])
        output_count = int(self.output["shape"][-1])
        if output_count < 1 or self.positive_class_index < 0 or self.positive_class_index >= output_count:
            raise ValueError("IMAGE_VALIDATION_POSITIVE_CLASS_INDEX does not match validator model output")

    def validate(self, rgb_image: Any) -> tuple[bool, float]:
        pixels = np.asarray(rgb_image.resize((self.width, self.height)), dtype=np.float32) / 255.0
        tensor = np.expand_dims(pixels, 0)
        if self.input["dtype"] != np.float32:
            scale, zero_point = self.input.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError("Validator model quantization metadata is unsupported")
            limits = np.iinfo(self.input["dtype"])
            tensor = np.clip(np.round(tensor / scale + zero_point), limits.min, limits.max).astype(self.input["dtype"])
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
        else:
            if np.all(raw >= 0) and np.all(raw <= 1) and abs(float(raw.sum()) - 1) < 0.01:
                scores = raw
            else:
                shifted = raw - np.max(raw)
                exp_values = np.exp(shifted)
                scores = exp_values / exp_values.sum()
            score = float(scores[self.positive_class_index])
        return score >= self.threshold, score
