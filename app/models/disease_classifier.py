from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.models.model_loader import load_tflite_interpreter

EXPECTED_CLASSES = (
    "Bacterial_spot",
    "Cercospora_Leaf_Spot",
    "Curl_Virus",
    "Healthy_Leaf",
    "Nutrition_Deficiency",
    "White_Spot",
)


class DiseaseClassifier:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels = tuple(json.loads(labels_path.read_text(encoding="utf-8")))
        if self.labels != EXPECTED_CLASSES:
            raise ValueError(f"Disease labels must match the trained model mapping: {EXPECTED_CLASSES}")
        self.interpreter: Any = load_tflite_interpreter(model_path)
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        shape = self.input["shape"]
        if len(shape) != 4 or int(shape[0]) != 1 or int(shape[-1]) != 3:
            raise ValueError(f"Unsupported disease model input shape: {shape}")
        self.height, self.width = int(shape[1]), int(shape[2])
        if int(self.output["shape"][-1]) != len(self.labels):
            raise ValueError("Disease model output count does not match class_names.json")

    def predict(self, rgb_image: Any) -> tuple[str, float]:
        image = rgb_image.resize((self.width, self.height), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        expected_dtype = self.input["dtype"]
        if expected_dtype != np.float32:
            scale, zero_point = self.input.get("quantization", (0.0, 0))
            if not scale:
                raise ValueError("Disease model uses unsupported quantization metadata")
            limits = np.iinfo(expected_dtype)
            pixels = np.clip(np.round(pixels / scale + zero_point), limits.min, limits.max).astype(expected_dtype)
        tensor = np.expand_dims(pixels, 0)
        self.interpreter.set_tensor(self.input["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output["index"]).astype(np.float32).reshape(-1)
        if len(raw) != len(self.labels) or not np.all(np.isfinite(raw)):
            raise ValueError("Disease model returned invalid output")
        if self.output["dtype"] != np.float32:
            scale, zero_point = self.output.get("quantization", (0.0, 0))
            if scale:
                raw = (raw - zero_point) * scale
        if np.all(raw >= 0) and np.all(raw <= 1) and abs(float(raw.sum()) - 1) < 0.01:
            probabilities = raw
        else:
            shifted = raw - np.max(raw)
            exp_values = np.exp(shifted)
            probabilities = exp_values / exp_values.sum()
        index = int(np.argmax(probabilities))
        return self.labels[index], float(probabilities[index])
