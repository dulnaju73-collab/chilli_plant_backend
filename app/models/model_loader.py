from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def load_tflite_interpreter(model_path: Path) -> Any:
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    try:
        tf = importlib.import_module("tensorflow")
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to load the configured TFLite model.") from exc
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=2)
    interpreter.allocate_tensors()
    return interpreter
