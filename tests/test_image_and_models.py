import io

import numpy as np
import pytest
from PIL import Image

from app.models.chilli_validator import ChilliImageValidator
from app.models.disease_classifier import DiseaseClassifier, EXPECTED_CLASSES
from app.utils.exceptions import InvalidImage
from app.utils.image_processing import decode_image


def png_bytes(color=(20, 140, 40)):
    stream = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(stream, format="PNG")
    return stream.getvalue()


def test_valid_png_is_decoded_to_rgb():
    decoded = decode_image(png_bytes())
    assert decoded.image_format == "PNG"
    assert decoded.image.mode == "RGB"
    assert decoded.image.size == (64, 64)


@pytest.mark.parametrize("data", [b"not an image", b"\x89PNG\r\n\x1a\ncorrupt"])
def test_corrupt_image_is_rejected(data):
    with pytest.raises(InvalidImage):
        decode_image(data)


def test_image_size_limit_is_enforced(monkeypatch):
    from app.utils import image_processing

    class SmallLimit:
        max_upload_size_bytes = 30
        max_image_width = 8000
        max_image_height = 8000

    monkeypatch.setattr(image_processing, "settings", SmallLimit())
    with pytest.raises(InvalidImage, match="upload limit"):
        decode_image(png_bytes())


class FakeInterpreter:
    def __init__(self, input_shape, output):
        self.input = {"shape": np.array(input_shape), "dtype": np.float32, "index": 0}
        self.output_detail = {"shape": np.array([1, len(output)]), "dtype": np.float32, "index": 1}
        self.output = np.array([output], dtype=np.float32)

    def get_input_details(self):
        return [self.input]

    def get_output_details(self):
        return [self.output_detail]

    def set_tensor(self, _, value):
        self.value = value

    def invoke(self):
        pass

    def get_tensor(self, _):
        return self.output


@pytest.mark.parametrize("class_index", range(6))
def test_disease_prediction_returns_each_supported_class(class_index):
    classifier = DiseaseClassifier.__new__(DiseaseClassifier)
    classifier.width = 4
    classifier.height = 4
    classifier.labels = EXPECTED_CLASSES
    probabilities = np.zeros(6, dtype=np.float32)
    probabilities[class_index] = 1
    classifier.input = {"shape": np.array([1, 4, 4, 3]), "dtype": np.float32, "index": 0}
    classifier.output = {"shape": np.array([1, 6]), "dtype": np.float32, "index": 1}
    classifier.interpreter = FakeInterpreter([1, 4, 4, 3], probabilities)
    disease, confidence = classifier.predict(Image.new("RGB", (64, 64), "green"))
    assert disease == EXPECTED_CLASSES[class_index]
    assert confidence == pytest.approx(1.0)


@pytest.mark.parametrize("output, expected", [([0.05, 0.95], True), ([0.9, 0.1], False)])
def test_validator_accepts_or_rejects_using_positive_class_score(output, expected):
    validator = ChilliImageValidator.__new__(ChilliImageValidator)
    validator.width = 4
    validator.height = 4
    validator.positive_class_index = 1
    validator.threshold = 0.7
    validator.input = {"shape": np.array([1, 4, 4, 3]), "dtype": np.float32, "index": 0}
    validator.output = {"shape": np.array([1, 2]), "dtype": np.float32, "index": 1}
    validator.interpreter = FakeInterpreter([1, 4, 4, 3], output)
    valid, confidence = validator.validate(Image.new("RGB", (64, 64), "green"))
    assert valid is expected
    assert confidence == pytest.approx(output[1])


@pytest.mark.parametrize(
    "raw_output, invert_score, expected_valid, expected_score",
    [
        ([0.92], False, True, 0.92),
        ([0.20], False, False, 0.20),
        ([0.08], True, True, 0.92),
        ([0.85], True, False, 0.15),
    ],
)
def test_validator_supports_single_output_sigmoid_and_score_inversion(raw_output, invert_score, expected_valid, expected_score):
    validator = ChilliImageValidator.__new__(ChilliImageValidator)
    validator.width = 4
    validator.height = 4
    validator.positive_class_index = 0
    validator.threshold = 0.7
    validator.invert_score = invert_score
    validator.input_scale = "0_1"
    validator.input = {"shape": np.array([1, 4, 4, 3]), "dtype": np.float32, "index": 0}
    validator.output = {"shape": np.array([1, 1]), "dtype": np.float32, "index": 1}
    validator.interpreter = FakeInterpreter([1, 4, 4, 3], raw_output)
    valid, confidence = validator.validate(Image.new("RGB", (64, 64), "green"))
    assert valid is expected_valid
    assert confidence == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "input_scale, expected_channel_value",
    [
        ("0_1", 128.0 / 255.0),
        ("raw_255", 128.0),
        ("neg1_1", 128.0 / 127.5 - 1.0),
    ],
)
def test_validator_applies_configured_input_scale(input_scale, expected_channel_value):
    validator = ChilliImageValidator.__new__(ChilliImageValidator)
    validator.width = 4
    validator.height = 4
    validator.positive_class_index = 0
    validator.threshold = 0.5
    validator.invert_score = False
    validator.input_scale = input_scale
    validator.input = {"shape": np.array([1, 4, 4, 3]), "dtype": np.float32, "index": 0}
    validator.output = {"shape": np.array([1, 1]), "dtype": np.float32, "index": 1}
    interpreter = FakeInterpreter([1, 4, 4, 3], [0.9])
    validator.interpreter = interpreter
    validator.validate(Image.new("RGB", (8, 8), (128, 128, 128)))
    assert interpreter.value.shape == (1, 4, 4, 3)
    assert np.allclose(interpreter.value, expected_channel_value, atol=1e-5)


def test_validator_rejects_unsupported_input_scale(tmp_path):
    with pytest.raises(ValueError, match="IMAGE_VALIDATION_INPUT_SCALE"):
        ChilliImageValidator(tmp_path / "missing.tflite", 1, 0.7, input_scale="bad_scale")
