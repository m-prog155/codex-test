import sys
from types import SimpleNamespace

from car_system.ocr.plate_ocr import PaddlePlateOCR


class FakeTextRecognizer:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, input, batch_size=1):
        self.calls.append({"input": input, "batch_size": batch_size})
        if isinstance(self.results, dict):
            return self.results.get(input[0], [])
        return self.results


def test_recognize_reads_textrecognition_result_payload() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    fake = FakeTextRecognizer(
        [
            {
                "rec_text": "ABC123",
                "rec_score": 0.91,
            }
        ]
    )
    ocr._ocr = fake

    result = ocr.recognize(image="plate-crop")

    assert result is not None
    assert result.text == "ABC123"
    assert result.confidence == 0.91
    assert fake.calls == [{"input": ["plate-crop"], "batch_size": 1}]


def test_recognize_returns_none_when_predictor_has_no_text() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    ocr._ocr = FakeTextRecognizer([])

    result = ocr.recognize(image="plate-crop")

    assert result is None


def test_recognize_prefers_more_plausible_candidate_text() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "AB", "rec_score": 0.8}],
            "enhanced": [{"rec_text": "#ANH16288", "rec_score": 0.76}],
            "contrast": [{"rec_text": "", "rec_score": 0.99}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original", "enhanced", "contrast"]  # type: ignore[method-assign]

    result = ocr.recognize(image="plate-crop")

    assert result is not None
    assert result.text == "ANH16288"
    assert result.confidence == 0.76


def test_recognize_normalizes_obvious_ambiguous_characters_conservatively() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "#A0V79I", "rec_score": 0.9}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize(image="plate-crop")

    assert result is not None
    assert result.text == "A0V791"
    assert result.confidence == 0.9


def test_recognize_raw_specialized_keeps_ambiguous_letters_in_standard_plate_text() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="specialized", model_dir="D:/models/plate_recognition")
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖A7Q653", "rec_score": 0.9}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is not None
    assert result.normalized_text == "皖A7Q653"


def test_recognize_raw_specialized_keeps_b_in_region_body() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="specialized", model_dir="D:/models/plate_recognition")
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖AB618B", "rec_score": 0.9}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is not None
    assert result.normalized_text == "皖AB618B"


def test_recognize_discards_overly_short_results() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "AEN", "rec_score": 0.95}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize(image="plate-crop")

    assert result is None


def test_recognize_raw_returns_unmodified_candidate_text() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖AY339S", "rec_score": 0.9}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is not None
    assert result.text == "皖AY339S"
    assert result.confidence == 0.9


def test_recognize_raw_specialized_rejects_short_plate_like_text() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="specialized", model_dir="D:/models/plate_recognition")
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖AKZ63", "rec_score": 0.99}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is None


def test_recognize_raw_specialized_rejects_nonstandard_single_plate_pattern() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="specialized", model_dir="D:/models/plate_recognition")
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖皖A8M333", "rec_score": 0.99}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is None


def test_recognize_raw_specialized_accepts_standard_single_plate_text() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="specialized", model_dir="D:/models/plate_recognition")
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖ANN665", "rec_score": 0.93}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is not None
    assert result.normalized_text == "皖ANN665"
    assert result.confidence == 0.93


def test_build_candidate_images_specialized_uses_only_original_image() -> None:
    import numpy as np

    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="specialized", model_dir="D:/models/plate_recognition")
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    candidates = ocr._build_candidate_images(image)

    assert len(candidates) == 1
    assert candidates[0] is image


def test_build_candidate_images_generic_keeps_augmented_candidates() -> None:
    import numpy as np

    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="generic")
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    candidates = ocr._build_candidate_images(image)

    assert len(candidates) == 3


def test_recognize_raw_specialized_rejects_low_confidence_result() -> None:
    ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir="D:/models/plate_recognition",
        min_confidence=0.93,
    )
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖ANN665", "rec_score": 0.92}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is None


def test_recognize_raw_specialized_accepts_result_at_min_confidence() -> None:
    ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir="D:/models/plate_recognition",
        min_confidence=0.93,
    )
    ocr._ocr = FakeTextRecognizer(
        {
            "original": [{"rec_text": "皖ANN665", "rec_score": 0.93}],
        }
    )
    ocr._build_candidate_images = lambda image: ["original"]  # type: ignore[method-assign]

    result = ocr.recognize_raw(image="plate-crop")

    assert result is not None
    assert result.normalized_text == "皖ANN665"


def test_load_generic_uses_default_text_recognition(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=FakeTextRecognition))

    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    ocr.load()

    assert captured_kwargs == {"model_name": "PP-OCRv5_mobile_rec"}
    assert ocr._ocr is not None


def test_load_specialized_passes_model_dir_and_omits_character_dict(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=FakeTextRecognition))

    ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir="D:/models/plate_recognition",
        character_dict_path="D:/models/plate_dict.txt",
    )
    ocr.load()

    assert captured_kwargs == {
        "model_dir": "D:/models/plate_recognition",
    }
    assert "character_dict_path" not in captured_kwargs
    assert ocr._ocr is not None


def test_load_specialized_requires_model_dir() -> None:
    ocr = PaddlePlateOCR(language="ch", use_angle_cls=False, mode="specialized")

    try:
        ocr.load()
    except ValueError as exc:
        assert "model_dir" in str(exc)
    else:
        raise AssertionError("Expected load() to require model_dir in specialized mode")


def test_recognize_auto_loads_specialized_model_and_returns_result(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def predict(self, input, batch_size=1):
            return [{"rec_text": "皖A0V79I", "rec_score": 0.93}]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=FakeTextRecognition))

    ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir="D:/models/plate_recognition",
        character_dict_path="D:/models/plate_dict.txt",
    )

    assert ocr._ocr is None

    result = ocr.recognize(image="plate-crop")

    assert result is not None
    assert result.text == "皖A0V79I"
    assert result.confidence == 0.93
    assert ocr._ocr is not None
    assert captured_kwargs == {
        "model_dir": "D:/models/plate_recognition",
    }
    assert "character_dict_path" not in captured_kwargs


def test_load_specialized_passes_explicit_model_name(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=FakeTextRecognition))

    ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir="D:/models/plate_recognition",
        model_name="PP-OCRv5_server_rec",
    )
    ocr.load()

    assert captured_kwargs == {
        "model_dir": "D:/models/plate_recognition",
        "model_name": "PP-OCRv5_server_rec",
    }


def test_load_specialized_falls_back_to_mobile_model_name_when_bare_model_dir_mismatches(monkeypatch) -> None:
    captured_calls: list[dict[str, object]] = []

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            captured_calls.append(kwargs)
            if kwargs == {"model_dir": "D:/models/plate_recognition"}:
                raise AssertionError("Model name mismatch")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=FakeTextRecognition))

    ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir="D:/models/plate_recognition",
    )
    ocr.load()

    assert captured_calls == [
        {"model_dir": "D:/models/plate_recognition"},
        {"model_dir": "D:/models/plate_recognition", "model_name": "PP-OCRv5_mobile_rec"},
    ]
