from car_system.config import AppConfig, DetectorConfig, OcrConfig, OcrProbeConfig, OutputConfig
from car_system.runtime import build_runtime


def test_build_runtime_returns_vehicle_and_plate_detectors() -> None:
    config = AppConfig(
        app_name="test-app",
        vehicle_detector=DetectorConfig(
            model_path="weights/vehicle.pt",
            confidence=0.25,
            labels=["car", "truck"],
        ),
        plate_detector=DetectorConfig(
            model_path="weights/plate.pt",
            confidence=0.30,
            labels=["plate"],
        ),
        ocr=OcrConfig(language="en", use_angle_cls=False),
        output=OutputConfig(directory="outputs", save_images=True, save_video=True),
    )

    vehicle_detector, plate_detector, ocr, probe_ocr, rescue_probe_ocr, secondary_rescue_probe_ocr = build_runtime(config)

    assert vehicle_detector.__class__.__name__ == "YoloDetector"
    assert plate_detector.__class__.__name__ == "YoloDetector"
    assert ocr.__class__.__name__ == "PaddlePlateOCR"
    assert probe_ocr is None
    assert rescue_probe_ocr is None
    assert secondary_rescue_probe_ocr is None


def test_build_runtime_passes_detector_device_settings() -> None:
    config = AppConfig(
        app_name="test-app",
        vehicle_detector=DetectorConfig(
            model_path="weights/vehicle.pt",
            confidence=0.25,
            labels=["car", "truck"],
            device="cpu",
        ),
        plate_detector=DetectorConfig(
            model_path="weights/plate.pt",
            confidence=0.30,
            labels=["plate"],
            device="cpu",
        ),
        ocr=OcrConfig(language="en", use_angle_cls=False),
        output=OutputConfig(directory="outputs", save_images=True, save_video=True),
    )

    vehicle_detector, plate_detector, _, probe_ocr, rescue_probe_ocr, secondary_rescue_probe_ocr = build_runtime(config)

    assert vehicle_detector.device == "cpu"
    assert plate_detector.device == "cpu"
    assert probe_ocr is None
    assert rescue_probe_ocr is None
    assert secondary_rescue_probe_ocr is None


def test_build_runtime_passes_specialized_ocr_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddlePlateOCR:
        def __init__(self, **kwargs):
            captured.setdefault("kwargs", []).append(kwargs)

    monkeypatch.setattr("car_system.runtime.PaddlePlateOCR", FakePaddlePlateOCR)

    config = AppConfig(
        app_name="test-app",
        vehicle_detector=DetectorConfig(
            model_path="weights/vehicle.pt",
            confidence=0.25,
            labels=["car", "truck"],
        ),
        plate_detector=DetectorConfig(
            model_path="weights/plate.pt",
            confidence=0.30,
            labels=["plate"],
        ),
        ocr=OcrConfig(
            language="ch",
            use_angle_cls=False,
            mode="specialized",
            model_dir="weights/plate_rec/inference",
            character_dict_path="weights/plate_rec/dicts/plate_dict.txt",
            probe=OcrProbeConfig(
                enabled=True,
                model_dir="weights/plate_rec_probe/inference",
                character_dict_path="weights/plate_rec_probe/dicts/plate_dict.txt",
                min_confidence=0.95,
            ),
            rescue_probe=OcrProbeConfig(
                enabled=True,
                model_dir="weights/plate_rec_rescue/inference",
                character_dict_path="weights/plate_rec_rescue/dicts/plate_dict.txt",
                min_confidence=0.90,
                rescue_requires_any_char=("D", "T", "Z"),
            ),
            secondary_rescue_probe=OcrProbeConfig(
                enabled=True,
                model_dir="weights/plate_rec_rescue_2/inference",
                character_dict_path="weights/plate_rec_rescue_2/dicts/plate_dict.txt",
                min_confidence=0.91,
                rescue_requires_any_char=("D", "T", "Z"),
                rescue_require_alpha_count=2,
            ),
        ),
        output=OutputConfig(directory="outputs", save_images=True, save_video=True),
    )

    _, _, ocr, probe_ocr, rescue_probe_ocr, secondary_rescue_probe_ocr = build_runtime(config)

    assert ocr.__class__.__name__ == "FakePaddlePlateOCR"
    assert probe_ocr.__class__.__name__ == "FakePaddlePlateOCR"
    assert rescue_probe_ocr.__class__.__name__ == "FakePaddlePlateOCR"
    assert secondary_rescue_probe_ocr.__class__.__name__ == "FakePaddlePlateOCR"
    assert captured["kwargs"] == [
        {
            "language": "ch",
            "use_angle_cls": False,
            "mode": "specialized",
            "model_dir": "weights/plate_rec/inference",
            "character_dict_path": "weights/plate_rec/dicts/plate_dict.txt",
            "min_confidence": 0.0,
        },
        {
            "language": "ch",
            "use_angle_cls": False,
            "mode": "specialized",
            "model_dir": "weights/plate_rec_probe/inference",
            "character_dict_path": "weights/plate_rec_probe/dicts/plate_dict.txt",
            "min_confidence": 0.95,
        },
        {
            "language": "ch",
            "use_angle_cls": False,
            "mode": "specialized",
            "model_dir": "weights/plate_rec_rescue/inference",
            "character_dict_path": "weights/plate_rec_rescue/dicts/plate_dict.txt",
            "min_confidence": 0.90,
        },
        {
            "language": "ch",
            "use_angle_cls": False,
            "mode": "specialized",
            "model_dir": "weights/plate_rec_rescue_2/inference",
            "character_dict_path": "weights/plate_rec_rescue_2/dicts/plate_dict.txt",
            "min_confidence": 0.91,
        },
    ]
