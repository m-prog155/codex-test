import numpy as np

from car_system.config import AppConfig, DetectorConfig, OcrConfig, OutputConfig
from car_system.pipeline.runner import PipelineRunner
from car_system.types import Detection, PlateRecognition


class FakeVehicleDetector:
    def predict(self, image):
        return [
            Detection(label="car", confidence=0.91, bbox=(10, 10, 140, 140)),
            Detection(label="truck", confidence=0.80, bbox=(180, 20, 310, 170)),
        ]


class FakePlateDetector:
    def predict(self, image):
        return [
            Detection(label="plate", confidence=0.95, bbox=(40, 90, 95, 115)),
            Detection(label="plate", confidence=0.93, bbox=(210, 110, 265, 140)),
        ]


class FakeOCR:
    def __init__(self) -> None:
        self.last_image_shape = None

    def recognize(self, image):
        self.last_image_shape = getattr(image, "shape", None)
        return PlateRecognition(text="ABC123", confidence=0.88)


def make_config() -> AppConfig:
    return AppConfig(
        app_name="test-app",
        vehicle_detector=DetectorConfig(
            model_path="weights/vehicle.pt",
            confidence=0.3,
            labels=["car", "truck"],
        ),
        plate_detector=DetectorConfig(
            model_path="weights/plate.pt",
            confidence=0.25,
            labels=["plate"],
        ),
        ocr=OcrConfig(language="en", use_angle_cls=False),
        output=OutputConfig(directory="outputs", save_images=True, save_video=True),
    )


def test_run_frame_attaches_ocr_result_to_plate_match() -> None:
    fake_ocr = FakeOCR()
    runner = PipelineRunner(
        config=make_config(),
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((160, 160, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="frame.jpg", frame_index=3)

    assert result.source_name == "frame.jpg"
    assert result.frame_index == 3
    assert len(result.matches) == 2
    assert len(result.detections) == 4
    assert result.matches[0].vehicle is not None
    assert result.matches[0].vehicle.label == "car"
    assert result.matches[0].recognition is not None
    assert result.matches[0].recognition.text == "ABC123"
    assert result.matches[1].vehicle is not None
    assert result.matches[1].vehicle.label == "truck"
    assert fake_ocr.last_image_shape is not None


class EdgeVehicleDetector:
    def predict(self, image):
        return [
            Detection(label="car", confidence=0.9, bbox=(0, 0, 12, 12)),
        ]


class EdgePlateDetector:
    def predict(self, image):
        return [
            Detection(label="plate", confidence=0.95, bbox=(1, 1, 10, 5)),
        ]


def test_run_frame_expands_plate_crop_but_clamps_to_image_bounds() -> None:
    fake_ocr = FakeOCR()
    runner = PipelineRunner(
        config=make_config(),
        vehicle_detector=EdgeVehicleDetector(),
        plate_detector=EdgePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert result.matches[0].recognition is not None
    assert fake_ocr.last_image_shape == (48, 168, 3)


def test_run_frame_records_raw_and_normalized_ocr_text() -> None:
    class DiagnosticOCR:
        def recognize_raw(self, image):
            return PlateRecognition(
                text="皖A1234O",
                confidence=0.91,
                raw_text="皖A1234O",
                normalized_text="皖A12340",
            )

    runner = PipelineRunner(
        config=make_config(),
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr_engine=DiagnosticOCR(),
    )
    image = np.zeros((160, 160, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="frame.jpg", frame_index=0)

    assert result.matches[0].recognition is not None
    assert result.matches[0].recognition.raw_text == "皖A1234O"
    assert result.matches[0].recognition.normalized_text == "皖A12340"
    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.status == "recognized"
    assert result.matches[0].diagnostic.crop_bbox == (36, 87, 99, 118)


def test_run_frame_marks_repeat_pattern_as_abnormal_text() -> None:
    class RepeatOCR:
        def recognize_raw(self, image):
            return PlateRecognition(
                text="皖A皖A8278",
                confidence=0.80,
                raw_text="皖A皖A8278",
                normalized_text="皖A皖A8278",
            )

    runner = PipelineRunner(
        config=make_config(),
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr_engine=RepeatOCR(),
    )
    image = np.zeros((160, 160, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="frame.jpg", frame_index=0)

    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.status == "ocr_abnormal_text"


def test_run_frame_uses_configured_crop_padding_ratios() -> None:
    config = make_config()
    config.ocr.crop_pad_x_ratio = 0.0
    config.ocr.crop_pad_y_ratio = 0.0

    fake_ocr = FakeOCR()
    runner = PipelineRunner(
        config=config,
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((160, 160, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="frame.jpg", frame_index=0)

    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.crop_bbox == (40, 90, 95, 115)


def test_run_frame_skips_rectification_when_disabled() -> None:
    config = make_config()
    config.ocr.enable_rectification = False

    fake_ocr = FakeOCR()
    runner = PipelineRunner(
        config=config,
        vehicle_detector=EdgeVehicleDetector(),
        plate_detector=EdgePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.rectified_shape is None
    assert fake_ocr.last_image_shape == (4, 11, 3)


def test_run_frame_safe_rectification_falls_back_to_plain_crop_when_rejected(monkeypatch) -> None:
    config = make_config()
    config.ocr.enable_rectification = True
    config.ocr.rectification_mode = "safe"

    class FakeRectifyResult:
        def __init__(self) -> None:
            self.image = np.zeros((25, 55, 3), dtype=np.uint8)
            self.applied = False
            self.reason = "center_offset"

    monkeypatch.setattr("car_system.pipeline.runner.safe_rectify_plate", lambda *args, **kwargs: FakeRectifyResult())

    fake_ocr = FakeOCR()
    runner = PipelineRunner(
        config=config,
        vehicle_detector=EdgeVehicleDetector(),
        plate_detector=EdgePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert fake_ocr.last_image_shape == (4, 11, 3)
    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.rectification_applied is False
    assert result.matches[0].diagnostic.rectification_reason == "center_offset"
    assert result.matches[0].diagnostic.rectification_mode == "safe"


def test_run_frame_safe_rectification_uses_rectified_image_when_applied(monkeypatch) -> None:
    config = make_config()
    config.ocr.enable_rectification = True
    config.ocr.rectification_mode = "safe"

    class FakeRectifyResult:
        def __init__(self) -> None:
            self.image = np.zeros((48, 168, 3), dtype=np.uint8)
            self.applied = True
            self.reason = "applied"

    monkeypatch.setattr("car_system.pipeline.runner.safe_rectify_plate", lambda *args, **kwargs: FakeRectifyResult())

    class SafeAppliedOCR:
        def __init__(self) -> None:
            self.shapes = []

        def recognize_raw(self, image):
            self.shapes.append(getattr(image, "shape", None))
            return PlateRecognition(text="ABC123", confidence=0.88, raw_text="ABC123", normalized_text="ABC123")

    fake_ocr = SafeAppliedOCR()
    runner = PipelineRunner(
        config=config,
        vehicle_detector=EdgeVehicleDetector(),
        plate_detector=EdgePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert fake_ocr.shapes == [(48, 168, 3), (4, 11, 3)]
    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.rectification_applied is True
    assert result.matches[0].diagnostic.rectification_reason == "applied"


def test_run_frame_safe_rectification_prefers_plain_text_when_plain_and_rectified_disagree(monkeypatch) -> None:
    config = make_config()
    config.ocr.enable_rectification = True
    config.ocr.rectification_mode = "safe"

    class FakeRectifyResult:
        def __init__(self) -> None:
            self.image = np.zeros((48, 168, 3), dtype=np.uint8)
            self.applied = True
            self.reason = "applied"

    class DualPathOCR:
        def __init__(self) -> None:
            self.shapes = []

        def recognize_raw(self, image):
            shape = getattr(image, "shape", None)
            self.shapes.append(shape)
            if shape == (48, 168, 3):
                return PlateRecognition(text="皖AS6554", confidence=0.99, raw_text="皖AS6554", normalized_text="皖AS6554")
            return PlateRecognition(text="皖AS654A", confidence=0.94, raw_text="皖AS654A", normalized_text="皖AS654A")

    monkeypatch.setattr("car_system.pipeline.runner.safe_rectify_plate", lambda *args, **kwargs: FakeRectifyResult())

    fake_ocr = DualPathOCR()
    runner = PipelineRunner(
        config=config,
        vehicle_detector=EdgeVehicleDetector(),
        plate_detector=EdgePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert result.matches[0].recognition is not None
    assert result.matches[0].recognition.text == "皖AS654A"
    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.rectification_applied is True
    assert result.matches[0].diagnostic.rectification_reason == "applied_disagreement_plain_used"
    assert fake_ocr.shapes == [(48, 168, 3), (4, 11, 3)]


def test_run_frame_safe_rectification_keeps_rectified_text_when_plain_is_null(monkeypatch) -> None:
    config = make_config()
    config.ocr.enable_rectification = True
    config.ocr.rectification_mode = "safe"

    class FakeRectifyResult:
        def __init__(self) -> None:
            self.image = np.zeros((48, 168, 3), dtype=np.uint8)
            self.applied = True
            self.reason = "applied"

    class DualPathOCR:
        def __init__(self) -> None:
            self.shapes = []

        def recognize_raw(self, image):
            shape = getattr(image, "shape", None)
            self.shapes.append(shape)
            if shape == (48, 168, 3):
                return PlateRecognition(text="皖AF3606", confidence=0.99, raw_text="皖AF3606", normalized_text="皖AF3606")
            return None

    monkeypatch.setattr("car_system.pipeline.runner.safe_rectify_plate", lambda *args, **kwargs: FakeRectifyResult())

    fake_ocr = DualPathOCR()
    runner = PipelineRunner(
        config=config,
        vehicle_detector=EdgeVehicleDetector(),
        plate_detector=EdgePlateDetector(),
        ocr_engine=fake_ocr,
    )
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert result.matches[0].recognition is not None
    assert result.matches[0].recognition.text == "皖AF3606"
    assert result.matches[0].diagnostic is not None
    assert result.matches[0].diagnostic.rectification_reason == "applied"
    assert fake_ocr.shapes == [(48, 168, 3), (4, 11, 3)]
