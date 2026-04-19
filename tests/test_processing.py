from pathlib import Path

import numpy as np

from car_system.config import AppConfig, DetectorConfig, OcrConfig, OutputConfig
from car_system.pipeline.processing import process_image_file, process_video_file
from car_system.types import Detection, PlateRecognition


class FakeVehicleDetector:
    def predict(self, image):
        return [
            Detection(label="car", confidence=0.91, bbox=(10, 10, 140, 140)),
        ]


class FakePlateDetector:
    def predict(self, image):
        return [
            Detection(label="plate", confidence=0.95, bbox=(40, 90, 95, 115)),
        ]


class FakeOCR:
    def recognize(self, image):
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


def test_process_image_file_saves_rendered_media_and_structured_outputs(tmp_path, monkeypatch) -> None:
    import car_system.pipeline.processing as processing

    image = np.zeros((160, 160, 3), dtype=np.uint8)
    source_path = tmp_path / "sample.jpg"
    source_path.write_bytes(b"placeholder")

    monkeypatch.setattr(processing, "load_image", lambda path: image)
    monkeypatch.setattr(processing, "annotate_frame", lambda frame, result: frame)

    def fake_save_image(path: str | Path, frame) -> Path:
        output_path = Path(path)
        output_path.write_bytes(b"image")
        return output_path

    monkeypatch.setattr(processing, "save_image", fake_save_image)

    artifacts = process_image_file(
        config=make_config(),
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr_engine=FakeOCR(),
        source_path=source_path,
        output_dir=tmp_path / "outputs",
    )

    assert artifacts["image_path"].exists()
    assert artifacts["json_path"].exists()
    assert artifacts["csv_path"].exists()
    assert artifacts["result"].matches[0].recognition.text == "ABC123"


def test_process_video_file_saves_video_and_tabular_outputs(tmp_path, monkeypatch) -> None:
    import car_system.pipeline.processing as processing

    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    source_path = tmp_path / "sample.mp4"
    source_path.write_bytes(b"placeholder")

    monkeypatch.setattr(processing, "iter_video_frames", lambda path: iter([(0, frame), (1, frame)]))
    monkeypatch.setattr(processing, "annotate_frame", lambda current_frame, result: current_frame)

    def fake_save_video(path: str | Path, frames, fps: float = 10.0) -> Path:
        output_path = Path(path)
        output_path.write_bytes(b"video")
        return output_path

    monkeypatch.setattr(processing, "save_video", fake_save_video)

    artifacts = process_video_file(
        config=make_config(),
        vehicle_detector=FakeVehicleDetector(),
        plate_detector=FakePlateDetector(),
        ocr_engine=FakeOCR(),
        source_path=source_path,
        output_dir=tmp_path / "outputs",
        fps=12.0,
    )

    assert artifacts["video_path"].exists()
    assert artifacts["json_path"].exists()
    assert artifacts["csv_path"].exists()
    assert len(artifacts["results"]) == 2
    assert artifacts["results"][0].matches[0].recognition.text == "ABC123"
