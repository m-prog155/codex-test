from types import SimpleNamespace

from car_system.detectors.yolo_detector import YoloDetector


class _FakeBox:
    def __init__(self) -> None:
        self.xyxy = [SimpleNamespace(tolist=lambda: [1.0, 2.0, 3.0, 4.0])]
        self.cls = [SimpleNamespace(item=lambda: 0)]
        self.conf = [SimpleNamespace(item=lambda: 0.9)]


class _FakeResult:
    def __init__(self) -> None:
        self.names = {0: "car"}
        self.boxes = [_FakeBox()]


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, image, **kwargs):
        self.calls.append({"image": image, **kwargs})
        return [_FakeResult()]


def test_predict_passes_device_to_ultralytics_model() -> None:
    detector = YoloDetector("weights/vehicle.pt", confidence=0.35, device="cpu")
    fake_model = _FakeModel()
    detector._model = fake_model

    detections = detector.predict("image-bytes")

    assert len(detections) == 1
    assert fake_model.calls == [
        {
            "image": "image-bytes",
            "conf": 0.35,
            "verbose": False,
            "device": "cpu",
        }
    ]
