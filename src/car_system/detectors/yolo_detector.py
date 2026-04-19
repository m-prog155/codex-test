from typing import Any

from car_system.types import Detection


class YoloDetector:
    def __init__(self, model_path: str, confidence: float = 0.35, device: str | None = None) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self._model: Any | None = None

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed.") from exc

        self._model = YOLO(self.model_path)

    def predict(self, image: Any) -> list[Detection]:
        if self._model is None:
            self.load()

        predict_kwargs = {
            "conf": self.confidence,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self._model.predict(image, **predict_kwargs)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                detections.append(
                    Detection(
                        label=str(names[class_id]),
                        confidence=score,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return detections
