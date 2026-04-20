from typing import Any

from car_system.config import AppConfig
from car_system.ocr.rectify import rectify_plate, safe_rectify_plate
from car_system.pipeline.matcher import match_plates_to_vehicles
from car_system.types import Detection, FrameResult, PlateDiagnostic, PlateRecognition


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    image_shape: tuple[int, ...],
    pad_x_ratio: float = 0.08,
    pad_y_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    image_height, image_width = image_shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_x = max(0, int(round(width * pad_x_ratio)))
    pad_y = max(0, int(round(height * pad_y_ratio)))

    left = min(max(0, x1 - pad_x), max(0, image_width - 1))
    top = min(max(0, y1 - pad_y), max(0, image_height - 1))
    right = min(image_width, max(left + 1, x2 + pad_x))
    bottom = min(image_height, max(top + 1, y2 + pad_y))
    return (left, top, right, bottom)


def _crop_bbox(image: Any, bbox: tuple[float, float, float, float]) -> Any:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return image[y1:y2, x1:x2]


def _diagnostic_status(raw_text: str | None, normalized_text: str | None) -> str:
    if raw_text is None:
        return "ocr_null"
    if not normalized_text:
        return "ocr_invalid_text"
    if len(raw_text) >= 4 and raw_text[:2] == raw_text[2:4]:
        return "ocr_abnormal_text"
    return "recognized"


def _recognize(ocr_engine: Any, image: Any) -> PlateRecognition | None:
    if hasattr(ocr_engine, "recognize_raw"):
        return ocr_engine.recognize_raw(image)
    return ocr_engine.recognize(image)


class PipelineRunner:
    def __init__(
        self,
        config: AppConfig,
        vehicle_detector: Any,
        plate_detector: Any,
        ocr_engine: Any,
        probe_ocr_engine: Any | None = None,
    ) -> None:
        self.config = config
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_engine = ocr_engine
        self.probe_ocr_engine = probe_ocr_engine

    def run_frame(self, image: Any, source_name: str, frame_index: int = 0) -> FrameResult:
        vehicle_detections: list[Detection] = [
            item for item in self.vehicle_detector.predict(image) if item.label in self.config.vehicle_detector.labels
        ]
        plate_detections: list[Detection] = [
            item for item in self.plate_detector.predict(image) if item.label in self.config.plate_detector.labels
        ]
        detections: list[Detection] = [*vehicle_detections, *plate_detections]

        matches = match_plates_to_vehicles(vehicle_detections, plate_detections)
        for match in matches:
            crop_bbox = _expand_bbox(
                match.plate.bbox,
                image.shape,
                pad_x_ratio=self.config.ocr.crop_pad_x_ratio,
                pad_y_ratio=self.config.ocr.crop_pad_y_ratio,
            )
            crop = _crop_bbox(image, crop_bbox)
            ocr_input = crop
            rectified_shape = None
            rectification_applied = False
            rectification_reason = None
            rectification_mode = "disabled"

            if self.config.ocr.enable_rectification and self.config.ocr.rectification_mode == "safe":
                rectification_mode = "safe"
                rectified = safe_rectify_plate(
                    crop,
                    min_area_ratio=self.config.ocr.safe_rect_min_area_ratio,
                    min_rectangularity=self.config.ocr.safe_rect_min_rectangularity,
                    max_center_offset=self.config.ocr.safe_rect_max_center_offset,
                )
                rectification_applied = rectified.applied
                rectification_reason = rectified.reason
                if rectified.applied:
                    ocr_input = rectified.image
                    rectified_shape = getattr(rectified.image, "shape", None)
            elif self.config.ocr.enable_rectification:
                rectification_mode = self.config.ocr.rectification_mode or "legacy"
                ocr_input = rectify_plate(crop)
                rectified_shape = getattr(ocr_input, "shape", None)

            raw_recognition = _recognize(self.ocr_engine, ocr_input)
            recognition_input = ocr_input
            if rectification_mode == "safe" and rectification_applied and raw_recognition is not None:
                rectified_text = raw_recognition.normalized_text or raw_recognition.text
                plain_recognition = _recognize(self.ocr_engine, crop)
                plain_text = None
                if plain_recognition is not None:
                    plain_text = plain_recognition.normalized_text or plain_recognition.text

                if plain_text and rectified_text and plain_text != rectified_text:
                    raw_recognition = plain_recognition
                    recognition_input = crop
                    rectification_reason = "applied_disagreement_plain_used"

            normalized_text: str | None = None
            diagnostic_status = None
            diagnostic_notes: list[str] = []
            diagnostic_confidence = raw_recognition.confidence if raw_recognition else None
            diagnostic_raw_text = raw_recognition.raw_text if raw_recognition else None
            diagnostic_normalized_text = None

            if raw_recognition is not None:
                normalized_text = raw_recognition.normalized_text or raw_recognition.text
                diagnostic_raw_text = raw_recognition.raw_text or raw_recognition.text
                diagnostic_normalized_text = normalized_text

            if self.probe_ocr_engine is not None and raw_recognition is not None:
                probe_recognition = _recognize(self.probe_ocr_engine, recognition_input)
                probe_text = None
                if probe_recognition is not None:
                    probe_text = probe_recognition.normalized_text or probe_recognition.text

                if probe_text and normalized_text and probe_text != normalized_text:
                    raw_recognition = None
                    normalized_text = None
                    diagnostic_status = "ocr_probe_disagreement"
                    diagnostic_notes.append(f"probe_disagreement:{probe_text}")

            if raw_recognition is not None:
                match.recognition = PlateRecognition(
                    text=normalized_text,
                    confidence=raw_recognition.confidence,
                    raw_text=raw_recognition.raw_text or raw_recognition.text,
                    normalized_text=normalized_text,
                )

            match.diagnostic = PlateDiagnostic(
                status=_diagnostic_status(
                    diagnostic_raw_text,
                    diagnostic_normalized_text if diagnostic_status is not None else normalized_text,
                ),
                crop_bbox=crop_bbox,
                crop_shape=getattr(crop, "shape", None),
                rectified_shape=rectified_shape,
                confidence=diagnostic_confidence,
                raw_text=diagnostic_raw_text,
                normalized_text=diagnostic_normalized_text if diagnostic_status is not None else normalized_text,
                rectification_mode=rectification_mode,
                rectification_applied=rectification_applied,
                rectification_reason=rectification_reason,
                notes=diagnostic_notes,
            )
            if diagnostic_status is not None:
                match.diagnostic.status = diagnostic_status

        return FrameResult(
            source_name=source_name,
            frame_index=frame_index,
            detections=detections,
            matches=matches,
        )
