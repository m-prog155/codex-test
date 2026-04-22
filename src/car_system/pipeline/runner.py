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


def _recognition_text(recognition: PlateRecognition | None) -> str | None:
    if recognition is None:
        return None
    return recognition.normalized_text or recognition.text


class PipelineRunner:
    def __init__(
        self,
        config: AppConfig,
        vehicle_detector: Any,
        plate_detector: Any,
        ocr_engine: Any,
        probe_ocr_engine: Any | None = None,
        rescue_probe_ocr_engine: Any | None = None,
        secondary_rescue_probe_ocr_engine: Any | None = None,
    ) -> None:
        self.config = config
        self.vehicle_detector = vehicle_detector
        self.plate_detector = plate_detector
        self.ocr_engine = ocr_engine
        self.probe_ocr_engine = probe_ocr_engine
        self.rescue_probe_ocr_engine = rescue_probe_ocr_engine
        self.secondary_rescue_probe_ocr_engine = secondary_rescue_probe_ocr_engine

    @staticmethod
    def _rescue_probe_matches_char_gate(
        text: str | None,
        required_chars: tuple[str, ...],
        *,
        require_alpha_count: int | None = None,
        reject_repeated_required_char: bool = False,
    ) -> bool:
        if not text:
            return False
        suffix = text[2:]
        if not required_chars:
            matched = True
        else:
            matched = any(char in suffix for char in required_chars)
        if not matched:
            return False
        if require_alpha_count is not None:
            alpha_count = sum(1 for char in suffix if char.isascii() and char.isalpha())
            if alpha_count != require_alpha_count:
                return False
        if reject_repeated_required_char:
            for char in required_chars:
                if suffix.count(char) > 1:
                    return False
        return True

    def _apply_rescue_probe(
        self,
        *,
        recognition_input: Any,
        rescue_engine: Any | None,
        rescue_config: Any,
        rescue_note_prefix: str,
        diagnostic_notes: list[str],
    ) -> PlateRecognition | None:
        if rescue_engine is None:
            return None
        rescue_recognition = _recognize(rescue_engine, recognition_input)
        rescue_text = _recognition_text(rescue_recognition)
        if rescue_recognition is not None and self._rescue_probe_matches_char_gate(
            rescue_text,
            rescue_config.rescue_requires_any_char,
            require_alpha_count=rescue_config.rescue_require_alpha_count,
            reject_repeated_required_char=rescue_config.rescue_reject_repeated_required_char,
        ):
            diagnostic_notes.append(f"{rescue_note_prefix}_rescue:{rescue_text}")
            return rescue_recognition
        if rescue_text:
            diagnostic_notes.append(f"{rescue_note_prefix}_rejected:{rescue_text}")
        return None

    def _apply_probe_policy(
        self,
        *,
        primary_recognition: PlateRecognition | None,
        probe_recognition: PlateRecognition | None,
    ) -> tuple[PlateRecognition | None, str | None, list[str]]:
        notes: list[str] = []
        probe_config = self.config.ocr.probe
        primary_text = _recognition_text(primary_recognition)
        probe_text = _recognition_text(probe_recognition)

        if primary_recognition is None:
            if (
                probe_recognition is not None
                and probe_text
                and probe_config.rescue_min_confidence is not None
                and probe_recognition.confidence >= probe_config.rescue_min_confidence
            ):
                notes.append(f"probe_rescue:{probe_text}")
                return probe_recognition, None, notes
            return None, None, notes

        if probe_recognition is None or not probe_text or not primary_text or primary_text == probe_text:
            return primary_recognition, None, notes

        if probe_config.disagreement_action == "keep_higher_confidence":
            stronger = primary_recognition
            stronger_text = primary_text
            if probe_recognition.confidence > primary_recognition.confidence:
                stronger = probe_recognition
                stronger_text = probe_text

            stronger_confidence = stronger.confidence
            confidence_gap = abs(primary_recognition.confidence - probe_recognition.confidence)
            min_confidence = probe_config.disagreement_min_confidence or 0.0
            if stronger_confidence >= min_confidence and confidence_gap >= probe_config.disagreement_min_gap:
                if stronger is primary_recognition:
                    notes.append(f"probe_disagreement_primary_kept:{probe_text}")
                else:
                    notes.append(f"probe_override:{primary_text}->{stronger_text}")
                return stronger, None, notes

            notes.append(f"probe_disagreement_ambiguous:{probe_text}")
            return None, "ocr_probe_disagreement", notes

        notes.append(f"probe_disagreement:{probe_text}")
        return None, "ocr_probe_disagreement", notes

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

            if self.probe_ocr_engine is not None:
                should_run_probe = raw_recognition is not None or self.config.ocr.probe.rescue_min_confidence is not None
                if should_run_probe:
                    probe_recognition = _recognize(self.probe_ocr_engine, recognition_input)
                    raw_recognition, diagnostic_status, probe_notes = self._apply_probe_policy(
                        primary_recognition=raw_recognition,
                        probe_recognition=probe_recognition,
                    )
                    diagnostic_notes.extend(probe_notes)
                    normalized_text = _recognition_text(raw_recognition)
                    if (
                        raw_recognition is not None
                        and diagnostic_notes
                        and any(note.startswith("probe_rescue:") or note.startswith("probe_override:") for note in diagnostic_notes)
                    ):
                        diagnostic_confidence = raw_recognition.confidence
                        diagnostic_raw_text = raw_recognition.raw_text or raw_recognition.text
                        diagnostic_normalized_text = normalized_text

            if raw_recognition is None:
                raw_recognition = self._apply_rescue_probe(
                    recognition_input=recognition_input,
                    rescue_engine=self.rescue_probe_ocr_engine,
                    rescue_config=self.config.ocr.rescue_probe,
                    rescue_note_prefix="rescue_probe",
                    diagnostic_notes=diagnostic_notes,
                )
                if raw_recognition is None:
                    raw_recognition = self._apply_rescue_probe(
                        recognition_input=recognition_input,
                        rescue_engine=self.secondary_rescue_probe_ocr_engine,
                        rescue_config=self.config.ocr.secondary_rescue_probe,
                        rescue_note_prefix="secondary_rescue_probe",
                        diagnostic_notes=diagnostic_notes,
                    )
                if raw_recognition is not None:
                    normalized_text = _recognition_text(raw_recognition)
                    diagnostic_confidence = raw_recognition.confidence
                    diagnostic_raw_text = raw_recognition.raw_text or raw_recognition.text
                    diagnostic_normalized_text = normalized_text

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
