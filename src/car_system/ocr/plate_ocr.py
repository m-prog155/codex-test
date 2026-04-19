from typing import Any

from car_system.types import PlateRecognition


PROVINCE_PREFIXES = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼")


class PaddlePlateOCR:
    def __init__(
        self,
        language: str = "ch",
        use_angle_cls: bool = False,
        mode: str = "generic",
        model_dir: str | None = None,
        character_dict_path: str | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.mode = mode
        self.model_dir = model_dir
        self.character_dict_path = character_dict_path
        self.min_confidence = min_confidence
        self._ocr: Any | None = None

    def load(self) -> None:
        if self.mode == "specialized" and not self.model_dir:
            raise ValueError("model_dir is required when mode='specialized'.")

        try:
            from paddleocr import TextRecognition
        except ImportError as exc:
            raise RuntimeError("PaddleOCR is not installed.") from exc

        # Generic mode keeps the existing recognition-only default model path.
        # We already crop the plate region upstream, so it is more stable here.
        kwargs: dict[str, str] = {"model_name": "PP-OCRv5_mobile_rec"}
        if self.mode == "specialized":
            kwargs["model_dir"] = self.model_dir

        self._ocr = TextRecognition(**kwargs)

    def _normalize_plate_text(self, text: str) -> str:
        cleaned = "".join(char for char in text.upper() if char.isalnum())
        if not cleaned:
            return ""

        if self.mode == "specialized":
            return cleaned

        normalized_chars = list(cleaned)
        for index, char in enumerate(normalized_chars):
            previous_is_digit = index > 0 and normalized_chars[index - 1].isdigit()
            next_is_digit = index + 1 < len(normalized_chars) and normalized_chars[index + 1].isdigit()
            if not (previous_is_digit or next_is_digit):
                continue

            if char in {"O", "Q", "D"}:
                normalized_chars[index] = "0"
            elif char in {"I", "L"}:
                normalized_chars[index] = "1"
            elif char == "B":
                normalized_chars[index] = "8"

        return "".join(normalized_chars)

    def _is_valid_plate_text(self, text: str) -> bool:
        if self.mode != "specialized":
            return 6 <= len(text) <= 8

        if len(text) != 7:
            return False
        if text[0] not in PROVINCE_PREFIXES:
            return False
        if not (text[1].isascii() and text[1].isalpha() and text[1].isupper()):
            return False
        return all(char.isascii() and char.isalnum() for char in text[2:])

    def _build_candidate_images(self, image: Any) -> list[Any]:
        if not hasattr(image, "shape"):
            return [image]

        if self.mode == "specialized":
            return [image]

        try:
            import cv2
        except ImportError:
            return [image]

        candidates = [image]

        enlarged = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        candidates.append(enlarged)

        grayscale = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(grayscale)
        enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        candidates.append(enhanced)

        return candidates

    def _recognize_single_candidate(self, candidate: Any) -> PlateRecognition | None:
        if hasattr(self._ocr, "predict") and not hasattr(self._ocr, "ocr"):
            return self._from_textrecognition_result(self._ocr.predict(input=[candidate], batch_size=1))

        raw = self._ocr.ocr(candidate, cls=self.use_angle_cls)
        if not raw or not raw[0]:
            return None

        texts: list[str] = []
        scores: list[float] = []
        for _, (text, score) in raw[0]:
            texts.append(text)
            scores.append(float(score))

        if not texts:
            return None

        average_score = sum(scores) / len(scores)
        return PlateRecognition(text="".join(texts), confidence=average_score)

    @staticmethod
    def _score_candidate(result: PlateRecognition) -> float:
        score = result.confidence
        text = result.text
        if not text:
            return score

        if text.isalnum():
            score += 0.05
        if 6 <= len(text) <= 8:
            score += 0.12
        elif 5 <= len(text) <= 9:
            score += 0.05
        if len(text) == 7:
            score += 0.03
        if any(char.isalpha() for char in text):
            score += 0.01
        if any(char.isdigit() for char in text):
            score += 0.01

        return score

    def _select_best_result(self, results: list[PlateRecognition]) -> PlateRecognition | None:
        if not results:
            return None

        best = max(results, key=self._score_candidate)
        return best

    @staticmethod
    def _from_textrecognition_result(raw: list[Any]) -> PlateRecognition | None:
        if not raw:
            return None

        first = raw[0]
        text = first.get("rec_text") if hasattr(first, "get") else getattr(first, "rec_text", None)
        score = first.get("rec_score") if hasattr(first, "get") else getattr(first, "rec_score", None)
        if not text:
            return None

        return PlateRecognition(text=str(text), confidence=float(score or 0.0))

    def recognize_raw(self, image: Any) -> PlateRecognition | None:
        if self._ocr is None:
            self.load()

        candidate_results: list[tuple[PlateRecognition, float]] = []
        for candidate in self._build_candidate_images(image):
            result = self._recognize_single_candidate(candidate)
            if result is None:
                continue

            normalized_text = self._normalize_plate_text(result.text)
            if not normalized_text:
                continue
            if not self._is_valid_plate_text(normalized_text):
                continue

            scored_result = PlateRecognition(
                text=result.text,
                confidence=result.confidence,
                raw_text=result.text,
                normalized_text=normalized_text,
            )
            candidate_results.append(
                (
                    scored_result,
                    self._score_candidate(PlateRecognition(text=normalized_text, confidence=result.confidence)),
                )
            )

        if not candidate_results:
            return None

        best_result, _ = max(candidate_results, key=lambda item: item[1])
        if best_result.confidence < self.min_confidence:
            return None
        return best_result

    def recognize(self, image: Any) -> PlateRecognition | None:
        raw_result = self.recognize_raw(image)
        if raw_result is None:
            return None

        normalized_text = raw_result.normalized_text or self._normalize_plate_text(raw_result.text)
        if not normalized_text:
            return None

        return PlateRecognition(
            text=normalized_text,
            confidence=raw_result.confidence,
            raw_text=raw_result.raw_text or raw_result.text,
            normalized_text=normalized_text,
        )
