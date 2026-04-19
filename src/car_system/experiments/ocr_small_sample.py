from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from car_system.data.ccpd import decode_ccpd_plate_indices
from car_system.ocr.plate_ocr import PaddlePlateOCR
from car_system.types import PlateRecognition

def subset_name_from_path(relative_path: str | Path) -> str:
    path = Path(relative_path)
    parts = path.parts
    return parts[0] if parts else ""


def normalize_full_plate_for_eval(text: str) -> str:
    return "".join(char.upper() for char in text if char.isascii() and char.isalnum())


def compute_char_match_counts(expected: str, predicted: str) -> tuple[int, int]:
    total = len(expected)
    correct = sum(1 for index, char in enumerate(expected) if index < len(predicted) and predicted[index] == char)
    return correct, total


def compare_plate_texts(expected: str, predicted: str | None, use_full_text: bool) -> dict[str, Any]:
    normalized_expected = expected if use_full_text else normalize_full_plate_for_eval(expected)
    normalized_predicted = "" if predicted is None else predicted
    if not use_full_text and predicted is not None:
        normalized_predicted = normalize_full_plate_for_eval(predicted)

    char_correct, char_total = compute_char_match_counts(normalized_expected, normalized_predicted)
    return {
        "exact_match": normalized_predicted == normalized_expected,
        "char_correct": char_correct,
        "char_total": char_total,
        "char_accuracy": char_correct / char_total if char_total else 0.0,
        "is_null": predicted is None,
    }


def sample_entries_by_subset(
    entries: list[Path],
    subsets: list[str],
    per_subset: int,
    seed: int,
) -> list[Path]:
    sampled_entries: list[Path] = []

    for subset in subsets:
        subset_entries = [entry for entry in entries if subset_name_from_path(entry) == subset]
        if len(subset_entries) > per_subset:
            subset_rng = random.Random(f"{seed}:{subset}")
            subset_entries = subset_rng.sample(subset_entries, per_subset)
        sampled_entries.extend(subset_entries)

    return sampled_entries


def load_bgr_image(path: str | Path) -> Any:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    return image


def crop_plate_region(
    image: Any,
    bbox: tuple[int, int, int, int],
    pad_x_ratio: float = 0.0,
    pad_y_ratio: float = 0.0,
) -> Any:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox

    pad_x = int(round((x2 - x1) * pad_x_ratio))
    pad_y = int(round((y2 - y1) * pad_y_ratio))

    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)

    return image[top:bottom, left:right]


class BaselinePlateOCR:
    def __init__(self, backend: PaddlePlateOCR | None = None) -> None:
        self._backend = backend or PaddlePlateOCR()

    def _ensure_loaded(self) -> None:
        if hasattr(self._backend, "load") and (not hasattr(self._backend, "_ocr") or getattr(self._backend, "_ocr") is None):
            self._backend.load()

    def recognize_raw(self, image: Any) -> PlateRecognition | None:
        if hasattr(self._backend, "recognize_raw"):
            return self._backend.recognize_raw(image)

        self._ensure_loaded()
        if hasattr(self._backend, "_recognize_single_candidate"):
            return self._backend._recognize_single_candidate(image)
        if hasattr(self._backend, "recognize"):
            return self._backend.recognize(image)
        return None

    def recognize(self, image: Any) -> PlateRecognition | None:
        result = self.recognize_raw(image)
        if result is None:
            return None

        cleaned = normalize_full_plate_for_eval(result.text)
        if not cleaned:
            return None

        return PlateRecognition(text=cleaned, confidence=result.confidence)


def _summarize_recognition_rows(rows: list[dict[str, Any]], text_prefix: str) -> dict[str, Any]:
    exact_match_key = f"{text_prefix}_exact_match"
    char_correct_key = f"{text_prefix}_char_correct"
    char_accuracy_key = f"{text_prefix}_char_accuracy"
    is_null_key = f"{text_prefix}_is_null"

    sample_count = len(rows)
    exact_match_count = sum(1 for row in rows if row.get(exact_match_key))
    char_correct = sum(int(row.get(char_correct_key, 0) or 0) for row in rows)
    char_total = sum(int(row.get("char_total", 0) or 0) for row in rows)
    null_count = sum(1 for row in rows if row.get(is_null_key))
    accuracies = [float(row[char_accuracy_key]) for row in rows if char_accuracy_key in row and row[char_accuracy_key] is not None]

    summary: dict[str, Any] = {
        "sample_count": sample_count,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / sample_count if sample_count else 0.0,
        "char_correct": char_correct,
        "char_total": char_total,
        "char_accuracy": char_correct / char_total if char_total else 0.0,
        "null_count": null_count,
        "null_rate": null_count / sample_count if sample_count else 0.0,
        "mean_row_char_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0.0,
    }

    return summary


def _path_to_posix_text(path: str | Path) -> str:
    return Path(path).as_posix()


def _recognize_plate_text(ocr: Any, image: Any, use_full_text: bool) -> tuple[str | None, float | None]:
    if use_full_text:
        if hasattr(ocr, "recognize_raw"):
            result = ocr.recognize_raw(image)
        elif hasattr(ocr, "recognize"):
            result = ocr.recognize(image)
        elif hasattr(ocr, "_recognize_single_candidate"):
            result = ocr._recognize_single_candidate(image)
        else:
            return None, None
    elif hasattr(ocr, "recognize"):
        result = ocr.recognize(image)
    elif hasattr(ocr, "recognize_raw"):
        result = ocr.recognize_raw(image)
    elif hasattr(ocr, "_recognize_single_candidate"):
        result = ocr._recognize_single_candidate(image)
    else:
        return None, None

    if result is None:
        return None, None

    text = result.text if use_full_text else normalize_full_plate_for_eval(result.text)
    return text, result.confidence


def evaluate_sample(
    dataset_root: str | Path,
    relative_path: str | Path,
    annotation: Any,
    baseline_ocr: Any,
    stabilized_ocr: Any,
    use_full_text: bool = False,
) -> dict[str, Any]:
    image_path = Path(dataset_root) / Path(relative_path)
    image = load_bgr_image(image_path)

    baseline_crop = crop_plate_region(image, annotation.bbox, pad_x_ratio=0.0, pad_y_ratio=0.0)
    stabilized_crop = crop_plate_region(image, annotation.bbox, pad_x_ratio=0.08, pad_y_ratio=0.12)

    gt_full_text = decode_ccpd_plate_indices(annotation.plate_indices)
    gt_eval_text = normalize_full_plate_for_eval(gt_full_text)
    char_total = len(gt_eval_text)

    baseline_text, baseline_confidence = _recognize_plate_text(baseline_ocr, baseline_crop, use_full_text=use_full_text)
    stabilized_text, stabilized_confidence = _recognize_plate_text(
        stabilized_ocr,
        stabilized_crop,
        use_full_text=use_full_text,
    )

    comparison_expected = gt_full_text if use_full_text else gt_eval_text
    baseline_comparison = compare_plate_texts(comparison_expected, baseline_text, use_full_text=use_full_text)
    stabilized_comparison = compare_plate_texts(comparison_expected, stabilized_text, use_full_text=use_full_text)

    return {
        "relative_path": _path_to_posix_text(relative_path),
        "subset": subset_name_from_path(relative_path),
        "gt_full_text": gt_full_text,
        "gt_eval_text": gt_eval_text,
        "baseline_text": baseline_text,
        "stabilized_text": stabilized_text,
        "baseline_exact_match": baseline_comparison["exact_match"],
        "stabilized_exact_match": stabilized_comparison["exact_match"],
        "baseline_char_correct": baseline_comparison["char_correct"],
        "stabilized_char_correct": stabilized_comparison["char_correct"],
        "char_total": baseline_comparison["char_total"],
        "baseline_char_accuracy": baseline_comparison["char_accuracy"],
        "stabilized_char_accuracy": stabilized_comparison["char_accuracy"],
        "baseline_is_null": baseline_text is None,
        "stabilized_is_null": stabilized_text is None,
        "baseline_confidence": baseline_confidence,
        "stabilized_confidence": stabilized_confidence,
    }


def build_summary(
    rows: list[dict[str, Any]],
    dataset_root: str | Path,
    split_file: str | Path,
    subsets: list[str],
    per_subset: int,
    seed: int,
    skipped: list[dict[str, str]],
) -> dict[str, Any]:
    rows_list = list(rows)
    grouped_rows: dict[str, list[dict[str, Any]]] = {subset: [] for subset in subsets}
    for row in rows_list:
        subset = row.get("subset", "")
        grouped_rows.setdefault(subset, []).append(row)

    per_subset_summary: dict[str, Any] = {}
    for subset in subsets:
        subset_rows = grouped_rows.get(subset, [])
        per_subset_summary[subset] = {
            "baseline": _summarize_recognition_rows(subset_rows, "baseline"),
            "stabilized": _summarize_recognition_rows(subset_rows, "stabilized"),
        }

    return {
        "dataset_root": _path_to_posix_text(dataset_root),
        "split_file": _path_to_posix_text(split_file),
        "subsets": list(subsets),
        "per_subset_target": per_subset,
        "seed": seed,
        "sample_count": len(rows_list),
        "skipped_count": len(skipped),
        "skipped": [
            {
                "relative_path": _path_to_posix_text(item["relative_path"]),
                "reason": item["reason"],
            }
            for item in skipped
        ],
        "baseline": _summarize_recognition_rows(rows_list, "baseline"),
        "stabilized": _summarize_recognition_rows(rows_list, "stabilized"),
        "per_subset": per_subset_summary,
    }
