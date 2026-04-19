from pathlib import Path

import numpy as np
import pytest

from car_system.experiments.ocr_small_sample import (
    compute_char_match_counts,
    crop_plate_region,
    compare_plate_texts,
    load_bgr_image,
    normalize_full_plate_for_eval,
    sample_entries_by_subset,
    subset_name_from_path,
)
from car_system.io.writers import write_json
from car_system.types import PlateRecognition


def test_subset_name_from_path_extracts_top_level_subset() -> None:
    assert subset_name_from_path("test/0.jpg") == "test"
    assert subset_name_from_path("train/abc/1.jpg") == "train"


def test_normalize_full_plate_for_eval_strips_chinese_and_uppercases() -> None:
    assert normalize_full_plate_for_eval("京A12345") == "A12345"
    assert normalize_full_plate_for_eval("粤B12O8Q7") == "B12O8Q7"


def test_normalize_full_plate_for_eval_uppercases_ascii_letters() -> None:
    assert normalize_full_plate_for_eval("粤b12o8q7") == "B12O8Q7"


def test_compute_char_match_counts_counts_position_matches() -> None:
    assert compute_char_match_counts("A12345", "A12X45") == (5, 6)
    assert compute_char_match_counts("A12345", "") == (0, 6)


def test_compare_plate_texts_uses_full_plate_text_when_requested() -> None:
    result = compare_plate_texts(expected="皖AY339S", predicted="皖AY339S", use_full_text=True)

    assert result["exact_match"] is True
    assert result["char_correct"] == 7
    assert result["char_total"] == 7
    assert result["char_accuracy"] == 1.0


def test_compare_plate_texts_preserves_eval_text_normalization_when_requested() -> None:
    result = compare_plate_texts(expected="皖AY339S", predicted="皖aY339s", use_full_text=False)

    assert result["exact_match"] is True
    assert result["char_correct"] == 6
    assert result["char_total"] == 6
    assert result["char_accuracy"] == 1.0


def test_sample_entries_by_subset_is_deterministic_and_preserves_subset_order() -> None:
    entries = [
        Path("ccpd_base/a1.jpg"),
        Path("ccpd_base/a2.jpg"),
        Path("ccpd_base/a3.jpg"),
        Path("ccpd_blur/b1.jpg"),
        Path("ccpd_blur/b2.jpg"),
        Path("ccpd_blur/b3.jpg"),
    ]

    first = sample_entries_by_subset(entries, ["ccpd_base", "ccpd_blur"], per_subset=2, seed=7)
    swapped = sample_entries_by_subset(entries, ["ccpd_blur", "ccpd_base"], per_subset=2, seed=7)

    assert first[:2] == swapped[2:]
    assert first[2:] == swapped[:2]
    assert [path.parts[0] for path in first] == ["ccpd_base", "ccpd_base", "ccpd_blur", "ccpd_blur"]
    assert [path.parts[0] for path in swapped] == ["ccpd_blur", "ccpd_blur", "ccpd_base", "ccpd_base"]


def test_sample_entries_by_subset_uses_seed_per_subset() -> None:
    entries = [
        Path("ccpd_base/a1.jpg"),
        Path("ccpd_base/a2.jpg"),
        Path("ccpd_base/a3.jpg"),
        Path("ccpd_base/a4.jpg"),
        Path("ccpd_base/a5.jpg"),
        Path("ccpd_blur/b1.jpg"),
        Path("ccpd_blur/b2.jpg"),
        Path("ccpd_blur/b3.jpg"),
        Path("ccpd_blur/b4.jpg"),
        Path("ccpd_blur/b5.jpg"),
    ]

    same_seed_first = sample_entries_by_subset(entries, ["ccpd_base", "ccpd_blur"], per_subset=2, seed=11)
    same_seed_second = sample_entries_by_subset(entries, ["ccpd_base", "ccpd_blur"], per_subset=2, seed=11)
    different_seed = sample_entries_by_subset(entries, ["ccpd_base", "ccpd_blur"], per_subset=2, seed=12)

    assert same_seed_first == same_seed_second
    assert same_seed_first != different_seed
    assert [path.parts[0] for path in same_seed_first] == ["ccpd_base", "ccpd_base", "ccpd_blur", "ccpd_blur"]
    assert [path.parts[0] for path in different_seed] == ["ccpd_base", "ccpd_base", "ccpd_blur", "ccpd_blur"]


class _FakeBackend:
    def __init__(self, result: PlateRecognition | None) -> None:
        self.result = result
        self.calls: list[object] = []

    def _recognize_single_candidate(self, candidate: object) -> PlateRecognition | None:
        self.calls.append(candidate)
        return self.result


def test_baseline_plate_ocr_cleans_short_raw_result_from_single_candidate(monkeypatch) -> None:
    monkeypatch.setattr("car_system.experiments.ocr_small_sample.PaddlePlateOCR", object)

    backend = _FakeBackend(PlateRecognition(text="#A1", confidence=0.91))
    baseline_ocr = getattr(__import__("car_system.experiments.ocr_small_sample", fromlist=["BaselinePlateOCR"]), "BaselinePlateOCR")(backend=backend)

    result = baseline_ocr.recognize("candidate-image")

    assert backend.calls == ["candidate-image"]
    assert result is not None
    assert result.text == "A1"
    assert result.confidence == 0.91


def test_evaluate_sample_returns_side_by_side_row(monkeypatch) -> None:
    module = __import__("car_system.experiments.ocr_small_sample", fromlist=["evaluate_sample"])

    monkeypatch.setattr(module, "load_bgr_image", lambda path: {"path": path})

    crop_calls: list[tuple[float, float]] = []

    def fake_crop_plate_region(image, bbox, pad_x_ratio=0.0, pad_y_ratio=0.0):
        crop_calls.append((pad_x_ratio, pad_y_ratio))
        return {"image": image, "bbox": bbox, "pad_x_ratio": pad_x_ratio, "pad_y_ratio": pad_y_ratio}

    monkeypatch.setattr(module, "crop_plate_region", fake_crop_plate_region)

    class _SingleCandidateBackend:
        def __init__(self, text: str | None, confidence: float = 0.88) -> None:
            self.text = text
            self.confidence = confidence
            self.calls: list[object] = []

        def _recognize_single_candidate(self, image: object):
            self.calls.append(image)
            if self.text is None:
                return None
            return PlateRecognition(text=self.text, confidence=self.confidence)

    class _EvalBackend:
        def __init__(self, text: str | None, confidence: float = 0.88) -> None:
            self.text = text
            self.confidence = confidence
            self.calls: list[object] = []

        def recognize(self, image: object):
            self.calls.append(image)
            if self.text is None:
                return None
            return PlateRecognition(text=self.text, confidence=self.confidence)

    baseline_ocr = module.BaselinePlateOCR(backend=_SingleCandidateBackend("#AY339S"))
    stabilized_ocr = _EvalBackend(None)

    annotation = __import__("car_system.data.ccpd", fromlist=["CcpdAnnotation"]).CcpdAnnotation(
        relative_path=Path("ccpd_base/example.jpg"),
        bbox=(10, 20, 30, 40),
        vertices=[(0, 0)] * 4,
        plate_indices=[0, 0, 22, 27, 27, 33, 16],
    )

    row = module.evaluate_sample(
        dataset_root=Path("D:/Projects/Car/data"),
        relative_path=annotation.relative_path,
        annotation=annotation,
        baseline_ocr=baseline_ocr,
        stabilized_ocr=stabilized_ocr,
    )

    assert row["gt_full_text"] == "皖AY339S"
    assert row["gt_eval_text"] == "AY339S"
    assert row["relative_path"] == "ccpd_base/example.jpg"
    assert row["baseline_text"] == "AY339S"
    assert row["stabilized_text"] is None
    assert row["baseline_exact_match"] is True
    assert row["stabilized_exact_match"] is False
    assert row["baseline_is_null"] is False
    assert row["stabilized_is_null"] is True
    assert row["baseline_char_correct"] == 6
    assert row["stabilized_char_correct"] == 0
    assert row["char_total"] == 6
    assert row["baseline_char_accuracy"] == 1.0
    assert row["stabilized_char_accuracy"] == 0.0
    assert row["baseline_confidence"] == 0.88
    assert row["stabilized_confidence"] is None
    assert crop_calls == [(0.0, 0.0), (0.08, 0.12)]


def test_evaluate_sample_normalizes_stabilized_full_plate_text_for_comparison(monkeypatch) -> None:
    module = __import__("car_system.experiments.ocr_small_sample", fromlist=["evaluate_sample"])

    monkeypatch.setattr(module, "load_bgr_image", lambda path: {"path": path})
    monkeypatch.setattr(module, "crop_plate_region", lambda image, bbox, pad_x_ratio=0.0, pad_y_ratio=0.0: image)

    class _NullBaseline:
        def recognize(self, image: object):
            return None

    class _EvalBackend:
        def recognize(self, image: object):
            return PlateRecognition(text="皖AY339S", confidence=0.93)

    annotation = __import__("car_system.data.ccpd", fromlist=["CcpdAnnotation"]).CcpdAnnotation(
        relative_path=Path("ccpd_base/example.jpg"),
        bbox=(10, 20, 30, 40),
        vertices=[(0, 0)] * 4,
        plate_indices=[0, 0, 22, 27, 27, 33, 16],
    )

    row = module.evaluate_sample(
        dataset_root=Path("D:/Projects/Car/data"),
        relative_path=annotation.relative_path,
        annotation=annotation,
        baseline_ocr=_NullBaseline(),
        stabilized_ocr=_EvalBackend(),
    )

    assert row["gt_eval_text"] == "AY339S"
    assert row["stabilized_text"] == "AY339S"
    assert row["stabilized_exact_match"] is True
    assert row["stabilized_char_correct"] == 6
    assert row["stabilized_char_accuracy"] == 1.0
    assert row["stabilized_is_null"] is False
    assert row["stabilized_confidence"] == 0.93


def test_evaluate_sample_preserves_full_text_when_requested(monkeypatch) -> None:
    module = __import__("car_system.experiments.ocr_small_sample", fromlist=["evaluate_sample"])

    monkeypatch.setattr(module, "load_bgr_image", lambda path: {"path": path})
    monkeypatch.setattr(module, "crop_plate_region", lambda image, bbox, pad_x_ratio=0.0, pad_y_ratio=0.0: image)

    class _FullTextBackend:
        def __init__(self, text: str, confidence: float = 0.93) -> None:
            self.text = text
            self.confidence = confidence

        def recognize(self, image: object):
            return PlateRecognition(text=self.text, confidence=self.confidence)

    annotation = __import__("car_system.data.ccpd", fromlist=["CcpdAnnotation"]).CcpdAnnotation(
        relative_path=Path("ccpd_base/example.jpg"),
        bbox=(10, 20, 30, 40),
        vertices=[(0, 0)] * 4,
        plate_indices=[0, 0, 22, 27, 27, 33, 16],
    )

    row = module.evaluate_sample(
        dataset_root=Path("D:/Projects/Car/data"),
        relative_path=annotation.relative_path,
        annotation=annotation,
        baseline_ocr=_FullTextBackend("皖AY339S"),
        stabilized_ocr=_FullTextBackend("皖AY339S"),
        use_full_text=True,
    )

    assert row["gt_full_text"] == "皖AY339S"
    assert row["baseline_text"] == "皖AY339S"
    assert row["stabilized_text"] == "皖AY339S"
    assert row["baseline_exact_match"] is True
    assert row["stabilized_exact_match"] is True
    assert row["baseline_char_correct"] == 7
    assert row["stabilized_char_correct"] == 7
    assert row["char_total"] == 7
    assert row["baseline_char_accuracy"] == 1.0
    assert row["stabilized_char_accuracy"] == 1.0


def test_build_summary_aggregates_overall_and_per_subset() -> None:
    module = __import__("car_system.experiments.ocr_small_sample", fromlist=["build_summary"])

    rows = [
        {
            "subset": "ccpd_base",
            "baseline_exact_match": True,
            "stabilized_exact_match": False,
            "baseline_char_correct": 6,
            "stabilized_char_correct": 5,
            "char_total": 6,
            "baseline_char_accuracy": 1.0,
            "stabilized_char_accuracy": 5 / 6,
            "baseline_is_null": False,
            "stabilized_is_null": False,
            "baseline_confidence": 0.9,
            "stabilized_confidence": 0.8,
        },
        {
            "subset": "ccpd_blur",
            "baseline_exact_match": False,
            "stabilized_exact_match": True,
            "baseline_char_correct": 4,
            "stabilized_char_correct": 6,
            "char_total": 6,
            "baseline_char_accuracy": 4 / 6,
            "stabilized_char_accuracy": 1.0,
            "baseline_is_null": True,
            "stabilized_is_null": False,
            "baseline_confidence": None,
            "stabilized_confidence": 0.95,
        },
    ]

    summary = module.build_summary(
        rows=rows,
        dataset_root=Path("D:/Projects/Car/data"),
        split_file=Path("D:/Projects/Car/splits/test.txt"),
        subsets=["ccpd_base", "ccpd_blur"],
        per_subset=1,
        seed=123,
        skipped=[{"relative_path": "missing.jpg", "reason": "bad sample"}],
    )

    assert summary["dataset_root"] == "D:/Projects/Car/data"
    assert summary["split_file"] == "D:/Projects/Car/splits/test.txt"
    assert summary["subsets"] == ["ccpd_base", "ccpd_blur"]
    assert summary["per_subset_target"] == 1
    assert summary["seed"] == 123
    assert summary["sample_count"] == 2
    assert summary["skipped_count"] == 1
    assert summary["skipped"] == [{"relative_path": "missing.jpg", "reason": "bad sample"}]

    assert summary["baseline"]["exact_match_count"] == 1
    assert summary["baseline"]["char_correct"] == 10
    assert summary["baseline"]["char_total"] == 12
    assert summary["baseline"]["null_count"] == 1
    assert summary["baseline"]["sample_count"] == 2

    assert summary["stabilized"]["exact_match_count"] == 1
    assert summary["stabilized"]["char_correct"] == 11
    assert summary["stabilized"]["char_total"] == 12
    assert summary["stabilized"]["null_count"] == 0

    assert summary["per_subset"]["ccpd_base"]["baseline"]["exact_match_count"] == 1
    assert summary["per_subset"]["ccpd_base"]["stabilized"]["exact_match_count"] == 0
    assert summary["per_subset"]["ccpd_blur"]["baseline"]["exact_match_count"] == 0
    assert summary["per_subset"]["ccpd_blur"]["stabilized"]["exact_match_count"] == 1


def test_build_summary_can_be_written_as_json(tmp_path) -> None:
    module = __import__("car_system.experiments.ocr_small_sample", fromlist=["build_summary"])

    summary = module.build_summary(
        rows=[
            {
                "subset": "ccpd_base",
                "baseline_exact_match": True,
                "stabilized_exact_match": True,
                "baseline_char_correct": 6,
                "stabilized_char_correct": 6,
                "char_total": 6,
                "baseline_char_accuracy": 1.0,
                "stabilized_char_accuracy": 1.0,
                "baseline_is_null": False,
                "stabilized_is_null": False,
                "baseline_confidence": 0.9,
                "stabilized_confidence": 0.95,
            }
        ],
        dataset_root=Path("D:/Projects/Car/data"),
        split_file=Path("D:/Projects/Car/splits/test.txt"),
        subsets=["ccpd_base"],
        per_subset=1,
        seed=123,
        skipped=[{"relative_path": "missing.jpg", "reason": "bad sample"}],
    )

    output_path = tmp_path / "summary.json"
    written = write_json(output_path, summary)

    assert written == output_path
    assert output_path.exists()


def test_load_bgr_image_raises_for_missing_file(monkeypatch) -> None:
    monkeypatch.setattr("cv2.imread", lambda path: None)

    with pytest.raises(FileNotFoundError):
        load_bgr_image(Path("missing.jpg"))


def test_crop_plate_region_applies_padding_and_clamps_to_bounds() -> None:
    image = np.arange(5 * 6 * 3, dtype=np.uint8).reshape(5, 6, 3)

    cropped = crop_plate_region(image, bbox=(1, 1, 4, 4), pad_x_ratio=0.5, pad_y_ratio=0.5)

    assert cropped.shape == (5, 6, 3)
    assert np.array_equal(cropped, image[0:5, 0:6])


def test_baseline_plate_ocr_loads_when_backend_has_no_ocr_attribute() -> None:
    module = __import__("car_system.experiments.ocr_small_sample", fromlist=["BaselinePlateOCR"])

    class _LazyBackend:
        def __init__(self) -> None:
            self.loaded = False
            self.calls: list[object] = []

        def load(self) -> None:
            self.loaded = True

        def _recognize_single_candidate(self, candidate: object) -> PlateRecognition | None:
            self.calls.append(candidate)
            return PlateRecognition(text="#A1", confidence=0.5)

    backend = _LazyBackend()
    result = module.BaselinePlateOCR(backend=backend).recognize("candidate-image")

    assert backend.loaded is True
    assert backend.calls == ["candidate-image"]
    assert result is not None
    assert result.text == "A1"
    assert result.confidence == 0.5


def test_build_summary_uses_stable_schema_for_empty_rows() -> None:
    module = __import__("car_system.experiments.ocr_small_sample", fromlist=["build_summary"])

    summary = module.build_summary(
        rows=[],
        dataset_root=Path("D:/Projects/Car/data"),
        split_file=Path("D:/Projects/Car/splits/test.txt"),
        subsets=["ccpd_base", "ccpd_blur"],
        per_subset=3,
        seed=7,
        skipped=[],
    )

    assert summary["baseline"]["mean_row_char_accuracy"] == 0.0
    assert summary["stabilized"]["mean_row_char_accuracy"] == 0.0
    assert summary["per_subset"]["ccpd_base"]["baseline"]["mean_row_char_accuracy"] == 0.0
    assert summary["per_subset"]["ccpd_base"]["stabilized"]["mean_row_char_accuracy"] == 0.0
    assert summary["per_subset"]["ccpd_blur"]["baseline"]["mean_row_char_accuracy"] == 0.0
    assert summary["per_subset"]["ccpd_blur"]["stabilized"]["mean_row_char_accuracy"] == 0.0
