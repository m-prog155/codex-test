# OCR Small Sample Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CCPD small-sample OCR evaluation script that compares baseline and stabilized OCR on the same ground-truth plate crops and writes reproducible summary and per-sample outputs.

**Architecture:** Extend the CCPD helper with filename-to-plate-text decoding, then add an experiment-local evaluation module that owns subset sampling, crop extraction, normalization, per-sample metrics, and side-by-side aggregation. Keep the “baseline OCR” approximation inside the experiment path so the production inference chain remains unchanged, and let the CLI script stay thin by delegating all non-argument logic into the reusable module.

**Tech Stack:** Python, OpenCV, PaddleOCR adapter, dataclasses, csv/json, pytest

---

## File Structure

- Modify: `D:\Projects\Car\src\car_system\data\ccpd.py`
  Purpose: add CCPD plate-index decoding helpers that convert filename annotations into full plate text.
- Create: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`
  Purpose: own sample selection, text normalization for evaluation, crop extraction, baseline/stabilized OCR runners, per-sample comparison, and summary building.
- Create: `D:\Projects\Car\scripts\evaluate_ocr_small_sample.py`
  Purpose: provide the CLI entry point that loads the dataset split, runs the evaluation, and writes `summary.json` plus `samples.csv`.
- Modify: `D:\Projects\Car\tests\test_ccpd.py`
  Purpose: verify CCPD plate decoding behavior.
- Create: `D:\Projects\Car\tests\test_ocr_small_sample_eval.py`
  Purpose: verify sampling, normalization, metric calculation, baseline/stabilized comparison, and summary generation without a real OCR dependency.

Note: this workspace is not a git repository, so the plan intentionally omits commit steps.

### Task 1: Add CCPD plate-text decoding helpers

**Files:**
- Modify: `D:\Projects\Car\src\car_system\data\ccpd.py`
- Modify: `D:\Projects\Car\tests\test_ccpd.py`

- [ ] **Step 1: Write the failing decode tests in `test_ccpd.py`**

```python
from car_system.data.ccpd import decode_ccpd_plate_indices, parse_ccpd_path


def test_decode_ccpd_plate_indices_returns_full_plate_text() -> None:
    assert decode_ccpd_plate_indices([0, 0, 22, 27, 27, 33, 16]) == "皖AY339S"


def test_decode_ccpd_plate_indices_rejects_unexpected_length() -> None:
    try:
        decode_ccpd_plate_indices([0, 1, 2])
    except ValueError as exc:
        assert "Unexpected CCPD plate index length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for short CCPD plate index list")


def test_decode_ccpd_annotation_round_trips_from_filename() -> None:
    annotation = parse_ccpd_path(
        "ccpd_base/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )
    assert decode_ccpd_plate_indices(annotation.plate_indices) == "皖AY339S"
```

- [ ] **Step 2: Run the CCPD tests and verify the new decode tests fail**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ccpd.py -q
```

Expected:

```text
An assertion failure showing the legacy `皖AW339S` expectation no longer matches the official CCPD decode result.
```

- [ ] **Step 3: Implement decode constants and helpers in `ccpd.py`**

Use the official CCPD tables exactly as defined in the README:

- `PROVINCES` ends with `["警", "学", "O"]`
- `ALPHABETS[22] == "Y"`
- `ADS[22] == "Y"`
- Do not add any `index == 22` compatibility special case

```python
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
]

ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "O",
]

ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "O",
]


def decode_ccpd_plate_indices(indices: list[int]) -> str:
    if len(indices) != 7:
        raise ValueError(f"Unexpected CCPD plate index length: {indices}")

    province_index, alphabet_index, *ads_indexes = indices
    try:
        province = PROVINCES[province_index]
        alphabet = ALPHABETS[alphabet_index]
        suffix = "".join(ADS[index] for index in ads_indexes)
    except IndexError as exc:
        raise ValueError(f"CCPD plate index out of range: {indices}") from exc

    return f"{province}{alphabet}{suffix}"
```

- [ ] **Step 4: Export the helper from the same module and keep existing parsing behavior unchanged**

```python
def parse_ccpd_path(relative_path: str | Path) -> CcpdAnnotation:
    rel_path = Path(relative_path)
    stem = rel_path.stem
    parts = stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Unexpected CCPD filename format: {rel_path}")

    bbox_left_top, bbox_right_bottom = parts[2].split("_")
    x1, y1 = _parse_point(bbox_left_top)
    x2, y2 = _parse_point(bbox_right_bottom)
    vertices = [_parse_point(value) for value in parts[3].split("_")]
    plate_indices = [int(value) for value in parts[4].split("_")]

    return CcpdAnnotation(
        relative_path=rel_path,
        bbox=(x1, y1, x2, y2),
        vertices=vertices,
        plate_indices=plate_indices,
    )
```

Expected outcome:

```text
Existing callers of parse_ccpd_path still work, and new callers can decode full plate text through decode_ccpd_plate_indices(annotation.plate_indices).
```

- [ ] **Step 5: Re-run the CCPD tests and verify they pass**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ccpd.py -q
```

Expected:

```text
all tests in `test_ccpd.py` pass
```

### Task 2: Add evaluation sampling and metric helpers

**Files:**
- Create: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`
- Create: `D:\Projects\Car\tests\test_ocr_small_sample_eval.py`

- [ ] **Step 1: Write the failing helper tests for subset sampling, normalization, and character matching**

```python
from pathlib import Path

from car_system.experiments.ocr_small_sample import (
    compute_char_match_counts,
    normalize_full_plate_for_eval,
    sample_entries_by_subset,
)


def test_normalize_full_plate_for_eval_removes_non_ascii_prefix() -> None:
    assert normalize_full_plate_for_eval("京A12345") == "A12345"
    assert normalize_full_plate_for_eval("粤B12O8Q7") == "B12O8Q7"


def test_compute_char_match_counts_uses_positional_comparison() -> None:
    assert compute_char_match_counts("A12345", "A12X45") == (5, 6)
    assert compute_char_match_counts("A12345", "") == (0, 6)


def test_sample_entries_by_subset_is_reproducible_per_seed() -> None:
    entries = [
        Path("ccpd_base/a.jpg"),
        Path("ccpd_base/b.jpg"),
        Path("ccpd_base/c.jpg"),
        Path("ccpd_blur/d.jpg"),
        Path("ccpd_blur/e.jpg"),
        Path("ccpd_blur/f.jpg"),
    ]

    first = sample_entries_by_subset(entries, subsets=["ccpd_base", "ccpd_blur"], per_subset=2, seed=7)
    second = sample_entries_by_subset(entries, subsets=["ccpd_base", "ccpd_blur"], per_subset=2, seed=7)
    third = sample_entries_by_subset(entries, subsets=["ccpd_base", "ccpd_blur"], per_subset=2, seed=9)

    assert first == second
    assert len(first) == 4
    assert [path.parts[0] for path in first] == ["ccpd_base", "ccpd_base", "ccpd_blur", "ccpd_blur"]
    assert first != third
```

- [ ] **Step 2: Run the new evaluation tests and verify they fail**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_eval.py -q
```

Expected:

```text
Module import failure for `car_system.experiments.ocr_small_sample`
```

- [ ] **Step 3: Implement the small reusable helpers in `ocr_small_sample.py`**

```python
from pathlib import Path
import random


def subset_name_from_path(relative_path: str | Path) -> str:
    path = Path(relative_path)
    return path.parts[0] if path.parts else ""


def normalize_full_plate_for_eval(text: str) -> str:
    normalized: list[str] = []
    for char in text.upper():
        if char.isascii() and char.isalnum():
            normalized.append(char)
    return "".join(normalized)


def compute_char_match_counts(expected: str, predicted: str) -> tuple[int, int]:
    total = len(expected)
    correct = 0
    for index, char in enumerate(expected):
        if index < len(predicted) and predicted[index] == char:
            correct += 1
    return correct, total


def sample_entries_by_subset(
    entries: list[Path],
    subsets: list[str],
    per_subset: int,
    seed: int,
) -> list[Path]:
    generator = random.Random(seed)
    sampled: list[Path] = []
    for subset in subsets:
        group = [entry for entry in entries if subset_name_from_path(entry) == subset]
        if len(group) <= per_subset:
            sampled.extend(group)
            continue

        selected_indexes = sorted(generator.sample(range(len(group)), per_subset))
        sampled.extend(group[index] for index in selected_indexes)
    return sampled
```

- [ ] **Step 4: Re-run the helper tests and verify they pass**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_eval.py -q
```

Expected:

```text
all tests in `test_ocr_small_sample_eval.py` pass
```

### Task 3: Add baseline/stabilized OCR comparison and summary building

**Files:**
- Modify: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`
- Create: `D:\Projects\Car\tests\test_ocr_small_sample_eval.py`

- [ ] **Step 1: Extend the test file with fake recognizer coverage for side-by-side evaluation**

```python
from pathlib import Path

from car_system.data.ccpd import parse_ccpd_path
from car_system.experiments import ocr_small_sample as eval_mod
from car_system.types import PlateRecognition


class FakeRecognizer:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def recognize(self, image):
        self.calls.append(image)
        return self.result


def test_baseline_plate_ocr_keeps_short_cleaned_results() -> None:
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def predict(self, input, batch_size=1):
            self.calls.append({"input": input, "batch_size": batch_size})
            return [{"rec_text": "#A1", "rec_score": 0.93}]

    from car_system.ocr.plate_ocr import PaddlePlateOCR

    backend = PaddlePlateOCR(language="ch", use_angle_cls=False)
    backend._ocr = FakeBackend()

    baseline = eval_mod.BaselinePlateOCR(backend)
    result = baseline.recognize("plate-crop")

    assert result is not None
    assert result.text == "A1"
    assert result.confidence == 0.93


def test_evaluate_sample_returns_side_by_side_metrics(monkeypatch, tmp_path: Path) -> None:
    relative_path = Path(
        "ccpd_base/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )
    image_path = tmp_path / relative_path
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"stub")

    monkeypatch.setattr(eval_mod, "load_bgr_image", lambda path: "image")
    monkeypatch.setattr(
        eval_mod,
        "crop_plate_region",
        lambda image, bbox, pad_x_ratio=0.0, pad_y_ratio=0.0: f"crop:{pad_x_ratio}:{pad_y_ratio}",
    )

    baseline = FakeRecognizer(PlateRecognition(text="AW339", confidence=0.70))
    stabilized = FakeRecognizer(PlateRecognition(text="AY339S", confidence=0.82))

    row = eval_mod.evaluate_sample(
        dataset_root=tmp_path,
        relative_path=relative_path,
        annotation=parse_ccpd_path(relative_path),
        baseline_ocr=baseline,
        stabilized_ocr=stabilized,
    )

    assert row["subset"] == "ccpd_base"
    assert row["gt_full_text"] == "皖AY339S"
    assert row["gt_eval_text"] == "AY339S"
    assert row["baseline_text"] == "AW339"
    assert row["stabilized_text"] == "AY339S"
    assert row["baseline_exact_match"] is False
    assert row["stabilized_exact_match"] is True
    assert row["baseline_is_null"] is False
    assert row["stabilized_is_null"] is False


def test_build_summary_aggregates_overall_and_per_subset_metrics() -> None:
    rows = [
        {
            "subset": "ccpd_base",
            "baseline_exact_match": False,
            "stabilized_exact_match": True,
            "baseline_is_null": False,
            "stabilized_is_null": False,
            "baseline_char_correct": 5,
            "stabilized_char_correct": 6,
            "char_total": 6,
        },
        {
            "subset": "ccpd_blur",
            "baseline_exact_match": False,
            "stabilized_exact_match": False,
            "baseline_is_null": False,
            "stabilized_is_null": True,
            "baseline_char_correct": 2,
            "stabilized_char_correct": 0,
            "char_total": 6,
        },
    ]

    summary = eval_mod.build_summary(
        rows=rows,
        dataset_root=Path("D:/plate_project/CCPD2019"),
        split_file=Path("D:/plate_project/CCPD2019/splits/test.txt"),
        subsets=["ccpd_base", "ccpd_blur"],
        per_subset=10,
        seed=42,
        skipped=[],
    )

    assert summary["sample_count"] == 2
    assert summary["baseline"]["exact_match_count"] == 0
    assert summary["stabilized"]["exact_match_count"] == 1
    assert summary["baseline"]["null_count"] == 0
    assert summary["stabilized"]["null_count"] == 1
    assert summary["per_subset"]["ccpd_base"]["stabilized"]["exact_match_rate"] == 1.0
```

- [ ] **Step 2: Run the expanded evaluation tests and verify they fail**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_eval.py -q
```

Expected:

```text
attribute error for missing `BaselinePlateOCR`
```

- [ ] **Step 3: Implement the baseline adapter and image/crop helpers in `ocr_small_sample.py`**

```python
import cv2
from pathlib import Path
from typing import Any

from car_system.ocr.plate_ocr import PaddlePlateOCR
from car_system.types import PlateRecognition


def load_bgr_image(path: str | Path) -> Any:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def crop_plate_region(
    image: Any,
    bbox: tuple[int, int, int, int],
    pad_x_ratio: float = 0.0,
    pad_y_ratio: float = 0.0,
) -> Any:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = int(round(box_width * pad_x_ratio))
    pad_y = int(round(box_height * pad_y_ratio))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    return image[top:bottom, left:right]


class BaselinePlateOCR:
    def __init__(self, backend: PaddlePlateOCR) -> None:
        self.backend = backend

    def recognize(self, image: Any) -> PlateRecognition | None:
        if self.backend._ocr is None:
            self.backend.load()

        raw = self.backend._recognize_single_candidate(image)
        if raw is None:
            return None

        cleaned = normalize_full_plate_for_eval(raw.text)
        if not cleaned:
            return None

        return PlateRecognition(text=cleaned, confidence=raw.confidence)
```

- [ ] **Step 4: Implement per-sample evaluation and summary aggregation in `ocr_small_sample.py`**

```python
from pathlib import Path

from car_system.data.ccpd import decode_ccpd_plate_indices


def evaluate_sample(
    dataset_root: Path,
    relative_path: Path,
    annotation,
    baseline_ocr,
    stabilized_ocr,
) -> dict[str, object]:
    gt_full_text = decode_ccpd_plate_indices(annotation.plate_indices)
    gt_eval_text = normalize_full_plate_for_eval(gt_full_text)

    image = load_bgr_image(dataset_root / relative_path)
    baseline_crop = crop_plate_region(image, annotation.bbox, pad_x_ratio=0.0, pad_y_ratio=0.0)
    stabilized_crop = crop_plate_region(image, annotation.bbox, pad_x_ratio=0.08, pad_y_ratio=0.12)

    baseline_result = baseline_ocr.recognize(baseline_crop)
    stabilized_result = stabilized_ocr.recognize(stabilized_crop)

    baseline_text = baseline_result.text if baseline_result else ""
    stabilized_text = stabilized_result.text if stabilized_result else ""

    baseline_char_correct, char_total = compute_char_match_counts(gt_eval_text, baseline_text)
    stabilized_char_correct, _ = compute_char_match_counts(gt_eval_text, stabilized_text)

    return {
        "relative_path": str(relative_path),
        "subset": subset_name_from_path(relative_path),
        "gt_full_text": gt_full_text,
        "gt_eval_text": gt_eval_text,
        "baseline_text": baseline_text,
        "stabilized_text": stabilized_text,
        "baseline_exact_match": baseline_text == gt_eval_text,
        "stabilized_exact_match": stabilized_text == gt_eval_text,
        "baseline_char_correct": baseline_char_correct,
        "stabilized_char_correct": stabilized_char_correct,
        "char_total": char_total,
        "baseline_char_accuracy": (baseline_char_correct / char_total) if char_total else 0.0,
        "stabilized_char_accuracy": (stabilized_char_correct / char_total) if char_total else 0.0,
        "baseline_is_null": baseline_text == "",
        "stabilized_is_null": stabilized_text == "",
        "baseline_confidence": baseline_result.confidence if baseline_result else None,
        "stabilized_confidence": stabilized_result.confidence if stabilized_result else None,
    }


def build_summary(
    rows: list[dict[str, object]],
    dataset_root: Path,
    split_file: Path,
    subsets: list[str],
    per_subset: int,
    seed: int,
    skipped: list[dict[str, object]],
) -> dict[str, object]:
    def summarize_mode_for_rows(sample_rows: list[dict[str, object]], mode_key: str) -> dict[str, object]:
        exact_count = sum(1 for row in sample_rows if row[f"{mode_key}_exact_match"])
        null_count = sum(1 for row in sample_rows if row[f"{mode_key}_is_null"])
        char_correct = sum(int(row[f"{mode_key}_char_correct"]) for row in sample_rows)
        char_total = sum(int(row["char_total"]) for row in sample_rows)
        sample_count = len(sample_rows)
        return {
            "sample_count": sample_count,
            "exact_match_count": exact_count,
            "exact_match_rate": (exact_count / sample_count) if sample_count else 0.0,
            "char_correct": char_correct,
            "char_total": char_total,
            "char_accuracy": (char_correct / char_total) if char_total else 0.0,
            "null_count": null_count,
        }

    subset_summaries: dict[str, object] = {}
    for subset in subsets:
        subset_rows = [row for row in rows if row["subset"] == subset]
        subset_summaries[subset] = {
            "sample_count": len(subset_rows),
            "baseline": summarize_mode_for_rows(subset_rows, "baseline"),
            "stabilized": summarize_mode_for_rows(subset_rows, "stabilized"),
        }

    return {
        "dataset_root": str(dataset_root),
        "split_file": str(split_file),
        "subsets": subsets,
        "per_subset_target": per_subset,
        "seed": seed,
        "sample_count": len(rows),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "baseline": summarize_mode_for_rows(rows, "baseline"),
        "stabilized": summarize_mode_for_rows(rows, "stabilized"),
        "per_subset": subset_summaries,
    }
```

- [ ] **Step 5: Re-run the evaluation tests and verify they pass**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ocr_small_sample_eval.py -q
```

Expected:

```text
all tests in `test_ocr_small_sample_eval.py` pass
```

### Task 4: Add the CLI script and wire outputs

**Files:**
- Create: `D:\Projects\Car\scripts\evaluate_ocr_small_sample.py`
- Modify: `D:\Projects\Car\src\car_system\experiments\ocr_small_sample.py`

- [ ] **Step 1: Write the CLI script with explicit arguments and defaults**

```python
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.data.ccpd import load_split_entries, parse_ccpd_path
from car_system.experiments.ocr_small_sample import (
    BaselinePlateOCR,
    build_summary,
    evaluate_sample,
    sample_entries_by_subset,
)
from car_system.io.writers import ensure_output_dir, write_csv, write_json
from car_system.ocr.plate_ocr import PaddlePlateOCR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate baseline vs stabilized OCR on a CCPD small sample.")
    parser.add_argument("--dataset-root", default="D:/plate_project/CCPD2019")
    parser.add_argument("--split-file", default="D:/plate_project/CCPD2019/splits/test.txt")
    parser.add_argument("--subsets", nargs="+", default=["ccpd_base", "ccpd_blur", "ccpd_db", "ccpd_rotate", "ccpd_tilt", "ccpd_weather", "ccpd_challenge"])
    parser.add_argument("--per-subset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "ocr_small_sample_eval"))
    return parser
```

- [ ] **Step 2: Add the main evaluation loop and file writing**

```python
def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    split_file = Path(args.split_file)
    output_dir = ensure_output_dir(args.output_dir)

    entries = load_split_entries(split_file)
    sampled_entries = sample_entries_by_subset(entries, subsets=args.subsets, per_subset=args.per_subset, seed=args.seed)

    stabilized_ocr = PaddlePlateOCR(language="ch", use_angle_cls=False)
    baseline_ocr = BaselinePlateOCR(stabilized_ocr)

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for relative_path in sampled_entries:
        try:
            annotation = parse_ccpd_path(relative_path)
            rows.append(
                evaluate_sample(
                    dataset_root=dataset_root,
                    relative_path=relative_path,
                    annotation=annotation,
                    baseline_ocr=baseline_ocr,
                    stabilized_ocr=stabilized_ocr,
                )
            )
        except Exception as exc:
            skipped.append({"relative_path": str(relative_path), "reason": str(exc)})

    summary = build_summary(
        rows=rows,
        dataset_root=dataset_root,
        split_file=split_file,
        subsets=args.subsets,
        per_subset=args.per_subset,
        seed=args.seed,
        skipped=skipped,
    )

    samples_path = write_csv(output_dir / "samples.csv", rows)
    summary_path = write_json(output_dir / "summary.json", summary)
    print(f"Samples CSV: {samples_path}")
    print(f"Summary JSON: {summary_path}")
    return 0
```

- [ ] **Step 3: Keep the script thin by moving any duplicated logic back into `ocr_small_sample.py`**

```text
Do not duplicate sampling, normalization, summary, or OCR-mode logic inside the script. The script should only parse args, call module functions, and write outputs.
```

- [ ] **Step 4: Run targeted tests to verify the new module still passes**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ccpd.py D:\Projects\Car\tests\test_ocr_small_sample_eval.py -q
```

Expected:

```text
targeted tests pass
```

### Task 5: Verify locally and document runtime limits honestly

**Files:**
- Verify only

- [ ] **Step 1: Run the targeted local test suite**

Run:

```powershell
python -m pytest D:\Projects\Car\tests\test_ccpd.py D:\Projects\Car\tests\test_ocr_small_sample_eval.py -q
```

Expected:

```text
all targeted tests pass
```

- [ ] **Step 2: Run the full local suite to guard against regressions**

Run:

```powershell
python -m pytest -q
```

Expected:

```text
all tests pass
```

- [ ] **Step 3: If PaddleOCR is importable locally, run a one-subset smoke command**

Run:

```powershell
@'
from paddleocr import TextRecognition
print(TextRecognition.__name__)
'@ | python -
```

Expected:

```text
TextRecognition
```

If the import succeeds, run:

```powershell
python D:\Projects\Car\scripts\evaluate_ocr_small_sample.py `
  --dataset-root D:\plate_project\CCPD2019 `
  --split-file D:\plate_project\CCPD2019\splits\test.txt `
  --subsets ccpd_base `
  --per-subset 1 `
  --seed 42 `
  --output-dir D:\Projects\Car\outputs\ocr_small_sample_eval_smoke
```

Expected:

```text
prints the paths to samples.csv and summary.json
```

- [ ] **Step 4: If PaddleOCR is not importable locally, stop after pytest and report the exact blocker**

```text
Report: "Unit tests passed, but the script smoke run was not executed because PaddleOCR TextRecognition is unavailable in the local environment."
```

## Self-Review

- Spec coverage: the plan covers CCPD plate decoding, normalized evaluation text, per-subset sampling, baseline vs stabilized OCR comparison, summary output, and tests for all core helpers.
- Placeholder scan: no unresolved placeholder markers remain in the actionable steps.
- Type consistency: the plan uses `decode_ccpd_plate_indices`, `normalize_full_plate_for_eval`, `compute_char_match_counts`, `BaselinePlateOCR`, `evaluate_sample`, and `build_summary` consistently across tasks.
