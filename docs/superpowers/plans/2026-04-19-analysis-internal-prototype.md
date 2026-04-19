# Analysis Internal Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first-stage analysis-focused internal prototype that runs a fixed image review set on the server, exports full diagnostics and intermediate artifacts, and generates a local batch report that makes pipeline errors attributable.

**Architecture:** Keep the current inference pipeline as the core execution path, but extend it with diagnostic metadata so every matched plate can expose crop, rectification, OCR raw text, normalized text, and an error status. Add one dedicated server-side review-set runner and one dedicated local HTML report generator so the diagnostic workflow stays separate from the current ad hoc smoke scripts.

**Tech Stack:** Python 3.13 local / Python 3.12 server, pytest, OpenCV, existing `car_system` package, YAML configs, HTML report generation, SSH-based server execution.

---

## File Map

### Existing files to modify

- Modify: `D:/Projects/Car/src/car_system/types.py`
- Modify: `D:/Projects/Car/src/car_system/ocr/plate_ocr.py`
- Modify: `D:/Projects/Car/src/car_system/pipeline/runner.py`
- Modify: `D:/Projects/Car/src/car_system/io/writers.py`
- Modify: `D:/Projects/Car/tests/test_runner.py`
- Modify: `D:/Projects/Car/tests/test_writers.py`

### New source files to create

- Create: `D:/Projects/Car/src/car_system/diagnostics/__init__.py`
- Create: `D:/Projects/Car/src/car_system/diagnostics/review_set.py`
- Create: `D:/Projects/Car/src/car_system/diagnostics/export.py`
- Create: `D:/Projects/Car/src/car_system/diagnostics/reporting.py`

### New scripts to create

- Create: `D:/Projects/Car/scripts/run_internal_review_set.py`
- Create: `D:/Projects/Car/scripts/build_internal_analysis_report.py`

### New config and docs files to create

- Create: `D:/Projects/Car/configs/review_sets/internal_analysis_v1.yaml`
- Create: `D:/Projects/Car/docs/internal-analysis-prototype-runbook.md`

### New tests to create

- Create: `D:/Projects/Car/tests/test_review_set.py`
- Create: `D:/Projects/Car/tests/test_diagnostic_export.py`
- Create: `D:/Projects/Car/tests/test_reporting.py`

---

### Task 1: Extend Core Types And OCR Metadata For Diagnostics

**Files:**
- Modify: `D:/Projects/Car/src/car_system/types.py`
- Modify: `D:/Projects/Car/src/car_system/ocr/plate_ocr.py`
- Modify: `D:/Projects/Car/src/car_system/pipeline/runner.py`
- Modify: `D:/Projects/Car/tests/test_runner.py`

- [ ] **Step 1: Write the failing tests for diagnostic OCR metadata**

Add these tests to `D:/Projects/Car/tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Run the test to confirm the current pipeline cannot satisfy it**

Run:

```powershell
pytest D:\Projects\Car\tests\test_runner.py -q
```

Expected: FAIL because `PlateRecognition` does not expose `raw_text` / `normalized_text`, `PlateMatch` has no `diagnostic`, and `PipelineRunner` does not assign a diagnostic status.

- [ ] **Step 3: Add the minimal diagnostic-capable domain model**

Update `D:/Projects/Car/src/car_system/types.py` to introduce explicit diagnostic metadata:

```python
from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    bbox: BBox


@dataclass(slots=True)
class PlateRecognition:
    text: str
    confidence: float
    raw_text: str | None = None
    normalized_text: str | None = None


@dataclass(slots=True)
class PlateDiagnostic:
    status: str
    crop_bbox: tuple[int, int, int, int]
    crop_shape: tuple[int, ...] | None = None
    rectified_shape: tuple[int, ...] | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PlateMatch:
    plate: Detection
    vehicle: Detection | None
    recognition: PlateRecognition | None = None
    diagnostic: PlateDiagnostic | None = None


@dataclass(slots=True)
class FrameResult:
    source_name: str
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    matches: list[PlateMatch] = field(default_factory=list)
```

- [ ] **Step 4: Teach the OCR layer and pipeline to populate diagnostic fields**

Update `D:/Projects/Car/src/car_system/ocr/plate_ocr.py` so `recognize_raw()` preserves the model output as `raw_text`, and `recognize()` returns normalized text while keeping the raw one:

```python
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
        candidate_results.append(
            (
                PlateRecognition(
                    text=result.text,
                    confidence=result.confidence,
                    raw_text=result.text,
                    normalized_text=normalized_text,
                ),
                self._score_candidate(PlateRecognition(text=normalized_text, confidence=result.confidence)),
            )
        )
    if not candidate_results:
        return None
    best_result, _ = max(candidate_results, key=lambda item: item[1])
    return best_result


def recognize(self, image: Any) -> PlateRecognition | None:
    raw_result = self.recognize_raw(image)
    if raw_result is None or not raw_result.normalized_text:
        return None
    return PlateRecognition(
        text=raw_result.normalized_text,
        confidence=raw_result.confidence,
        raw_text=raw_result.raw_text,
        normalized_text=raw_result.normalized_text,
    )
```

Update `D:/Projects/Car/src/car_system/pipeline/runner.py` so it uses `recognize_raw()` when available and attaches a diagnostic:

```python
def _diagnostic_status(raw_text: str | None, normalized_text: str | None) -> str:
    if raw_text is None:
        return "ocr_null"
    if normalized_text is None or not normalized_text:
        return "ocr_invalid_text"
    if len(raw_text) >= 4 and raw_text[:2] == raw_text[2:4]:
        return "ocr_abnormal_text"
    return "recognized"


for match in matches:
    crop_bbox = _expand_bbox(match.plate.bbox, image.shape)
    crop = _crop_bbox(image, crop_bbox)
    rectified = rectify_plate(crop)
    if hasattr(self.ocr_engine, "recognize_raw"):
        raw_recognition = self.ocr_engine.recognize_raw(rectified)
    else:
        raw_recognition = self.ocr_engine.recognize(rectified)
    normalized_text = None
    if raw_recognition is not None:
        normalized_text = raw_recognition.normalized_text or raw_recognition.text
        match.recognition = PlateRecognition(
            text=normalized_text,
            confidence=raw_recognition.confidence,
            raw_text=raw_recognition.raw_text or raw_recognition.text,
            normalized_text=normalized_text,
        )
    match.diagnostic = PlateDiagnostic(
        status=_diagnostic_status(
            raw_recognition.raw_text if raw_recognition else None,
            normalized_text,
        ),
        crop_bbox=crop_bbox,
        crop_shape=getattr(crop, "shape", None),
        rectified_shape=getattr(rectified, "shape", None),
        raw_text=raw_recognition.raw_text if raw_recognition else None,
        normalized_text=normalized_text,
    )
```

- [ ] **Step 5: Run the focused tests and record the checkpoint**

Run:

```powershell
pytest D:\Projects\Car\tests\test_runner.py -q
```

Expected: PASS.

Checkpoint note: this workspace is not a git repository, so store the passing command and output in the execution log instead of attempting a commit.

---

### Task 2: Export Diagnostic Rows, JSON, And Intermediate Artifact Metadata

**Files:**
- Create: `D:/Projects/Car/src/car_system/diagnostics/export.py`
- Modify: `D:/Projects/Car/src/car_system/io/writers.py`
- Modify: `D:/Projects/Car/tests/test_writers.py`
- Create: `D:/Projects/Car/tests/test_diagnostic_export.py`

- [ ] **Step 1: Write failing tests for diagnostic export payloads**

Create `D:/Projects/Car/tests/test_diagnostic_export.py` with:

```python
from pathlib import Path

import numpy as np

from car_system.diagnostics.export import export_frame_diagnostics
from car_system.types import (
    Detection,
    FrameResult,
    PlateDiagnostic,
    PlateMatch,
    PlateRecognition,
)


def test_export_frame_diagnostics_writes_crop_rectified_and_manifest(tmp_path) -> None:
    vehicle = Detection(label="car", confidence=0.91, bbox=(10, 10, 120, 120))
    plate = Detection(label="plate", confidence=0.96, bbox=(40, 90, 95, 115))
    recognition = PlateRecognition(
        text="皖A12340",
        confidence=0.89,
        raw_text="皖A1234O",
        normalized_text="皖A12340",
    )
    diagnostic = PlateDiagnostic(
        status="recognized",
        crop_bbox=(36, 87, 99, 118),
        crop_shape=(31, 63, 3),
        rectified_shape=(31, 63, 3),
        raw_text="皖A1234O",
        normalized_text="皖A12340",
    )
    result = FrameResult(
        source_name="sample.jpg",
        frame_index=0,
        detections=[vehicle, plate],
        matches=[PlateMatch(plate=plate, vehicle=vehicle, recognition=recognition, diagnostic=diagnostic)],
    )

    source = np.zeros((160, 160, 3), dtype=np.uint8)
    crop = np.zeros((31, 63, 3), dtype=np.uint8)
    rectified = np.ones((31, 63, 3), dtype=np.uint8)

    payload = export_frame_diagnostics(
        output_dir=tmp_path,
        frame_result=result,
        source_image=source,
        crops=[crop],
        rectified_images=[rectified],
    )

    assert Path(payload["diagnostics"][0]["crop_path"]).exists()
    assert Path(payload["diagnostics"][0]["rectified_path"]).exists()
    assert payload["diagnostics"][0]["raw_text"] == "皖A1234O"
    assert payload["diagnostics"][0]["normalized_text"] == "皖A12340"
```

Append to `D:/Projects/Car/tests/test_writers.py`:

```python
def test_frame_result_to_rows_includes_diagnostic_status() -> None:
    result = make_frame_result()
    result.matches[0].diagnostic = PlateDiagnostic(
        status="recognized",
        crop_bbox=(36, 87, 99, 118),
        raw_text="ABC12O",
        normalized_text="ABC120",
    )

    rows = frame_result_to_rows(result)

    assert rows[0]["diagnostic_status"] == "recognized"
    assert rows[0]["ocr_raw_text"] == "ABC12O"
    assert rows[0]["ocr_normalized_text"] == "ABC120"
```

- [ ] **Step 2: Run the focused tests and verify the current code fails**

Run:

```powershell
pytest D:\Projects\Car\tests\test_writers.py D:\Projects\Car\tests\test_diagnostic_export.py -q
```

Expected: FAIL because diagnostic export does not exist and CSV rows do not include diagnostic fields.

- [ ] **Step 3: Add a dedicated diagnostic exporter**

Create `D:/Projects/Car/src/car_system/diagnostics/export.py`:

```python
from pathlib import Path

from car_system.io.media import save_image
from car_system.io.writers import write_json


def export_frame_diagnostics(output_dir, frame_result, source_image, crops, rectified_images):
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for index, match in enumerate(frame_result.matches):
        crop_path = target_dir / f"{Path(frame_result.source_name).stem}_match_{index}_crop.jpg"
        rectified_path = target_dir / f"{Path(frame_result.source_name).stem}_match_{index}_rectified.jpg"
        save_image(crop_path, crops[index])
        save_image(rectified_path, rectified_images[index])
        diagnostics.append(
            {
                "match_index": index,
                "status": match.diagnostic.status if match.diagnostic else "missing",
                "crop_path": str(crop_path),
                "rectified_path": str(rectified_path),
                "raw_text": match.diagnostic.raw_text if match.diagnostic else None,
                "normalized_text": match.diagnostic.normalized_text if match.diagnostic else None,
            }
        )
    payload = {
        "source_name": frame_result.source_name,
        "frame_index": frame_result.frame_index,
        "diagnostics": diagnostics,
    }
    write_json(target_dir / f"{Path(frame_result.source_name).stem}_diagnostics.json", payload)
    return payload
```

- [ ] **Step 4: Extend structured row export to include diagnostic columns**

Update `D:/Projects/Car/src/car_system/io/writers.py`:

```python
def frame_result_to_rows(result: FrameResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for match in result.matches:
        rows.append(
            {
                "source_name": result.source_name,
                "frame_index": result.frame_index,
                "vehicle_label": match.vehicle.label if match.vehicle else "",
                "vehicle_confidence": match.vehicle.confidence if match.vehicle else None,
                "plate_confidence": match.plate.confidence,
                "plate_text": match.recognition.text if match.recognition else "",
                "ocr_confidence": match.recognition.confidence if match.recognition else None,
                "diagnostic_status": match.diagnostic.status if match.diagnostic else "",
                "ocr_raw_text": match.diagnostic.raw_text if match.diagnostic else "",
                "ocr_normalized_text": match.diagnostic.normalized_text if match.diagnostic else "",
            }
        )
    return rows
```

- [ ] **Step 5: Run the export-focused tests and record the checkpoint**

Run:

```powershell
pytest D:\Projects\Car\tests\test_writers.py D:\Projects\Car\tests\test_diagnostic_export.py -q
```

Expected: PASS.

Checkpoint note: keep the generated sample manifest shape stable because later report tasks depend on these keys.

---

### Task 3: Create A Fixed Review-Set Format And Server Batch Runner

**Files:**
- Create: `D:/Projects/Car/src/car_system/diagnostics/review_set.py`
- Create: `D:/Projects/Car/scripts/run_internal_review_set.py`
- Create: `D:/Projects/Car/configs/review_sets/internal_analysis_v1.yaml`
- Create: `D:/Projects/Car/tests/test_review_set.py`

- [ ] **Step 1: Write failing tests for review-set loading**

Create `D:/Projects/Car/tests/test_review_set.py`:

```python
from pathlib import Path

from car_system.diagnostics.review_set import load_review_set


def test_load_review_set_preserves_fixed_sample_order(tmp_path) -> None:
    review_yaml = tmp_path / "review.yaml"
    review_yaml.write_text(
        \"\"\"
dataset_root: /root/autodl-tmp/datasets/ccpd_yolo_mvp/images/test
samples:
  - category: easy
    relative_path: ccpd_base/example_a.jpg
  - category: blur
    relative_path: ccpd_blur/example_b.jpg
\"\"\".strip(),
        encoding="utf-8",
    )

    review_set = load_review_set(review_yaml)

    assert review_set.dataset_root == Path("/root/autodl-tmp/datasets/ccpd_yolo_mvp/images/test")
    assert [item.category for item in review_set.samples] == ["easy", "blur"]
    assert [item.relative_path.as_posix() for item in review_set.samples] == [
        "ccpd_base/example_a.jpg",
        "ccpd_blur/example_b.jpg",
    ]
```

- [ ] **Step 2: Run the test to confirm the loader does not exist yet**

Run:

```powershell
pytest D:\Projects\Car\tests\test_review_set.py -q
```

Expected: FAIL because `car_system.diagnostics.review_set` is missing.

- [ ] **Step 3: Implement the fixed review-set loader**

Create `D:/Projects/Car/src/car_system/diagnostics/review_set.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ReviewSample:
    category: str
    relative_path: Path


@dataclass(slots=True)
class ReviewSet:
    dataset_root: Path
    samples: list[ReviewSample]


def load_review_set(path: str | Path) -> ReviewSet:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    dataset_root = Path(payload["dataset_root"])
    samples = [
        ReviewSample(
            category=item["category"],
            relative_path=Path(item["relative_path"]),
        )
        for item in payload["samples"]
    ]
    return ReviewSet(dataset_root=dataset_root, samples=samples)
```

- [ ] **Step 4: Add the first curated review-set config and batch runner**

Create `D:/Projects/Car/configs/review_sets/internal_analysis_v1.yaml`:

```yaml
dataset_root: /root/autodl-tmp/datasets/ccpd_yolo_mvp/images/test
samples:
  - category: easy
    relative_path: ccpd_base/REPLACE_WITH_REAL_FILENAME_A.jpg
  - category: easy
    relative_path: ccpd_base/REPLACE_WITH_REAL_FILENAME_B.jpg
  - category: blur
    relative_path: ccpd_blur/REPLACE_WITH_REAL_FILENAME_C.jpg
  - category: tilt
    relative_path: ccpd_tilt/REPLACE_WITH_REAL_FILENAME_D.jpg
  - category: failure
    relative_path: ccpd_blur/REPLACE_WITH_REAL_FILENAME_E.jpg
```

Create `D:/Projects/Car/scripts/run_internal_review_set.py`:

```python
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.config import load_config
from car_system.diagnostics.export import export_frame_diagnostics
from car_system.diagnostics.review_set import load_review_set
from car_system.io.media import load_image
from car_system.pipeline.runner import PipelineRunner
from car_system.runtime import build_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--review-set", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    vehicle_detector, plate_detector, ocr_engine = build_runtime(config)
    review_set = load_review_set(args.review_set)
    runner = PipelineRunner(config, vehicle_detector, plate_detector, ocr_engine)

    for sample in review_set.samples:
        source_path = review_set.dataset_root / sample.relative_path
        image = load_image(source_path)
        result = runner.run_frame(image=image, source_name=sample.relative_path.name, frame_index=0)
        # Step 4 keeps runner output and export loop simple; refine later if needed.
        print(sample.category, sample.relative_path.as_posix(), len(result.matches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

After adding the skeleton, replace the placeholder filenames in the YAML with five real files from the server dataset before any execution run.

- [ ] **Step 5: Run the loader test and a dry execution checkpoint**

Run:

```powershell
pytest D:\Projects\Car\tests\test_review_set.py -q
python D:\Projects\Car\scripts\run_internal_review_set.py --help
```

Expected: the test passes and the script prints CLI usage.

Checkpoint note: do not execute the server batch yet; that belongs after the local report path exists.

---

### Task 4: Build The Local Batch Diagnostic Report Generator

**Files:**
- Create: `D:/Projects/Car/src/car_system/diagnostics/reporting.py`
- Create: `D:/Projects/Car/scripts/build_internal_analysis_report.py`
- Create: `D:/Projects/Car/tests/test_reporting.py`

- [ ] **Step 1: Write failing tests for summary metrics and HTML generation**

Create `D:/Projects/Car/tests/test_reporting.py`:

```python
from pathlib import Path

from car_system.diagnostics.reporting import build_report_summary, render_html_report


def test_build_report_summary_counts_statuses_and_accuracy() -> None:
    rows = [
        {
            "category": "easy",
            "diagnostic_status": "recognized",
            "gt_text": "皖A12345",
            "ocr_normalized_text": "皖A12345",
        },
        {
            "category": "blur",
            "diagnostic_status": "ocr_null",
            "gt_text": "皖A12346",
            "ocr_normalized_text": "",
        },
    ]

    summary = build_report_summary(rows)

    assert summary["total_samples"] == 2
    assert summary["status_counts"]["recognized"] == 1
    assert summary["status_counts"]["ocr_null"] == 1
    assert summary["exact_plate_accuracy"] == 0.5
    assert summary["null_rate"] == 0.5


def test_render_html_report_includes_failure_section(tmp_path) -> None:
    summary = {
        "total_samples": 2,
        "status_counts": {"recognized": 1, "ocr_null": 1},
        "exact_plate_accuracy": 0.5,
        "char_accuracy": 0.5,
        "null_rate": 0.5,
        "by_category": {"easy": {"samples": 1}, "blur": {"samples": 1}},
    }
    failures = [
        {
            "source_name": "failure.jpg",
            "diagnostic_status": "ocr_null",
            "gt_text": "皖A12346",
            "ocr_normalized_text": "",
            "crop_path": "failure_crop.jpg",
            "rectified_path": "failure_rectified.jpg",
        }
    ]

    html = render_html_report(summary, failures)

    assert "failure.jpg" in html
    assert "ocr_null" in html
    assert "皖A12346" in html
```

- [ ] **Step 2: Run the report tests and confirm they fail**

Run:

```powershell
pytest D:\Projects\Car\tests\test_reporting.py -q
```

Expected: FAIL because no reporting module exists yet.

- [ ] **Step 3: Implement summary computation and HTML rendering**

Create `D:/Projects/Car/src/car_system/diagnostics/reporting.py`:

```python
from collections import Counter, defaultdict


def _char_accuracy(expected: str, predicted: str) -> tuple[int, int]:
    total = len(expected)
    correct = sum(1 for index, char in enumerate(expected) if index < len(predicted) and predicted[index] == char)
    return correct, total


def build_report_summary(rows):
    total = len(rows)
    status_counts = Counter(row.get("diagnostic_status", "missing") for row in rows)
    exact_matches = 0
    null_count = 0
    char_correct = 0
    char_total = 0
    by_category = defaultdict(lambda: {"samples": 0, "recognized": 0, "ocr_null": 0})

    for row in rows:
        category = row.get("category", "uncategorized")
        status = row.get("diagnostic_status", "missing")
        gt_text = row.get("gt_text", "")
        predicted = row.get("ocr_normalized_text", "") or ""
        by_category[category]["samples"] += 1
        by_category[category][status] = by_category[category].get(status, 0) + 1
        if not predicted:
            null_count += 1
        if predicted == gt_text:
            exact_matches += 1
        correct, total_chars = _char_accuracy(gt_text, predicted)
        char_correct += correct
        char_total += total_chars

    return {
        "total_samples": total,
        "status_counts": dict(status_counts),
        "exact_plate_accuracy": exact_matches / total if total else 0.0,
        "char_accuracy": char_correct / char_total if char_total else 0.0,
        "null_rate": null_count / total if total else 0.0,
        "by_category": dict(by_category),
    }


def render_html_report(summary, failures):
    failure_rows = "".join(
        f"<tr><td>{item['source_name']}</td><td>{item['diagnostic_status']}</td><td>{item['gt_text']}</td><td>{item['ocr_normalized_text']}</td></tr>"
        for item in failures
    )
    return f\"\"\"
    <html>
      <body>
        <h1>Internal Analysis Report</h1>
        <p>Total samples: {summary['total_samples']}</p>
        <p>Exact plate accuracy: {summary['exact_plate_accuracy']:.4f}</p>
        <p>Character accuracy: {summary['char_accuracy']:.4f}</p>
        <p>Null rate: {summary['null_rate']:.4f}</p>
        <h2>Failures</h2>
        <table>{failure_rows}</table>
      </body>
    </html>
    \"\"\"
```

- [ ] **Step 4: Wire the report builder CLI**

Create `D:/Projects/Car/scripts/build_internal_analysis_report.py`:

```python
import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.diagnostics.reporting import build_report_summary, render_html_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = build_report_summary(rows)
    failures = [row for row in rows if row.get("diagnostic_status") != "recognized"]
    html = render_html_report(summary, failures)
    Path(args.output_html).write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the reporting tests and keep the CLI stable**

Run:

```powershell
pytest D:\Projects\Car\tests\test_reporting.py -q
python D:\Projects\Car\scripts\build_internal_analysis_report.py --help
```

Expected: tests pass and the script prints usage.

Checkpoint note: the first report can be visually ugly; correctness and traceability matter more than CSS.

---

### Task 5: Integrate The Review Workflow And Document The Runbook

**Files:**
- Modify: `D:/Projects/Car/scripts/run_internal_review_set.py`
- Modify: `D:/Projects/Car/scripts/build_internal_analysis_report.py`
- Create: `D:/Projects/Car/docs/internal-analysis-prototype-runbook.md`
- Test: `D:/Projects/Car/tests/test_review_set.py`
- Test: `D:/Projects/Car/tests/test_reporting.py`

- [ ] **Step 1: Write the runbook-first acceptance checklist**

Create `D:/Projects/Car/docs/internal-analysis-prototype-runbook.md` with:

```markdown
# Internal Analysis Prototype Runbook

## Server execution

1. Confirm server Python path:
   `/root/miniconda3/bin/python --version`
2. Run the fixed review set:
   `/root/miniconda3/bin/python scripts/run_internal_review_set.py --config configs/remote_mvp_specialized.yaml --review-set configs/review_sets/internal_analysis_v1.yaml --output-dir outputs/internal_analysis_v1`

## Local report build

1. Pull the CSV/JSON/artifact directory from the server.
2. Build the report:
   `python scripts/build_internal_analysis_report.py --input-csv outputs/internal_analysis_v1/review_results.csv --output-html outputs/internal_analysis_v1/report.html`

## Acceptance

- Report exists
- Report shows summary metrics
- Report shows at least one failure table
- Every failure row can be traced back to crop and rectified images
```

- [ ] **Step 2: Finish the server runner so it writes one review CSV and diagnostics directory**

Update `D:/Projects/Car/scripts/run_internal_review_set.py` so it no longer just prints counts:

```python
import csv
from car_system.data.ccpd import parse_ccpd_path, decode_ccpd_plate_indices
from car_system.diagnostics.export import export_frame_diagnostics
from car_system.io.media import load_image

rows = []
for sample in review_set.samples:
    source_path = review_set.dataset_root / sample.relative_path
    annotation = parse_ccpd_path(sample.relative_path)
    gt_text = decode_ccpd_plate_indices(annotation.plate_indices)
    image = load_image(source_path)
    result = runner.run_frame(image=image, source_name=sample.relative_path.name, frame_index=0)
    export_payload = export_frame_diagnostics(
        output_dir=Path(args.output_dir) / Path(sample.relative_path).stem,
        frame_result=result,
        source_image=image,
        crops=[],
        rectified_images=[],
    )
    for diagnostic in export_payload["diagnostics"]:
        rows.append(
            {
                "category": sample.category,
                "source_name": sample.relative_path.name,
                "gt_text": gt_text,
                "diagnostic_status": diagnostic["status"],
                "ocr_raw_text": diagnostic["raw_text"] or "",
                "ocr_normalized_text": diagnostic["normalized_text"] or "",
                "crop_path": diagnostic["crop_path"],
                "rectified_path": diagnostic["rectified_path"],
            }
        )

with (Path(args.output_dir) / "review_results.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
    if rows:
        writer.writeheader()
        writer.writerows(rows)
```

When implementing this step for real, do not leave `crops=[]` and `rectified_images=[]`; instead plumb the actual crop and rectified arrays out of the pipeline before calling `export_frame_diagnostics`.

- [ ] **Step 3: Run all local tests before any server execution**

Run:

```powershell
pytest D:\Projects\Car\tests\test_runner.py D:\Projects\Car\tests\test_writers.py D:\Projects\Car\tests\test_review_set.py D:\Projects\Car\tests\test_diagnostic_export.py D:\Projects\Car\tests\test_reporting.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the first full server-to-local diagnostic cycle**

Server run:

```bash
cd /root/autodl-tmp/car-project
/root/miniconda3/bin/python scripts/run_internal_review_set.py \
  --config configs/remote_mvp_specialized.yaml \
  --review-set configs/review_sets/internal_analysis_v1.yaml \
  --output-dir outputs/internal_analysis_v1
```

Local report build:

```powershell
python D:\Projects\Car\scripts\build_internal_analysis_report.py `
  --input-csv D:\Projects\Car\outputs\internal_analysis_v1\review_results.csv `
  --output-html D:\Projects\Car\outputs\internal_analysis_v1\report.html
```

Expected: server writes `review_results.csv` plus per-sample artifact directories; local writes `report.html`.

- [ ] **Step 5: Record the first bottleneck decision instead of a commit**

Write a short execution note at the end of `D:/Projects/Car/docs/internal-analysis-prototype-runbook.md`:

```markdown
## First diagnostic conclusion

- Dominant error type: `REPLACE_AFTER_FIRST_RUN`
- Recommended next investment: `OCR` or `detection/cropping`
- Evidence: reference the failing categories and top representative samples
```

Checkpoint note: this workspace has no usable git repository, so the durable handoff is the report directory plus the runbook conclusion block.

---

## Self-Review

### Spec coverage

- Fixed review set: covered by Task 3.
- Server-side inference on a fixed set: covered by Task 3 and Task 5.
- Local batch report: covered by Task 4 and Task 5.
- Diagnostic fields and error attribution: covered by Task 1 and Task 2.
- Acceptance based on analysis capability rather than model lift: covered by Task 5 runbook.

### Placeholder scan

- The only allowed placeholders are the real review-set filenames and the first-run conclusion block, both of which are explicitly marked as post-implementation substitutions tied to concrete execution data.
- No `TODO` / `TBD` markers remain in the task steps themselves.

### Type consistency

- `PlateRecognition` consistently carries `raw_text` and `normalized_text`.
- `PlateMatch` consistently carries `diagnostic`.
- Report rows consistently use `diagnostic_status`, `ocr_raw_text`, `ocr_normalized_text`, and `gt_text`.

