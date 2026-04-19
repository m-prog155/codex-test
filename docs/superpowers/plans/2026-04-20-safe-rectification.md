# Safe Rectification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe, fallback-first rectification branch for specialized OCR so we can A/B audit whether geometry-standardized plate inputs reduce wrong non-null outputs without regressing the fixed review set.

**Architecture:** Keep the current plain-crop specialized OCR path as the control. Add a new rectification mode that tries to find a reliable inner plate quad, returns structured success/failure metadata, and falls back to the plain crop whenever the quad is not trustworthy. Wire that metadata through the runner and diagnostic export so audits can explain whether a sample used rectification or why it fell back.

**Tech Stack:** Python, OpenCV (`cv2`), NumPy, pytest, YAML config, existing internal review/audit scripts

---

### Task 1: Add Config And Diagnostic Surface For Safe Rectification

**Files:**
- Modify: `D:/Projects/Car/src/car_system/config.py`
- Modify: `D:/Projects/Car/src/car_system/types.py`
- Modify: `D:/Projects/Car/tests/test_config.py`

- [ ] **Step 1: Write the failing config and diagnostic tests**

Add these tests to `D:/Projects/Car/tests/test_config.py`:

```python
def test_load_config_reads_safe_rectification_mode_and_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "ocr": {
                    "language": "ch",
                    "use_angle_cls": False,
                    "rectification_mode": "safe",
                    "safe_rect_min_area_ratio": 0.18,
                    "safe_rect_min_rectangularity": 0.72,
                    "safe_rect_max_center_offset": 0.28,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ocr.rectification_mode == "safe"
    assert config.ocr.safe_rect_min_area_ratio == 0.18
    assert config.ocr.safe_rect_min_rectangularity == 0.72
    assert config.ocr.safe_rect_max_center_offset == 0.28


def test_load_config_defaults_safe_rectification_to_disabled_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("ocr:\n  language: ch\n  use_angle_cls: false\n", encoding="utf-8")

    config = load_config(config_path)

    assert config.ocr.rectification_mode == "disabled"
    assert config.ocr.safe_rect_min_area_ratio == 0.12
    assert config.ocr.safe_rect_min_rectangularity == 0.7
    assert config.ocr.safe_rect_max_center_offset == 0.35
```

Add this assertion to `test_local_specialized_config_vehicle_labels_match_final_scope`:

```python
assert config.ocr.rectification_mode == "disabled"
```

- [ ] **Step 2: Run the targeted config tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_config.py -q
```

Expected: FAIL because `OcrConfig` does not expose `rectification_mode` or the safe-rectification thresholds yet.

- [ ] **Step 3: Add the config fields and diagnostic metadata**

Update `D:/Projects/Car/src/car_system/config.py` so `OcrConfig` carries:

```python
@dataclass(slots=True)
class OcrConfig:
    language: str
    use_angle_cls: bool
    mode: str = "generic"
    model_dir: str | None = None
    character_dict_path: str | None = None
    min_confidence: float = 0.0
    enable_rectification: bool = True
    rectification_mode: str = "disabled"
    crop_pad_x_ratio: float = 0.08
    crop_pad_y_ratio: float = 0.12
    safe_rect_min_area_ratio: float = 0.12
    safe_rect_min_rectangularity: float = 0.70
    safe_rect_max_center_offset: float = 0.35
```

Parse the new YAML keys in `load_config()`:

```python
    ocr = OcrConfig(
        language=str(ocr_raw.get("language", "ch")),
        use_angle_cls=bool(ocr_raw.get("use_angle_cls", False)),
        mode=str(ocr_raw.get("mode", "generic")),
        model_dir=str(ocr_raw["model_dir"]) if ocr_raw.get("model_dir") else None,
        character_dict_path=str(ocr_raw["character_dict_path"]) if ocr_raw.get("character_dict_path") else None,
        min_confidence=float(ocr_raw.get("min_confidence", 0.0)),
        enable_rectification=bool(ocr_raw.get("enable_rectification", True)),
        rectification_mode=str(ocr_raw.get("rectification_mode", "disabled")),
        crop_pad_x_ratio=float(ocr_raw.get("crop_pad_x_ratio", 0.08)),
        crop_pad_y_ratio=float(ocr_raw.get("crop_pad_y_ratio", 0.12)),
        safe_rect_min_area_ratio=float(ocr_raw.get("safe_rect_min_area_ratio", 0.12)),
        safe_rect_min_rectangularity=float(ocr_raw.get("safe_rect_min_rectangularity", 0.70)),
        safe_rect_max_center_offset=float(ocr_raw.get("safe_rect_max_center_offset", 0.35)),
    )
```

Extend `D:/Projects/Car/src/car_system/types.py` so `PlateDiagnostic` can carry safe-rectification provenance:

```python
@dataclass(slots=True)
class PlateDiagnostic:
    status: str
    crop_bbox: tuple[int, int, int, int]
    crop_shape: tuple[int, ...] | None = None
    rectified_shape: tuple[int, ...] | None = None
    confidence: float | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    rectification_mode: str | None = None
    rectification_applied: bool = False
    rectification_reason: str | None = None
    notes: list[str] = field(default_factory=list)
```

- [ ] **Step 4: Re-run the targeted config tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify git status and record that this workspace is not a repo**

Run:

```powershell
git rev-parse --show-toplevel
```

Expected: FAIL with `fatal: not a git repository`. Do not attempt a commit in this workspace; keep the task log in the plan/checkpoints instead.

### Task 2: Build Safe Rectification As A Fallback-First Primitive

**Files:**
- Modify: `D:/Projects/Car/src/car_system/ocr/rectify.py`
- Modify: `D:/Projects/Car/tests/test_rectify.py`

- [ ] **Step 1: Write failing rectification tests**

Add these tests to `D:/Projects/Car/tests/test_rectify.py`:

```python
def test_safe_rectify_plate_returns_success_for_canonical_skewed_plate() -> None:
    canonical, skewed = _make_skewed_plate_crop()

    result = safe_rectify_plate(skewed)

    assert result.applied is True
    assert result.reason == "applied"
    assert result.image.shape == canonical.shape
    mean_abs_diff = np.abs(result.image.astype(np.int16) - canonical.astype(np.int16)).mean()
    assert mean_abs_diff < 35.0


def test_safe_rectify_plate_falls_back_when_no_valid_quad_found() -> None:
    image = np.zeros((31, 63, 3), dtype=np.uint8)
    image[:, :32] = (255, 255, 255)

    result = safe_rectify_plate(image)

    assert result.applied is False
    assert result.reason in {"no_quad", "low_score"}
    assert result.image.shape == image.shape


def test_safe_rectify_plate_rejects_quad_with_large_center_offset() -> None:
    image = np.zeros((48, 168, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 4), (167, 44), (255, 255, 255), thickness=-1)

    result = safe_rectify_plate(image, max_center_offset=0.10)

    assert result.applied is False
    assert result.reason == "center_offset"
```

- [ ] **Step 2: Run the targeted rectification tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_rectify.py -q
```

Expected: FAIL because `safe_rectify_plate()` and the structured result object do not exist yet.

- [ ] **Step 3: Implement the safe rectification result object and fallback logic**

In `D:/Projects/Car/src/car_system/ocr/rectify.py`, add a focused result type and a new safe entrypoint:

```python
from dataclasses import dataclass


@dataclass(slots=True)
class RectifyResult:
    image: np.ndarray
    applied: bool
    reason: str
```

Add a score-aware quad finder:

```python
def _quad_diagnostics(points: np.ndarray, contour_area: float, image_shape: tuple[int, ...]) -> dict[str, float]:
    image_height, image_width = image_shape[:2]
    image_area = max(1.0, float(image_height * image_width))
    rect = cv2.minAreaRect(points.astype(np.float32))
    width, height = rect[1]
    short_edge = min(width, height)
    long_edge = max(width, height)
    if short_edge <= 0 or long_edge <= 0:
        return {"area_ratio": 0.0, "rectangularity": 0.0, "center_offset": 1.0, "aspect_ratio": 0.0}

    box_area = max(1.0, float(width * height))
    center_x = float(points[:, 0].mean())
    center_y = float(points[:, 1].mean())
    diagonal = max(1.0, float(np.hypot(image_width, image_height)))
    return {
        "area_ratio": float(contour_area) / image_area,
        "rectangularity": float(contour_area) / box_area,
        "center_offset": float(np.hypot(center_x - image_width / 2.0, center_y - image_height / 2.0)) / diagonal,
        "aspect_ratio": long_edge / short_edge,
    }
```

Keep the candidate sweep focused in `_find_plate_quad()`:

```python
def _find_plate_quad(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 160)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # keep the existing min-area contour sweep and return the best candidate quad
```

Add the safe API:

```python
def safe_rectify_plate(
    image: Any,
    output_size: tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    min_area_ratio: float = 0.12,
    min_rectangularity: float = 0.70,
    max_center_offset: float = 0.35,
) -> RectifyResult:
    color_image = _ensure_color_image(image)
    if color_image.size == 0:
        return RectifyResult(image=color_image, applied=False, reason="empty")

    plate_quad = _find_plate_quad(color_image)
    if plate_quad is None:
        return RectifyResult(image=color_image, applied=False, reason="no_quad")

    contour_area = float(cv2.contourArea(plate_quad.astype(np.float32)))
    score = _quad_diagnostics(plate_quad, contour_area, color_image.shape)
    if score["area_ratio"] < min_area_ratio:
        return RectifyResult(image=color_image, applied=False, reason="area_ratio")
    if score["rectangularity"] < min_rectangularity:
        return RectifyResult(image=color_image, applied=False, reason="rectangularity")
    if score["center_offset"] > max_center_offset:
        return RectifyResult(image=color_image, applied=False, reason="center_offset")

    rectified = warp_plate_from_vertices(
        color_image,
        [(int(x), int(y)) for x, y in plate_quad],
        output_size=output_size,
    )
    if rectified.size == 0:
        return RectifyResult(image=color_image, applied=False, reason="warp_failed")
    return RectifyResult(image=rectified, applied=True, reason="applied")
```

Keep the old `rectify_plate()` API for backward compatibility by making it call `safe_rectify_plate(...).image`.

- [ ] **Step 4: Re-run the targeted rectification tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_rectify.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify git status and record that this workspace is not a repo**

Run:

```powershell
git rev-parse --show-toplevel
```

Expected: FAIL with `fatal: not a git repository`. Do not attempt a commit in this workspace.

### Task 3: Wire Safe Rectification Through Runner And Diagnostic Export

**Files:**
- Modify: `D:/Projects/Car/src/car_system/pipeline/runner.py`
- Modify: `D:/Projects/Car/src/car_system/diagnostics/export.py`
- Modify: `D:/Projects/Car/tests/test_runner.py`
- Modify: `D:/Projects/Car/tests/test_diagnostic_export.py`

- [ ] **Step 1: Write the failing runner and export tests**

Add these tests to `D:/Projects/Car/tests/test_runner.py`:

```python
def test_run_frame_safe_rectification_falls_back_to_plain_crop_when_rejected(monkeypatch) -> None:
    config = make_config()
    config.ocr.enable_rectification = True
    config.ocr.rectification_mode = "safe"

    class FakeRectifyResult:
        def __init__(self):
            self.image = np.zeros((25, 55, 3), dtype=np.uint8)
            self.applied = False
            self.reason = "center_offset"

    monkeypatch.setattr("car_system.pipeline.runner.safe_rectify_plate", lambda *args, **kwargs: FakeRectifyResult())

    fake_ocr = FakeOCR()
    runner = PipelineRunner(config=config, vehicle_detector=EdgeVehicleDetector(), plate_detector=EdgePlateDetector(), ocr_engine=fake_ocr)
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert fake_ocr.last_image_shape == (4, 11, 3)
    assert result.matches[0].diagnostic.rectification_applied is False
    assert result.matches[0].diagnostic.rectification_reason == "center_offset"
    assert result.matches[0].diagnostic.rectification_mode == "safe"


def test_run_frame_safe_rectification_uses_rectified_image_when_applied(monkeypatch) -> None:
    config = make_config()
    config.ocr.enable_rectification = True
    config.ocr.rectification_mode = "safe"

    class FakeRectifyResult:
        def __init__(self):
            self.image = np.zeros((48, 168, 3), dtype=np.uint8)
            self.applied = True
            self.reason = "applied"

    monkeypatch.setattr("car_system.pipeline.runner.safe_rectify_plate", lambda *args, **kwargs: FakeRectifyResult())

    fake_ocr = FakeOCR()
    runner = PipelineRunner(config=config, vehicle_detector=EdgeVehicleDetector(), plate_detector=EdgePlateDetector(), ocr_engine=fake_ocr)
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = runner.run_frame(image=image, source_name="edge.jpg", frame_index=0)

    assert fake_ocr.last_image_shape == (48, 168, 3)
    assert result.matches[0].diagnostic.rectification_applied is True
    assert result.matches[0].diagnostic.rectification_reason == "applied"
```

Add this test to `D:/Projects/Car/tests/test_diagnostic_export.py`:

```python
def test_export_frame_diagnostics_writes_rectification_metadata(tmp_path) -> None:
    plate = Detection(label="plate", confidence=0.96, bbox=(40, 90, 95, 115))
    diagnostic = PlateDiagnostic(
        status="recognized",
        crop_bbox=(36, 87, 99, 118),
        rectified_shape=(48, 168, 3),
        rectification_mode="safe",
        rectification_applied=True,
        rectification_reason="applied",
    )
    result = FrameResult(source_name="sample.jpg", frame_index=0, detections=[plate], matches=[PlateMatch(plate=plate, vehicle=None, diagnostic=diagnostic)])

    crop = np.zeros((31, 63, 3), dtype=np.uint8)
    rectified = np.ones((48, 168, 3), dtype=np.uint8)
    payload = export_frame_diagnostics(tmp_path, result, np.zeros((160, 160, 3), dtype=np.uint8), [crop], [rectified])

    assert payload["diagnostics"][0]["rectification_mode"] == "safe"
    assert payload["diagnostics"][0]["rectification_applied"] is True
    assert payload["diagnostics"][0]["rectification_reason"] == "applied"
```

- [ ] **Step 2: Run the targeted runner/export tests and verify they fail**

Run:

```powershell
python -m pytest tests/test_runner.py tests/test_diagnostic_export.py -q
```

Expected: FAIL because the runner and diagnostics export do not yet understand safe-rectification metadata.

- [ ] **Step 3: Implement the runner fallback and export plumbing**

In `D:/Projects/Car/src/car_system/pipeline/runner.py`, replace the current one-line rectification branch with explicit mode handling:

```python
plain_crop = _crop_bbox(image, crop_bbox)
ocr_input = plain_crop
rectified_shape = None
rectification_applied = False
rectification_reason = None

if self.config.ocr.enable_rectification and self.config.ocr.rectification_mode == "safe":
    rectified = safe_rectify_plate(
        plain_crop,
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
    ocr_input = rectify_plate(plain_crop)
    rectified_shape = getattr(ocr_input, "shape", None)
```

Record the metadata in `PlateDiagnostic`:

```python
match.diagnostic = PlateDiagnostic(
    status=_diagnostic_status(...),
    crop_bbox=crop_bbox,
    crop_shape=getattr(plain_crop, "shape", None),
    rectified_shape=rectified_shape,
    confidence=raw_recognition.confidence if raw_recognition else None,
    raw_text=raw_recognition.raw_text if raw_recognition else None,
    normalized_text=normalized_text,
    rectification_mode=self.config.ocr.rectification_mode if self.config.ocr.enable_rectification else "disabled",
    rectification_applied=rectification_applied,
    rectification_reason=rectification_reason,
)
```

In `D:/Projects/Car/src/car_system/diagnostics/export.py`, persist the new keys:

```python
diagnostics.append(
    {
        "match_index": index,
        "status": match.diagnostic.status if match.diagnostic else "missing",
        "crop_path": str(crop_path),
        "rectified_path": str(rectified_path),
        "confidence": match.diagnostic.confidence if match.diagnostic else None,
        "raw_text": match.diagnostic.raw_text if match.diagnostic else None,
        "normalized_text": match.diagnostic.normalized_text if match.diagnostic else None,
        "rectification_mode": match.diagnostic.rectification_mode if match.diagnostic else None,
        "rectification_applied": match.diagnostic.rectification_applied if match.diagnostic else False,
        "rectification_reason": match.diagnostic.rectification_reason if match.diagnostic else None,
    }
)
```

- [ ] **Step 4: Re-run the targeted runner/export tests and then the focused suite**

Run:

```powershell
python -m pytest tests/test_runner.py tests/test_diagnostic_export.py -q
python -m pytest tests/test_config.py tests/test_runtime.py tests/test_plate_ocr.py tests/test_runner.py tests/test_rectify.py tests/test_writers.py tests/test_review_set.py tests/test_diagnostic_export.py tests/test_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify git status and record that this workspace is not a repo**

Run:

```powershell
git rev-parse --show-toplevel
```

Expected: FAIL with `fatal: not a git repository`. Do not attempt a commit in this workspace.

### Task 4: Add A/B Config, Audit, And Documentation Updates

**Files:**
- Modify: `D:/Projects/Car/configs/plate_ocr_specialized.local.yaml`
- Modify: `D:/Projects/Car/configs/remote_mvp_specialized.yaml`
- Modify: `D:/Projects/Car/docs/ocr-root-cause-audit-2026-04-19.md`
- Modify: `D:/Projects/Car/docs/internal-analysis-prototype-runbook.md`

- [ ] **Step 1: Write the failing config assertions for the new safe-rectification mode**

Extend `D:/Projects/Car/tests/test_config.py`:

```python
def test_local_specialized_config_keeps_safe_rectification_disabled_by_default() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "plate_ocr_specialized.local.yaml")

    assert config.ocr.rectification_mode == "disabled"
    assert config.ocr.enable_rectification is False
```

- [ ] **Step 2: Run the config test and verify the current baseline still passes**

Run:

```powershell
python -m pytest tests/test_config.py::test_local_specialized_config_keeps_safe_rectification_disabled_by_default -q
```

Expected: PASS once Task 1 is complete. This locks in the control path before adding any A/B config.

- [ ] **Step 3: Add an explicit A/B config variant and document the audit commands**

Keep the existing defaults in:

```yaml
# D:/Projects/Car/configs/plate_ocr_specialized.local.yaml
enable_rectification: false
rectification_mode: disabled
```

Keep the existing defaults in:

```yaml
# D:/Projects/Car/configs/remote_mvp_specialized.yaml
enable_rectification: false
rectification_mode: disabled
```

Create or document an audit-only override by duplicating the remote config into a temporary A/B file during implementation, then run:

```powershell
/root/miniconda3/bin/python scripts/run_internal_review_set.py --config configs/remote_mvp_specialized_safe.yaml --review-set configs/review_sets/internal_analysis_v1.yaml --output-dir outputs/internal_analysis_v1_safe
```

and:

```powershell
/root/miniconda3/bin/python scripts/audit_specialized_ocr_confidence.py --config configs/remote_mvp_specialized_safe.yaml --sample-count 500 --output-dir outputs/ocr_confidence_audit_plaincrop_500_safe
```

Document the comparison rule in `D:/Projects/Car/docs/ocr-root-cause-audit-2026-04-19.md`:

```markdown
- only promote `safe_rectification` if 500-sample `wrong` drops below `33`
- reject promotion if fixed-review `blur` or `fn` loses its exact match
```

Update the runbook with the local report command:

```powershell
python scripts/build_internal_analysis_report.py --input-csv outputs/internal_analysis_v1_safe/review_results.csv --output-html outputs/internal_analysis_v1_safe/report.html
```

- [ ] **Step 4: Run the focused test suite and verify no regression before the remote audit**

Run:

```powershell
python -m pytest tests/test_config.py tests/test_runtime.py tests/test_plate_ocr.py tests/test_runner.py tests/test_rectify.py tests/test_writers.py tests/test_review_set.py tests/test_diagnostic_export.py tests/test_reporting.py -q
```

Expected: PASS.

- [ ] **Step 5: Verify git status and record that this workspace is not a repo**

Run:

```powershell
git rev-parse --show-toplevel
```

Expected: FAIL with `fatal: not a git repository`. Do not attempt a commit in this workspace.

## Self-Review

### Spec coverage

- Goal and fallback-first behavior: covered by Tasks 2 and 3.
- Configurable A/B rollout: covered by Tasks 1 and 4.
- Safe-rectification-only-for-specialized flow: covered by Tasks 1 and 3.
- Diagnostic visibility for applied vs fallback: covered by Task 3.
- Acceptance gates (`wrong < 33`, fixed-review exact retention): covered by Task 4.

### Placeholder scan

- No placeholder markers or defer-later notes remain.
- Each code-changing step includes explicit code snippets, file paths, and commands.

### Type consistency

- `rectification_mode`, `rectification_applied`, and `rectification_reason` are introduced in Task 1 and reused consistently in Tasks 3 and 4.
- `safe_rectify_plate()` returns a structured `RectifyResult` in Task 2 and is consumed with the same fields in Task 3.
