# OCR Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve OCR stability by expanding plate crops, adding a few lightweight OCR preprocessing candidates, and selecting more plausible recognition results.

**Architecture:** Keep the current PaddleOCR `TextRecognition` backend and strengthen the data around it. The runner will own deterministic crop expansion, the OCR module will own candidate generation and result ranking, and tests will stay at unit level with fake OCR objects.

**Tech Stack:** Python, NumPy, OpenCV, PaddleOCR adapter, pytest

---

## File Structure

- Modify: `D:\Projects\Car\src\car_system\pipeline\runner.py`
- Modify: `D:\Projects\Car\src\car_system\ocr\plate_ocr.py`
- Modify: `D:\Projects\Car\src\car_system\ocr\rectify.py`
- Modify: `D:\Projects\Car\tests\test_runner.py`
- Modify: `D:\Projects\Car\tests\test_plate_ocr.py`
- Modify: `D:\Projects\Car\tests\test_processing.py`

### Task 1: Add crop expansion in the runner

**Files:**
- Modify: `D:\Projects\Car\src\car_system\pipeline\runner.py`
- Modify: `D:\Projects\Car\tests\test_runner.py`

- [x] Write a failing test that verifies a plate crop expands outward but stays inside the image boundary.
- [x] Run `python -m pytest tests/test_runner.py -q` and verify it fails for the new crop-expansion expectation.
- [x] Implement a small helper in `runner.py` that expands the bounding box before cropping.
- [x] Re-run `python -m pytest tests/test_runner.py -q` and verify it passes.

### Task 2: Add OCR candidate preprocessing and ranking

**Files:**
- Modify: `D:\Projects\Car\src\car_system\ocr\plate_ocr.py`
- Modify: `D:\Projects\Car\src\car_system\ocr\rectify.py`
- Modify: `D:\Projects\Car\tests\test_plate_ocr.py`

- [x] Write a failing test that verifies the OCR adapter picks the most plausible candidate result among multiple OCR attempts.
- [x] Write a failing test that verifies obviously ambiguous characters are normalized conservatively.
- [x] Run `python -m pytest tests/test_plate_ocr.py -q` and verify the new tests fail.
- [x] Implement candidate generation with at most three images: original, enlarged, and contrast-enhanced grayscale.
- [x] Implement result normalization and scoring logic using OCR confidence plus simple mainland-plate plausibility checks.
- [x] Keep the fallback behavior conservative: if no rule confidently improves the text, preserve the original text.
- [x] Re-run `python -m pytest tests/test_plate_ocr.py -q` and verify it passes.

### Task 3: Verify pipeline compatibility

**Files:**
- Modify: `D:\Projects\Car\tests\test_processing.py`

- [x] Extend the processing tests only if needed so they continue to validate image and video flow under the updated OCR path.
- [x] Run `python -m pytest tests/test_processing.py -q` and verify it passes.

### Task 4: Run full local verification

**Files:**
- Verify only

- [x] Run `python -m pytest -q`.
- [x] Confirm all tests pass and there are no regressions in the existing pipeline behavior.
