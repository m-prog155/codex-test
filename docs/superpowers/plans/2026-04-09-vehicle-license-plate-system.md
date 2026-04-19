# Vehicle And License Plate System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python PC-side MVP for vehicle and license plate detection, OCR recognition, visualization, saving, and baseline evaluation.

**Architecture:** The system uses a thin application layer over a detection module, an OCR module, and a pipeline orchestrator. The first deliverable is a local MVP with image and video support, followed by experiment and reporting utilities.

**Tech Stack:** Python 3.11+, Streamlit, Ultralytics YOLO, PaddleOCR, OpenCV, PyYAML, pandas, pytest

---

## File Structure

- Create: `D:\Projects\Car\app.py`
- Create: `D:\Projects\Car\requirements.txt`
- Create: `D:\Projects\Car\README.md`
- Create: `D:\Projects\Car\configs\default.yaml`
- Create: `D:\Projects\Car\src\car_system\__init__.py`
- Create: `D:\Projects\Car\src\car_system\config.py`
- Create: `D:\Projects\Car\src\car_system\types.py`
- Create: `D:\Projects\Car\src\car_system\detectors\__init__.py`
- Create: `D:\Projects\Car\src\car_system\detectors\yolo_detector.py`
- Create: `D:\Projects\Car\src\car_system\ocr\__init__.py`
- Create: `D:\Projects\Car\src\car_system\ocr\plate_ocr.py`
- Create: `D:\Projects\Car\src\car_system\ocr\rectify.py`
- Create: `D:\Projects\Car\src\car_system\pipeline\__init__.py`
- Create: `D:\Projects\Car\src\car_system\pipeline\matcher.py`
- Create: `D:\Projects\Car\src\car_system\pipeline\runner.py`
- Create: `D:\Projects\Car\src\car_system\io\__init__.py`
- Create: `D:\Projects\Car\src\car_system\io\media.py`
- Create: `D:\Projects\Car\src\car_system\io\writers.py`
- Create: `D:\Projects\Car\src\car_system\ui\__init__.py`
- Create: `D:\Projects\Car\src\car_system\ui\view_models.py`
- Create: `D:\Projects\Car\scripts\run_image.py`
- Create: `D:\Projects\Car\scripts\run_video.py`
- Create: `D:\Projects\Car\scripts\evaluate_dataset.py`
- Create: `D:\Projects\Car\scripts\summarize_results.py`
- Create: `D:\Projects\Car\tests\test_config.py`
- Create: `D:\Projects\Car\tests\test_matcher.py`
- Create: `D:\Projects\Car\tests\test_runner.py`
- Create: `D:\Projects\Car\tests\test_writers.py`

### Task 1: Establish project structure and config loading

**Files:**
- Create: `D:\Projects\Car\requirements.txt`
- Create: `D:\Projects\Car\configs\default.yaml`
- Create: `D:\Projects\Car\src\car_system\config.py`
- Create: `D:\Projects\Car\tests\test_config.py`

- [ ] Write a failing config loading test
- [ ] Run the config test and verify it fails because loader code is missing
- [ ] Implement the minimal config loader and default config schema
- [ ] Re-run the config test and verify it passes

### Task 2: Define pipeline data structures and plate matching

**Files:**
- Create: `D:\Projects\Car\src\car_system\types.py`
- Create: `D:\Projects\Car\src\car_system\pipeline\matcher.py`
- Create: `D:\Projects\Car\tests\test_matcher.py`

- [ ] Write a failing matcher test for vehicle and plate association
- [ ] Run the matcher test and verify it fails because the matcher is missing
- [ ] Implement minimal detection dataclasses and nearest-vehicle matching
- [ ] Re-run the matcher test and verify it passes

### Task 3: Add detector and OCR adapters

**Files:**
- Create: `D:\Projects\Car\src\car_system\detectors\yolo_detector.py`
- Create: `D:\Projects\Car\src\car_system\ocr\plate_ocr.py`
- Create: `D:\Projects\Car\src\car_system\ocr\rectify.py`
- Create: `D:\Projects\Car\tests\test_runner.py`

- [ ] Write a failing pipeline runner test using fake detector and OCR outputs
- [ ] Run the runner test and verify it fails because the runner is missing
- [ ] Implement detector and OCR adapter interfaces with optional plate rectification
- [ ] Implement the pipeline runner with dependency injection for testability
- [ ] Re-run the runner test and verify it passes

### Task 4: Add image and video entry points

**Files:**
- Create: `D:\Projects\Car\src\car_system\io\media.py`
- Create: `D:\Projects\Car\src\car_system\io\writers.py`
- Create: `D:\Projects\Car\scripts\run_image.py`
- Create: `D:\Projects\Car\scripts\run_video.py`
- Create: `D:\Projects\Car\tests\test_writers.py`

- [ ] Write a failing writer test for JSON and CSV result export
- [ ] Run the writer test and verify it fails because the writer is missing
- [ ] Implement output writers and media helpers
- [ ] Implement minimal CLI scripts for image and video execution
- [ ] Re-run the writer test and verify it passes

### Task 5: Add the Streamlit MVP

**Files:**
- Create: `D:\Projects\Car\app.py`
- Create: `D:\Projects\Car\src\car_system\ui\view_models.py`

- [ ] Build a minimal Streamlit page for image and video upload
- [ ] Render annotated outputs and structured tables
- [ ] Allow saving outputs to a configurable directory

### Task 6: Add experiment scripts and documentation

**Files:**
- Create: `D:\Projects\Car\scripts\evaluate_dataset.py`
- Create: `D:\Projects\Car\scripts\summarize_results.py`
- Create: `D:\Projects\Car\README.md`

- [ ] Implement evaluation script entry points for dataset inference logs
- [ ] Implement result aggregation and summary export
- [ ] Document local setup, remote setup, and MVP usage in the README
- [ ] Add a short implementation summary suitable for the thesis implementation chapter
