# Vehicle And License Plate Detection System

## Overview

This repository contains the code for a PC-side vehicle and license plate analysis pipeline built around:

- vehicle detection
- license plate detection
- license plate OCR
- image and video annotation
- structured result export
- lightweight experiment and review utilities

The current implementation uses a dual-detector architecture:

1. vehicle detector
2. plate detector
3. OCR recognizer
4. matching, rendering, and export

This repository is intentionally code-only. Datasets, trained weights, generated outputs, and thesis materials are kept out of Git.

## Repository Layout

```text
assets/                   small code-related assets
configs/                  runtime and experiment configs
scripts/                  training, evaluation, and inference entry points
src/car_system/           core package
tests/                    unit and integration tests
app.py                    Streamlit demo entry
requirements.txt          local dependency baseline
```

## What Is Not In Git

The following are expected to live outside the repository:

- training and evaluation datasets
- trained YOLO and PaddleOCR weights
- generated outputs and review artifacts
- local experiment caches and temporary files

The default `.gitignore` already excludes the main large or generated directories used during development.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full test suite:

```bash
python -m pytest tests -q
```

Run image inference:

```bash
python scripts/run_image.py --source path/to/image.jpg
```

Run video inference:

```bash
python scripts/run_video.py --source path/to/video.mp4
```

Run the Streamlit demo:

```bash
streamlit run app.py
```

## Configuration

Default config file:

```text
configs/default.yaml
```

The pipeline expects external model paths for:

- `vehicle_detector.model_path`
- `plate_detector.model_path`
- `ocr.model_dir` when using specialized OCR mode

Recommended setup:

- vehicle detector: Ultralytics YOLO model trained on vehicle categories
- plate detector: Ultralytics YOLO model trained on license plate boxes
- OCR: PaddleOCR or a specialized PaddleOCR recognition model

## Useful Scripts

- `scripts/prepare_ccpd_dataset.py`: build YOLO-style detection data from CCPD
- `scripts/prepare_ccpd_ocr_dataset.py`: build OCR-ready plate text data
- `scripts/train_plate_detector.py`: train the plate detector
- `scripts/train_plate_ocr.py`: train the OCR recognizer
- `scripts/evaluate_plate_ocr_model.py`: run OCR evaluation
- `scripts/audit_pipeline_sample_list.py`: audit one config on a fixed sample-path list
- `scripts/run_internal_review_set.py`: generate a fixed diagnostic review set
- `scripts/build_internal_analysis_report.py`: build a local HTML review report

## CI

GitHub Actions runs the repository test suite on pushes and pull requests using a lightweight dependency set suitable for the current tests.

## Current Limits

- model weights are not bundled in the repository
- production deployment is out of scope here
- end-to-end quality depends heavily on external training data and model checkpoints
