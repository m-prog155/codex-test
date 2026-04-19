# Vehicle And License Plate Detection System

## Overview

This project implements a PC-side graduation design system for:

- vehicle detection and type classification
- license plate detection
- license plate OCR recognition
- image and video result visualization
- result export and basic experiment summary

The system does not include RK3588 deployment, RKNN conversion, NPU acceleration, C++ rewriting, or edge-side optimization.

## Architecture

The project now uses a dual-detector pipeline:

1. vehicle detector
2. plate detector
3. OCR recognizer
4. result matching, visualization, and export

This is intentionally more stable than a single joint detector because it reduces data dependency and makes training and debugging easier.

## Project Structure

```text
configs/                  runtime config
docs/                     design, plan, and thesis notes
scripts/                  CLI entry points
src/car_system/           core package
src/car_system/detectors/ detector backends
src/car_system/ocr/       OCR backends and optional rectification
src/car_system/pipeline/  matching and processing orchestration
src/car_system/io/        media loading, rendering, and export
src/car_system/experiments/ experiment summary helpers
tests/                    unit and integration tests
```

## Current Capabilities

- image pipeline
- video pipeline
- annotated image export
- annotated video export
- JSON result export
- CSV result export
- experiment summary JSON
- file-level summary CSV
- Streamlit demo entry

## Configuration

Default config file: `configs/default.yaml`

The system expects separate weight paths for:

- `vehicle_detector.model_path`
- `plate_detector.model_path`

Recommended first setup:

- vehicle detector: a YOLO model trained or fine-tuned on vehicle categories
- plate detector: a YOLO model trained or fine-tuned on license plate boxes
- OCR: PaddleOCR

## Local Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Run image inference:

```bash
python scripts/run_image.py --source path/to/image.jpg
```

Run video inference:

```bash
python scripts/run_video.py --source path/to/video.mp4
```

Generate summary JSON from exported CSV files:

```bash
python scripts/evaluate_dataset.py --input outputs
```

Generate per-file summary CSV:

```bash
python scripts/summarize_results.py --input outputs
```

Generate training comparison CSV from Ultralytics run directories:

```bash
python scripts/summarize_training_runs.py --input runs/plate_detector
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Remote Runtime Notes

Verified runtime:

- Ubuntu 22.04
- Python 3.12
- PyTorch 2.8.0 + CUDA 12.8
- RTX 5090

Remote project path currently used during development:

```text
/root/autodl-tmp/car-project
```

## Known Gaps

- the repository does not yet contain actual trained detector weights
- OCR is integrated, but final plate recognition quality depends on real model weights and sample data
- evaluation currently summarizes exported inference results; formal benchmark metrics still depend on the final datasets

## Recommendation For Thesis Progress

Use the current codebase as the stable engineering baseline, then complete the thesis in this order:

1. prepare dual-model weights
2. run image and video examples
3. export CSV and JSON results
4. run experiment summary scripts
5. capture screenshots, tables, and sample outputs for the thesis
