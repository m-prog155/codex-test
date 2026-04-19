# Vehicle And License Plate System Design

**Date:** 2026-04-09

## Goal

Build a PC-only deep learning system that supports image and video input, detects vehicle targets, classifies vehicle types, detects license plate regions, recognizes plate text, visualizes results, saves outputs, and provides basic experiment evaluation.

## Scope

### Included

- Python implementation only
- PC-side inference, evaluation, and demo UI
- Vehicle detection and coarse vehicle type classification
- License plate detection
- License plate OCR recognition
- Result visualization and export
- Basic experiment scripts and summary statistics

### Excluded

- RK3588 deployment
- RKNN conversion
- NPU acceleration
- C++ inference rewrite
- Edge-side multithread optimization
- Complex hardware adaptation

## Design Principles

- Prefer mature, reproducible libraries over custom model changes
- Keep the MVP small enough to demo early
- Separate detection, OCR, application, and experiment logic into focused modules
- Make optional enhancements additive, not foundational

## Recommended Technical Route

### Detection

- Use YOLO family models for object detection
- Use two separate detectors: one for vehicles and one for license plates
- Vehicle type is determined by configured vehicle detection labels such as `car`, `bus`, `truck`, `van`, and `motorcycle`

### Recognition

- Use PaddleOCR first for license plate text recognition
- Keep LPRNet as a later alternative only if OCR accuracy is clearly insufficient

### Optional Optimization

- Add plate perspective correction as an optional preprocessing step before OCR
- Keep image enhancement outside the MVP unless it directly improves OCR in a measurable way

## System Architecture

### Core Flow

1. Load image or video frame
2. Run the vehicle detector
3. Run the plate detector
4. Match each plate to the nearest vehicle when possible
5. Crop plate image
6. Optionally rectify the crop
7. Run OCR on the crop
8. Render annotated results
9. Save structured results and optional rendered media

### Module Boundaries

- `config`: runtime and model configuration
- `detectors`: model loading and detection inference
- `ocr`: plate OCR and optional rectification
- `pipeline`: orchestration and result merging
- `io`: image/video loading and output saving
- `ui`: Streamlit-based MVP demo
- `experiments`: evaluation and statistics scripts

## Data Strategy

- Use public datasets and pre-trained weights first
- Keep dataset preparation scripts lightweight and repeatable
- Support local sample folders for quick demo images and videos

## Error Handling

- Fail clearly when weights or model paths are missing
- Skip unreadable frames instead of crashing the whole run
- Preserve partial results for long video jobs
- Return empty predictions when no objects are found

## Testing Strategy

- Unit tests for config loading, prediction parsing, plate-to-vehicle matching, and result serialization
- Small integration tests for image pipeline behavior with mocked model outputs
- Keep UI tests minimal and focus on pipeline correctness

## MVP Definition

The MVP is successful when a user can select an image or video, run the pipeline, see annotated vehicle and plate results, read recognized plate text, and save structured outputs without any edge deployment work.

## MVP Feature Breakdown

### Must Have In The First Working Version

- Load one image file and run the full pipeline
- Load one video file and process frames sequentially
- Detect vehicles and license plates with separate detector outputs
- Recognize plate text from cropped plate regions with PaddleOCR
- Display annotated results with bounding boxes and labels
- Save structured inference results as JSON and CSV
- Save annotated image outputs and optional processed video outputs
- Provide a simple Streamlit interface for image and video demo

### Should Have After The First End-To-End Demo

- Associate each plate with the nearest vehicle detection
- Save run metadata such as model name, thresholds, and timestamps
- Add a command-line entry point for batch image or video execution
- Add a small evaluation script for dataset-level inference summaries

### Optional Only If Time Remains

- Perspective rectification before OCR
- OCR confidence filtering and post-processing cleanup
- Side-by-side comparison views for different models or thresholds
- Aggregate experiment charts for thesis figures

### Explicitly Deferred

- Live camera streaming
- User accounts or database-backed history
- Complex front-end interaction
- Real-time multi-thread acceleration
- Training UI
- Any deployment outside a normal PC environment
