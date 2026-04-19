from pathlib import Path

import numpy as np

from car_system.diagnostics.export import build_match_artifacts, export_frame_diagnostics
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
        confidence=0.89,
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
    assert payload["diagnostics"][0]["confidence"] == 0.89
    assert payload["diagnostics"][0]["raw_text"] == "皖A1234O"
    assert payload["diagnostics"][0]["normalized_text"] == "皖A12340"


def test_build_match_artifacts_uses_diagnostic_crop_bbox() -> None:
    vehicle = Detection(label="car", confidence=0.91, bbox=(10, 10, 120, 120))
    plate = Detection(label="plate", confidence=0.96, bbox=(40, 90, 95, 115))
    result = FrameResult(
        source_name="sample.jpg",
        frame_index=0,
        detections=[vehicle, plate],
        matches=[
            PlateMatch(
                plate=plate,
                vehicle=vehicle,
                diagnostic=PlateDiagnostic(
                    status="recognized",
                    crop_bbox=(36, 87, 99, 118),
                    rectified_shape=(48, 168, 3),
                ),
            )
        ],
    )
    source = np.zeros((160, 160, 3), dtype=np.uint8)
    source[87:118, 36:99] = 255

    crops, rectified_images = build_match_artifacts(result, source)

    assert len(crops) == 1
    assert len(rectified_images) == 1
    assert crops[0].shape == (31, 63, 3)
    assert rectified_images[0].shape == (48, 168, 3)


def test_build_match_artifacts_preserves_crop_when_rectification_was_disabled() -> None:
    plate = Detection(label="plate", confidence=0.96, bbox=(40, 90, 95, 115))
    result = FrameResult(
        source_name="sample.jpg",
        frame_index=0,
        detections=[plate],
        matches=[
            PlateMatch(
                plate=plate,
                vehicle=None,
                diagnostic=PlateDiagnostic(status="recognized", crop_bbox=(40, 90, 95, 115), rectified_shape=None),
            )
        ],
    )
    source = np.zeros((160, 160, 3), dtype=np.uint8)
    source[90:115, 40:95] = 255

    crops, rectified_images = build_match_artifacts(result, source)

    assert crops[0].shape == (25, 55, 3)
    assert rectified_images[0].shape == (25, 55, 3)


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
    result = FrameResult(
        source_name="sample.jpg",
        frame_index=0,
        detections=[plate],
        matches=[PlateMatch(plate=plate, vehicle=None, diagnostic=diagnostic)],
    )

    crop = np.zeros((31, 63, 3), dtype=np.uint8)
    rectified = np.ones((48, 168, 3), dtype=np.uint8)
    payload = export_frame_diagnostics(
        output_dir=tmp_path,
        frame_result=result,
        source_image=np.zeros((160, 160, 3), dtype=np.uint8),
        crops=[crop],
        rectified_images=[rectified],
    )

    assert payload["diagnostics"][0]["rectification_mode"] == "safe"
    assert payload["diagnostics"][0]["rectification_applied"] is True
    assert payload["diagnostics"][0]["rectification_reason"] == "applied"
