from pathlib import Path
from typing import Any

from car_system.io.media import save_image
from car_system.io.writers import write_json
from car_system.ocr.rectify import rectify_plate, safe_rectify_plate
from car_system.types import FrameResult


def build_match_artifacts(frame_result: FrameResult, source_image: Any) -> tuple[list[Any], list[Any]]:
    crops: list[Any] = []
    rectified_images: list[Any] = []
    for match in frame_result.matches:
        if match.diagnostic is None:
            raise ValueError("match diagnostic is required to rebuild crop artifacts")
        x1, y1, x2, y2 = match.diagnostic.crop_bbox
        crop = source_image[y1:y2, x1:x2]
        rectified = crop
        if match.diagnostic.rectification_applied:
            if match.diagnostic.rectification_mode == "safe":
                rectified = safe_rectify_plate(crop).image
            elif match.diagnostic.rectified_shape is not None:
                rectified = rectify_plate(crop)
        elif match.diagnostic.rectified_shape is not None:
            rectified = rectify_plate(crop)
        crops.append(crop)
        rectified_images.append(rectified)
    return crops, rectified_images


def export_frame_diagnostics(
    output_dir: str | Path,
    frame_result: FrameResult,
    source_image: Any,
    crops: list[Any],
    rectified_images: list[Any],
) -> dict[str, Any]:
    del source_image
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if len(crops) != len(frame_result.matches) or len(rectified_images) != len(frame_result.matches):
        raise ValueError("crop and rectified image counts must match frame_result.matches")

    diagnostics: list[dict[str, Any]] = []
    stem = Path(frame_result.source_name).stem
    for index, match in enumerate(frame_result.matches):
        crop_path = target_dir / f"{stem}_match_{index}_crop.jpg"
        rectified_path = target_dir / f"{stem}_match_{index}_rectified.jpg"
        save_image(crop_path, crops[index])
        save_image(rectified_path, rectified_images[index])
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

    payload = {
        "source_name": frame_result.source_name,
        "frame_index": frame_result.frame_index,
        "diagnostics": diagnostics,
    }
    write_json(target_dir / f"{stem}_diagnostics.json", payload)
    return payload
