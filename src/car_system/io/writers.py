import csv
import json
from pathlib import Path

from car_system.types import FrameResult


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def frame_results_to_rows(results: list[FrameResult]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.extend(frame_result_to_rows(result))
    return rows


def frame_result_to_dict(result: FrameResult) -> dict[str, object]:
    return {
        "source_name": result.source_name,
        "frame_index": result.frame_index,
        "detections": [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "bbox": list(detection.bbox),
            }
            for detection in result.detections
        ],
        "matches": [
            {
                "vehicle": (
                    {
                        "label": match.vehicle.label,
                        "confidence": match.vehicle.confidence,
                        "bbox": list(match.vehicle.bbox),
                    }
                    if match.vehicle
                    else None
                ),
                "plate": {
                    "label": match.plate.label,
                    "confidence": match.plate.confidence,
                    "bbox": list(match.plate.bbox),
                },
                "recognition": (
                    {
                        "text": match.recognition.text,
                        "confidence": match.recognition.confidence,
                        "raw_text": match.recognition.raw_text,
                        "normalized_text": match.recognition.normalized_text,
                    }
                    if match.recognition
                    else None
                ),
                "diagnostic": (
                    {
                        "status": match.diagnostic.status,
                        "crop_bbox": list(match.diagnostic.crop_bbox),
                        "crop_shape": list(match.diagnostic.crop_shape) if match.diagnostic.crop_shape else None,
                        "rectified_shape": list(match.diagnostic.rectified_shape)
                        if match.diagnostic.rectified_shape
                        else None,
                        "raw_text": match.diagnostic.raw_text,
                        "normalized_text": match.diagnostic.normalized_text,
                        "notes": list(match.diagnostic.notes),
                    }
                    if match.diagnostic
                    else None
                ),
            }
            for match in result.matches
        ],
    }


def frame_results_to_dict(results: list[FrameResult]) -> list[dict[str, object]]:
    return [frame_result_to_dict(result) for result in results]


def write_json(path: str | Path, payload: object) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output_path = Path(path)
    fieldnames = list(rows[0].keys()) if rows else []
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return output_path
