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


def build_review_rows(
    category: str,
    source_name: str,
    gt_text: str,
    diagnostics: list[dict[str, str | None]],
) -> list[dict[str, str]]:
    if not diagnostics:
        return [
            {
                "category": category,
                "source_name": source_name,
                "gt_text": gt_text,
                "diagnostic_status": "plate_missed",
                "ocr_confidence": "",
                "ocr_raw_text": "",
                "ocr_normalized_text": "",
                "crop_path": "",
                "rectified_path": "",
                "rectification_mode": "",
                "rectification_applied": "",
                "rectification_reason": "",
            }
        ]

    rows: list[dict[str, str]] = []
    for diagnostic in diagnostics:
        rectification_applied = diagnostic.get("rectification_applied", "")
        if rectification_applied in (None, ""):
            rectification_applied_text = ""
        else:
            rectification_applied_text = str(rectification_applied)
        rows.append(
            {
                "category": category,
                "source_name": source_name,
                "gt_text": gt_text,
                "diagnostic_status": diagnostic.get("status", "") or "",
                "ocr_confidence": str(diagnostic.get("confidence", "") or ""),
                "ocr_raw_text": diagnostic.get("raw_text", "") or "",
                "ocr_normalized_text": diagnostic.get("normalized_text", "") or "",
                "crop_path": diagnostic.get("crop_path", "") or "",
                "rectified_path": diagnostic.get("rectified_path", "") or "",
                "rectification_mode": diagnostic.get("rectification_mode", "") or "",
                "rectification_applied": rectification_applied_text,
                "rectification_reason": diagnostic.get("rectification_reason", "") or "",
            }
        )
    return rows


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
