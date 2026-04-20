from __future__ import annotations

from collections import Counter
from pathlib import Path
import shutil
from statistics import mean, median
from typing import Any

from car_system.types import PlateMatch


def pick_best_recognized_match(matches: list[PlateMatch]) -> PlateMatch | None:
    recognized = [match for match in matches if match.recognition is not None and match.recognition.text]
    if not recognized:
        return None

    recognized.sort(
        key=lambda match: (
            getattr(match.recognition, "confidence", 0.0),
            getattr(match.plate, "confidence", 0.0),
        ),
        reverse=True,
    )
    return recognized[0]


def _subset_name(relative_path: str | Path) -> str:
    path = Path(relative_path)
    if not path.parts:
        return ""
    if len(path.parts) == 1:
        return path.parent.as_posix()
    return path.parts[0]


def build_sample_audit_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    summary: dict[str, object] = {"sample_count": len(rows)}

    for status in ["exact", "wrong", "null"]:
        selected = [row for row in rows if row.get("status") == status]
        confidences = [
            float(row["confidence"])
            for row in selected
            if row.get("confidence") not in (None, "")
        ]
        summary[status] = {
            "count": len(selected),
            "rate": (len(selected) / len(rows)) if rows else 0.0,
            "mean_confidence": mean(confidences) if confidences else None,
            "median_confidence": median(confidences) if confidences else None,
            "min_confidence": min(confidences) if confidences else None,
            "max_confidence": max(confidences) if confidences else None,
        }

    return summary


def build_sample_audit_row(
    *,
    relative_path: str | Path,
    gt_text: str,
    best_match: PlateMatch | None,
) -> dict[str, Any]:
    relative_path_obj = Path(relative_path)
    if best_match is None or best_match.recognition is None or not best_match.recognition.text:
        return {
            "relative_path": relative_path_obj.as_posix(),
            "subset": _subset_name(relative_path_obj),
            "gt_text": gt_text,
            "predicted_text": None,
            "status": "null",
            "confidence": None,
            "raw_text": None,
            "normalized_text": None,
            "diagnostic_status": best_match.diagnostic.status if best_match and best_match.diagnostic else None,
            "rectification_mode": best_match.diagnostic.rectification_mode if best_match and best_match.diagnostic else None,
            "rectification_applied": (
                best_match.diagnostic.rectification_applied if best_match and best_match.diagnostic else False
            ),
            "rectification_reason": (
                best_match.diagnostic.rectification_reason if best_match and best_match.diagnostic else None
            ),
        }

    predicted_text = best_match.recognition.text
    diagnostic = best_match.diagnostic
    return {
        "relative_path": relative_path_obj.as_posix(),
        "subset": _subset_name(relative_path_obj),
        "gt_text": gt_text,
        "predicted_text": predicted_text,
        "status": "exact" if predicted_text == gt_text else "wrong",
        "confidence": best_match.recognition.confidence,
        "raw_text": best_match.recognition.raw_text,
        "normalized_text": best_match.recognition.normalized_text,
        "diagnostic_status": diagnostic.status if diagnostic else None,
        "rectification_mode": diagnostic.rectification_mode if diagnostic else None,
        "rectification_applied": diagnostic.rectification_applied if diagnostic else False,
        "rectification_reason": diagnostic.rectification_reason if diagnostic else None,
    }


def filter_audit_rows(
    rows: list[dict[str, Any]],
    *,
    statuses: list[str] | tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    allowed_statuses = set(statuses) if statuses else None
    selected = [
        row for row in rows if allowed_statuses is None or str(row.get("status", "")).strip() in allowed_statuses
    ]
    if limit is not None:
        return selected[:limit]
    return selected


def build_sample_path_list(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(relative_path).strip()
        for row in rows
        if (relative_path := row.get("relative_path"))
    ]


def copy_audit_sample_images(
    *,
    rows: list[dict[str, Any]],
    dataset_root: str | Path,
    export_root: str | Path,
) -> list[Path]:
    dataset_root_path = Path(dataset_root)
    export_root_path = Path(export_root)
    written: list[Path] = []

    for relative_path in build_sample_path_list(rows):
        source_path = dataset_root_path / Path(relative_path)
        destination_path = export_root_path / "images" / Path(relative_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        written.append(destination_path)

    return written


def build_hard_case_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    status_counts = Counter(str(row.get("status", "")).strip() for row in rows if row.get("status"))

    prefix_transitions = Counter()
    mismatch_positions = Counter()
    character_confusions = Counter()
    length_pairs = Counter()

    for row in rows:
        gt_text = str(row.get("gt_text") or "")
        predicted_text = str(row.get("predicted_text") or "")
        if not gt_text or not predicted_text:
            continue

        length_pairs[(len(gt_text), len(predicted_text))] += 1

        if row.get("status") != "wrong":
            continue

        prefix_transitions[(gt_text[:2], predicted_text[:2])] += 1
        for position, (gt_char, predicted_char) in enumerate(zip(gt_text, predicted_text)):
            if gt_char == predicted_char:
                continue
            mismatch_positions[position] += 1
            character_confusions[(gt_char, predicted_char)] += 1

    return {
        "sample_count": len(rows),
        "status_counts": dict(status_counts),
        "length_pairs": [
            {
                "gt_length": gt_length,
                "pred_length": pred_length,
                "count": count,
            }
            for (gt_length, pred_length), count in sorted(
                length_pairs.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "prefix_transitions": [
            {
                "gt_prefix": gt_prefix,
                "pred_prefix": pred_prefix,
                "count": count,
            }
            for (gt_prefix, pred_prefix), count in sorted(
                prefix_transitions.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "mismatch_positions": [
            {
                "position": position,
                "count": count,
            }
            for position, count in sorted(mismatch_positions.items(), key=lambda item: (-item[1], item[0]))
        ],
        "character_confusions": [
            {
                "gt_char": gt_char,
                "pred_char": pred_char,
                "count": count,
            }
            for (gt_char, pred_char), count in sorted(
                character_confusions.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
    }
