from __future__ import annotations

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
