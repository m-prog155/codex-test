import csv
import json
from collections import Counter
from pathlib import Path


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _result_csv_files(input_dir: str | Path) -> list[Path]:
    directory = Path(input_dir)
    return sorted(
        path
        for path in directory.glob("*.csv")
        if path.name not in {"summary.csv", "summary.json", "file_summaries.csv"}
    )


def _compute_summary(rows: list[dict[str, str]], file_name: str | None = None) -> dict[str, object]:
    total = len(rows)
    recognized_rows = [row for row in rows if row.get("plate_text", "").strip()]
    ocr_scores = [float(row["ocr_confidence"]) for row in recognized_rows if str(row.get("ocr_confidence", "")).strip()]

    vehicle_counts = Counter()
    for row in rows:
        label = row.get("vehicle_label", "").strip()
        if label:
            vehicle_counts[label] += 1

    summary: dict[str, object] = {
        "plate_detection_count": total,
        "recognized_plate_count": len(recognized_rows),
        "recognition_rate": (len(recognized_rows) / total) if total else 0.0,
        "average_ocr_confidence": round(sum(ocr_scores) / len(ocr_scores), 4) if ocr_scores else 0.0,
        "vehicle_counts": dict(vehicle_counts),
    }
    if file_name is not None:
        summary["file_name"] = file_name
    return summary


def build_directory_summary(input_dir: str | Path) -> dict[str, object]:
    files = _result_csv_files(input_dir)
    all_rows: list[dict[str, str]] = []
    for path in files:
        all_rows.extend(_load_csv_rows(path))

    summary = _compute_summary(all_rows)
    summary["file_count"] = len(files)
    return summary


def build_file_summaries(input_dir: str | Path) -> list[dict[str, object]]:
    files = _result_csv_files(input_dir)
    return [_compute_summary(_load_csv_rows(path), file_name=path.name) for path in files]


def write_summary_json(path: str | Path, payload: dict[str, object]) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_file_summaries_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output_path = Path(path)
    fieldnames = ["file_name", "plate_detection_count", "recognized_plate_count", "recognition_rate", "average_ocr_confidence"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path
