import csv
from pathlib import Path


METRIC_FIELDS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def _load_results_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if str(value).strip() else 0.0


def _to_int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    return int(float(value)) if str(value).strip() else 0


def summarize_training_run(run_dir: str | Path) -> dict[str, object]:
    run_path = Path(run_dir)
    rows = _load_results_rows(run_path / "results.csv")
    if not rows:
        raise ValueError(f"No rows found in {run_path / 'results.csv'}")

    best_row = max(rows, key=lambda row: _to_float(row, METRIC_FIELDS["map50_95"]))
    final_row = rows[-1]

    return {
        "run_name": run_path.name,
        "best_epoch": _to_int(best_row, "epoch"),
        "best_precision": _to_float(best_row, METRIC_FIELDS["precision"]),
        "best_recall": _to_float(best_row, METRIC_FIELDS["recall"]),
        "best_map50": _to_float(best_row, METRIC_FIELDS["map50"]),
        "best_map50_95": _to_float(best_row, METRIC_FIELDS["map50_95"]),
        "final_epoch": _to_int(final_row, "epoch"),
        "final_map50_95": _to_float(final_row, METRIC_FIELDS["map50_95"]),
        "best_weights_path": str((run_path / "weights" / "best.pt").resolve()),
    }


def summarize_training_runs(input_dir: str | Path) -> list[dict[str, object]]:
    base_dir = Path(input_dir)
    run_dirs = sorted(path for path in base_dir.iterdir() if path.is_dir() and (path / "results.csv").exists())
    return [summarize_training_run(path) for path in run_dirs]


def write_training_summaries_csv(path: str | Path, rows: list[dict[str, object]]) -> Path:
    output_path = Path(path)
    fieldnames = [
        "run_name",
        "best_epoch",
        "best_precision",
        "best_recall",
        "best_map50",
        "best_map50_95",
        "final_epoch",
        "final_map50_95",
        "best_weights_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return output_path
