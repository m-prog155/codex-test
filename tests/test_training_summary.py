import csv
from pathlib import Path

from car_system.experiments.training_summary import (
    summarize_training_runs,
    write_training_summaries_csv,
)


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_training_runs_extracts_best_and_final_metrics(tmp_path) -> None:
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    (run_a / "weights").mkdir()
    (run_a / "weights" / "best.pt").write_text("best", encoding="utf-8")
    (run_a / "weights" / "last.pt").write_text("last", encoding="utf-8")
    write_results_csv(
        run_a / "results.csv",
        [
            {
                "epoch": 1,
                "metrics/precision(B)": 0.91,
                "metrics/recall(B)": 0.90,
                "metrics/mAP50(B)": 0.97,
                "metrics/mAP50-95(B)": 0.61,
            },
            {
                "epoch": 2,
                "metrics/precision(B)": 0.95,
                "metrics/recall(B)": 0.94,
                "metrics/mAP50(B)": 0.98,
                "metrics/mAP50-95(B)": 0.75,
            },
            {
                "epoch": 3,
                "metrics/precision(B)": 0.94,
                "metrics/recall(B)": 0.93,
                "metrics/mAP50(B)": 0.975,
                "metrics/mAP50-95(B)": 0.71,
            },
        ],
    )

    run_b = tmp_path / "run_b"
    run_b.mkdir()
    (run_b / "weights").mkdir()
    (run_b / "weights" / "best.pt").write_text("best", encoding="utf-8")
    write_results_csv(
        run_b / "results.csv",
        [
            {
                "epoch": 1,
                "metrics/precision(B)": 0.89,
                "metrics/recall(B)": 0.88,
                "metrics/mAP50(B)": 0.95,
                "metrics/mAP50-95(B)": 0.55,
            },
            {
                "epoch": 2,
                "metrics/precision(B)": 0.92,
                "metrics/recall(B)": 0.91,
                "metrics/mAP50(B)": 0.965,
                "metrics/mAP50-95(B)": 0.68,
            },
        ],
    )

    summaries = summarize_training_runs(tmp_path)

    assert [summary["run_name"] for summary in summaries] == ["run_a", "run_b"]
    assert summaries[0]["best_epoch"] == 2
    assert summaries[0]["best_map50_95"] == 0.75
    assert summaries[0]["final_epoch"] == 3
    assert summaries[0]["final_map50_95"] == 0.71
    assert summaries[0]["best_weights_path"].endswith("run_a\\weights\\best.pt")
    assert summaries[1]["best_epoch"] == 2
    assert summaries[1]["best_map50_95"] == 0.68


def test_write_training_summaries_csv_outputs_compact_table(tmp_path) -> None:
    output_path = tmp_path / "training_summary.csv"
    rows = [
        {
            "run_name": "ccpd_yolo26n_quick",
            "best_epoch": 50,
            "best_precision": 1.0,
            "best_recall": 0.99857,
            "best_map50": 0.995,
            "best_map50_95": 0.7791,
            "final_epoch": 50,
            "final_map50_95": 0.7791,
            "best_weights_path": "/tmp/best.pt",
        }
    ]

    write_training_summaries_csv(output_path, rows)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        data = list(csv.DictReader(handle))

    assert data == [
        {
            "run_name": "ccpd_yolo26n_quick",
            "best_epoch": "50",
            "best_precision": "1.0",
            "best_recall": "0.99857",
            "best_map50": "0.995",
            "best_map50_95": "0.7791",
            "final_epoch": "50",
            "final_map50_95": "0.7791",
            "best_weights_path": "/tmp/best.pt",
        }
    ]
