import csv

from car_system.experiments.summary import build_directory_summary, build_file_summaries


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_build_directory_summary_aggregates_plate_and_ocr_metrics(tmp_path) -> None:
    rows_a = [
        {
            "source_name": "a.jpg",
            "frame_index": 0,
            "vehicle_label": "car",
            "vehicle_confidence": 0.91,
            "plate_confidence": 0.95,
            "plate_text": "ABC123",
            "ocr_confidence": 0.90,
        },
        {
            "source_name": "b.jpg",
            "frame_index": 0,
            "vehicle_label": "truck",
            "vehicle_confidence": 0.84,
            "plate_confidence": 0.93,
            "plate_text": "",
            "ocr_confidence": "",
        },
    ]
    rows_b = [
        {
            "source_name": "video.mp4",
            "frame_index": 1,
            "vehicle_label": "car",
            "vehicle_confidence": 0.88,
            "plate_confidence": 0.91,
            "plate_text": "XYZ789",
            "ocr_confidence": 0.80,
        }
    ]

    write_rows(tmp_path / "run_a.csv", rows_a)
    write_rows(tmp_path / "run_b.csv", rows_b)

    summary = build_directory_summary(tmp_path)

    assert summary["file_count"] == 2
    assert summary["plate_detection_count"] == 3
    assert summary["recognized_plate_count"] == 2
    assert summary["recognition_rate"] == 2 / 3
    assert summary["average_ocr_confidence"] == 0.85
    assert summary["vehicle_counts"]["car"] == 2
    assert summary["vehicle_counts"]["truck"] == 1


def test_build_file_summaries_returns_one_summary_per_csv(tmp_path) -> None:
    rows = [
        {
            "source_name": "sample.jpg",
            "frame_index": 0,
            "vehicle_label": "car",
            "vehicle_confidence": 0.91,
            "plate_confidence": 0.95,
            "plate_text": "ABC123",
            "ocr_confidence": 0.90,
        }
    ]

    write_rows(tmp_path / "sample.csv", rows)

    summaries = build_file_summaries(tmp_path)

    assert len(summaries) == 1
    assert summaries[0]["file_name"] == "sample.csv"
    assert summaries[0]["plate_detection_count"] == 1
    assert summaries[0]["recognized_plate_count"] == 1
