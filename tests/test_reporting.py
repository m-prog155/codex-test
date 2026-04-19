from car_system.diagnostics.reporting import build_report_summary, render_html_report, select_failure_rows


def test_build_report_summary_counts_statuses_and_accuracy() -> None:
    rows = [
        {
            "category": "easy",
            "diagnostic_status": "recognized",
            "gt_text": "皖A12345",
            "ocr_normalized_text": "皖A12345",
        },
        {
            "category": "blur",
            "diagnostic_status": "ocr_null",
            "gt_text": "皖A12346",
            "ocr_normalized_text": "",
        },
    ]

    summary = build_report_summary(rows)

    assert summary["total_samples"] == 2
    assert summary["status_counts"]["recognized"] == 1
    assert summary["status_counts"]["ocr_null"] == 1
    assert summary["exact_plate_accuracy"] == 0.5
    assert summary["null_rate"] == 0.5


def test_render_html_report_includes_failure_section() -> None:
    summary = {
        "total_samples": 2,
        "status_counts": {"recognized": 1, "ocr_null": 1},
        "exact_plate_accuracy": 0.5,
        "char_accuracy": 0.5,
        "null_rate": 0.5,
        "by_category": {"easy": {"samples": 1}, "blur": {"samples": 1}},
    }
    failures = [
        {
            "source_name": "failure.jpg",
            "diagnostic_status": "ocr_null",
            "gt_text": "皖A12346",
            "ocr_confidence": "",
            "ocr_normalized_text": "",
            "crop_path": "failure_crop.jpg",
            "rectified_path": "failure_rectified.jpg",
        }
    ]

    html = render_html_report(summary, failures)

    assert "failure.jpg" in html
    assert "ocr_null" in html
    assert "皖A12346" in html
    assert "OCR confidence" in html


def test_select_failure_rows_keeps_wrong_recognized_text() -> None:
    rows = [
        {
            "source_name": "wrong.jpg",
            "diagnostic_status": "recognized",
            "gt_text": "皖A12345",
            "ocr_normalized_text": "皖A12340",
        },
        {
            "source_name": "correct.jpg",
            "diagnostic_status": "recognized",
            "gt_text": "皖A12345",
            "ocr_normalized_text": "皖A12345",
        },
    ]

    failures = select_failure_rows(rows)

    assert [row["source_name"] for row in failures] == ["wrong.jpg"]
