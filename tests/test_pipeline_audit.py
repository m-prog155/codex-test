from pathlib import Path

from car_system.experiments.pipeline_audit import (
    build_sample_audit_row,
    build_hard_case_summary,
    build_sample_audit_summary,
    build_sample_path_list,
    copy_audit_sample_images,
    filter_audit_rows,
    pick_best_recognized_match,
)
from car_system.types import Detection, PlateDiagnostic, PlateMatch, PlateRecognition


def test_pick_best_recognized_match_prefers_highest_ocr_confidence_then_plate_confidence() -> None:
    low_ocr = PlateMatch(
        plate=Detection(label="plate", confidence=0.98, bbox=(0.0, 0.0, 10.0, 4.0)),
        vehicle=None,
        recognition=PlateRecognition(text="皖A12345", confidence=0.91),
    )
    high_ocr_low_plate = PlateMatch(
        plate=Detection(label="plate", confidence=0.80, bbox=(0.0, 0.0, 10.0, 4.0)),
        vehicle=None,
        recognition=PlateRecognition(text="皖A54321", confidence=0.95),
    )
    high_ocr_high_plate = PlateMatch(
        plate=Detection(label="plate", confidence=0.97, bbox=(0.0, 0.0, 10.0, 4.0)),
        vehicle=None,
        recognition=PlateRecognition(text="皖ANN665", confidence=0.95),
    )

    chosen = pick_best_recognized_match(
        [
            PlateMatch(
                plate=Detection(label="plate", confidence=0.99, bbox=(0.0, 0.0, 10.0, 4.0)),
                vehicle=None,
                recognition=None,
            ),
            low_ocr,
            high_ocr_low_plate,
            high_ocr_high_plate,
        ]
    )

    assert chosen is high_ocr_high_plate


def test_build_sample_audit_summary_counts_statuses_and_confidence_distribution() -> None:
    rows = [
        {
            "relative_path": "a.jpg",
            "subset": "test",
            "gt_text": "皖ANN665",
            "predicted_text": "皖ANN665",
            "status": "exact",
            "confidence": 0.99,
        },
        {
            "relative_path": "b.jpg",
            "subset": "test",
            "gt_text": "皖AB618B",
            "predicted_text": "皖AB6188",
            "status": "wrong",
            "confidence": 0.95,
        },
        {
            "relative_path": "c.jpg",
            "subset": "test",
            "gt_text": "皖AF3606",
            "predicted_text": None,
            "status": "null",
            "confidence": None,
        },
        {
            "relative_path": "d.jpg",
            "subset": "test",
            "gt_text": "皖A05066",
            "predicted_text": "皖A05066",
            "status": "exact",
            "confidence": 0.97,
        },
    ]

    summary = build_sample_audit_summary(rows)

    assert summary["sample_count"] == 4
    assert summary["exact"]["count"] == 2
    assert summary["exact"]["rate"] == 0.5
    assert summary["exact"]["mean_confidence"] == 0.98
    assert summary["exact"]["median_confidence"] == 0.98
    assert summary["wrong"]["count"] == 1
    assert summary["wrong"]["min_confidence"] == 0.95
    assert summary["null"]["count"] == 1
    assert summary["null"]["mean_confidence"] is None


def test_filter_audit_rows_and_build_sample_path_list_select_requested_statuses() -> None:
    rows = [
        {
            "relative_path": "test/exact.jpg",
            "subset": "test",
            "gt_text": "皖ANN665",
            "predicted_text": "皖ANN665",
            "status": "exact",
            "confidence": 0.99,
        },
        {
            "relative_path": "test/wrong.jpg",
            "subset": "test",
            "gt_text": "皖AB12Z9",
            "predicted_text": "皖AB1279",
            "status": "wrong",
            "confidence": 0.96,
        },
        {
            "relative_path": "test/null.jpg",
            "subset": "test",
            "gt_text": "皖AF3606",
            "predicted_text": None,
            "status": "null",
            "confidence": None,
        },
    ]

    selected = filter_audit_rows(rows, statuses=["wrong", "null"])

    assert [row["relative_path"] for row in selected] == ["test/wrong.jpg", "test/null.jpg"]
    assert build_sample_path_list(selected) == ["test/wrong.jpg", "test/null.jpg"]


def test_build_hard_case_summary_reports_prefix_transitions_positions_and_confusions() -> None:
    rows = [
        {
            "relative_path": "test/a.jpg",
            "subset": "test",
            "gt_text": "皖AB12Z9",
            "predicted_text": "皖AB1279",
            "status": "wrong",
            "confidence": 0.97,
        },
        {
            "relative_path": "test/b.jpg",
            "subset": "test",
            "gt_text": "晋LD0QB1",
            "predicted_text": "皖A00Q81",
            "status": "wrong",
            "confidence": 0.95,
        },
        {
            "relative_path": "test/c.jpg",
            "subset": "test",
            "gt_text": "皖AF3606",
            "predicted_text": None,
            "status": "null",
            "confidence": None,
        },
    ]

    summary = build_hard_case_summary(rows)

    assert summary["sample_count"] == 3
    assert summary["status_counts"] == {"wrong": 2, "null": 1}
    assert summary["length_pairs"] == [{"gt_length": 7, "pred_length": 7, "count": 2}]
    assert summary["prefix_transitions"] == [
        {"gt_prefix": "晋L", "pred_prefix": "皖A", "count": 1},
        {"gt_prefix": "皖A", "pred_prefix": "皖A", "count": 1},
    ]
    assert summary["mismatch_positions"] == [
        {"position": 5, "count": 2},
        {"position": 0, "count": 1},
        {"position": 1, "count": 1},
        {"position": 2, "count": 1},
    ]
    assert summary["character_confusions"][:4] == [
        {"gt_char": "B", "pred_char": "8", "count": 1},
        {"gt_char": "D", "pred_char": "0", "count": 1},
        {"gt_char": "L", "pred_char": "A", "count": 1},
        {"gt_char": "Z", "pred_char": "7", "count": 1},
    ]


def test_copy_audit_sample_images_preserves_relative_structure(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    source_file = dataset_root / "test" / "sample.jpg"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"fake-image")
    export_root = tmp_path / "export"

    written = copy_audit_sample_images(
        rows=[
            {
                "relative_path": "test/sample.jpg",
                "status": "wrong",
            }
        ],
        dataset_root=dataset_root,
        export_root=export_root,
    )

    assert written == [export_root / "images" / "test" / "sample.jpg"]
    assert (export_root / "images" / "test" / "sample.jpg").read_bytes() == b"fake-image"


def test_build_sample_audit_row_includes_recognition_and_diagnostic_fields() -> None:
    row = build_sample_audit_row(
        relative_path=Path("test/sample.jpg"),
        gt_text="皖A7Q653",
        best_match=PlateMatch(
            plate=Detection(label="plate", confidence=0.97, bbox=(10.0, 20.0, 110.0, 50.0)),
            vehicle=None,
            recognition=PlateRecognition(
                text="皖A70653",
                confidence=0.96,
                raw_text="皖A70653",
                normalized_text="皖A70653",
            ),
            diagnostic=PlateDiagnostic(
                status="recognized",
                crop_bbox=(8, 18, 112, 52),
                crop_shape=(34, 104, 3),
                rectified_shape=(34, 104, 3),
                confidence=0.96,
                raw_text="皖A70653",
                normalized_text="皖A70653",
                rectification_mode="safe",
                rectification_applied=True,
                rectification_reason="applied_disagreement_plain_used",
            ),
        ),
    )

    assert row == {
        "relative_path": "test/sample.jpg",
        "subset": "test",
        "gt_text": "皖A7Q653",
        "predicted_text": "皖A70653",
        "status": "wrong",
        "confidence": 0.96,
        "raw_text": "皖A70653",
        "normalized_text": "皖A70653",
        "diagnostic_status": "recognized",
        "rectification_mode": "safe",
        "rectification_applied": True,
        "rectification_reason": "applied_disagreement_plain_used",
    }
