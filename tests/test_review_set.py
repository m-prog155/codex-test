from pathlib import Path

from car_system.diagnostics.review_set import build_review_rows, load_review_set


def test_load_review_set_preserves_fixed_sample_order(tmp_path) -> None:
    review_yaml = tmp_path / "review.yaml"
    review_yaml.write_text(
        """
dataset_root: /root/autodl-tmp/datasets/ccpd_yolo_mvp/images/test
samples:
  - category: easy
    relative_path: ccpd_base/example_a.jpg
  - category: blur
    relative_path: ccpd_blur/example_b.jpg
""".strip(),
        encoding="utf-8",
    )

    review_set = load_review_set(review_yaml)

    assert review_set.dataset_root == Path("/root/autodl-tmp/datasets/ccpd_yolo_mvp/images/test")
    assert [item.category for item in review_set.samples] == ["easy", "blur"]
    assert [item.relative_path.as_posix() for item in review_set.samples] == [
        "ccpd_base/example_a.jpg",
        "ccpd_blur/example_b.jpg",
    ]


def test_build_review_rows_keeps_category_ground_truth_and_paths() -> None:
    diagnostics = [
        {
            "status": "ocr_null",
            "confidence": 0.42,
            "raw_text": "",
            "normalized_text": "",
            "crop_path": "/tmp/crop.jpg",
            "rectified_path": "/tmp/rectified.jpg",
            "rectification_mode": "safe",
            "rectification_applied": True,
            "rectification_reason": "applied",
        }
    ]

    rows = build_review_rows(
        category="blur",
        source_name="sample.jpg",
        gt_text="皖A12345",
        diagnostics=diagnostics,
    )

    assert rows == [
        {
            "category": "blur",
            "source_name": "sample.jpg",
            "gt_text": "皖A12345",
            "diagnostic_status": "ocr_null",
            "ocr_confidence": "0.42",
            "ocr_raw_text": "",
            "ocr_normalized_text": "",
            "crop_path": "/tmp/crop.jpg",
            "rectified_path": "/tmp/rectified.jpg",
            "rectification_mode": "safe",
            "rectification_applied": "True",
            "rectification_reason": "applied",
        }
    ]


def test_build_review_rows_preserves_false_rectification_flag() -> None:
    diagnostics = [
        {
            "status": "recognized",
            "confidence": 0.91,
            "raw_text": "皖A12345",
            "normalized_text": "皖A12345",
            "crop_path": "/tmp/crop.jpg",
            "rectified_path": "/tmp/rectified.jpg",
            "rectification_mode": "safe",
            "rectification_applied": False,
            "rectification_reason": "low_score",
        }
    ]

    rows = build_review_rows(
        category="blur",
        source_name="sample.jpg",
        gt_text="皖A12345",
        diagnostics=diagnostics,
    )

    assert rows[0]["rectification_applied"] == "False"
    assert rows[0]["rectification_reason"] == "low_score"


def test_build_review_rows_emits_plate_missed_row_when_no_diagnostics() -> None:
    rows = build_review_rows(
        category="challenge",
        source_name="missing.jpg",
        gt_text="皖A00001",
        diagnostics=[],
    )

    assert rows == [
        {
            "category": "challenge",
            "source_name": "missing.jpg",
            "gt_text": "皖A00001",
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
