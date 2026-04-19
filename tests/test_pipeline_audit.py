from car_system.experiments.pipeline_audit import build_sample_audit_summary, pick_best_recognized_match
from car_system.types import Detection, PlateMatch, PlateRecognition


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
