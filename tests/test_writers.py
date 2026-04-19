import json

from car_system.io.writers import frame_result_to_dict, frame_result_to_rows, write_csv, write_json
from car_system.types import Detection, FrameResult, PlateDiagnostic, PlateMatch, PlateRecognition


def make_frame_result() -> FrameResult:
    vehicle = Detection(label="car", confidence=0.91, bbox=(10, 10, 120, 120))
    plate = Detection(label="plate", confidence=0.96, bbox=(40, 90, 95, 115))
    recognition = PlateRecognition(text="ABC123", confidence=0.89)
    return FrameResult(
        source_name="sample.jpg",
        frame_index=0,
        detections=[vehicle, plate],
        matches=[PlateMatch(plate=plate, vehicle=vehicle, recognition=recognition)],
    )


def test_frame_result_to_dict_contains_serializable_match_data() -> None:
    result = make_frame_result()

    payload = frame_result_to_dict(result)

    assert payload["source_name"] == "sample.jpg"
    assert payload["frame_index"] == 0
    assert payload["matches"][0]["vehicle"]["label"] == "car"
    assert payload["matches"][0]["recognition"]["text"] == "ABC123"


def test_write_json_and_csv_persist_frame_result(tmp_path) -> None:
    result = make_frame_result()
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "result.csv"

    write_json(json_path, frame_result_to_dict(result))
    write_csv(csv_path, frame_result_to_rows(result))

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    csv_text = csv_path.read_text(encoding="utf-8")

    assert loaded["matches"][0]["recognition"]["text"] == "ABC123"
    assert "plate_text" in csv_text
    assert "ABC123" in csv_text


def test_frame_result_to_rows_includes_diagnostic_status() -> None:
    result = make_frame_result()
    result.matches[0].diagnostic = PlateDiagnostic(
        status="recognized",
        crop_bbox=(36, 87, 99, 118),
        raw_text="ABC12O",
        normalized_text="ABC120",
    )

    rows = frame_result_to_rows(result)

    assert rows[0]["diagnostic_status"] == "recognized"
    assert rows[0]["ocr_raw_text"] == "ABC12O"
    assert rows[0]["ocr_normalized_text"] == "ABC120"
