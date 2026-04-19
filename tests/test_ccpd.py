from pathlib import Path

from car_system.data.ccpd import (
    ADS,
    ALPHABETS,
    PROVINCES,
    bbox_to_yolo,
    decode_ccpd_plate_indices,
    load_split_entries,
    parse_ccpd_path,
    sample_split_entries,
    write_dataset_yaml,
)


def test_parse_ccpd_path_reads_bbox_and_vertices() -> None:
    sample = Path(
        "ccpd_base/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )

    annotation = parse_ccpd_path(sample)

    assert annotation.relative_path == sample
    assert annotation.bbox == (154, 383, 386, 473)
    assert annotation.vertices == [(386, 473), (177, 454), (154, 383), (363, 402)]
    assert annotation.plate_indices == [0, 0, 22, 27, 27, 33, 16]


def test_decode_ccpd_plate_indices_decodes_known_sequence() -> None:
    assert decode_ccpd_plate_indices([0, 0, 22, 27, 27, 33, 16]) == "皖AY339S"


def test_ccpd_lookup_tables_match_the_official_tail() -> None:
    assert PROVINCES[-3:] == ["警", "学", "O"]
    assert ALPHABETS[22] == "Y"
    assert ADS[22] == "Y"


def test_decode_ccpd_plate_indices_rejects_out_of_range_province_indices() -> None:
    for indices in ([-1, 0, 0, 0, 0, 0, 0], [len(PROVINCES), 0, 0, 0, 0, 0, 0]):
        try:
            decode_ccpd_plate_indices(indices)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for out-of-range province index")


def test_decode_ccpd_plate_indices_rejects_out_of_range_alphabet_indices() -> None:
    for indices in ([0, -1, 0, 0, 0, 0, 0], [0, len(ALPHABETS), 0, 0, 0, 0, 0]):
        try:
            decode_ccpd_plate_indices(indices)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for out-of-range alphabet index")


def test_decode_ccpd_plate_indices_rejects_out_of_range_ad_indices() -> None:
    for indices in (
        [0, 0, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, len(ADS)],
    ):
        try:
            decode_ccpd_plate_indices(indices)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for out-of-range ad index")


def test_decode_ccpd_plate_indices_requires_seven_indices() -> None:
    for indices in ([0, 0, 22, 27, 27, 33], [0, 0, 22, 27, 27, 33, 16, 1]):
        try:
            decode_ccpd_plate_indices(indices)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError for invalid CCPD plate index length")


def test_parse_then_decode_ccpd_plate_indices_matches_sample_plate_text() -> None:
    sample = Path(
        "ccpd_base/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )

    annotation = parse_ccpd_path(sample)

    assert decode_ccpd_plate_indices(annotation.plate_indices) == "皖AY339S"


def test_bbox_to_yolo_normalizes_coordinates() -> None:
    values = bbox_to_yolo((154, 383, 386, 473), image_width=720, image_height=1160)

    assert len(values) == 4
    assert round(values[0], 6) == round(((154 + 386) / 2) / 720, 6)
    assert round(values[1], 6) == round(((383 + 473) / 2) / 1160, 6)
    assert round(values[2], 6) == round((386 - 154) / 720, 6)
    assert round(values[3], 6) == round((473 - 383) / 1160, 6)


def test_load_split_entries_reads_non_empty_lines(tmp_path: Path) -> None:
    split_file = tmp_path / "train.txt"
    split_file.write_text("ccpd_base/a.jpg\n\nccpd_blur/b.jpg\n", encoding="utf-8")

    entries = load_split_entries(split_file)

    assert entries == [Path("ccpd_base/a.jpg"), Path("ccpd_blur/b.jpg")]


def test_sample_split_entries_is_reproducible_and_preserves_original_order() -> None:
    entries = [Path(f"ccpd_base/{name}.jpg") for name in ["a", "b", "c", "d", "e", "f", "g"]]

    sampled = sample_split_entries(entries, limit=3, seed=7)

    assert sampled == [
        Path("ccpd_base/b.jpg"),
        Path("ccpd_base/c.jpg"),
        Path("ccpd_base/d.jpg"),
    ]


def test_write_dataset_yaml_creates_basic_yolo_config(tmp_path: Path) -> None:
    output_path = tmp_path / "dataset.yaml"

    write_dataset_yaml(output_path, dataset_root=tmp_path, class_names=["plate"])

    content = output_path.read_text(encoding="utf-8")

    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "test: images/test" in content
    assert "names:" in content
    assert "- plate" in content
