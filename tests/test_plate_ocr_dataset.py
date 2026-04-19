from pathlib import Path

import cv2
import numpy as np

from car_system.datasets.plate_ocr_dataset import (
    build_plate_full_text,
    export_recognition_split,
    order_plate_vertices,
    warp_plate_from_vertices,
    write_plate_dictionary,
)


def _canonical_plate_image() -> np.ndarray:
    image = np.full((48, 168, 3), 128, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (11, 11), (0, 0, 255), thickness=-1)
    cv2.rectangle(image, (156, 0), (167, 11), (0, 255, 0), thickness=-1)
    cv2.rectangle(image, (156, 36), (167, 47), (255, 0, 0), thickness=-1)
    cv2.rectangle(image, (0, 36), (11, 47), (0, 255, 255), thickness=-1)
    return image


def _make_skewed_plate_source() -> tuple[np.ndarray, list[tuple[int, int]]]:
    plate = _canonical_plate_image()
    vertices = [(62, 55), (154, 42), (165, 128), (48, 139)]
    matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [167, 0], [167, 47], [0, 47]]),
        np.float32(vertices),
    )
    source = cv2.warpPerspective(plate, matrix, (220, 200))
    return source, vertices


def _expected_dictionary_lines() -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for chars in (
        [
            "皖",
            "沪",
            "津",
            "渝",
            "冀",
            "晋",
            "蒙",
            "辽",
            "吉",
            "黑",
            "苏",
            "浙",
            "京",
            "闽",
            "赣",
            "鲁",
            "豫",
            "鄂",
            "湘",
            "粤",
            "桂",
            "琼",
            "川",
            "贵",
            "云",
            "藏",
            "陕",
            "甘",
            "青",
            "宁",
            "新",
            "警",
            "学",
            "O",
        ],
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "O",
        ],
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "O",
        ],
    ):
        for char in chars:
            if char == "O" or char in seen:
                continue
            seen.add(char)
            ordered.append(char)
    return ordered


def test_build_plate_full_text_decodes_ccpd_indices() -> None:
    assert build_plate_full_text([0, 0, 22, 27, 27, 33, 16]) == "皖AY339S"


def test_order_plate_vertices_orders_corners_from_top_left() -> None:
    vertices = [(165, 128), (48, 139), (154, 42), (62, 55)]

    assert order_plate_vertices(vertices) == [(62, 55), (154, 42), (165, 128), (48, 139)]


def test_order_plate_vertices_handles_ties_without_repeating_points() -> None:
    vertices = [(0, 2), (2, 0), (4, 2), (2, 4)]

    ordered = order_plate_vertices(vertices)

    assert ordered == [(0, 2), (2, 0), (4, 2), (2, 4)]
    assert len({tuple(point) for point in ordered}) == 4


def test_warp_plate_from_vertices_rectifies_plate_to_default_size() -> None:
    source, vertices = _make_skewed_plate_source()

    warped = warp_plate_from_vertices(source, vertices)

    assert warped.shape == (48, 168, 3)
    assert warped[4:10, 4:10, 2].mean() > 180
    assert warped[4:10, 158:164, 1].mean() > 180
    assert warped[38:44, 158:164, 0].mean() > 180
    assert warped[38:44, 4:10, 1].mean() > 180
    assert warped[38:44, 4:10, 2].mean() > 180


def test_export_recognition_split_writes_warped_images_and_label_file(tmp_path: Path) -> None:
    source_root = tmp_path / "ccpd"
    output_root = tmp_path / "exported"
    relative_path = Path("ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg")
    source_image = source_root / relative_path
    source_image.parent.mkdir(parents=True, exist_ok=True)

    source, vertices = _make_skewed_plate_source()
    cv2.imwrite(str(source_image), source)

    count = export_recognition_split(
        source_root=source_root,
        output_root=output_root,
        split_name="train",
        entries=[relative_path],
    )

    export_name = f"{relative_path.parent.name}__{relative_path.name}"
    image_path = output_root / "images" / "train" / export_name
    label_path = output_root / "train.txt"
    exported_image = cv2.imread(str(image_path))

    assert count == 1
    assert image_path.exists()
    assert label_path.read_text(encoding="utf-8") == f"images/train/{export_name}\t皖AY339S\n"
    assert exported_image is not None
    assert exported_image.shape == (48, 168, 3)


def test_export_recognition_split_honors_custom_output_size(tmp_path: Path) -> None:
    source_root = tmp_path / "ccpd"
    output_root = tmp_path / "exported"
    relative_path = Path("ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg")
    source_image = source_root / relative_path
    source_image.parent.mkdir(parents=True, exist_ok=True)

    source, vertices = _make_skewed_plate_source()
    cv2.imwrite(str(source_image), source)

    count = export_recognition_split(
        source_root=source_root,
        output_root=output_root,
        split_name="train",
        entries=[relative_path],
        output_width=96,
        output_height=32,
    )

    export_name = f"{relative_path.parent.name}__{relative_path.name}"
    image_path = output_root / "images" / "train" / export_name
    exported_image = cv2.imread(str(image_path))

    assert count == 1
    assert image_path.exists()
    assert exported_image is not None
    assert exported_image.shape == (32, 96, 3)


def test_export_recognition_split_raises_for_missing_source_image_without_writing_partial_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "ccpd"
    output_root = tmp_path / "exported"
    valid_relative_path = Path(
        "ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    )
    missing_relative_path = Path(
        "ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-16.jpg"
    )
    source_image = source_root / valid_relative_path
    source_image.parent.mkdir(parents=True, exist_ok=True)

    source, _ = _make_skewed_plate_source()
    cv2.imwrite(str(source_image), source)

    try:
        export_recognition_split(
            source_root=source_root,
            output_root=output_root,
            split_name="train",
            entries=[valid_relative_path, missing_relative_path],
        )
    except FileNotFoundError as exc:
        assert str(missing_relative_path) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing source image")

    assert not (output_root / "images" / "train").exists()
    assert not (output_root / "train.txt").exists()


def test_export_recognition_split_raises_when_imread_fails_without_writing_partial_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "ccpd"
    output_root = tmp_path / "exported"
    relative_path = Path("ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg")
    source_image = source_root / relative_path
    source_image.parent.mkdir(parents=True, exist_ok=True)

    source, _ = _make_skewed_plate_source()
    cv2.imwrite(str(source_image), source)

    def _fake_imread(_: str) -> None:
        return None

    monkeypatch.setattr(cv2, "imread", _fake_imread)

    try:
        export_recognition_split(
            source_root=source_root,
            output_root=output_root,
            split_name="train",
            entries=[relative_path],
        )
    except RuntimeError as exc:
        assert "Failed to read source image" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when cv2.imread fails")

    assert not (output_root / "images" / "train").exists()
    assert not (output_root / "train.txt").exists()


def test_export_recognition_split_clears_stale_images_on_reexport(tmp_path: Path) -> None:
    source_root = tmp_path / "ccpd"
    output_root = tmp_path / "exported"
    relative_path = Path("ccpd_blur/025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg")
    source_image = source_root / relative_path
    source_image.parent.mkdir(parents=True, exist_ok=True)

    source, _ = _make_skewed_plate_source()
    cv2.imwrite(str(source_image), source)

    first_count = export_recognition_split(
        source_root=source_root,
        output_root=output_root,
        split_name="train",
        entries=[relative_path],
    )

    export_name = f"{relative_path.parent.name}__{relative_path.name}"
    image_path = output_root / "images" / "train" / export_name
    label_path = output_root / "train.txt"

    assert first_count == 1
    assert image_path.exists()
    assert label_path.read_text(encoding="utf-8") == f"images/train/{export_name}\t皖AY339S\n"

    stale_image = output_root / "images" / "train" / "obsolete.jpg"
    stale_image.parent.mkdir(parents=True, exist_ok=True)
    stale_image.write_bytes(b"stale")

    second_count = export_recognition_split(
        source_root=source_root,
        output_root=output_root,
        split_name="train",
        entries=[relative_path],
    )

    assert second_count == 1
    assert image_path.exists()
    assert not stale_image.exists()
    assert sorted(path.name for path in (output_root / "images" / "train").glob("*")) == [export_name]
    assert label_path.read_text(encoding="utf-8") == f"images/train/{export_name}\t皖AY339S\n"


def test_write_plate_dictionary_deduplicates_characters_and_excludes_placeholder_o(tmp_path: Path) -> None:
    output_path = tmp_path / "plate_dict.txt"

    write_plate_dictionary(output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert lines == _expected_dictionary_lines()
