from dataclasses import dataclass
from pathlib import Path
import random

import yaml


PROVINCES = [
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
]

# Official CCPD mapping tail:
# provinces end with 警, 学, O; do not reintroduce any legacy index-22 compatibility branch.
ALPHABETS = [
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
]

ADS = [
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
]


@dataclass(slots=True)
class CcpdAnnotation:
    relative_path: Path
    bbox: tuple[int, int, int, int]
    vertices: list[tuple[int, int]]
    plate_indices: list[int]


def _parse_point(value: str) -> tuple[int, int]:
    x_raw, y_raw = value.split("&")
    return int(x_raw), int(y_raw)


def parse_ccpd_path(relative_path: str | Path) -> CcpdAnnotation:
    rel_path = Path(relative_path)
    stem = rel_path.stem
    parts = stem.split("-")
    if len(parts) < 7:
        raise ValueError(f"Unexpected CCPD filename format: {rel_path}")

    bbox_left_top, bbox_right_bottom = parts[2].split("_")
    x1, y1 = _parse_point(bbox_left_top)
    x2, y2 = _parse_point(bbox_right_bottom)

    vertices = [_parse_point(value) for value in parts[3].split("_")]
    plate_indices = [int(value) for value in parts[4].split("_")]

    return CcpdAnnotation(
        relative_path=rel_path,
        bbox=(x1, y1, x2, y2),
        vertices=vertices,
        plate_indices=plate_indices,
    )


def decode_ccpd_plate_indices(plate_indices: list[int]) -> str:
    if len(plate_indices) != 7:
        raise ValueError(f"Expected 7 CCPD plate indices, got {len(plate_indices)}")

    province_index, letter_index, *ad_indices = plate_indices
    if not 0 <= province_index < len(PROVINCES):
        raise ValueError(f"Province index out of range: {province_index}")
    if not 0 <= letter_index < len(ALPHABETS):
        raise ValueError(f"Letter index out of range: {letter_index}")

    decoded = [PROVINCES[province_index], ALPHABETS[letter_index]]
    for index in ad_indices:
        if not 0 <= index < len(ADS):
            raise ValueError(f"Character index out of range: {index}")
        decoded.append(ADS[index])
    return "".join(decoded)


def bbox_to_yolo(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return (
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height,
    )


def load_split_entries(path: str | Path) -> list[Path]:
    split_path = Path(path)
    entries = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        cleaned = line.strip()
        if cleaned:
            entries.append(Path(cleaned))
    return entries


def sample_split_entries(entries: list[Path], limit: int | None, seed: int = 42) -> list[Path]:
    if limit is None or limit >= len(entries):
        return list(entries)

    generator = random.Random(seed)
    selected_indexes = sorted(generator.sample(range(len(entries)), limit))
    return [entries[index] for index in selected_indexes]


def write_dataset_yaml(path: str | Path, dataset_root: str | Path, class_names: list[str]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(Path(dataset_root)),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return output_path
