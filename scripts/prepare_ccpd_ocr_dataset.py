from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.data.ccpd import load_split_entries
from car_system.datasets.plate_ocr_dataset import export_recognition_split, write_plate_dictionary


DEFAULT_SOURCE_ROOT = Path("D:/plate_project/CCPD2019")
DEFAULT_OUTPUT_ROOT = Path("outputs/plate_ocr_dataset")
DEFAULT_OUTPUT_WIDTH = 168
DEFAULT_OUTPUT_HEIGHT = 48


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CCPD splits for PaddleOCR recognition training.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-width", type=int, default=DEFAULT_OUTPUT_WIDTH)
    parser.add_argument("--output-height", type=int, default=DEFAULT_OUTPUT_HEIGHT)
    return parser


def _call_export_recognition_split(
    source_root: Path,
    output_root: Path,
    split_name: str,
    entries: list[Path],
    output_width: int,
    output_height: int,
) -> int:
    return export_recognition_split(
        source_root=source_root,
        output_root=output_root,
        split_name=split_name,
        entries=entries,
        output_width=output_width,
        output_height=output_height,
    )


def main() -> int:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    split_root = source_root / "splits"

    for split_name in ("train", "val", "test"):
        entries = load_split_entries(split_root / f"{split_name}.txt")
        _call_export_recognition_split(
            source_root=source_root,
            output_root=output_root,
            split_name=split_name,
            entries=entries,
            output_width=args.output_width,
            output_height=args.output_height,
        )

    dictionary_path = write_plate_dictionary(output_root / "dicts" / "plate_dict.txt")
    print(f"Dictionary: {dictionary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
