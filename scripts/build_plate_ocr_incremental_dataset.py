from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.data.ccpd import load_split_entries
from car_system.datasets.plate_ocr_dataset import export_recognition_split


DEFAULT_BASE_DATASET_ROOT = Path("outputs/plate_ocr_dataset")
DEFAULT_SOURCE_ROOT = Path("D:/plate_project/ccpd_yolo_mvp/images")
DEFAULT_SOURCE_SPLIT_FILE = Path("D:/plate_project/ccpd_yolo_mvp/ocr_splits/test.txt")
DEFAULT_OUTPUT_ROOT = Path("outputs/plate_ocr_incremental_v1")
DEFAULT_INCLUDE_SUBSETS = (
    "ccpd_challenge",
    "ccpd_blur",
    "ccpd_tilt",
    "ccpd_db",
    "ccpd_fn",
    "ccpd_rotate",
)
DEFAULT_OUTPUT_WIDTH = 168
DEFAULT_OUTPUT_HEIGHT = 48


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a self-contained OCR dataset by adding difficult CCPD subsets onto the base OCR dataset."
    )
    parser.add_argument("--base-dataset-root", type=Path, default=DEFAULT_BASE_DATASET_ROOT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--source-split-file",
        type=Path,
        action="append",
        default=None,
        help="Source split files containing difficult examples to fold into OCR train/val.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--protected-sample-list",
        type=Path,
        action="append",
        default=[],
        help="Sample-path lists that must be excluded from incremental train/val.",
    )
    parser.add_argument("--include-subsets", default=",".join(DEFAULT_INCLUDE_SUBSETS))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-val-per-subset", type=int, default=200)
    parser.add_argument("--output-width", type=int, default=DEFAULT_OUTPUT_WIDTH)
    parser.add_argument("--output-height", type=int, default=DEFAULT_OUTPUT_HEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def infer_source_subset(path: Path) -> str:
    if "__" in path.name:
        return path.name.split("__", 1)[0]
    if len(path.parts) > 1:
        return path.parts[0]
    return ""


def _read_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_text_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        target.hardlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def _materialize_base_split_images(base_dataset_root: Path, output_root: Path, split_name: str) -> list[str]:
    lines = _read_text_lines(base_dataset_root / f"{split_name}.txt")
    for line in lines:
        image_rel = Path(line.split("\t", 1)[0])
        _link_or_copy_file(base_dataset_root / image_rel, output_root / image_rel)
    return lines


def _load_protected_entries(paths: list[Path]) -> set[Path]:
    protected: set[Path] = set()
    for path in paths:
        protected.update(load_split_entries(path))
    return protected


def _load_unique_entries(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    entries: list[Path] = []
    for path in paths:
        for entry in load_split_entries(path):
            key = entry.as_posix()
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return entries


def partition_incremental_entries(
    entries: list[Path],
    *,
    include_subsets: tuple[str, ...],
    protected_entries: set[Path],
    val_ratio: float,
    max_val_per_subset: int,
    seed: int,
) -> tuple[list[Path], list[Path], dict[str, object]]:
    eligible_by_subset: dict[str, list[Path]] = defaultdict(list)
    protected_excluded = 0
    subset_excluded = 0
    protected_keys = {entry.as_posix() for entry in protected_entries}

    for entry in entries:
        subset = infer_source_subset(entry)
        if subset not in include_subsets:
            subset_excluded += 1
            continue
        if entry.as_posix() in protected_keys:
            protected_excluded += 1
            continue
        eligible_by_subset[subset].append(entry)

    generator = random.Random(seed)
    train_entries: list[Path] = []
    val_entries: list[Path] = []
    train_counts: dict[str, int] = {}
    val_counts: dict[str, int] = {}

    for subset in include_subsets:
        subset_entries = list(eligible_by_subset.get(subset, []))
        if not subset_entries:
            continue
        generator.shuffle(subset_entries)
        desired_val = int(round(len(subset_entries) * val_ratio))
        if val_ratio > 0.0 and len(subset_entries) >= 2 and desired_val == 0:
            desired_val = 1
        val_count = min(max_val_per_subset, desired_val, max(0, len(subset_entries) - 1))
        current_val = subset_entries[:val_count]
        current_train = subset_entries[val_count:]
        val_entries.extend(current_val)
        train_entries.extend(current_train)
        train_counts[subset] = len(current_train)
        val_counts[subset] = len(current_val)

    return train_entries, val_entries, {
        "source_entry_count": len(entries),
        "protected_sample_count": len(protected_entries),
        "protected_excluded_count": protected_excluded,
        "subset_excluded_count": subset_excluded,
        "eligible_counts_by_subset": {subset: len(paths) for subset, paths in eligible_by_subset.items()},
        "train_extra_count": len(train_entries),
        "val_extra_count": len(val_entries),
        "train_extra_by_subset": train_counts,
        "val_extra_by_subset": val_counts,
    }


def _copy_dictionary(base_dataset_root: Path, output_root: Path) -> None:
    dict_source = base_dataset_root / "dicts" / "plate_dict.txt"
    dict_target = output_root / "dicts" / "plate_dict.txt"
    dict_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dict_source, dict_target)


def main() -> int:
    args = build_parser().parse_args()
    base_dataset_root = Path(args.base_dataset_root)
    source_root = Path(args.source_root)
    source_split_files = (
        [Path(path) for path in args.source_split_file]
        if args.source_split_file is not None
        else [DEFAULT_SOURCE_SPLIT_FILE]
    )
    output_root = Path(args.output_root)
    protected_lists = [Path(path) for path in args.protected_sample_list]
    include_subsets = tuple(item.strip() for item in args.include_subsets.split(",") if item.strip())

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_train_lines = _materialize_base_split_images(base_dataset_root, output_root, "train")
    base_val_lines = _materialize_base_split_images(base_dataset_root, output_root, "val")
    base_test_lines = _materialize_base_split_images(base_dataset_root, output_root, "test")
    _write_text_lines(output_root / "train_base.txt", base_train_lines)
    _write_text_lines(output_root / "val_base.txt", base_val_lines)
    _write_text_lines(output_root / "test.txt", base_test_lines)
    _copy_dictionary(base_dataset_root, output_root)

    protected_entries = _load_protected_entries(protected_lists)
    source_entries = _load_unique_entries(source_split_files)
    train_extra_entries, val_extra_entries, incremental_summary = partition_incremental_entries(
        source_entries,
        include_subsets=include_subsets,
        protected_entries=protected_entries,
        val_ratio=args.val_ratio,
        max_val_per_subset=args.max_val_per_subset,
        seed=args.seed,
    )

    export_recognition_split(
        source_root=source_root,
        output_root=output_root,
        split_name="train_extra",
        entries=train_extra_entries,
        output_width=args.output_width,
        output_height=args.output_height,
    )
    export_recognition_split(
        source_root=source_root,
        output_root=output_root,
        split_name="val_extra",
        entries=val_extra_entries,
        output_width=args.output_width,
        output_height=args.output_height,
    )

    train_extra_lines = _read_text_lines(output_root / "train_extra.txt")
    val_extra_lines = _read_text_lines(output_root / "val_extra.txt")
    _write_text_lines(output_root / "train.txt", base_train_lines + train_extra_lines)
    _write_text_lines(output_root / "val.txt", base_val_lines + val_extra_lines)

    summary = {
        "base_counts": {
            "train": len(base_train_lines),
            "val": len(base_val_lines),
            "test": len(base_test_lines),
        },
        "incremental": incremental_summary,
        "source_split_files": [path.as_posix() for path in source_split_files],
        "protected_sample_lists": [path.as_posix() for path in protected_lists],
        "include_subsets": list(include_subsets),
        "output_width": args.output_width,
        "output_height": args.output_height,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
