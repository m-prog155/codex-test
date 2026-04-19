import argparse
import shutil
from pathlib import Path
import sys

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.data.ccpd import (
    bbox_to_yolo,
    load_split_entries,
    parse_ccpd_path,
    sample_split_entries,
    write_dataset_yaml,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare CCPD2019 into YOLO detection format.")
    parser.add_argument("--source-root", required=True, help="Path to the CCPD2019 dataset root.")
    parser.add_argument("--output-root", required=True, help="Output directory for YOLO-ready images and labels.")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of hard-linking them.")
    parser.add_argument("--limit-train", type=int, default=None, help="Optional train split size cap for quick experiments.")
    parser.add_argument("--limit-val", type=int, default=None, help="Optional val split size cap for quick experiments.")
    parser.add_argument("--limit-test", type=int, default=None, help="Optional test split size cap for quick experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when limiting split sizes.")
    return parser


def _link_or_copy_image(source: Path, target: Path, copy_file: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if copy_file:
        shutil.copy2(source, target)
    else:
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def _write_label_file(path: Path, values: tuple[float, float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cx, cy, width, height = values
    path.write_text(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n", encoding="utf-8")


def _build_export_name(relative_path: Path) -> str:
    if len(relative_path.parts) == 1:
        return relative_path.name
    prefix = "__".join(relative_path.parts[:-1])
    return f"{prefix}__{relative_path.name}"


def _prepare_split(
    source_root: Path,
    output_root: Path,
    split_name: str,
    entries: list[Path],
    copy_file: bool,
) -> int:
    count = 0
    for rel_path in entries:
        annotation = parse_ccpd_path(rel_path)
        source_image = source_root / annotation.relative_path
        if not source_image.exists():
            continue

        with Image.open(source_image) as image:
            width, height = image.size

        yolo_bbox = bbox_to_yolo(annotation.bbox, image_width=width, image_height=height)
        export_name = _build_export_name(annotation.relative_path)
        target_image = output_root / "images" / split_name / export_name
        target_label = output_root / "labels" / split_name / f"{Path(export_name).stem}.txt"

        _link_or_copy_image(source_image, target_image, copy_file=copy_file)
        _write_label_file(target_label, yolo_bbox)
        count += 1
    return count


def main() -> int:
    args = build_parser().parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    split_root = source_root / "splits"

    train_entries = load_split_entries(split_root / "train.txt")
    val_entries = load_split_entries(split_root / "val.txt")
    test_entries = load_split_entries(split_root / "test.txt")

    train_entries = sample_split_entries(train_entries, limit=args.limit_train, seed=args.seed)
    val_entries = sample_split_entries(val_entries, limit=args.limit_val, seed=args.seed)
    test_entries = sample_split_entries(test_entries, limit=args.limit_test, seed=args.seed)

    train_count = _prepare_split(source_root, output_root, "train", train_entries, copy_file=args.copy)
    val_count = _prepare_split(source_root, output_root, "val", val_entries, copy_file=args.copy)
    test_count = _prepare_split(source_root, output_root, "test", test_entries, copy_file=args.copy)

    dataset_yaml = write_dataset_yaml(output_root / "dataset.yaml", dataset_root=output_root, class_names=["plate"])

    print(f"Prepared train images: {train_count}")
    print(f"Prepared val images: {val_count}")
    print(f"Prepared test images: {test_count}")
    print(f"Dataset YAML: {dataset_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
