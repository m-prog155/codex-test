import random
import shutil
from pathlib import Path

import yaml


def _paired_stems(image_dir: Path, label_dir: Path) -> list[str]:
    image_stems = {path.stem for path in image_dir.glob("*") if path.is_file()}
    label_stems = {path.stem for path in label_dir.glob("*.txt")}
    return sorted(image_stems & label_stems)


def _copy_pair(stem: str, image_dir: Path, label_dir: Path, target_root: Path, split: str) -> None:
    target_image_dir = target_root / "images" / split
    target_label_dir = target_root / "labels" / split
    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    source_image = next(path for path in image_dir.glob(f"{stem}.*") if path.is_file())
    source_label = label_dir / f"{stem}.txt"

    shutil.copy2(source_image, target_image_dir / source_image.name)
    shutil.copy2(source_label, target_label_dir / source_label.name)


def prepare_yolo_dataset(
    image_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 7,
) -> dict[str, int]:
    image_root = Path(image_dir)
    label_root = Path(label_dir)
    target_root = Path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)

    stems = _paired_stems(image_root, label_root)
    rng = random.Random(seed)
    rng.shuffle(stems)

    train_count = int(len(stems) * train_ratio)
    train_stems = stems[:train_count]
    val_stems = stems[train_count:]

    for stem in train_stems:
        _copy_pair(stem, image_root, label_root, target_root, "train")
    for stem in val_stems:
        _copy_pair(stem, image_root, label_root, target_root, "val")

    return {
        "total_pairs": len(stems),
        "train_pairs": len(train_stems),
        "val_pairs": len(val_stems),
    }


def write_dataset_yaml(output_path: str | Path, dataset_root: str | Path, class_names: list[str]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(Path(dataset_root)),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path
