from pathlib import Path

from PIL import Image
import yaml

from scripts.prepare_ccpd_dataset import _prepare_split
from car_system.datasets.yolo_dataset import prepare_yolo_dataset, write_dataset_yaml


def _create_dummy_pair(images_dir: Path, labels_dir: Path, stem: str) -> None:
    (images_dir / f"{stem}.jpg").write_bytes(b"image")
    (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")


def test_prepare_yolo_dataset_creates_train_val_structure(tmp_path: Path) -> None:
    source_images = tmp_path / "source_images"
    source_labels = tmp_path / "source_labels"
    source_images.mkdir()
    source_labels.mkdir()

    for stem in ["img1", "img2", "img3", "img4", "img5"]:
        _create_dummy_pair(source_images, source_labels, stem)

    output_dir = tmp_path / "prepared"
    stats = prepare_yolo_dataset(
        image_dir=source_images,
        label_dir=source_labels,
        output_dir=output_dir,
        train_ratio=0.8,
        seed=7,
    )

    assert stats["total_pairs"] == 5
    assert stats["train_pairs"] == 4
    assert stats["val_pairs"] == 1
    assert (output_dir / "images" / "train").exists()
    assert (output_dir / "images" / "val").exists()
    assert (output_dir / "labels" / "train").exists()
    assert (output_dir / "labels" / "val").exists()


def test_write_dataset_yaml_records_dataset_root_and_class_names(tmp_path: Path) -> None:
    dataset_root = tmp_path / "prepared"
    dataset_root.mkdir()
    output_path = tmp_path / "dataset.yaml"

    write_dataset_yaml(
        output_path=output_path,
        dataset_root=dataset_root,
        class_names=["car", "truck", "bus"],
    )

    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["path"] == str(dataset_root)
    assert payload["train"] == "images/train"
    assert payload["val"] == "images/val"
    assert payload["names"] == ["car", "truck", "bus"]


def test_prepare_split_preserves_duplicate_filenames_from_different_ccpd_subdirs(tmp_path: Path) -> None:
    source_root = tmp_path / "ccpd"
    sample_name = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"
    blur_path = source_root / "ccpd_blur" / sample_name
    challenge_path = source_root / "ccpd_challenge" / sample_name
    blur_path.parent.mkdir(parents=True, exist_ok=True)
    challenge_path.parent.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (100, 100), color=(255, 255, 255)).save(blur_path)
    Image.new("RGB", (100, 100), color=(0, 0, 0)).save(challenge_path)

    output_root = tmp_path / "prepared"
    count = _prepare_split(
        source_root=source_root,
        output_root=output_root,
        split_name="test",
        entries=[Path(f"ccpd_blur/{sample_name}"), Path(f"ccpd_challenge/{sample_name}")],
        copy_file=False,
    )

    exported_images = sorted(path.name for path in (output_root / "images" / "test").glob("*.jpg"))
    exported_labels = sorted(path.name for path in (output_root / "labels" / "test").glob("*.txt"))

    assert count == 2
    assert exported_images == [
        f"ccpd_blur__{sample_name}",
        f"ccpd_challenge__{sample_name}",
    ]
    assert exported_labels == [
        f"ccpd_blur__{Path(sample_name).stem}.txt",
        f"ccpd_challenge__{Path(sample_name).stem}.txt",
    ]
