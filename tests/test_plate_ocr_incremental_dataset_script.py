from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import scripts.build_plate_ocr_incremental_dataset as script


def _write_image(path: Path, shape: tuple[int, int, int] = (200, 220, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, 128, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_label_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def test_build_plate_ocr_incremental_dataset_build_parser_uses_expected_defaults() -> None:
    args = script.build_parser().parse_args([])

    assert args.base_dataset_root == Path("outputs/plate_ocr_dataset")
    assert args.source_root == Path("D:/plate_project/ccpd_yolo_mvp/images")
    assert args.output_root == Path("outputs/plate_ocr_incremental_v1")
    assert args.source_split_file is None
    assert args.protected_sample_list == []
    assert args.include_subsets == "ccpd_challenge,ccpd_blur,ccpd_tilt,ccpd_db,ccpd_fn,ccpd_rotate"
    assert args.val_ratio == 0.1
    assert args.max_val_per_subset == 200
    assert args.output_width == 168
    assert args.output_height == 48
    assert args.seed == 42


def test_build_plate_ocr_incremental_dataset_build_parser_allows_custom_source_split_files() -> None:
    args = script.build_parser().parse_args(
        [
            "--source-split-file",
            "custom/train.txt",
            "--source-split-file",
            "custom/test.txt",
        ]
    )

    assert args.source_split_file == [Path("custom/train.txt"), Path("custom/test.txt")]


def test_infer_source_subset_handles_prefixed_and_nested_paths() -> None:
    assert script.infer_source_subset(Path("test/ccpd_blur__sample.jpg")) == "ccpd_blur"
    assert script.infer_source_subset(Path("ccpd_tilt/sample.jpg")) == "ccpd_tilt"
    assert script.infer_source_subset(Path("plain.jpg")) == ""


def test_partition_incremental_entries_filters_protected_and_non_target_subsets() -> None:
    entries = [
        Path("test/ccpd_blur__a.jpg"),
        Path("test/ccpd_blur__b.jpg"),
        Path("test/ccpd_challenge__c.jpg"),
        Path("test/ccpd_rotate__d.jpg"),
        Path("test/ccpd_base__e.jpg"),
    ]

    train_entries, val_entries, summary = script.partition_incremental_entries(
        entries,
        include_subsets=("ccpd_blur", "ccpd_challenge", "ccpd_rotate"),
        protected_entries={Path("test/ccpd_rotate__d.jpg")},
        val_ratio=0.5,
        max_val_per_subset=1,
        seed=7,
    )

    selected = {path.as_posix() for path in train_entries + val_entries}
    assert "test/ccpd_base__e.jpg" not in selected
    assert "test/ccpd_rotate__d.jpg" not in selected
    assert len(train_entries) == 2
    assert len(val_entries) == 1
    assert summary["protected_excluded_count"] == 1
    assert summary["subset_excluded_count"] == 1
    assert summary["eligible_counts_by_subset"] == {"ccpd_blur": 2, "ccpd_challenge": 1}


def test_build_plate_ocr_incremental_dataset_main_creates_augmented_dataset(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base_root = tmp_path / "base_dataset"
    source_root = tmp_path / "source_images"
    output_root = tmp_path / "incremental_dataset"

    _write_image(base_root / "images" / "train" / "base_train.jpg", shape=(48, 168, 3))
    _write_image(base_root / "images" / "val" / "base_val.jpg", shape=(48, 168, 3))
    _write_image(base_root / "images" / "test" / "base_test.jpg", shape=(48, 168, 3))
    _write_label_file(base_root / "train.txt", ["images/train/base_train.jpg\t皖A12345"])
    _write_label_file(base_root / "val.txt", ["images/val/base_val.jpg\t皖A54321"])
    _write_label_file(base_root / "test.txt", ["images/test/base_test.jpg\t皖A99999"])
    _write_label_file(base_root / "dicts" / "plate_dict.txt", ["皖", "A", "1"])

    blur_a = "test/ccpd_blur__0022-0_4-337&385_411&410-411&410_339&410_337&385_409&385-0_0_13_24_9_31_30-74-6.jpg"
    blur_b = "test/ccpd_blur__0023-0_0-290&386_367&412-367&412_290&411_290&386_367&387-0_0_23_26_5_31_24-69-2.jpg"
    challenge_a = "test/ccpd_challenge__0045-2_1-270&420_364&456-361&456_274&449_270&420_357&427-0_7_7_15_17_31_29-35-24.jpg"
    challenge_b = "test/ccpd_challenge__0046-0_0-250&410_352&448-352&448_252&447_250&410_350&411-0_0_4_26_31_30_29-66-14.jpg"
    for rel_path in (blur_a, blur_b, challenge_a, challenge_b):
        _write_image(source_root / Path(rel_path))

    source_split = tmp_path / "ocr_test.txt"
    _write_label_file(source_split, [blur_a, blur_b, challenge_a, challenge_b])

    protected = tmp_path / "protected.txt"
    _write_label_file(protected, [challenge_b])

    parsed_args = Namespace(
        base_dataset_root=base_root,
        source_root=source_root,
        source_split_file=[source_split],
        output_root=output_root,
        protected_sample_list=[protected],
        include_subsets="ccpd_blur,ccpd_challenge",
        val_ratio=0.5,
        max_val_per_subset=1,
        output_width=96,
        output_height=32,
        seed=11,
    )
    monkeypatch.setattr(script, "build_parser", lambda: SimpleNamespace(parse_args=lambda *args, **kwargs: parsed_args))

    exit_code = script.main()

    assert exit_code == 0
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["base_counts"] == {"train": 1, "val": 1, "test": 1}
    assert summary["incremental"]["protected_excluded_count"] == 1
    assert summary["incremental"]["eligible_counts_by_subset"] == {"ccpd_blur": 2, "ccpd_challenge": 1}
    assert summary["incremental"]["train_extra_count"] == 2
    assert summary["incremental"]["val_extra_count"] == 1

    train_lines = (output_root / "train.txt").read_text(encoding="utf-8").splitlines()
    val_lines = (output_root / "val.txt").read_text(encoding="utf-8").splitlines()
    test_lines = (output_root / "test.txt").read_text(encoding="utf-8").splitlines()

    assert train_lines[0] == "images/train/base_train.jpg\t皖A12345"
    assert val_lines[0] == "images/val/base_val.jpg\t皖A54321"
    assert test_lines == ["images/test/base_test.jpg\t皖A99999"]
    assert len(train_lines) == 3
    assert len(val_lines) == 2
    assert (output_root / "images" / "train" / "base_train.jpg").exists()
    assert (output_root / "images" / "val" / "base_val.jpg").exists()
    assert (output_root / "images" / "train_extra").exists()
    assert (output_root / "images" / "val_extra").exists()
    assert (output_root / "dicts" / "plate_dict.txt").exists()
