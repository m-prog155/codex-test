from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import scripts.build_plate_ocr_focus_labels as focus_script


def test_build_plate_ocr_focus_labels_build_parser_uses_expected_defaults() -> None:
    args = focus_script.build_parser().parse_args([])

    assert args.dataset_root == Path("outputs/plate_ocr_dataset")
    assert args.output_root == Path("outputs/plate_ocr_focus_v1")
    assert args.default_province == "皖"
    assert args.subset_bonus == 2
    assert args.non_default_province_bonus == 2
    assert args.targeted_char_bonus == 1
    assert args.tail_letter_bonus == 1
    assert args.max_multiplier == 6


def test_build_focus_multiplier_rewards_targeted_subset_chars_and_non_default_province() -> None:
    multiplier = focus_script.build_focus_multiplier(text="苏AZ7B21", subset="ccpd_challenge")

    assert multiplier == 6


def test_parse_label_lines_extracts_subset_from_exported_train_prefix(tmp_path: Path) -> None:
    label_file = tmp_path / "train.txt"
    label_file.write_text(
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21\n"
        "images/train/train__ccpd_base__plain.jpg\t皖A12345\n",
        encoding="utf-8",
    )

    rows = focus_script.parse_label_lines(label_file)

    assert rows == [
        {"image_path": "images/train/train__ccpd_challenge__focus.jpg", "text": "苏AZ7B21", "subset": "ccpd_challenge"},
        {"image_path": "images/train/train__ccpd_base__plain.jpg", "text": "皖A12345", "subset": "ccpd_base"},
    ]


def test_build_focus_multiplier_keeps_plain_sample_at_base_weight() -> None:
    multiplier = focus_script.build_focus_multiplier(text="皖A12345", subset="ccpd_base")

    assert multiplier == 1


def test_build_focus_multiplier_honors_custom_bonus_weights() -> None:
    multiplier = focus_script.build_focus_multiplier(
        text="苏AZ7B21",
        subset="ccpd_challenge",
        subset_bonus=1,
        non_default_province_bonus=1,
        targeted_char_bonus=1,
        tail_letter_bonus=0,
        max_multiplier=4,
    )

    assert multiplier == 4


def test_build_focused_train_lines_duplicates_rows_by_multiplier() -> None:
    rows = [
        {"image_path": "images/train/plain.jpg", "text": "皖A12345", "subset": "ccpd_base"},
        {"image_path": "images/train/focus.jpg", "text": "苏AZ7B21", "subset": "ccpd_challenge"},
    ]

    lines, summary = focus_script.build_focused_train_lines(rows)

    assert lines.count("images/train/plain.jpg\t皖A12345") == 1
    assert lines.count("images/train/focus.jpg\t苏AZ7B21") == 6
    assert summary["base_rows"] == 2
    assert summary["focused_rows"] == 7
    assert summary["multiplier_histogram"] == {"1": 1, "6": 1}


def test_build_plate_ocr_focus_labels_main_writes_focused_train_and_copies_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "focused"
    dict_path = dataset_root / "dicts" / "plate_dict.txt"
    dict_path.parent.mkdir(parents=True)
    dict_path.write_text("皖\nA\n1\n", encoding="utf-8")
    (dataset_root / "train.txt").write_text(
        "images/train/train__ccpd_base__plain.jpg\t皖A12345\n"
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21\n",
        encoding="utf-8",
    )
    (dataset_root / "val.txt").write_text("images/val/keep.jpg\t皖A99999\n", encoding="utf-8")
    (dataset_root / "test.txt").write_text("images/test/keep.jpg\t皖A88888\n", encoding="utf-8")

    monkeypatch.setattr(
        focus_script,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: Namespace(
                dataset_root=dataset_root,
                    output_root=output_root,
                    default_province="皖",
                    boosted_subsets="ccpd_challenge,ccpd_blur,ccpd_tilt,ccpd_db,ccpd_fn",
                    targeted_chars="BDGHJLQRWZ",
                    subset_bonus=2,
                    non_default_province_bonus=2,
                    targeted_char_bonus=1,
                    tail_letter_bonus=1,
                    max_multiplier=6,
                )
            ),
        )

    exit_code = focus_script.main()

    assert exit_code == 0
    assert (output_root / "train.txt").read_text(encoding="utf-8").splitlines() == [
        "images/train/train__ccpd_base__plain.jpg\t皖A12345",
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21",
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21",
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21",
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21",
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21",
        "images/train/train__ccpd_challenge__focus.jpg\t苏AZ7B21",
    ]
    assert (output_root / "val.txt").read_text(encoding="utf-8") == "images/val/keep.jpg\t皖A99999\n"
    assert (output_root / "test.txt").read_text(encoding="utf-8") == "images/test/keep.jpg\t皖A88888\n"
    assert (output_root / "dicts" / "plate_dict.txt").read_text(encoding="utf-8") == "皖\nA\n1\n"
