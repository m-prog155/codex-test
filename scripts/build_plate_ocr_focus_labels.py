from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("outputs/plate_ocr_dataset")
DEFAULT_OUTPUT_ROOT = Path("outputs/plate_ocr_focus_v1")
DEFAULT_BOOSTED_SUBSETS = ("ccpd_challenge", "ccpd_blur", "ccpd_tilt", "ccpd_db", "ccpd_fn")
DEFAULT_TARGETED_CHARS = tuple("BDGHJLQRWZ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a focused OCR training label set from the base OCR dataset.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--default-province", default="皖")
    parser.add_argument("--boosted-subsets", default=",".join(DEFAULT_BOOSTED_SUBSETS))
    parser.add_argument("--targeted-chars", default="".join(DEFAULT_TARGETED_CHARS))
    parser.add_argument("--subset-bonus", type=int, default=2)
    parser.add_argument("--non-default-province-bonus", type=int, default=2)
    parser.add_argument("--targeted-char-bonus", type=int, default=1)
    parser.add_argument("--tail-letter-bonus", type=int, default=1)
    parser.add_argument("--max-multiplier", type=int, default=6)
    return parser


def parse_label_lines(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        image_path, text = stripped.split("\t", 1)
        image_name = Path(image_path).name
        parts = image_name.split("__")
        if len(parts) >= 3 and parts[0] in {"train", "val", "test"}:
            subset = parts[1]
        elif "__" in image_name:
            subset = parts[0]
        else:
            subset = ""
        rows.append({"image_path": image_path, "text": text, "subset": subset})
    return rows


def build_focus_multiplier(
    *,
    text: str,
    subset: str,
    default_province: str = "皖",
    boosted_subsets: tuple[str, ...] = DEFAULT_BOOSTED_SUBSETS,
    targeted_chars: tuple[str, ...] = DEFAULT_TARGETED_CHARS,
    subset_bonus: int = 2,
    non_default_province_bonus: int = 2,
    targeted_char_bonus: int = 1,
    tail_letter_bonus: int = 1,
    max_multiplier: int = 6,
) -> int:
    multiplier = 1
    if subset in boosted_subsets:
        multiplier += subset_bonus
    if text and text[0] != default_province:
        multiplier += non_default_province_bonus
    if any(char in targeted_chars for char in text[2:]):
        multiplier += targeted_char_bonus
    if any(char.isascii() and char.isalpha() for char in text[-2:]):
        multiplier += tail_letter_bonus
    return min(max_multiplier, multiplier)


def build_focused_train_lines(
    rows: list[dict[str, str]],
    *,
    default_province: str = "皖",
    boosted_subsets: tuple[str, ...] = DEFAULT_BOOSTED_SUBSETS,
    targeted_chars: tuple[str, ...] = DEFAULT_TARGETED_CHARS,
    subset_bonus: int = 2,
    non_default_province_bonus: int = 2,
    targeted_char_bonus: int = 1,
    tail_letter_bonus: int = 1,
    max_multiplier: int = 6,
) -> tuple[list[str], dict[str, object]]:
    lines: list[str] = []
    histogram: dict[str, int] = {}
    for row in rows:
        multiplier = build_focus_multiplier(
            text=row["text"],
            subset=row["subset"],
            default_province=default_province,
            boosted_subsets=boosted_subsets,
            targeted_chars=targeted_chars,
            subset_bonus=subset_bonus,
            non_default_province_bonus=non_default_province_bonus,
            targeted_char_bonus=targeted_char_bonus,
            tail_letter_bonus=tail_letter_bonus,
            max_multiplier=max_multiplier,
        )
        histogram[str(multiplier)] = histogram.get(str(multiplier), 0) + 1
        line = f"{row['image_path']}\t{row['text']}"
        lines.extend([line] * multiplier)

    return lines, {
        "base_rows": len(rows),
        "focused_rows": len(lines),
        "multiplier_histogram": histogram,
    }


def _write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    boosted_subsets = tuple(item.strip() for item in args.boosted_subsets.split(",") if item.strip())
    targeted_chars = tuple(char for char in args.targeted_chars if char.strip())

    train_rows = parse_label_lines(dataset_root / "train.txt")
    focused_train_lines, summary = build_focused_train_lines(
        train_rows,
        default_province=args.default_province,
        boosted_subsets=boosted_subsets,
        targeted_chars=targeted_chars,
        subset_bonus=args.subset_bonus,
        non_default_province_bonus=args.non_default_province_bonus,
        targeted_char_bonus=args.targeted_char_bonus,
        tail_letter_bonus=args.tail_letter_bonus,
        max_multiplier=args.max_multiplier,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_text(output_root / "train.txt", focused_train_lines)
    shutil.copy2(dataset_root / "val.txt", output_root / "val.txt")
    shutil.copy2(dataset_root / "test.txt", output_root / "test.txt")
    dict_output = output_root / "dicts" / "plate_dict.txt"
    dict_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_root / "dicts" / "plate_dict.txt", dict_output)
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                **summary,
                "default_province": args.default_province,
                "boosted_subsets": list(boosted_subsets),
                "targeted_chars": list(targeted_chars),
                "subset_bonus": args.subset_bonus,
                "non_default_province_bonus": args.non_default_province_bonus,
                "targeted_char_bonus": args.targeted_char_bonus,
                "tail_letter_bonus": args.tail_letter_bonus,
                "max_multiplier": args.max_multiplier,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
