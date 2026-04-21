from __future__ import annotations

import argparse
import csv
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
    parser.add_argument(
        "--transition-guidance-csv",
        type=Path,
        action="append",
        default=[],
        help="Audit CSV files used only to derive extra weighting guidance from existing train rows.",
    )
    parser.add_argument("--guidance-subset-bonus", type=int, default=1)
    parser.add_argument("--guidance-char-bonus", type=int, default=1)
    parser.add_argument("--guidance-province-bonus", type=int, default=2)
    parser.add_argument(
        "--guidance-char-source",
        choices=("both", "gt"),
        default="both",
        help="Whether transition guidance boosts both GT/predicted mismatched chars or GT chars only.",
    )
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


def load_transition_guidance_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                rows.append({key: (value or "") for key, value in row.items()})
    return rows


def _infer_guidance_subset(relative_path: str) -> str:
    image_name = Path(relative_path).name
    if "__" in image_name:
        return image_name.split("__", 1)[0]
    return ""


def derive_transition_guidance(
    rows: list[dict[str, str]],
    *,
    default_province: str = "皖",
    guidance_char_source: str = "both",
) -> dict[str, tuple[str, ...]]:
    subsets: set[str] = set()
    targeted_chars: set[str] = set()
    targeted_provinces: set[str] = set()

    for row in rows:
        subset = _infer_guidance_subset(row.get("relative_path", ""))
        if subset:
            subsets.add(subset)

        gt_text = row.get("gt_text", "")
        predicted_text = row.get("conditional_predicted_text", "") or row.get("predicted_text", "")

        if gt_text and gt_text[0] != default_province:
            targeted_provinces.add(gt_text[0])

        max_length = min(len(gt_text), len(predicted_text))
        for index in range(2, max_length):
            gt_char = gt_text[index]
            predicted_char = predicted_text[index]
            if gt_char == predicted_char:
                continue
            if gt_char.strip():
                targeted_chars.add(gt_char)
            if guidance_char_source == "both" and predicted_char.strip():
                targeted_chars.add(predicted_char)

    return {
        "subsets": tuple(sorted(subsets)),
        "targeted_chars": tuple(sorted(targeted_chars)),
        "targeted_provinces": tuple(sorted(targeted_provinces)),
    }


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
    guidance_subsets: tuple[str, ...] = (),
    guidance_chars: tuple[str, ...] = (),
    guidance_provinces: tuple[str, ...] = (),
    guidance_subset_bonus: int = 1,
    guidance_char_bonus: int = 1,
    guidance_province_bonus: int = 2,
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
    if subset in guidance_subsets:
        multiplier += guidance_subset_bonus
    if text and text[0] in guidance_provinces:
        multiplier += guidance_province_bonus
    if any(char in guidance_chars for char in text[2:]):
        multiplier += guidance_char_bonus
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
    guidance_subsets: tuple[str, ...] = (),
    guidance_chars: tuple[str, ...] = (),
    guidance_provinces: tuple[str, ...] = (),
    guidance_subset_bonus: int = 1,
    guidance_char_bonus: int = 1,
    guidance_province_bonus: int = 2,
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
            guidance_subsets=guidance_subsets,
            guidance_chars=guidance_chars,
            guidance_provinces=guidance_provinces,
            guidance_subset_bonus=guidance_subset_bonus,
            guidance_char_bonus=guidance_char_bonus,
            guidance_province_bonus=guidance_province_bonus,
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
    guidance_paths = [Path(path) for path in getattr(args, "transition_guidance_csv", [])]
    guidance_rows = load_transition_guidance_rows(guidance_paths) if guidance_paths else []
    guidance_char_source = getattr(args, "guidance_char_source", "both")
    guidance = derive_transition_guidance(
        guidance_rows,
        default_province=args.default_province,
        guidance_char_source=guidance_char_source,
    )

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
        guidance_subsets=guidance["subsets"],
        guidance_chars=guidance["targeted_chars"],
        guidance_provinces=guidance["targeted_provinces"],
        guidance_subset_bonus=getattr(args, "guidance_subset_bonus", 1),
        guidance_char_bonus=getattr(args, "guidance_char_bonus", 1),
        guidance_province_bonus=getattr(args, "guidance_province_bonus", 2),
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
                "guidance": {
                    "transition_guidance_csv": [path.as_posix() for path in guidance_paths],
                    "subsets": list(guidance["subsets"]),
                    "targeted_chars": list(guidance["targeted_chars"]),
                    "targeted_provinces": list(guidance["targeted_provinces"]),
                    "guidance_subset_bonus": getattr(args, "guidance_subset_bonus", 1),
                    "guidance_char_bonus": getattr(args, "guidance_char_bonus", 1),
                    "guidance_province_bonus": getattr(args, "guidance_province_bonus", 2),
                    "guidance_char_source": guidance_char_source,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
