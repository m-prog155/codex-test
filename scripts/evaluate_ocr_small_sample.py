from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.data.ccpd import load_split_entries, parse_ccpd_path
from car_system.experiments.ocr_small_sample import (
    BaselinePlateOCR,
    build_summary,
    evaluate_sample,
    sample_entries_by_subset,
)
from car_system.io.writers import ensure_output_dir, write_csv, write_json
from car_system.ocr.plate_ocr import PaddlePlateOCR


DEFAULT_DATASET_ROOT = Path("D:/plate_project/CCPD2019")
DEFAULT_SPLIT_FILE = DEFAULT_DATASET_ROOT / "splits" / "test.txt"
DEFAULT_SUBSETS = [
    "ccpd_base",
    "ccpd_blur",
    "ccpd_db",
    "ccpd_rotate",
    "ccpd_tilt",
    "ccpd_weather",
    "ccpd_challenge",
]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ocr_small_sample_eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a small CCPD OCR sample.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--per-subset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ocr-mode", choices=["generic", "specialized"], default="generic")
    parser.add_argument("--ocr-model-dir", type=Path, default=None)
    parser.add_argument("--ocr-dict-path", type=Path, default=None)
    return parser


def _entry_relative_path(entry: Any) -> str:
    if isinstance(entry, Path):
        return entry.as_posix()
    return str(entry)


def _validate_ocr_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.ocr_mode == "generic":
        if args.ocr_model_dir is not None or args.ocr_dict_path is not None:
            parser.error("--ocr-model-dir and --ocr-dict-path require --ocr-mode specialized.")
        return

    if args.ocr_model_dir is None:
        parser.error("--ocr-model-dir is required when --ocr-mode specialized is set.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _validate_ocr_args(parser, args)

    split_entries = load_split_entries(args.split_file)
    sampled_entries = sample_entries_by_subset(
        split_entries,
        args.subsets,
        per_subset=args.per_subset,
        seed=args.seed,
    )

    stabilized_ocr_kwargs: dict[str, Any] = {
        "language": "ch",
        "use_angle_cls": False,
        "mode": args.ocr_mode,
    }
    if args.ocr_model_dir is not None:
        stabilized_ocr_kwargs["model_dir"] = str(args.ocr_model_dir)
    if args.ocr_dict_path is not None:
        stabilized_ocr_kwargs["character_dict_path"] = str(args.ocr_dict_path)

    stabilized_ocr = PaddlePlateOCR(**stabilized_ocr_kwargs)
    baseline_ocr = BaselinePlateOCR(stabilized_ocr)

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for entry in sampled_entries:
        relative_path = _entry_relative_path(entry)
        try:
            annotation = parse_ccpd_path(entry)
        except ValueError as exc:
            skipped.append({"relative_path": relative_path, "reason": str(exc)})
            continue

        try:
            row = evaluate_sample(
                dataset_root=args.dataset_root,
                relative_path=annotation.relative_path,
                annotation=annotation,
                baseline_ocr=baseline_ocr,
                stabilized_ocr=stabilized_ocr,
            )
        except FileNotFoundError as exc:
            skipped.append({"relative_path": relative_path, "reason": str(exc)})
            continue

        rows.append(row)

    summary = build_summary(
        rows=rows,
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        subsets=args.subsets,
        per_subset=args.per_subset,
        seed=args.seed,
        skipped=skipped,
    )

    output_dir = ensure_output_dir(args.output_dir)
    samples_csv = output_dir / "samples.csv"
    summary_json = output_dir / "summary.json"

    write_csv(samples_csv, rows)
    write_json(summary_json, summary)

    print(f"Samples CSV: {samples_csv}")
    print(f"Summary JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
