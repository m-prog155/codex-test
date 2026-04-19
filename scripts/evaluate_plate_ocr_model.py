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
from car_system.experiments.ocr_small_sample import BaselinePlateOCR, build_summary, evaluate_sample
from car_system.io.writers import ensure_output_dir, write_csv, write_json
from car_system.ocr.plate_ocr import PaddlePlateOCR


DEFAULT_DATASET_ROOT = Path("D:/plate_project/CCPD2019")
DEFAULT_SPLIT_FILE = DEFAULT_DATASET_ROOT / "splits" / "test.txt"
DEFAULT_OUTPUT_DIR = Path("outputs/plate_ocr_eval")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generic vs specialized plate OCR on CCPD.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--generic-model", type=Path, default=None)
    parser.add_argument("--specialized-model", type=Path, required=True)
    parser.add_argument("--dict-path", type=Path, required=True)
    parser.set_defaults(use_full_text=True)
    parser.add_argument("--use-eval-text", action="store_false", dest="use_full_text")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _entry_relative_path(entry: Any) -> str:
    if isinstance(entry, Path):
        return entry.as_posix()
    return str(entry)


def _build_generic_ocr(args: argparse.Namespace) -> PaddlePlateOCR:
    if args.generic_model is None:
        return PaddlePlateOCR(language="ch", use_angle_cls=False)

    return PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir=str(args.generic_model),
    )


def main() -> int:
    args = build_parser().parse_args()

    generic_ocr = _build_generic_ocr(args)
    stabilized_ocr = PaddlePlateOCR(
        language="ch",
        use_angle_cls=False,
        mode="specialized",
        model_dir=str(args.specialized_model),
        character_dict_path=str(args.dict_path),
    )
    baseline_ocr = BaselinePlateOCR(backend=generic_ocr)

    split_entries = load_split_entries(args.split_file)
    if args.limit is not None:
        split_entries = split_entries[: args.limit]

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for entry in split_entries:
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
                use_full_text=args.use_full_text,
            )
        except FileNotFoundError as exc:
            skipped.append({"relative_path": relative_path, "reason": str(exc)})
            continue

        rows.append(row)

    summary = build_summary(
        rows=rows,
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        subsets=[],
        per_subset=0,
        seed=0,
        skipped=skipped,
    )

    output_dir = ensure_output_dir(args.output_dir)
    samples_csv = output_dir / "samples.csv"
    summary_json = output_dir / "summary.json"

    write_csv(samples_csv, rows)
    write_json(summary_json, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
