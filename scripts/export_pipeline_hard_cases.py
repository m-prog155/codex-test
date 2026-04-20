from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.experiments.pipeline_audit import (
    build_hard_case_summary,
    build_sample_path_list,
    copy_audit_sample_images,
    filter_audit_rows,
)
from car_system.io.writers import ensure_output_dir, write_csv, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a filtered hard-case subset from an audit rows.csv file."
    )
    parser.add_argument("--rows-csv", required=True, help="Audit rows.csv produced by the pipeline audit.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the hard-case export.")
    parser.add_argument(
        "--status",
        action="append",
        choices=["exact", "wrong", "null"],
        help="Statuses to keep. Can be provided multiple times. Defaults to wrong only.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of rows to export.")
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Optional dataset root. When set, matching source images are copied into output_dir/images.",
    )
    return parser


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = build_parser().parse_args()

    rows = _load_rows(args.rows_csv)
    selected = filter_audit_rows(rows, statuses=args.status or ["wrong"], limit=args.limit)
    output_dir = ensure_output_dir(args.output_dir)

    write_csv(output_dir / "selected_rows.csv", selected)
    (output_dir / "sample_paths.txt").write_text(
        "\n".join(build_sample_path_list(selected)) + ("\n" if selected else ""),
        encoding="utf-8",
    )
    write_json(output_dir / "summary.json", build_hard_case_summary(selected))

    if args.dataset_root:
        written = copy_audit_sample_images(
            rows=selected,
            dataset_root=args.dataset_root,
            export_root=output_dir,
        )
        print(f"Copied images: {len(written)}")

    print(f"Selected rows CSV: {output_dir / 'selected_rows.csv'}")
    print(f"Sample path list: {output_dir / 'sample_paths.txt'}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
