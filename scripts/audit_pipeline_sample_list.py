from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.config import load_config, resolve_config_path
from car_system.data.ccpd import decode_ccpd_plate_indices, parse_ccpd_path
from car_system.experiments.pipeline_audit import (
    build_sample_audit_row,
    build_sample_audit_summary,
    pick_best_recognized_match,
)
from car_system.io.media import load_image
from car_system.io.writers import ensure_output_dir, write_csv, write_json
from car_system.pipeline.runner import PipelineRunner
from car_system.runtime import build_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit one pipeline config against a sample-path list.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing the sample images.")
    parser.add_argument("--sample-list", required=True, help="Text file containing relative sample paths.")
    parser.add_argument("--output-dir", required=True, help="Directory to write rows.csv and summary.json.")
    return parser


def _load_sample_paths(path: str | Path) -> list[Path]:
    sample_list_path = Path(path)
    return [Path(line.strip()) for line in sample_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _subset_name(relative_path: Path) -> str:
    if not relative_path.parts:
        return ""
    if len(relative_path.parts) == 1:
        return relative_path.parent.as_posix()
    return relative_path.parts[0]


def main() -> int:
    args = build_parser().parse_args()

    config = load_config(resolve_config_path(args.config))
    vehicle_detector, plate_detector, ocr_engine, probe_ocr_engine = build_runtime(config)
    runner = PipelineRunner(config, vehicle_detector, plate_detector, ocr_engine, probe_ocr_engine=probe_ocr_engine)
    dataset_root = Path(args.dataset_root)
    sample_paths = _load_sample_paths(args.sample_list)

    rows: list[dict[str, Any]] = []
    for relative_path in sample_paths:
        annotation = parse_ccpd_path(relative_path)
        gt_text = decode_ccpd_plate_indices(annotation.plate_indices)
        image = load_image(dataset_root / relative_path)
        result = runner.run_frame(image=image, source_name=relative_path.name, frame_index=0)
        best_match = pick_best_recognized_match(result.matches)
        rows.append(build_sample_audit_row(relative_path=relative_path, gt_text=gt_text, best_match=best_match))

    output_dir = ensure_output_dir(args.output_dir)
    write_csv(output_dir / "rows.csv", rows)
    write_json(output_dir / "summary.json", build_sample_audit_summary(rows))
    print(f"Rows CSV: {output_dir / 'rows.csv'}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
