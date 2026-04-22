import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.config import load_config
from car_system.data.ccpd import decode_ccpd_plate_indices, parse_ccpd_path
from car_system.diagnostics.export import build_match_artifacts, export_frame_diagnostics
from car_system.diagnostics.review_set import build_review_rows, load_review_set
from car_system.io.media import load_image
from car_system.pipeline.runner import PipelineRunner
from car_system.runtime import build_runtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--review-set", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    (
        vehicle_detector,
        plate_detector,
        ocr_engine,
        probe_ocr_engine,
        rescue_probe_ocr_engine,
        secondary_rescue_probe_ocr_engine,
    ) = build_runtime(config)
    review_set = load_review_set(args.review_set)
    runner = PipelineRunner(
        config,
        vehicle_detector,
        plate_detector,
        ocr_engine,
        probe_ocr_engine=probe_ocr_engine,
        rescue_probe_ocr_engine=rescue_probe_ocr_engine,
        secondary_rescue_probe_ocr_engine=secondary_rescue_probe_ocr_engine,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for sample in review_set.samples:
        source_path = review_set.dataset_root / sample.relative_path
        annotation = parse_ccpd_path(sample.relative_path)
        gt_text = decode_ccpd_plate_indices(annotation.plate_indices)
        image = load_image(source_path)
        result = runner.run_frame(image=image, source_name=sample.relative_path.name, frame_index=0)
        crops, rectified_images = build_match_artifacts(result, image)
        export_payload = export_frame_diagnostics(
            output_dir=output_dir / Path(sample.relative_path).stem,
            frame_result=result,
            source_image=image,
            crops=crops,
            rectified_images=rectified_images,
        )
        rows.extend(
            build_review_rows(
                category=sample.category,
                source_name=sample.relative_path.name,
                gt_text=gt_text,
                diagnostics=export_payload["diagnostics"],
            )
        )

    csv_path = output_dir / "review_results.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
