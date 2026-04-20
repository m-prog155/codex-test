import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.config import load_config, resolve_config_path
from car_system.pipeline.processing import process_image_file
from car_system.runtime import build_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the image inference pipeline.")
    parser.add_argument("--config", default=None, help="Path to the YAML config file. Defaults to CAR_SYSTEM_CONFIG or configs/default.yaml.")
    parser.add_argument("--source", required=True, help="Path to an input image.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(resolve_config_path(args.config))
    vehicle_detector, plate_detector, ocr_engine, probe_ocr_engine = build_runtime(config)
    artifacts = process_image_file(
        config=config,
        vehicle_detector=vehicle_detector,
        plate_detector=plate_detector,
        ocr_engine=ocr_engine,
        probe_ocr_engine=probe_ocr_engine,
        source_path=Path(args.source),
        output_dir=args.output_dir,
    )

    print(f"Image processed: {args.source}")
    print(f"JSON: {artifacts['json_path']}")
    print(f"CSV: {artifacts['csv_path']}")
    print(f"Annotated image: {artifacts['image_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
