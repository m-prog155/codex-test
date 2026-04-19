import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.experiments.summary import build_directory_summary, write_summary_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate saved inference outputs.")
    parser.add_argument("--input", required=True, help="Directory containing inference results.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = build_directory_summary(args.input)
    output_path = Path(args.output) if args.output else Path(args.input) / "summary.json"
    write_summary_json(output_path, summary)
    print(f"Summary JSON: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
