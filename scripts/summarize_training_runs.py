import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.experiments.training_summary import summarize_training_runs, write_training_summaries_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize training runs from Ultralytics results.csv files.")
    parser.add_argument("--input", required=True, help="Directory containing training run subdirectories.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summaries = summarize_training_runs(args.input)
    output_path = Path(args.output) if args.output else Path(args.input) / "training_summaries.csv"
    write_training_summaries_csv(output_path, summaries)
    print(f"Training summary CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
