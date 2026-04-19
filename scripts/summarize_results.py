import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.experiments.summary import build_file_summaries, write_file_summaries_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize experiment outputs.")
    parser.add_argument("--input", required=True, help="Directory containing result files.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summaries = build_file_summaries(args.input)
    output_path = Path(args.output) if args.output else Path(args.input) / "file_summaries.csv"
    write_file_summaries_csv(output_path, summaries)
    print(f"File summary CSV: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
