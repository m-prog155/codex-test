import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from car_system.diagnostics.reporting import build_report_summary, render_html_report, select_failure_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    with Path(args.input_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = build_report_summary(rows)
    failures = select_failure_rows(rows)
    html = render_html_report(summary, failures)
    Path(args.output_html).write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
