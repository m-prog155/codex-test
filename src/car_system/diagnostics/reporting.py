from collections import Counter, defaultdict
from typing import Any


def _char_accuracy(expected: str, predicted: str) -> tuple[int, int]:
    total = len(expected)
    correct = sum(
        1 for index, char in enumerate(expected) if index < len(predicted) and predicted[index] == char
    )
    return correct, total


def build_report_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    status_counts = Counter(row.get("diagnostic_status", "missing") for row in rows)
    exact_matches = 0
    null_count = 0
    char_correct = 0
    char_total = 0
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"samples": 0})

    for row in rows:
        category = str(row.get("category", "uncategorized"))
        status = str(row.get("diagnostic_status", "missing"))
        gt_text = str(row.get("gt_text", ""))
        predicted = str(row.get("ocr_normalized_text", "") or "")

        category_bucket = by_category[category]
        category_bucket["samples"] += 1
        category_bucket[status] = category_bucket.get(status, 0) + 1

        if not predicted:
            null_count += 1
        if predicted == gt_text:
            exact_matches += 1

        correct, total_chars = _char_accuracy(gt_text, predicted)
        char_correct += correct
        char_total += total_chars

    return {
        "total_samples": total,
        "status_counts": dict(status_counts),
        "exact_plate_accuracy": exact_matches / total if total else 0.0,
        "char_accuracy": char_correct / char_total if char_total else 0.0,
        "null_rate": null_count / total if total else 0.0,
        "by_category": dict(by_category),
    }


def select_failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in rows:
        predicted = str(row.get("ocr_normalized_text", "") or "")
        gt_text = str(row.get("gt_text", ""))
        status = str(row.get("diagnostic_status", "missing"))
        if status != "recognized" or predicted != gt_text:
            failures.append(row)
    return failures


def render_html_report(summary: dict[str, Any], failures: list[dict[str, Any]]) -> str:
    failure_rows = "".join(
        (
            "<tr>"
            f"<td>{item['source_name']}</td>"
            f"<td>{item['diagnostic_status']}</td>"
            f"<td>{item['gt_text']}</td>"
            f"<td>{item['ocr_normalized_text']}</td>"
            f"<td>{item.get('ocr_confidence', '')}</td>"
            "</tr>"
        )
        for item in failures
    )
    return f"""
    <html>
      <body>
        <h1>Internal Analysis Report</h1>
        <p>Total samples: {summary['total_samples']}</p>
        <p>Exact plate accuracy: {summary['exact_plate_accuracy']:.4f}</p>
        <p>Character accuracy: {summary['char_accuracy']:.4f}</p>
        <p>Null rate: {summary['null_rate']:.4f}</p>
        <h2>Failures</h2>
        <table>
          <tr>
            <th>Source</th>
            <th>Status</th>
            <th>Ground truth</th>
            <th>OCR text</th>
            <th>OCR confidence</th>
          </tr>
          {failure_rows}
        </table>
      </body>
    </html>
    """
