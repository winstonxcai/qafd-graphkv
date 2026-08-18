"""Append one completed 250-question experiment to the Markdown/CSV ledger."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


BASELINE_EM = 0.732
TARGET_EM = 0.782
EXPECTED_QUESTIONS = 250
CSV_COLUMNS = [
    "current_time",
    "attempt_number",
    "strategy_name",
    "short_strategy_description",
    "EM",
    "F1",
    "avg_latency",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--attempt-number", required=True, type=int)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--description", required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict:
    rows = json.loads(path.read_text())
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one summary row in {path}, got {len(rows)}")
    row = rows[0]
    if row["questions"] != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Attempt used {row['questions']} questions; expected {EXPECTED_QUESTIONS}"
        )
    return row


def existing_attempts(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row["attempt_number"]) for row in csv.DictReader(handle)}


def main() -> None:
    args = parse_args()
    if args.attempt_number in existing_attempts(args.csv):
        raise ValueError(f"Attempt {args.attempt_number} is already logged")

    summary = read_summary(args.summary)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    csv_row = {
        "current_time": timestamp,
        "attempt_number": args.attempt_number,
        "strategy_name": args.strategy_name,
        "short_strategy_description": args.description,
        "EM": f"{summary['em']:.6f}",
        "F1": f"{summary['f1']:.6f}",
        "avg_latency": f"{summary['avg_seconds']:.6f}",
    }

    write_header = not args.csv.exists() or args.csv.stat().st_size == 0
    with args.csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    hits = round(summary["em"] * EXPECTED_QUESTIONS)
    delta = summary["em"] - BASELINE_EM
    outcome = (
        "target met"
        if summary["em"] >= TARGET_EM
        else "target not met; continue experimentation"
    )
    entry = f"""
### Attempt {args.attempt_number} — {args.strategy_name}

- Completed: {timestamp}
- Strategy: {args.description}
- Result: EM `{summary['em']:.6f}` ({hits}/{EXPECTED_QUESTIONS}), F1 `{summary['f1']:.6f}`, average latency `{summary['avg_seconds']:.6f} s`.
- Delta from sequential: `{delta:+.6f}` EM.
- Outcome: {outcome}.
"""
    with args.markdown.open("a") as handle:
        handle.write(entry)


if __name__ == "__main__":
    main()
