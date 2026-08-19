"""Log one matched Sequential versus QAFD+GraphKV 250-question attempt."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


EXPECTED_QUESTIONS = 250
TARGET_DELTA_F1 = 0.05
CSV_COLUMNS = [
    "timestamp",
    "attempt_number",
    "strategy_name",
    "strategy_description",
    "sequential_em",
    "sequential_f1",
    "sequential_avg_latency",
    "qafd_graphkv_em",
    "qafd_graphkv_f1",
    "qafd_graphkv_avg_latency",
    "delta_em",
    "delta_f1",
    "latency_ratio",
]


def read_summary(path: Path) -> dict:
    rows = json.loads(path.read_text())
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row in {path}, got {len(rows)}")
    row = rows[0]
    if row["questions"] != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Attempt used {row['questions']} questions; expected {EXPECTED_QUESTIONS}"
        )
    return row


def read_predictions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"Prediction file {path} has {len(rows)} rows; expected {EXPECTED_QUESTIONS}"
        )
    return rows


def validate_alignment(sequential: list[dict], qafd_graphkv: list[dict]) -> None:
    expected_qids = list(range(EXPECTED_QUESTIONS))
    sequential_qids = [row["qid"] for row in sequential]
    qafd_qids = [row["qid"] for row in qafd_graphkv]
    if sequential_qids != expected_qids or qafd_qids != expected_qids:
        raise ValueError("Predictions must preserve QID order 0..249")
    for index, (sequential_row, qafd_row) in enumerate(zip(sequential, qafd_graphkv)):
        if sequential_row["question"] != qafd_row["question"]:
            raise ValueError(f"Question mismatch at QID {index}")
        if sequential_row["answers"] != qafd_row["answers"]:
            raise ValueError(f"Answer metadata mismatch at QID {index}")


def existing_attempts(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {int(row["attempt_number"]) for row in csv.DictReader(handle)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential-summary", required=True, type=Path)
    parser.add_argument("--qafd-summary", required=True, type=Path)
    parser.add_argument("--sequential-predictions", required=True, type=Path)
    parser.add_argument("--qafd-predictions", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--attempt-number", required=True, type=int)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--changes", required=True)
    parser.add_argument("--matched-config", required=True)
    parser.add_argument("--hyperparameters", required=True)
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--next-experiment", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attempt_number in existing_attempts(args.csv):
        raise ValueError(f"Attempt {args.attempt_number} is already logged")

    sequential = read_summary(args.sequential_summary)
    qafd = read_summary(args.qafd_summary)
    validate_alignment(
        read_predictions(args.sequential_predictions),
        read_predictions(args.qafd_predictions),
    )

    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    delta_em = qafd["em"] - sequential["em"]
    delta_f1 = qafd["f1"] - sequential["f1"]
    latency_ratio = qafd["avg_seconds"] / sequential["avg_seconds"]
    row = {
        "timestamp": timestamp,
        "attempt_number": args.attempt_number,
        "strategy_name": args.strategy_name,
        "strategy_description": args.description,
        "sequential_em": f"{sequential['em']:.6f}",
        "sequential_f1": f"{sequential['f1']:.6f}",
        "sequential_avg_latency": f"{sequential['avg_seconds']:.6f}",
        "qafd_graphkv_em": f"{qafd['em']:.6f}",
        "qafd_graphkv_f1": f"{qafd['f1']:.6f}",
        "qafd_graphkv_avg_latency": f"{qafd['avg_seconds']:.6f}",
        "delta_em": f"{delta_em:+.6f}",
        "delta_f1": f"{delta_f1:+.6f}",
        "latency_ratio": f"{latency_ratio:.6f}",
    }
    write_header = not args.csv.exists() or args.csv.stat().st_size == 0
    with args.csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    outcome = (
        "success; validate with a reproducibility run"
        if delta_f1 >= TARGET_DELTA_F1
        else "target not met"
    )
    entry = f"""
## Attempt {args.attempt_number} — {args.strategy_name}

**Timestamp:** {timestamp}  
**Hypothesis:** {args.hypothesis}  
**Changes:** {args.changes}  
**Matched Sequential configuration:** {args.matched_config}  
**Important hyperparameters:** {args.hyperparameters}  

### Results

| Method | EM | F1 | Avg Latency |
|---|---:|---:|---:|
| Sequential | {sequential['em']:.6f} | {sequential['f1']:.6f} | {sequential['avg_seconds']:.6f} s |
| QAFD + GraphKV | {qafd['em']:.6f} | {qafd['f1']:.6f} | {qafd['avg_seconds']:.6f} s |

**Delta EM:** {delta_em:+.6f}  
**Delta F1:** {delta_f1:+.6f}  
**Latency ratio:** {latency_ratio:.6f}x  
**Outcome:** {outcome}.  

### Interpretation

{args.interpretation}

### Next Experiment

{args.next_experiment}
"""
    with args.markdown.open("a") as handle:
        handle.write(entry)


if __name__ == "__main__":
    main()
