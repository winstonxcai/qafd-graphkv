"""Combine aligned benchmark predictions into a multi-view answer artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.test4_benchmark import score_prediction


def combine_rows(rows_by_source: list[list[dict]]) -> list[dict]:
    if not rows_by_source:
        raise ValueError("At least one prediction source is required")
    row_count = len(rows_by_source[0])
    if any(len(rows) != row_count for rows in rows_by_source):
        raise ValueError("Prediction sources have different row counts")

    combined = []
    for index in range(row_count):
        source_rows = [rows[index] for rows in rows_by_source]
        identity = (source_rows[0]["qid"], source_rows[0]["question"])
        if any((row["qid"], row["question"]) != identity for row in source_rows[1:]):
            raise ValueError(f"Prediction sources are misaligned at row {index}")
        generated = "\n\nAlternative candidate:\n".join(
            row["generated"] for row in source_rows
        )
        em, f1 = score_prediction(generated, source_rows[0]["answers"])
        combined.append(
            {
                "qid": identity[0],
                "question": identity[1],
                "answers": source_rows[0]["answers"],
                "generated": generated,
                "em": em,
                "f1": f1,
                "seconds": sum(row["seconds"] for row in source_rows),
            }
        )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strategy-name", required=True)
    args = parser.parse_args()

    rows_by_source = [
        [json.loads(line) for line in path.read_text().splitlines()]
        for path in args.source
    ]
    combined = combine_rows(rows_by_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.strategy_name}.jsonl"
    with output_path.open("w") as handle:
        for row in combined:
            handle.write(json.dumps(row) + "\n")

    questions = len(combined)
    total_seconds = sum(row["seconds"] for row in combined)
    summary = [
        {
            "method": args.strategy_name,
            "questions": questions,
            "em": sum(row["em"] for row in combined) / questions,
            "f1": sum(row["f1"] for row in combined) / questions,
            "avg_seconds": total_seconds / questions,
            "total_seconds": total_seconds,
        }
    ]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
