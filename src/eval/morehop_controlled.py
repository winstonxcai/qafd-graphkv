"""Freeze and aggregate a matched MoreHopQA topology comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.eval.graphkv_faithful_morehop import load_rows, sha256
from src.eval.morehop_scoring import score_both


METHODS = (
    "sequential",
    "block_rag",
    "graphkv_top1",
    "graphkv_top2",
    "graphkv_top3",
    "graphkv_full",
)


def proportional_allocation(counts: dict[int, int], total: int) -> dict[int, int]:
    population = sum(counts.values())
    exact = {key: total * value / population for key, value in counts.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def select_questions(rows: list[dict], total: int, seed: str) -> list[dict]:
    counts = Counter(int(row["no_of_hops"]) for row in rows)
    allocation = proportional_allocation(dict(counts), total)
    selected_ids = set()
    for hop, amount in sorted(allocation.items()):
        candidates = [row for row in rows if int(row["no_of_hops"]) == hop]
        candidates.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['_id']}".encode("utf-8")
            ).hexdigest()
        )
        selected_ids.update(row["_id"] for row in candidates[:amount])
    return [row for row in rows if row["_id"] in selected_ids]


def write_manifest(dataset: Path, output: Path, total: int, seed: str) -> None:
    rows = load_rows(dataset)
    if total <= 0 or total > len(rows):
        raise ValueError("question count must be within the dataset size")
    selected = select_questions(rows, total, seed)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256(dataset),
        "selection": "proportional-by-hop SHA-256 ordering",
        "selection_seed": seed,
        "question_count": len(selected),
        "hop_counts": dict(
            sorted(Counter(int(row["no_of_hops"]) for row in selected).items())
        ),
        "selected_question_ids": [row["_id"] for row in selected],
        "frozen_contract": {
            "model": "ldsjmdy/Tulu3-Block-FT",
            "model_revision": "62eecbdfa4e12821a487e91ed2ef847a85426efe",
            "graphkv_revision": "ba01bbcc98fc9a352582dd1e75ab88fa6cac0dfb",
            "passages": "released order; all released documents retained",
            "prompt": "upstream MoreHopQA Tulu prompt",
            "max_new_tokens": 256,
            "decoding": "greedy; EOS termination",
            "primary_replication_score": "public whole-response-fallback normalized containment",
            "integrity_score": "explicit-final-answer normalized containment",
            "methods": list(METHODS),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def paired_bootstrap(
    candidate: list[float], baseline: list[float], seed: int, samples: int = 20_000
) -> tuple[float, float]:
    rng = random.Random(seed)
    differences = [left - right for left, right in zip(candidate, baseline, strict=True)]
    count = len(differences)
    draws = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def load_method_rows(root: Path, method: str, expected_ids: list[str]) -> list[dict]:
    path = root / method / f"{method}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    actual_ids = [row["_id"] for row in rows]
    if actual_ids != expected_ids:
        raise ValueError(f"{method} question IDs/order do not match the frozen manifest")
    return rows


def aggregate(
    root: Path,
    manifest_path: Path,
    *,
    noncomparable_latency_methods: set[str] | None = None,
    exploratory_methods: set[str] | None = None,
    execution_note: str | None = None,
) -> None:
    noncomparable_latency_methods = noncomparable_latency_methods or set()
    exploratory_methods = exploratory_methods or set()
    manifest = json.loads(manifest_path.read_text())
    expected_ids = manifest["selected_question_ids"]
    method_rows = {
        method: load_method_rows(root, method, expected_ids) for method in METHODS
    }
    for offset, question_id in enumerate(expected_ids):
        hashes = {
            method_rows[method][offset]["prompt_hash"] for method in METHODS
        }
        if len(hashes) != 1:
            raise ValueError(f"prompt mismatch for {question_id}")

    scored: dict[str, list[dict]] = {}
    for method, rows in method_rows.items():
        scored[method] = [
            score_both(row["generated"], row["answers"]) for row in rows
        ]

    baseline_paper = [
        row["paper_compat_accuracy"] for row in scored["sequential"]
    ]
    baseline_strict = [row["strict_final_accuracy"] for row in scored["sequential"]]
    summaries = []
    for method in METHODS:
        paper = [row["paper_compat_accuracy"] for row in scored[method]]
        strict = [row["strict_final_accuracy"] for row in scored[method]]
        paper_ci = paired_bootstrap(paper, baseline_paper, 20260909)
        strict_ci = paired_bootstrap(strict, baseline_strict, 20260910)
        summaries.append(
            {
                "method": method,
                "analysis_role": (
                    "exploratory" if method in exploratory_methods else "preregistered control"
                ),
                "questions": len(expected_ids),
                "paper_compat_accuracy": sum(paper) / len(paper),
                "paper_delta_vs_sequential": sum(
                    left - right
                    for left, right in zip(paper, baseline_paper, strict=True)
                )
                / len(paper),
                "paper_delta_ci_low": paper_ci[0],
                "paper_delta_ci_high": paper_ci[1],
                "strict_final_accuracy": sum(strict) / len(strict),
                "strict_delta_vs_sequential": sum(
                    left - right
                    for left, right in zip(strict, baseline_strict, strict=True)
                )
                / len(strict),
                "strict_delta_ci_low": strict_ci[0],
                "strict_delta_ci_high": strict_ci[1],
                "explicit_answer_rate": sum(
                    row["has_explicit_answer"] for row in scored[method]
                )
                / len(scored[method]),
                "avg_latency_seconds": sum(
                    float(row["latency_seconds"]) for row in method_rows[method]
                )
                / len(expected_ids),
                "latency_comparable": method not in noncomparable_latency_methods,
            }
        )

    fields = list(summaries[0])
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (root / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Matched MoreHopQA topology comparison",
        "",
        f"Frozen manifest: `{manifest_path}` (`{sha256(manifest_path)}`). All methods used the same {len(expected_ids)} questions, prompt hashes, document order, model revision, greedy decoding, and 256-token cap.",
        "",
        "The paper-compatible and strict-final metrics are separate analyses. They are never combined into one table cell.",
        "",
        *( [execution_note, ""] if execution_note else [] ),
        "| Method | Role | Paper-compatible accuracy | Δ vs Sequential (95% CI) | Strict-final accuracy | Δ vs Sequential (95% CI) | Explicit `Answer:` | Avg latency (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['analysis_role']} | {row['paper_compat_accuracy']:.3f} | {row['paper_delta_vs_sequential']:+.3f} [{row['paper_delta_ci_low']:+.3f}, {row['paper_delta_ci_high']:+.3f}] | {row['strict_final_accuracy']:.3f} | {row['strict_delta_vs_sequential']:+.3f} [{row['strict_delta_ci_low']:+.3f}, {row['strict_delta_ci_high']:+.3f}] | {row['explicit_answer_rate']:.3f} | {row['avg_latency_seconds']:.3f}{'' if row['latency_comparable'] else '*'} |"
        )
    lines.extend(
        [
            "",
            "This 100-question calibration run is an implementation/topology check, not a confirmation result. Confidence intervals are paired question-level bootstrap intervals with 20,000 resamples.",
            "",
            "`*` Mixed-hardware timing is recorded for completeness but excluded from latency comparisons.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--dataset", required=True, type=Path)
    manifest_parser.add_argument("--output", required=True, type=Path)
    manifest_parser.add_argument("--questions", type=int, default=100)
    manifest_parser.add_argument("--seed", default="20260909:morehop-controlled-100")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--root", required=True, type=Path)
    aggregate_parser.add_argument("--manifest", required=True, type=Path)
    aggregate_parser.add_argument(
        "--noncomparable-latency-method", action="append", default=[]
    )
    aggregate_parser.add_argument("--exploratory-method", action="append", default=[])
    aggregate_parser.add_argument("--execution-note")
    args = parser.parse_args()
    if args.command == "manifest":
        write_manifest(args.dataset, args.output, args.questions, args.seed)
    else:
        aggregate(
            args.root,
            args.manifest,
            noncomparable_latency_methods=set(args.noncomparable_latency_method),
            exploratory_methods=set(args.exploratory_method),
            execution_note=args.execution_note,
        )


if __name__ == "__main__":
    main()
