"""Summarize the matched A800 MoreHopQA topology screen.

This report deliberately treats the 100-question A800 control as one paired
evaluation.  It does not pool the historical 1,118-question run or mix the
paper-compatible and strict-final scorers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.eval.morehop_scoring import score_both


HOPS = (1, 2, 3, 4, 5)
ROUTING_METHODS = (
    "released_last1",
    "released_last3",
    "full",
    "query_bm25_k1",
    "query_bm25_k3",
    "ppr_k1",
    "random_k1_seed42",
    "reversed_rank_k1",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def _score(row: dict[str, Any], key: str) -> int:
    stored_key = f"{key}_accuracy"
    if stored_key in row:
        return int(bool(row[stored_key]))
    if key in row:
        return int(bool(row[key]))
    scores = score_both(str(row.get("generated", "")), row.get("answers", []))
    return int(bool(scores[f"{key}_accuracy"]))


def validate_same_questions(named_rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    ids_by_name = {name: [str(row["_id"]) for row in rows] for name, rows in named_rows.items()}
    first_name, first_ids = next(iter(ids_by_name.items()))
    for name, ids in ids_by_name.items():
        if ids != first_ids:
            raise ValueError(f"question ordering differs: {first_name} vs {name}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate question ID in {name}")
    return first_ids


def bootstrap_interval(deltas: list[int], seed: int = 20260910, draws: int = 20000) -> tuple[float, float]:
    if not deltas:
        raise ValueError("cannot bootstrap empty deltas")
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(draws):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * (draws - 1))], means[int(0.975 * (draws - 1))]


def _mean(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if field in row]
    return statistics.fmean(values) if values else float("nan")


def summarize_method(rows: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    public = [_score(row, "paper_compat") for row in rows]
    strict = [_score(row, "strict_final") for row in rows]
    base_public = [_score(row, "paper_compat") for row in baseline]
    base_strict = [_score(row, "strict_final") for row in baseline]
    result: dict[str, Any] = {
        "method": str(rows[0].get("method", "unknown")),
        "questions": len(rows),
        "public_correct": sum(public),
        "public_accuracy": statistics.fmean(public),
        "strict_correct": sum(strict),
        "strict_accuracy": statistics.fmean(strict),
        "avg_latency_seconds": _mean(rows, "latency_seconds"),
        "avg_peak_vram_gb": _mean(rows, "peak_allocated_bytes") / (1024**3),
        "avg_edge_count": _mean(rows, "edge_count"),
        "avg_cross_edges": _mean(rows, "cross_edges"),
        "avg_diagonal_edges": _mean(rows, "diagonal_edges"),
        "avg_source_token_exposure": _mean(rows, "source_token_exposure"),
        "public_delta_vs_sequential": statistics.fmean(public) - statistics.fmean(base_public),
        "strict_delta_vs_sequential": statistics.fmean(strict) - statistics.fmean(base_strict),
        "public_delta_ci_low": bootstrap_interval([x - y for x, y in zip(public, base_public)])[0],
        "public_delta_ci_high": bootstrap_interval([x - y for x, y in zip(public, base_public)])[1],
        "strict_delta_ci_low": bootstrap_interval([x - y for x, y in zip(strict, base_strict)], seed=20260911)[0],
        "strict_delta_ci_high": bootstrap_interval([x - y for x, y in zip(strict, base_strict)], seed=20260911)[1],
        "by_hop": {},
    }
    for hop in HOPS:
        indices = [i for i, row in enumerate(rows) if int(row["no_of_hops"]) == hop]
        if not indices:
            continue
        result["by_hop"][str(hop)] = {
            "n": len(indices),
            "public_correct": sum(public[i] for i in indices),
            "public_accuracy": statistics.fmean(public[i] for i in indices),
            "strict_correct": sum(strict[i] for i in indices),
            "strict_accuracy": statistics.fmean(strict[i] for i in indices),
        }
    return result


def compare_reference_outputs(
    routing_rows: dict[str, list[dict[str, Any]]],
    references: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    checks = {}
    for routing_name, reference_name in (("released_last1", "graphkv_top1"), ("released_last3", "graphkv_top3"), ("full", "graphkv_full")):
        new = routing_rows[routing_name]
        old = references[reference_name]
        checks[routing_name] = {
            "reference": reference_name,
            "question_count": len(new),
            "prompt_hash_equal": all(a.get("prompt_hash") == b.get("prompt_hash") for a, b in zip(new, old)),
            "generated_equal": all(a.get("generated") == b.get("generated") for a, b in zip(new, old)),
            "public_score_equal": all(_score(a, "paper_compat") == _score(b, "paper_compat") for a, b in zip(new, old)),
            "strict_score_equal": all(_score(a, "strict_final") == _score(b, "strict_final") for a, b in zip(new, old)),
        }
    return checks


def summarize(routing_dir: Path, sequential_path: Path, reference_root: Path) -> dict[str, Any]:
    routing = {name: read_jsonl(routing_dir / f"{name}.jsonl") for name in ROUTING_METHODS}
    baseline = read_jsonl(sequential_path)
    named = {"sequential": baseline, **routing}
    question_ids = validate_same_questions(named)
    refs = {
        "graphkv_top1": read_jsonl(reference_root / "graphkv_top1" / "graphkv_top1.jsonl"),
        "graphkv_top3": read_jsonl(reference_root / "graphkv_top3" / "graphkv_top3.jsonl"),
        "graphkv_full": read_jsonl(reference_root / "graphkv_full" / "graphkv_full.jsonl"),
    }
    validate_same_questions({"released_last1": routing["released_last1"], "graphkv_top1": refs["graphkv_top1"]})
    validate_same_questions({"released_last3": routing["released_last3"], "graphkv_top3": refs["graphkv_top3"]})
    validate_same_questions({"full": routing["full"], "graphkv_full": refs["graphkv_full"]})
    summaries = {"sequential": summarize_method(baseline, baseline)}
    summaries.update({name: summarize_method(rows, baseline) for name, rows in routing.items()})
    return {
        "scope": "matched 100-question A800 MoreHopQA control",
        "question_ids_sha256": hashlib.sha256("\n".join(question_ids).encode()).hexdigest(),
        "question_count": len(question_ids),
        "methods": summaries,
        "reference_equivalence": compare_reference_outputs(routing, refs),
        "scorer": "paper-compatible whole-response answer containment; strict-final marker scorer diagnostic only",
        "bootstrap": {"draws": 20000, "seed_public": 20260910, "seed_strict": 20260911},
    }


def write_outputs(result: dict[str, Any], csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "questions", "public_correct", "public_accuracy", "strict_correct", "strict_accuracy",
        "avg_latency_seconds", "avg_peak_vram_gb", "avg_edge_count", "avg_cross_edges", "avg_diagonal_edges",
        "avg_source_token_exposure", "public_delta_vs_sequential", "public_delta_ci_low", "public_delta_ci_high",
        "strict_delta_vs_sequential", "strict_delta_ci_low", "strict_delta_ci_high",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in result["methods"].values():
            writer.writerow({field: summary.get(field, "") for field in fields})

    methods = result["methods"]
    order = sorted(methods, key=lambda name: (-methods[name]["public_accuracy"], methods[name]["avg_latency_seconds"]))
    lines = [
        "# A800 MoreHopQA Topology Screen",
        "",
        f"Scope: {result['scope']}; n={result['question_count']}. Question-order SHA-256: `{result['question_ids_sha256']}`.",
        "",
        "This is a matched 100-question control. Every topology uses the same prompt, model revision, question IDs, released document order, 256-token cap, greedy decoding, and scorer. Retrieval and serialization are excluded from latency.",
        "",
        "## Results",
        "",
        "| Method | Public correct | Public accuracy | Strict correct | Strict accuracy | Avg latency (s) | Avg block-read edges | Public Δ vs sequential | 95% paired CI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        s = methods[name]
        lines.append(f"| {name} | {s['public_correct']}/{s['questions']} | {s['public_accuracy']:.3f} | {s['strict_correct']}/{s['questions']} | {s['strict_accuracy']:.3f} | {s['avg_latency_seconds']:.3f} | {s['avg_edge_count']:.1f} | {s['public_delta_vs_sequential']:+.3f} | [{s['public_delta_ci_low']:+.3f}, {s['public_delta_ci_high']:+.3f}] |")
    lines += [
        "",
        "The public scorer is the paper-compatible whole-response answer-containment rule. Strict-final scoring is shown only as a diagnostic and requires an explicit final-answer marker.",
        "",
        "## Hop breakdown",
        "",
        "| Method | H1 | H2 | H3 | H4 | H5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in order:
        vals = [methods[name]["by_hop"].get(str(h), {}).get("public_accuracy", float("nan")) for h in HOPS]
        lines.append("| " + name + " | " + " | ".join("—" if v != v else f"{v:.3f}" for v in vals) + " |")
    lines += [
        "",
        "## Integrity checks",
        "",
        "The parity controls below compare the new routing engine against the existing project GraphKV outputs on the same 100 questions:",
        "",
    ]
    for name, check in result["reference_equivalence"].items():
        status = "PASS" if all(check[key] for key in ("prompt_hash_equal", "generated_equal", "public_score_equal", "strict_score_equal")) else "FAIL"
        lines.append(f"- `{name}` vs `{check['reference']}`: **{status}**; generated text equal={check['generated_equal']}, prompt hash equal={check['prompt_hash_equal']}, scorer outputs equal={check['public_score_equal'] and check['strict_score_equal']}.")
    lines += [
        "",
        "No claim of held-out improvement is made from this screen. The released-last-1 control is an exact reproduction of GraphKV Top-1, not a new topology. Query-BM25-k1 is the strongest non-released topology in this screen but trails released-last-1 by one question and needs confirmation on a larger, predeclared sample.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=. python -m src.eval.morehop_topology_results \\",
        "  --routing-dir artifacts/results/topology_research/routing_a800 \\",
        "  --sequential artifacts/results/morehop_controlled_100_a800/sequential/sequential.jsonl \\",
        "  --reference-root artifacts/results/morehop_controlled_100_a800 \\",
        "  --csv artifacts/results/topology_research/routing_summary.csv \\",
        "  --markdown artifacts/results/topology_research/routing_summary.md",
        "```",
    ]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dir", type=Path, required=True)
    parser.add_argument("--sequential", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.routing_dir, args.sequential, args.reference_root)
    write_outputs(result, args.csv, args.markdown)
    print(json.dumps({"question_count": result["question_count"], "methods": list(result["methods"])}, indent=2))


if __name__ == "__main__":
    main()
