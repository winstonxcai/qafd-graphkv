"""Faithful Graph-KV Task-1 reproduction for MoreHopQA.

This follows the upstream GraphKV MoreHopQA path: the released
``with_human_verification_ascend.jsonl`` file, original document order, the
Tulu prompt, five Graph-KV controls, and the upstream evaluator's final
``Answer:`` extraction.  It is intentionally independent of QAFD code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.eval.morehop_scoring import (
    normalize_answer,
    paper_compat_prediction,
    score_both,
)


ROOT = Path(__file__).resolve().parents[2]
GRAPHKV_ROOT = ROOT / "third_party" / "GraphKV"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def extract_morehop_answer(generated: str) -> str:
    """Backward-compatible name for the public evaluator's fallback scorer."""
    return paper_compat_prediction(generated)


def answer_containment(generated: str, answer: str | list[str]) -> float:
    return float(score_both(generated, answer)["paper_compat_accuracy"])


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"dataset is empty: {path}")
    seen = set()
    for index, row in enumerate(rows):
        stable_id = row.get("_id")
        if not stable_id or stable_id in seen:
            raise ValueError(f"missing or duplicate _id at dataset index {index}: {stable_id}")
        seen.add(stable_id)
        hop = int(row["no_of_hops"])
        if hop not in range(1, 6):
            raise ValueError(f"unsupported no_of_hops={hop} at dataset index {index}")
        if not row.get("question") or not row.get("answers"):
            raise ValueError(f"missing question or answer at dataset index {index}")
        # The released file has 10 documents for nearly every row, but four
        # rows contain expanded document sets (18/26/34/42).  The upstream
        # MoreHopQA path preserves those rows, so do not truncate them here.
        if not row.get("documents"):
            raise ValueError(f"dataset index {index} has no documents")
        if not all(isinstance(document, str) for document in row["documents"]):
            raise ValueError(f"dataset index {index} contains a non-string document")
    return rows


def build_prefix() -> str:
    return (
        "<|user|>\nYou are an intelligent AI assistant. Please answer questions "
        "based on the user's instructions. Below are some reference documents "
        "that may help you in answering the user's question.\n\n"
    )


def build_suffix(question: str) -> str:
    return (
        "Please write a high-quality answer for the given question using only "
        "the provided search documents. (If the answer is a date, format is as "
        "follows: YYYY-MM-DD (ISO standard).) After thinking step by step, give "
        f"your final answer following 'Answer:' \n Question: {question} \n<|assistant|>\n"
    )


def upstream_blocks(row: dict) -> list[str]:
    # MoreHopQA's upstream docs2blocks path preserves the released document order.
    contexts = [document + "\n" for document in row["documents"]]
    return [build_prefix(), "", *contexts, build_suffix(row["question"])]


def prompt_hash(blocks: list[str]) -> str:
    return hashlib.sha256("".join(blocks).encode("utf-8")).hexdigest()


METHODS = {
    "sequential": ("/generate_vanilla", None),
    "block_rag": ("/generate_block", None),
    "graphkv_top1": ("/generate_gapemp_appr", {"top_k": 1}),
    "graphkv_top3": ("/generate_gapemp_appr", {"top_k": 3}),
    "graphkv_full": ("/generate_gapemp", None),
}


def request(port: int, endpoint: str, blocks: list[str], extra: dict | None) -> tuple[str, float]:
    payload = {"blocks": blocks}
    if extra:
        payload.update(extra)
    started = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{port}{endpoint}", json=payload, timeout=1800
    )
    response.raise_for_status()
    result = response.json()
    if result.get("ret", 0) != 0:
        raise RuntimeError(result.get("message", "Graph-KV server failed"))
    return result["generated"], time.perf_counter() - started


def load_completed(path: Path, expected_ids: list[str]) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for offset, row in enumerate(rows):
        if row.get("_id") != expected_ids[offset]:
            raise ValueError(f"non-contiguous or mismatched output: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--hop", type=int, choices=range(1, 6), help="only evaluate this hop bucket")
    parser.add_argument(
        "--qid-manifest",
        type=Path,
        help="optional JSON manifest containing selected_question_ids",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        parser.error(f"unknown methods: {unknown}")

    all_rows = load_rows(args.dataset)
    rows = [row for row in all_rows if args.hop is None or int(row["no_of_hops"]) == args.hop]
    selection_manifest_sha256 = None
    if args.qid_manifest:
        selection = json.loads(args.qid_manifest.read_text())
        selected_ids = selection.get("selected_question_ids")
        if not isinstance(selected_ids, list) or not selected_ids:
            parser.error("qid manifest must contain a non-empty selected_question_ids list")
        if len(selected_ids) != len(set(selected_ids)):
            parser.error("qid manifest contains duplicate question IDs")
        by_id = {row["_id"]: row for row in rows}
        missing = [question_id for question_id in selected_ids if question_id not in by_id]
        if missing:
            parser.error(f"qid manifest contains IDs absent from selected dataset: {missing[:5]}")
        rows = [by_id[question_id] for question_id in selected_ids]
        selection_manifest_sha256 = sha256(args.qid_manifest)
    if not rows:
        parser.error("selected hop bucket is empty")
    expected_ids = [row["_id"] for row in rows]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))
    manifest = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "dataset_rows": len(all_rows),
        "selected_rows": len(rows),
        "hop": args.hop,
        "qid_manifest": str(args.qid_manifest.resolve()) if args.qid_manifest else None,
        "qid_manifest_sha256": selection_manifest_sha256,
        "selected_question_ids": expected_ids,
        "hop_counts_all": dict(sorted(Counter(int(row["no_of_hops"]) for row in all_rows).items())),
        "hop_counts_selected": dict(sorted(Counter(int(row["no_of_hops"]) for row in rows).items())),
        "methods": methods,
        "model": "ldsjmdy/Tulu3-Block-FT",
        "retrieved_chunks": 10,
        "document_order": "released MoreHopQA ascend file order",
        "max_new_tokens": max_new_tokens,
        "primary_replication_score": "paper_compat_accuracy: public GraphKV rag_eval.py final Answer: extraction with whole-response fallback plus normalized substring containment",
        "integrity_audit_score": "strict_final_accuracy: require an explicit final Answer: marker; missing markers score incorrect",
        "graphkv_submodule_commit": git_revision(GRAPHKV_ROOT),
        "project_commit": git_revision(ROOT),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    completed = {}
    handles = {}
    try:
        for method in methods:
            path = args.output_dir / f"{method}.jsonl"
            previous = load_completed(path, expected_ids)
            completed[method] = previous
            handles[method] = path.open("a", encoding="utf-8")

        for offset, row in enumerate(rows):
            blocks = upstream_blocks(row)
            digest = prompt_hash(blocks)
            for method in methods:
                if offset < len(completed[method]):
                    continue
                endpoint, extra = METHODS[method]
                generated, seconds = request(args.port, endpoint, blocks, extra)
                scores = score_both(generated, row["answers"])
                result = {
                    "dataset_index": all_rows.index(row),
                    "_id": row["_id"],
                    "no_of_hops": int(row["no_of_hops"]),
                    "question": row["question"],
                    "answers": row["answers"],
                    "documents": row["documents"],
                    "prompt_hash": digest,
                    "method": method,
                    "generated": generated,
                    **scores,
                    "accuracy": scores["paper_compat_accuracy"],
                    "latency_seconds": seconds,
                    "max_new_tokens": max_new_tokens,
                }
                handles[method].write(json.dumps(result, ensure_ascii=False) + "\n")
                handles[method].flush()
                print(json.dumps({"method": method, "_id": row["_id"], "hop": row["no_of_hops"], "accuracy": result["accuracy"], "latency_seconds": seconds}), flush=True)
    finally:
        for handle in handles.values():
            handle.close()

    summaries = []
    for method in methods:
        output_rows = [json.loads(line) for line in (args.output_dir / f"{method}.jsonl").read_text().splitlines() if line.strip()]
        if len(output_rows) != len(rows):
            raise ValueError(f"incomplete output for {method}: {len(output_rows)}/{len(rows)}")
        summaries.append({
            "method": method,
            "questions": len(output_rows),
            "paper_compat_accuracy": sum(
                score_both(row["generated"], row["answers"])["paper_compat_accuracy"]
                for row in output_rows
            ) / len(output_rows),
            "strict_final_accuracy": sum(
                score_both(row["generated"], row["answers"])["strict_final_accuracy"]
                for row in output_rows
            ) / len(output_rows),
            "explicit_answer_rate": sum(
                score_both(row["generated"], row["answers"])["has_explicit_answer"]
                for row in output_rows
            ) / len(output_rows),
            "avg_latency_seconds": sum(row["latency_seconds"] for row in output_rows) / len(output_rows),
            "total_latency_seconds": sum(row["latency_seconds"] for row in output_rows),
            "by_hop": {
                str(hop): {
                    "questions": sum(int(row["no_of_hops"]) == hop for row in output_rows),
                    "accuracy": (
                        sum(row["accuracy"] for row in output_rows if int(row["no_of_hops"]) == hop)
                        / sum(int(row["no_of_hops"]) == hop for row in output_rows)
                    ),
                }
                for hop in sorted({int(row["no_of_hops"]) for row in output_rows})
            },
        })
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = [
        "# Faithful Graph-KV MoreHopQA reproduction",
        "",
        f"Dataset: `{args.dataset}`; selected hop: `{args.hop}`; rows: `{len(rows)}`.",
        f"The protocol uses the official processed MoreHopQA release, ten documents, Tulu3-Block-FT, the upstream MoreHopQA prompt, a {max_new_tokens}-token greedy cap, and the upstream final `Answer:` scorer.",
        "",
        "| Method | Questions | Paper-compatible accuracy | Strict-final accuracy | Explicit-answer rate | Average latency (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {item['method']} | {item['questions']} | {item['paper_compat_accuracy']:.4f} | {item['strict_final_accuracy']:.4f} | {item['explicit_answer_rate']:.4f} | {item['avg_latency_seconds']:.3f} |"
        for item in summaries
    )
    lines.extend(["", "## Accuracy by hop", "", "| Method | Hop-1 | Hop-2 | Hop-3 | Hop-4 | Hop-5 |", "|---|---:|---:|---:|---:|---:|"])
    for item in summaries:
        values = [item["by_hop"].get(str(hop), {}).get("accuracy", "—") for hop in range(1, 6)]
        values = [f"{value:.4f}" if isinstance(value, float) else value for value in values]
        lines.append(f"| {item['method']} | " + " | ".join(values) + " |")
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
