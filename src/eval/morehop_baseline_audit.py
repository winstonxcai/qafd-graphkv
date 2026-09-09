"""Audit reproducible MoreHopQA baseline artifacts.

The command is deliberately standard-library-only and reads historical output
files without modifying them.  It keeps the 100-question A800 control, the
full 1,118-question 256-token run, and the available 1,024-token output in
separate scopes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.eval.morehop_scoring import (
    containment,
    normalize_answer,
    paper_compat_prediction,
    score_both,
    strict_final_prediction,
)


HOPS = (1, 2, 3, 4, 5)
METHODS = ("sequential", "block_rag", "graphkv_top1", "graphkv_top2", "graphkv_top3", "graphkv_full")


def _answers(value: str | list[str]) -> list[str]:
    return [value] if isinstance(value, str) else value


def public_prediction(generated: str) -> str:
    return paper_compat_prediction(generated)


def strict_prediction(generated: str) -> str | None:
    return strict_final_prediction(generated)


def contains(prediction: str | None, answers: str | list[str]) -> bool:
    return bool(containment(prediction, answers))


def score_row(row: dict[str, Any]) -> dict[str, bool]:
    scores = score_both(str(row.get("generated", "")), row.get("answers", []))
    return {
        "paper_compat": bool(scores["paper_compat_accuracy"]),
        "strict_final": bool(scores["strict_final_accuracy"]),
        "has_explicit_answer": bool(scores["has_explicit_answer"]),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty prediction file: {path}")
    return rows


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot summarize an empty set")
    method = rows[0].get("method", "unknown")
    by_hop: dict[str, dict[str, Any]] = {}
    for hop in HOPS:
        subset = [r for r in rows if int(r["no_of_hops"]) == hop]
        if not subset:
            by_hop[str(hop)] = {"correct": None, "denominator": 0, "paper_compat": None, "strict_final": None}
            continue
        scores = [score_row(r) for r in subset]
        by_hop[str(hop)] = {
            "correct": sum(s["paper_compat"] for s in scores),
            "strict_correct": sum(s["strict_final"] for s in scores),
            "denominator": len(subset),
            "paper_compat": sum(s["paper_compat"] for s in scores) / len(subset),
            "strict_final": sum(s["strict_final"] for s in scores) / len(subset),
        }
    scores = [score_row(r) for r in rows]
    return {
        "method": method,
        "questions": len(rows),
        "paper_compat_correct": sum(s["paper_compat"] for s in scores),
        "paper_compat_denominator": len(rows),
        "paper_compat_accuracy": sum(s["paper_compat"] for s in scores) / len(rows),
        "strict_final_correct": sum(s["strict_final"] for s in scores),
        "strict_final_denominator": len(rows),
        "strict_final_accuracy": sum(s["strict_final"] for s in scores) / len(rows),
        "explicit_answer_count": sum(s["has_explicit_answer"] for s in scores),
        "explicit_answer_denominator": len(rows),
        "by_hop": by_hop,
        "avg_latency_seconds": sum(float(r["latency_seconds"]) for r in rows if "latency_seconds" in r) / max(1, sum("latency_seconds" in r for r in rows)),
    }


def load_scope(files: dict[str, list[Path]], expected_count: int | None = None) -> dict[str, Any]:
    summaries = []
    ids_by_method: dict[str, list[str]] = {}
    for method, paths in files.items():
        rows = [row for path in paths for row in read_jsonl(path)]
        if expected_count is not None and len(rows) != expected_count:
            raise ValueError(f"{method}: expected {expected_count} rows, found {len(rows)}")
        ids_by_method[method] = [str(r["_id"]) for r in rows]
        summaries.append(summarize(rows))
    if len({tuple(v) for v in ids_by_method.values()}) > 1:
        raise ValueError("methods in one scope do not have identical question IDs")
    ids = next(iter(ids_by_method.values()), [])
    all_rows = [row for method in files for path in files[method] for row in read_jsonl(path)]
    manifest_paths = sorted({
        path.parent / "manifest.json"
        for paths in files.values()
        for path in paths
        if (path.parent / "manifest.json").exists()
    })
    return {
        "summaries": summaries,
        "question_ids": ids,
        "normalization_difference": normalization_difference(all_rows),
        "manifests": provenance(manifest_paths),
    }


def provenance(manifest_paths: Iterable[Path]) -> dict[str, Any]:
    records = []
    for path in manifest_paths:
        data = json.loads(path.read_text())
        records.append({
            "path": str(path),
            "dataset_sha256": data.get("dataset_sha256", "unavailable"),
            "model": data.get("model", "unavailable"),
            "model_revision": data.get("model_revision", "unavailable"),
            "graphkv_revision": data.get("graphkv_submodule_commit", data.get("graphkv_revision", "unavailable")),
            "project_commit": data.get("project_commit", "unavailable"),
        })
    return {"records": records, "missing_fields_are_unavailable": True}


def find_method_files(root: Path, methods: Iterable[str] = METHODS) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    for method in methods:
        candidates = sorted(root.glob(f"**/{method}.jsonl"))
        if candidates:
            found[method] = candidates
    return found


def cache_topology_finding() -> dict[str, Any]:
    """Static interpretation of the upstream ``pcw.py`` block-read paths."""
    return {
        "full": {
            "read_graph": "complete directed block-read graph",
            "n_blocks": "n",
            "ordered_pairs": "n*n",
            "diagonal_pairs": "n",
            "off_diagonal_pairs": "n*(n-1)",
            "raw_sources_exposed": "all n context blocks",
            "reprocessed_sources": "all n context blocks",
        },
        "top_k": {
            "read_graph": "n-by-k directed block-read graph",
            "n_blocks": "n",
            "source_blocks": "last k context blocks (context_input_ids[-top_k:])",
            "ordered_pairs": "n*k",
            "diagonal_pairs": "k",
            "off_diagonal_pairs": "k*(n-1)",
            "raw_sources_exposed": "all n context blocks",
            "reprocessed_sources": "all n context blocks; context2_outputs feeds full context_input_ids",
        },
        "source": "third_party/GraphKV/pcw.py:135-181 and :212-260",
    }


def _upstream_normalize_answer(text: str) -> str:
    """Upstream GraphKV order: punctuation is removed before articles."""
    without_punctuation = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", without_punctuation).split())


def _score_with_normalizer(row: dict[str, Any], normalizer) -> tuple[bool, bool]:
    generated = str(row.get("generated", ""))
    answers = _answers(row.get("answers", []))
    public = public_prediction(generated)
    strict = strict_prediction(generated)
    return (
        any((gold := normalizer(answer)) and gold in normalizer(public) for answer in answers),
        any((gold := normalizer(answer)) and strict is not None and gold in normalizer(strict) for answer in answers),
    )


def normalization_difference(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = {"rows_affected": 0, "paper_compat_score_changes": 0, "strict_final_score_changes": 0}
    for row in rows:
        ours = score_row(row)
        upstream_public, upstream_strict = _score_with_normalizer(row, _upstream_normalize_answer)
        if ours["paper_compat"] != upstream_public or ours["strict_final"] != upstream_strict:
            counts["rows_affected"] += 1
        counts["paper_compat_score_changes"] += int(ours["paper_compat"] != upstream_public)
        counts["strict_final_score_changes"] += int(ours["strict_final"] != upstream_strict)
    return counts


def audit(controlled: Path, full: Path, full_1024: Path | None = None) -> dict[str, Any]:
    controlled_files = find_method_files(controlled)
    full_files = find_method_files(full)
    if set(controlled_files) != set(METHODS):
        raise ValueError(f"controlled scope missing methods: {sorted(set(METHODS) - set(controlled_files))}")
    if set(full_files) != {m for m in METHODS if m != "graphkv_top2"}:
        raise ValueError("full 256 scope must contain sequential, block_rag, top1, top3, and full")
    result: dict[str, Any] = {
        "scopes": {
            "controlled_100_a800_256": load_scope(controlled_files),
            "full_1118_256": load_scope(full_files),
        },
        "cache_topology": cache_topology_finding(),
        "separation_rule": "No subset, hop bucket, method, or generation cap is mixed across scopes.",
        "available_1024": None,
    }
    if full_1024 is not None:
        rows = read_jsonl(full_1024)
        full_ids = set(result["scopes"]["full_1118_256"]["question_ids"])
        output_ids = {str(row["_id"]) for row in rows}
        is_full = output_ids == full_ids
        result["available_1024"] = {
            "path": str(full_1024),
            "rows": len(rows),
            "hop_counts": dict(sorted(Counter(int(r["no_of_hops"]) for r in rows).items())),
            "scope": summarize(rows),
            "dataset_scope": "full_dataset" if is_full else "subset",
            "id_equality_with_full_256_scope": is_full,
            "normalization_difference": normalization_difference(rows),
            "comparability": (
                "same question IDs as full 256 scope; compare as a separate 1,024-token cap"
                if is_full else "question IDs differ from full scope; subset-only result"
            ),
        }
    return result


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# MoreHopQA baseline audit", "", "This report is generated from historical artifacts read-only. Scopes and generation caps are kept separate; dataset scope is determined from actual question-ID equality.", ""]
    for scope_name, scope in result["scopes"].items():
        lines += [f"## {scope_name}", "", "| Method | Paper-compatible | Strict-final | Explicit `Answer:` | Avg latency (s) |", "|---|---:|---:|---:|---:|"]
        for row in scope["summaries"]:
            lines.append(f"| {row['method']} | {row['paper_compat_correct']}/{row['paper_compat_denominator']} ({row['paper_compat_accuracy']:.4f}) | {row['strict_final_correct']}/{row['strict_final_denominator']} ({row['strict_final_accuracy']:.4f}) | {row['explicit_answer_count']}/{row['explicit_answer_denominator']} | {row['avg_latency_seconds']:.3f} |")
        lines += ["", "| Method | H1 | H2 | H3 | H4 | H5 |", "|---|---:|---:|---:|---:|---:|"]
        for row in scope["summaries"]:
            cells = []
            for hop in HOPS:
                h = row["by_hop"][str(hop)]
                cells.append("—" if not h["denominator"] else f"{h['correct']}/{h['denominator']} ({h['paper_compat']:.4f}); {h['strict_correct']}/{h['denominator']} ({h['strict_final']:.4f})")
            lines.append(f"| {row['method']} | " + " | ".join(cells) + " |")
        lines.append("")
    if result["available_1024"]:
        partial = result["available_1024"]
        s = partial["scope"]
        lines += ["## 1,024-token output", "", f"The available file has {partial['rows']} rows with hop counts {partial['hop_counts']}; ID comparison labels its scope `{partial['dataset_scope']}` (full equality: `{partial['id_equality_with_full_256_scope']}`). {partial['comparability']}.", "", f"| Method | Paper-compatible | Strict-final |", "|---|---:|---:|"]
        lines.append(f"| {s['method']} | {s['paper_compat_correct']}/{s['paper_compat_denominator']} ({s['paper_compat_accuracy']:.4f}) | {s['strict_final_correct']}/{s['strict_final_denominator']} ({s['strict_final_accuracy']:.4f}) |")
        lines.append("")
    normalization_lines = []
    for scope_name, scope in result["scopes"].items():
        diff = scope.get("normalization_difference", {})
        normalization_lines.append(f"{scope_name}: {diff.get('rows_affected', 0)} rows affected ({diff.get('paper_compat_score_changes', 0)} public, {diff.get('strict_final_score_changes', 0)} strict score changes)")
    if result.get("available_1024"):
        diff = result["available_1024"].get("normalization_difference", {})
        normalization_lines.append(f"1,024-token output: {diff.get('rows_affected', 0)} rows affected ({diff.get('paper_compat_score_changes', 0)} public, {diff.get('strict_final_score_changes', 0)} strict score changes)")
    lines += ["## Findings", "", "- The public evaluator scores normalized answer containment after the final `Answer:` marker, but falls back to the entire generated response when no marker exists. Strict-final scoring requires the marker; generated fallback wording can therefore inflate the public score and should not be described as exact final-answer accuracy.", "- The complete 1,118-row 256-token summaries are computed from the five hop files. Sequential is the strongest reproducible matched baseline under the public scorer in that scope; exact counts and denominators are the generated table values, not hard-coded claims.", "- Our shared scorer removes articles before punctuation; upstream `rag_eval.py` removes punctuation before articles. Affected rows: " + "; ".join(normalization_lines) + ".", "- Latency is end-to-end client request time around the HTTP call, including server-side work and generation; it is not a prefill-only, decode-only, or GPU-kernel timing. A separately measured cache-construction scope is unavailable.", "- Manifest evidence records the dataset hash and available revisions; missing model revisions or provenance fields are explicitly `unavailable`.", "", "## Cache/token topology", "", "Full has n*n ordered block-read pairs, including n diagonal/self pairs and n*(n-1) off-diagonal pairs. Top-k reprocesses all n documents: `context2_outputs` is fed the full `context_input_ids`; its source side is the last k blocks, yielding n*k ordered pairs, including k diagonal pairs and k*(n-1) off-diagonal pairs. These are the defined block-read graph counts.", "", "JSON and Markdown exports intentionally record evidence and uncertainty separately; claims about paper tables require the authors' stored predictions, not regenerated fallback-scored outputs.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controlled", type=Path, required=True, help="100-question control directory")
    parser.add_argument("--full", type=Path, required=True, help="directory containing the five complete 1,118-row hop outputs")
    parser.add_argument("--full-1024", type=Path, help="optional available 1,024-token JSONL output")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.controlled, args.full, args.full_1024)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_out.write_text(render_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
