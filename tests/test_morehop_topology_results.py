import json
from pathlib import Path

from src.eval.morehop_topology_results import bootstrap_interval, summarize_method, validate_same_questions


def _row(qid: str, hop: int, public: int, strict: int, method: str) -> dict:
    return {
        "_id": qid,
        "no_of_hops": hop,
        "paper_compat_accuracy": public,
        "strict_final_accuracy": strict,
        "latency_seconds": 1.0,
        "peak_allocated_bytes": 1024**3,
        "edge_count": 10,
        "cross_edges": 9,
        "diagonal_edges": 1,
        "source_token_exposure": 100,
        "method": method,
    }


def test_question_order_is_strictly_paired():
    rows = [_row("a", 1, 1, 1, "x"), _row("b", 2, 0, 0, "x")]
    assert validate_same_questions({"a": rows, "b": [dict(rows[0]), dict(rows[1])]}) == ["a", "b"]


def test_summary_and_bootstrap_are_deterministic():
    baseline = [_row("a", 1, 0, 0, "seq"), _row("b", 2, 1, 1, "seq")]
    candidate = [_row("a", 1, 1, 0, "candidate"), _row("b", 2, 1, 1, "candidate")]
    summary = summarize_method(candidate, baseline)
    assert summary["public_accuracy"] == 1.0
    assert summary["public_delta_vs_sequential"] == 0.5
    assert bootstrap_interval([1, 0], seed=7, draws=1000) == bootstrap_interval([1, 0], seed=7, draws=1000)


def test_summary_includes_hop_buckets():
    baseline = [_row("a", 1, 0, 0, "seq"), _row("b", 2, 1, 1, "seq")]
    candidate = [_row("a", 1, 1, 1, "candidate"), _row("b", 2, 0, 0, "candidate")]
    summary = summarize_method(candidate, baseline)
    assert summary["by_hop"]["1"]["public_accuracy"] == 1.0
    assert summary["by_hop"]["2"]["public_accuracy"] == 0.0
