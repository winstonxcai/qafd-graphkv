import json
from pathlib import Path

from src.eval.morehop_baseline_audit import (
    audit,
    cache_topology_finding,
    public_prediction,
    render_markdown,
    score_row,
    strict_prediction,
)


def _write_scope(root: Path, ids: list[str], methods: tuple[str, ...]) -> None:
    for method in methods:
        directory = root / method
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(json.dumps({"dataset_sha256": "fixture", "model": "fixture"}))
        rows = [
            {"_id": qid, "no_of_hops": i + 1, "answers": [f"answer-{i}"],
             "generated": f"Answer: answer-{i}", "method": method, "latency_seconds": 1.0}
            for i, qid in enumerate(ids)
        ]
        (directory / f"{method}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_public_fallback_and_strict_scorers_are_distinct():
    generated = "Reasoning says the answer is 42, but no marker."
    assert public_prediction(generated) == generated
    assert strict_prediction(generated) is None
    row = {"generated": generated, "answers": "42", "no_of_hops": 1, "_id": "x", "method": "sequential"}
    assert score_row(row) == {"paper_compat": True, "strict_final": False, "has_explicit_answer": False}


def test_audit_uses_fixture_ids_and_dynamic_1024_scope(tmp_path):
    methods = ("sequential", "block_rag", "graphkv_top1", "graphkv_top2", "graphkv_top3", "graphkv_full")
    controlled = tmp_path / "controlled"
    full = tmp_path / "full"
    _write_scope(controlled, [f"q{i}" for i in range(100)], methods)
    _write_scope(full, ["a", "b", "c"], tuple(m for m in methods if m != "graphkv_top2"))
    output = tmp_path / "full1024.jsonl"
    output.write_text(json.dumps({"_id": "a", "no_of_hops": 1, "answers": "x", "generated": "Answer: x", "method": "graphkv_full"}) + "\n")
    result = audit(controlled, full, output)
    assert result["scopes"]["full_1118_256"]["summaries"][0]["paper_compat_denominator"] == 3
    assert result["available_1024"]["dataset_scope"] == "subset"
    assert result["available_1024"]["id_equality_with_full_256_scope"] is False


def test_cache_topology_has_defined_block_read_counts():
    topology = cache_topology_finding()
    assert topology["full"]["ordered_pairs"] == "n*n"
    assert topology["full"]["diagonal_pairs"] == "n"
    assert topology["top_k"]["ordered_pairs"] == "n*k"
    assert topology["top_k"]["diagonal_pairs"] == "k"
    assert "reprocesses all n documents" in render_markdown({"scopes": {}, "available_1024": None, "cache_topology": topology})
