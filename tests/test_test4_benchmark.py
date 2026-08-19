import sys
import types

sys.modules.setdefault("igraph", types.SimpleNamespace(Graph=object))

from src.eval.test4_benchmark import build_suffix, remap_adjacency


def test_concise_prompt_requests_answer_only():
    suffix = build_suffix("Who wrote it?", "concise")

    assert "shortest answer phrase" in suffix
    assert "Who wrote it?" in suffix
    assert suffix.endswith("<|assistant|>\n")


def test_multihop_prompt_requests_link_verification():
    suffix = build_suffix("Where was the author born?", "multihop")

    assert "verify each hop" in suffix
    assert "Where was the author born?" in suffix


def test_remap_adjacency_tracks_reordered_passages():
    adjacency = [{1}, {0, 2}, {1}]

    assert remap_adjacency(adjacency, [2, 0, 1]) == [{2}, {2}, {0, 1}]
