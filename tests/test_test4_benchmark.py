import sys
import types

sys.modules.setdefault("igraph", types.SimpleNamespace(Graph=object))

from src.eval.test4_benchmark import build_suffix, remap_adjacency, select_graph_indices


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


def test_best_edge_selection_prefers_highest_scoring_connected_pair():
    adjacency = [{1}, {0}, {3}, {2}, set()]
    scores = [0.9, 0.1, 0.7, 0.6, 0.8]

    assert select_graph_indices(adjacency, scores, 2, "best_edge") == [2, 3]


def test_top_component_selection_fills_from_retrieval_when_component_is_small():
    adjacency = [{1}, {0}, set(), set()]
    scores = [0.9, 0.8, 0.7, 0.6]

    assert select_graph_indices(adjacency, scores, 3, "top_component") == [0, 1, 2]
