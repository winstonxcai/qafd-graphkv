import pytest

from src.graph.morehop_query_topology import (
    build_query_topology,
    build_source_topology,
    random_source_indices,
    released_last_k_source_indices,
    reverse_rank_source_indices,
)


def test_query_topology_uses_only_question_and_document_text():
    result = build_query_topology("capital capital", ["the capital is Paris", "a river"], top_k=1)
    assert result["neighbors"] == [[0], [0]]
    assert result["node_scores"][0] > result["node_scores"][1]
    assert result["method"] == "bm25_query"


def test_stable_ties_keep_original_indices_and_repeat_sources_for_all_targets():
    result = build_query_topology("missing", ["same", "same", ""], top_k=2)
    assert result["neighbors"] == [[0, 1], [0, 1], [0, 1]]
    assert result["node_scores"] == [0.0, 0.0, 0.0]


def test_empty_text_is_finite_and_diagonal_is_reported_separately():
    result = build_query_topology("", ["", "word"], top_k=1)
    assert result["node_scores"] == [0.0, 0.0]
    assert result["edge_count"] == 2
    assert result["diagonal_edge_count"] == 1
    assert result["config"]["edge_count_includes_diagonal"] is True


@pytest.mark.parametrize("top_k", [0, -1, 3])
def test_invalid_k(top_k):
    with pytest.raises(ValueError):
        build_query_topology("q", ["a", "b"], top_k=top_k)


def test_empty_document_collection_rejects_default_k():
    with pytest.raises(ValueError):
        build_query_topology("q", [], top_k=1)


def test_source_controls_preserve_exact_sets_and_order():
    documents = ["a", "b", "c", "d"]
    assert random_source_indices(documents, 2, seed=7) == random_source_indices(documents, 2, seed=7)
    assert reverse_rank_source_indices([1, 3, 0]) == [0, 3, 1]
    assert released_last_k_source_indices(documents, 2) == [2, 3]
    result = build_source_topology(documents, [3, 1], method="reverse_rank")
    assert result["neighbors"] == [[3, 1]] * 4
    assert result["config"]["source_indices"] == [3, 1]
