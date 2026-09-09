import math

import pytest

from src.graph.morehop_path_topology import build_path_topology, stage_neighbors


def test_query_seed_and_global_neighbors_are_normalized_and_self_included():
    result = build_path_topology("red", ["red apple", "blue apple", "green"], top_k=2, graph_k=2)
    assert result["neighbors"] == [[0, 1]] * 3
    assert math.isclose(sum(result["node_scores"]), 1.0, rel_tol=0, abs_tol=1e-10)
    assert all(math.isfinite(score) for score in result["node_scores"])
    assert result["diagnostic_graph"]["document_only"] is True


def test_ppr_converges_with_dangling_nodes_and_zero_overlap_is_uniform():
    result = build_path_topology("missing words", ["alpha", "beta"], alpha=0.85, graph_k=0)
    assert result["node_scores"] == pytest.approx([0.5, 0.5])
    assert result["diagnostic_graph"]["dangling_nodes"] == [0, 1]
    assert result["config"]["iterations"] > 0


def test_ties_break_by_document_index():
    result = build_path_topology("", ["same", "same", "other"], top_k=2)
    assert result["neighbors"] == [[0, 1], [0, 1], [0, 1]]


def test_no_labels_or_hop_depth_appear_in_result():
    result = build_path_topology("alpha", ["alpha", "beta"])
    assert "gold" not in result and "hop" not in result
    assert result["method"] == "query_seeded_ppr_tfidf_cosine_knn"


def test_stages_are_explicit_and_read_only_from_immediately_previous_layer():
    assert stage_neighbors(5, [[0, 1], [2], [3, 4]]) == [[], [], [0, 1], [2], [2]]


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        build_path_topology("q", ["d"], alpha=1.1)
    with pytest.raises(ValueError):
        stage_neighbors(2, [[2]])
    with pytest.raises(ValueError):
        stage_neighbors(2, [[0], [0, 1]])
    with pytest.raises(ValueError):
        build_path_topology("q", ["d"], alpha=1.0)
    with pytest.raises(ValueError):
        build_path_topology("q", ["d"], top_k=2)
