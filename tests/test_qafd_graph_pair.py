import sys
import types

sys.modules.setdefault("igraph", types.SimpleNamespace(Graph=object))

from src.eval.qafd_graph_pair_benchmark import choose_center_and_neighbors


def test_best_edge_uses_highest_scoring_endpoint_as_center():
    adjacency = [{1}, {0}, {3}, {2}]
    scores = [0.8, 0.7, 0.95, 0.1]

    center, neighbors = choose_center_and_neighbors(
        adjacency, scores, "best_edge", max_neighbors=2
    )

    assert center == 0
    assert neighbors == [1]


def test_isolated_center_falls_back_without_dropping_question():
    center, neighbors = choose_center_and_neighbors(
        [set(), set(), set()], [0.9, 0.8, 0.7], "top_seed", max_neighbors=1
    )

    assert center == 0
    assert neighbors == [1]
