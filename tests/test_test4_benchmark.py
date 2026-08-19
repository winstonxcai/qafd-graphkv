import sys
import types

sys.modules.setdefault("igraph", types.SimpleNamespace(Graph=object))

from src.eval.test4_benchmark import (
    PassageGraph,
    build_suffix,
    graph_diameter,
    remap_adjacency,
    resolve_order_key,
    select_graph_indices,
)


class FakeVertex:
    def __init__(self, index, name):
        self.index = index
        self.name = name

    def __getitem__(self, key):
        if key == "name":
            return self.name
        raise KeyError(key)


class FakeGraph:
    def __init__(self):
        self.vs = [
            FakeVertex(0, "chunk-0"),
            FakeVertex(1, "chunk-1"),
            FakeVertex(2, "chunk-2"),
            FakeVertex(3, "entity-a"),
            FakeVertex(4, "entity-b"),
            FakeVertex(5, "entity-c"),
        ]
        self.edges = {
            0: {3},
            1: {4},
            2: {5},
            3: {0, 4},
            4: {1, 3, 5},
            5: {2, 4},
        }

    def neighbors(self, node):
        return list(self.edges[node])


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


def test_sequential_can_use_the_matched_qafd_order():
    assert resolve_order_key("sequential", "qafd_h1") == "qafd_h1"
    assert resolve_order_key("sequential", "retrieval") == "sequential"
    assert resolve_order_key("qafd_h1", "qafd_h0") == "qafd_h1"


def test_passage_graph_bounded_search_reuse_preserves_hop_semantics():
    graph = FakeGraph()

    assert PassageGraph(graph, 0).adjacency([0, 1, 2]) == [set(), set(), set()]
    assert PassageGraph(graph, 1).adjacency([0, 1, 2]) == [
        {1},
        {0, 2},
        {1},
    ]
    assert PassageGraph(graph, 2).adjacency([0, 1, 2]) == [
        {1, 2},
        {0, 2},
        {0, 1},
    ]


def test_graph_diameter_uses_largest_finite_component_diameter():
    assert graph_diameter([{1}, {0, 2}, {1}, set(), set()]) == 2
    assert graph_diameter([set(), set()]) == 0
