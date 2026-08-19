import sys
import types

import pytest

sys.modules.setdefault("igraph", types.SimpleNamespace(Graph=object))

from src.eval.qafd_graph_pair_benchmark import (
    choose_center_and_neighbors,
    choose_component_stars,
    fill_star_neighbors,
    query_overlap,
    query_conditioned_center,
)


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


def test_query_overlap_ignores_question_stopwords():
    assert query_overlap("Who was born in Paris?", "The artist was born in Paris.") == 2


def test_query_conditioned_center_is_shared_prompt_text():
    assert query_conditioned_center("passage", "question?", enabled=True) == (
        "Question-directed graph integration:\n"
        "Question: question?\n"
        "Use the linked neighbor evidence while reading this center passage.\n"
        "passage"
    )
    assert query_conditioned_center("passage", "question?", enabled=False) == "passage"


def test_query_conditioned_center_can_add_latent_integration_checkpoint():
    center = query_conditioned_center(
        "passage", "question?", enabled=True, integration_checkpoint=True
    )

    assert center.startswith("Question-directed graph integration:")
    assert center.endswith("shortest supported answer for the final response.\n")


def test_bridge_target_directs_edge_away_from_question_facing_passage():
    adjacency = [{1}, {0, 2}, {1}]
    scores = [0.9, 0.8, 0.7]
    passages = [
        "Alice founded Acme Industries.",
        "Acme Industries employed Bob Smith.",
        "An unrelated passage.",
    ]
    center, neighbors = choose_center_and_neighbors(
        adjacency,
        scores,
        "bridge_target",
        max_neighbors=2,
        passages=passages,
        question="Who founded Acme Industries?",
    )

    assert center == 1
    assert neighbors == [0, 2]


def test_bridge_target_requires_query_inputs():
    with pytest.raises(ValueError, match="requires the question"):
        choose_center_and_neighbors(
            [{1}, {0}], [0.9, 0.8], "bridge_target", max_neighbors=1
        )


def test_component_stars_keep_highest_scoring_components_separate():
    adjacency = [{1, 2}, {0}, {0}, {4}, {3}, set()]
    scores = [0.9, 0.8, 0.7, 0.95, 0.6, 0.99]

    stars = choose_component_stars(
        adjacency, scores, max_stars=2, max_neighbors=2
    )

    assert stars == [(0, [1, 2]), (3, [4])]


def test_component_stars_fall_back_without_graph_edges():
    stars = choose_component_stars(
        [set(), set(), set()], [0.9, 0.8, 0.7], max_stars=2, max_neighbors=1
    )

    assert stars == [(0, [1])]


def test_sparse_star_fills_from_wider_qafd_topology_by_score():
    neighbors = fill_star_neighbors(
        center=0,
        neighbors=[1],
        fill_adjacency=[{1, 2, 3, 4}, {0}, {0}, {0}, {0}],
        scores=[1.0, 0.7, 0.9, 0.8, 0.6],
        min_neighbors=4,
        max_neighbors=4,
    )

    assert neighbors == [1, 2, 3, 4]


def test_dense_star_is_not_reordered_by_fill_policy():
    neighbors = fill_star_neighbors(
        center=0,
        neighbors=[3, 1, 2, 4],
        fill_adjacency=[{1, 2, 3, 4}, {0}, {0}, {0}, {0}],
        scores=[1.0, 0.7, 0.9, 0.8, 0.6],
        min_neighbors=4,
        max_neighbors=4,
    )

    assert neighbors == [3, 1, 2, 4]
