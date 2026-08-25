"""QAFD h<=1 soft-prior matrix construction for CSA."""

from __future__ import annotations

from src.graph.sparsify import score_passage_edges


def qafd_prior_matrix(graph, passage_vertices, entity_scores) -> tuple[list[list[float]], list[dict]]:
    edges = score_passage_edges(
        graph=graph,
        passage_vertices=list(passage_vertices),
        entity_scores=entity_scores,
        max_hops=1,
        decay_lambda=1.0,
    )
    count = len(passage_vertices)
    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    audit = []
    for edge in edges:
        matrix[edge.src][edge.dst] = float(edge.score)
        matrix[edge.dst][edge.src] = float(edge.score)
        audit.append(edge.to_dict())
    return matrix, audit

