"""Graph controls using target->source cache-read adjacency.

Degree swaps preserve each labelled node's in/out degree and the exact diagonal.
A global-source star has no alternative degree-preserving rewiring: its labelled
degree sequence determines its adjacency. Random-source controls instead match
the degree distribution and edge budget, not each labelled node's degree.
"""
from __future__ import annotations

import random


def validate_neighbors(neighbors: list[list[int]]) -> list[list[int]]:
    n = len(neighbors)
    if not n:
        raise ValueError("graph must contain a node")
    result = []
    for sources in neighbors:
        if any(type(j) is not int or not 0 <= j < n for j in sources):
            raise ValueError("invalid source index")
        if len(sources) != len(set(sources)):
            raise ValueError("duplicate source")
        result.append(sorted(sources))
    return result


def reverse_neighbors(neighbors: list[list[int]]) -> list[list[int]]:
    neighbors = validate_neighbors(neighbors)
    result = [[] for _ in neighbors]
    for i, sources in enumerate(neighbors):
        for j in sources:
            result[j].append(i)
    return result


def degree_preserving_control(neighbors: list[list[int]], seed: int, attempts: int = 1000) -> dict:
    neighbors = validate_neighbors(neighbors)
    if attempts < 0:
        raise ValueError("attempts must be nonnegative")
    original = {(i, j) for i, sources in enumerate(neighbors) for j in sources}
    diagonal = {(i, j) for i, j in original if i == j}
    edges = original - diagonal
    rng = random.Random(seed)
    swaps = 0
    if len(edges) >= 2:
        for _ in range(attempts):
            (a, b), (c, d) = rng.sample(sorted(edges), 2)
            if a == c or b == d or a == d or c == b:
                continue
            replacement = {(a, d), (c, b)}
            if replacement & edges:
                continue
            edges.difference_update({(a, b), (c, d)})
            edges.update(replacement)
            swaps += 1
    final = edges | diagonal
    return {
        "neighbors": [[j for j in range(len(neighbors)) if (i, j) in final] for i in range(len(neighbors))],
        "seed": seed,
        "attempts": attempts,
        "accepted_swaps": swaps,
        "changed_edges": len(final - original),
        "distinct_control": final != original,
    }


def graph_statistics(neighbors: list[list[int]]) -> dict:
    neighbors = validate_neighbors(neighbors)
    n = len(neighbors)
    diagonal = sum(i in sources for i, sources in enumerate(neighbors))
    total = sum(map(len, neighbors))
    # Components and diameter exclude self loops and use cache-read direction.
    reverse = reverse_neighbors(neighbors)
    weak = [(set(neighbors[i]) | set(reverse[i])) - {i} for i in range(n)]
    remaining = set(range(n))
    components = []
    while remaining:
        todo = [min(remaining)]
        component = set()
        while todo:
            node = todo.pop()
            if node in component:
                continue
            component.add(node)
            todo.extend(weak[node] - component)
        remaining -= component
        components.append(sorted(component))
    reachable = 0
    diameter = 0
    for start in range(n):
        distances = {start: 0}
        todo = [start]
        for node in todo:
            for other in neighbors[node]:
                if other not in distances:
                    distances[other] = distances[node] + 1
                    todo.append(other)
        reachable += len(distances) - 1
        diameter = max(diameter, max(distances.values()))
    return {
        "nodes": n, "edges_including_diagonal": total,
        "diagonal_edges": diagonal, "cross_edges": total - diagonal,
        "full_pair_ratio": total / (n * n),
        "full_cross_edge_ratio": (total-diagonal)/(n*(n-1)) if n > 1 else 0.0,
        "sources_per_target": list(map(len, neighbors)),
        "targets_per_source": list(map(len, reverse)),
        "weak_components": components,
        "weak_component_count": len(components),
        "isolated_nodes": sum(not (set(neighbors[i]) | set(reverse[i])) - {i} for i in range(n)),
        "finite_directed_diameter": diameter,
        "unreachable_ordered_pair_fraction": 1-reachable/(n*(n-1)) if n > 1 else 0.0,
    }
