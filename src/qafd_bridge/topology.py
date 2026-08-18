"""Bounded passage-topology analysis for serialized QAFD selections.

The graph adapter is deliberately duck-typed: this module can operate on an
igraph graph without importing QAFD, keeping the GraphKV boundary clean.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from itertools import combinations
from math import comb
from typing import Any


@dataclass(frozen=True)
class PassageTopology:
    selected_passages: int
    mapped_passages: int
    selected_entities: int
    edges: int
    components: int
    largest_component: int
    average_degree: float
    edge_density: float
    diameter: int | None
    average_shortest_path: float | None
    isolated_passages: int
    unreachable_pairs: int
    hop_histogram: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _kind(graph, vertex: int) -> str:
    name = str(graph.vs[vertex]["name"])
    return name.split("-", 1)[0]


def _entity_path(graph, source: int, target: int, max_hops: int):
    """Find a shortest bounded path represented by entity node names.

    A path ``passage-entity-passage`` has entity-hop length 0. Each additional
    entity-to-entity transition increases the entity-hop length by one.
    """
    queue = deque([(source, (), -1)])
    seen = {(source, -1)}
    while queue:
        node, path, entity_hops = queue.popleft()
        for neighbor in graph.neighbors(node):
            kind = _kind(graph, neighbor)
            if neighbor == target and path:
                return path
            if kind != "entity":
                continue
            next_hops = entity_hops + 1
            if next_hops > max_hops:
                continue
            state = (neighbor, next_hops)
            if state in seen:
                continue
            seen.add(state)
            queue.append((neighbor, path + (str(graph.vs[neighbor]["name"]),), next_hops))
    return None


def _components(node_count: int, edges: list[tuple[int, int]]):
    adjacency = [[] for _ in range(node_count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    seen = set()
    groups = []
    for start in range(node_count):
        if start in seen:
            continue
        group = set([start])
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    group.add(neighbor)
                    queue.append(neighbor)
        groups.append(group)
    return adjacency, groups


def analyze_passages(
    graph,
    passage_vertices: list[int],
    max_hops: int = 4,
    selected_count: int | None = None,
) -> PassageTopology:
    edges: list[tuple[int, int]] = []
    hop_histogram: Counter[int] = Counter()
    selected_entities: set[str] = set()
    for left, right in combinations(range(len(passage_vertices)), 2):
        path = _entity_path(graph, passage_vertices[left], passage_vertices[right], max_hops)
        if path is None:
            continue
        edges.append((left, right))
        hop_histogram[len(path) - 1] += 1
        selected_entities.update(path)

    adjacency, groups = _components(len(passage_vertices), edges)
    distances = []
    for start in range(len(passage_vertices)):
        distances_from_start = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distances_from_start:
                    distances_from_start[neighbor] = distances_from_start[node] + 1
                    queue.append(neighbor)
        distances.extend(
            distance
            for node, distance in distances_from_start.items()
            if node > start
        )

    possible_edges = comb(len(passage_vertices), 2)
    return PassageTopology(
        selected_passages=selected_count if selected_count is not None else len(passage_vertices),
        mapped_passages=len(passage_vertices),
        selected_entities=len(selected_entities),
        edges=len(edges),
        components=len(groups),
        largest_component=max((len(group) for group in groups), default=0),
        average_degree=(2 * len(edges) / len(passage_vertices)) if passage_vertices else 0.0,
        edge_density=(len(edges) / possible_edges) if possible_edges else 0.0,
        diameter=max(distances) if distances else (0 if passage_vertices else None),
        average_shortest_path=(sum(distances) / len(distances)) if distances else None,
        isolated_passages=sum(not neighbors for neighbors in adjacency),
        unreachable_pairs=possible_edges - len(distances),
        hop_histogram=dict(sorted(hop_histogram.items())),
    )


def analyze_record(graph, passages: list[dict[str, Any]], k: int, max_hops: int = 4):
    """Analyze the first k serialized passages by matching graph content."""
    selected = passages[:k]
    content_to_vertex = {}
    for vertex in range(graph.vcount()):
        content = graph.vs[vertex]["content"]
        if content and content not in content_to_vertex:
            content_to_vertex[content] = vertex
    vertices = [content_to_vertex[p["text"]] for p in selected if p["text"] in content_to_vertex]
    return analyze_passages(graph, vertices, max_hops=max_hops, selected_count=len(selected))
