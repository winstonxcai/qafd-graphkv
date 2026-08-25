"""Query-weighted passage-graph sparsification for recursive GraphKV.

The module is intentionally duck-typed around ``igraph.Graph`` so the graph
selection policy can be tested without importing either upstream project.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import exp, isfinite, log
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ScoredPassageEdge:
    src: int
    dst: int
    score: float
    entity_hops: int
    path: tuple[str, ...]
    path_terms: tuple[dict[str, float | str], ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["path"] = list(self.path)
        value["path_terms"] = [dict(term) for term in self.path_terms]
        return value


@dataclass(frozen=True)
class FinalPassageEdge:
    edge: ScoredPassageEdge
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.edge.to_dict(), "provenance": self.provenance}


@dataclass(frozen=True)
class SparsePassageGraph:
    candidate_edges: tuple[ScoredPassageEdge, ...]
    final_edges: tuple[FinalPassageEdge, ...]
    adjacency: tuple[tuple[int, ...], ...]
    candidate_stats: dict[str, Any]
    final_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_edges": [edge.to_dict() for edge in self.candidate_edges],
            "final_edges": [edge.to_dict() for edge in self.final_edges],
            "adjacency": [list(neighbors) for neighbors in self.adjacency],
            "candidate_stats": self.candidate_stats,
            "final_stats": self.final_stats,
        }


def _name(graph, vertex: int) -> str:
    return str(graph.vs[vertex]["name"])


def _label(graph, vertex: int) -> str:
    try:
        return str(graph.vs[vertex]["content"] or _name(graph, vertex))
    except (KeyError, TypeError):
        return _name(graph, vertex)


def _is_entity(graph, vertex: int) -> bool:
    return _name(graph, vertex).startswith("entity-")


def _is_passage(graph, vertex: int) -> bool:
    return _name(graph, vertex).startswith("chunk-")


def _passage_entities(graph, passage: int) -> tuple[int, ...]:
    return tuple(sorted(node for node in graph.neighbors(passage) if _is_entity(graph, node)))


def _entity_neighbors(graph, entity: int) -> tuple[int, ...]:
    return tuple(sorted(node for node in graph.neighbors(entity) if _is_entity(graph, node)))


def entity_idf(graph, entity: int, passage_count: int | None = None) -> float:
    """Return global passage-frequency IDF for one entity vertex."""
    if passage_count is None:
        passage_count = sum(_is_passage(graph, vertex) for vertex in range(graph.vcount()))
    passage_degree = len({node for node in graph.neighbors(entity) if _is_passage(graph, node)})
    return log((passage_count + 1) / (passage_degree + 1))


def bounded_entity_paths(
    graph,
    source_passage: int,
    target_passage: int,
    max_hops: int,
) -> Iterable[tuple[int, ...]]:
    """Yield simple entity paths with at most ``max_hops`` entity edges."""
    if max_hops not in (0, 1, 2):
        raise ValueError("max_hops must be one of 0, 1, or 2")
    targets = set(_passage_entities(graph, target_passage))
    for start in _passage_entities(graph, source_passage):
        stack = [(start, (start,))]
        while stack:
            current, path = stack.pop()
            if current in targets:
                yield path
            if len(path) - 1 >= max_hops:
                continue
            for neighbor in reversed(_entity_neighbors(graph, current)):
                if neighbor not in path:
                    stack.append((neighbor, path + (neighbor,)))


def score_entity_path(
    graph,
    path: tuple[int, ...],
    entity_scores: Mapping[str, float],
    decay_lambda: float,
    passage_count: int | None = None,
    idf_cache: dict[int, float] | None = None,
) -> tuple[float, tuple[dict[str, float | str], ...]]:
    if not path:
        raise ValueError("entity path must not be empty")
    if decay_lambda < 0:
        raise ValueError("decay_lambda must be non-negative")
    terms = []
    weighted_sum = 0.0
    idf_cache = {} if idf_cache is None else idf_cache
    for entity in path:
        entity_name = _name(graph, entity)
        flow = max(float(entity_scores.get(entity_name, 0.0)), 0.0)
        if entity not in idf_cache:
            idf_cache[entity] = entity_idf(graph, entity, passage_count)
        idf = idf_cache[entity]
        contribution = flow * idf
        weighted_sum += contribution
        terms.append(
            {
                "entity": entity_name,
                "label": _label(graph, entity),
                "flow": flow,
                "idf": idf,
                "contribution": contribution,
            }
        )
    score = exp(-decay_lambda * len(path)) * weighted_sum / len(path)
    return score, tuple(terms)


def score_passage_edges(
    graph,
    passage_vertices: list[int],
    entity_scores: Mapping[str, float],
    max_hops: int,
    decay_lambda: float,
) -> list[ScoredPassageEdge]:
    """Score each reachable passage pair by its best bounded entity path."""
    passage_count = sum(_is_passage(graph, vertex) for vertex in range(graph.vcount()))
    idf_cache: dict[int, float] = {}
    edges = []
    for src in range(len(passage_vertices)):
        for dst in range(src + 1, len(passage_vertices)):
            candidates = []
            for path in bounded_entity_paths(
                graph, passage_vertices[src], passage_vertices[dst], max_hops
            ):
                score, terms = score_entity_path(
                    graph,
                    path,
                    entity_scores,
                    decay_lambda,
                    passage_count,
                    idf_cache,
                )
                names = tuple(_name(graph, entity) for entity in path)
                candidates.append((score, names, terms))
            if not candidates:
                continue
            score, names, terms = min(
                candidates,
                key=lambda item: (-item[0], len(item[1]), item[1]),
            )
            edges.append(
                ScoredPassageEdge(
                    src=src,
                    dst=dst,
                    score=score,
                    entity_hops=len(names) - 1,
                    path=names,
                    path_terms=terms,
                )
            )
    return edges


def mutual_top_b(
    edges: Iterable[ScoredPassageEdge], node_count: int, top_b: int
) -> set[tuple[int, int]]:
    if top_b < 1:
        raise ValueError("top_b must be positive")
    edges = list(edges)
    incident: list[list[tuple[int, ScoredPassageEdge]]] = [[] for _ in range(node_count)]
    for edge in edges:
        if edge.score <= 0 or not isfinite(edge.score):
            continue
        incident[edge.src].append((edge.dst, edge))
        incident[edge.dst].append((edge.src, edge))
    selected = []
    for node, candidates in enumerate(incident):
        ranked = sorted(
            candidates,
            key=lambda item: (-item[1].score, len(item[1].path), item[0]),
        )[:top_b]
        selected.append({neighbor for neighbor, _edge in ranked})
    return {
        (edge.src, edge.dst)
        for edge in edges
        if edge.dst in selected[edge.src] and edge.src in selected[edge.dst]
    }


class _DisjointSet:
    def __init__(self, count: int):
        self.parent = list(range(count))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        self.parent[right_root] = left_root
        return True


def maximum_spanning_forest(
    edges: Iterable[ScoredPassageEdge], node_count: int
) -> set[tuple[int, int]]:
    """Kruskal maximum forest over the original candidate graph."""
    disjoint = _DisjointSet(node_count)
    selected = set()
    for edge in sorted(
        edges,
        key=lambda item: (-item.score, len(item.path), item.src, item.dst),
    ):
        if disjoint.union(edge.src, edge.dst):
            selected.add((edge.src, edge.dst))
    return selected


def graph_stats(node_count: int, edges: Iterable[tuple[int, int]]) -> dict[str, Any]:
    adjacency = [set() for _ in range(node_count)]
    edge_list = list(edges)
    for left, right in edge_list:
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(node_count))
    components = []
    diameter = 0
    for root in range(node_count):
        if root not in unseen:
            continue
        component = set()
        queue = deque([root])
        unseen.remove(root)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
        for start in component:
            distances = {start: 0}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            diameter = max(diameter, max(distances.values(), default=0))
    return {
        "nodes": node_count,
        "edges": len(edge_list),
        "components": len(components),
        "largest_component": max((len(component) for component in components), default=0),
        "average_degree": 2 * len(edge_list) / node_count if node_count else 0.0,
        "isolated_passages": sum(not neighbors for neighbors in adjacency),
        "diameter": diameter,
    }


def build_sparse_passage_graph(
    graph,
    passage_vertices: list[int],
    entity_scores: Mapping[str, float],
    max_hops: int,
    decay_lambda: float,
    top_b: int,
    restore_msf: bool,
) -> SparsePassageGraph:
    candidate_edges = score_passage_edges(
        graph, passage_vertices, entity_scores, max_hops, decay_lambda
    )
    mutual = mutual_top_b(candidate_edges, len(passage_vertices), top_b)
    msf = maximum_spanning_forest(candidate_edges, len(passage_vertices)) if restore_msf else set()
    selected = mutual | msf
    edge_by_pair = {(edge.src, edge.dst): edge for edge in candidate_edges}
    final_edges = []
    for pair in sorted(selected):
        if pair in mutual and pair in msf:
            provenance = "mutual+msf"
        elif pair in mutual:
            provenance = "mutual"
        else:
            provenance = "msf"
        final_edges.append(FinalPassageEdge(edge_by_pair[pair], provenance))
    adjacency = [set() for _ in passage_vertices]
    for left, right in selected:
        adjacency[left].add(right)
        adjacency[right].add(left)
    candidate_pairs = [(edge.src, edge.dst) for edge in candidate_edges]
    return SparsePassageGraph(
        candidate_edges=tuple(candidate_edges),
        final_edges=tuple(final_edges),
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency),
        candidate_stats=graph_stats(len(passage_vertices), candidate_pairs),
        final_stats=graph_stats(len(passage_vertices), selected),
    )
