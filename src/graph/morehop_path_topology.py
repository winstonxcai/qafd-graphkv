"""Small, deterministic lexical propagation/path-topology primitives."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


_TOKEN = re.compile(r"\w+", re.UNICODE)
_TOLERANCE = 1e-12
_MAX_ITERATIONS = 10_000


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


def _tfidf(
    texts: Sequence[str],
    document_frequency: dict[str, int] | None = None,
    document_count: int | None = None,
) -> list[dict[str, float]]:
    tokenized = [_tokens(text) for text in texts]
    if document_frequency is None:
        document_frequency = {}
        for tokens in tokenized:
            for token in set(tokens):
                document_frequency[token] = document_frequency.get(token, 0) + 1
        document_count = len(texts)
    count = document_count if document_count is not None else len(texts)
    vectors = []
    for tokens in tokenized:
        term_frequency: dict[str, float] = {}
        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0.0) + 1.0
        vector = {
            token: frequency * (math.log((count + 1.0) / (document_frequency.get(token, 0) + 1.0)) + 1.0)
            for token, frequency in term_frequency.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        vectors.append({token: value / norm for token, value in vector.items()} if norm else {})
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def _knn(vectors: list[dict[str, float]], graph_k: int) -> tuple[list[list[int]], list[list[tuple[int, float]]]]:
    neighbors: list[list[int]] = []
    weighted: list[list[tuple[int, float]]] = []
    for source, vector in enumerate(vectors):
        candidates = [(target, _cosine(vector, vectors[target])) for target in range(len(vectors)) if target != source]
        candidates = [(target, score) for target, score in candidates if score > 0.0 and math.isfinite(score)]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        chosen = candidates[:graph_k]
        neighbors.append([target for target, _ in chosen])
        weighted.append(chosen)
    return neighbors, weighted


def _query_seed(
    question: str,
    document_vectors: list[dict[str, float]],
    document_frequency: dict[str, int],
) -> list[float]:
    query = _tfidf([question], document_frequency, len(document_vectors))[0]
    similarities = [_cosine(query, vector) for vector in document_vectors]
    total = sum(score for score in similarities if score > 0.0 and math.isfinite(score))
    if total <= 0.0:
        return [1.0 / len(document_vectors)] * len(document_vectors) if document_vectors else []
    return [score / total if score > 0.0 and math.isfinite(score) else 0.0 for score in similarities]


def _ppr(seed: list[float], weighted: list[list[tuple[int, float]]], alpha: float) -> tuple[list[float], int]:
    scores = seed[:]
    for iteration in range(1, _MAX_ITERATIONS + 1):
        next_scores = [(1.0 - alpha) * value for value in seed]
        dangling = 0.0
        for source, outgoing in enumerate(weighted):
            if not outgoing:
                dangling += scores[source]
                continue
            normalizer = sum(weight for _, weight in outgoing)
            if normalizer <= 0.0 or not math.isfinite(normalizer):
                dangling += scores[source]
                continue
            for target, weight in outgoing:
                next_scores[target] += alpha * scores[source] * weight / normalizer
        for target, value in enumerate(seed):
            next_scores[target] += alpha * dangling * value
        if sum(abs(left - right) for left, right in zip(next_scores, scores)) <= _TOLERANCE:
            return [value if math.isfinite(value) and value >= 0.0 else 0.0 for value in next_scores], iteration
        scores = next_scores
    return scores, _MAX_ITERATIONS


def build_path_topology(
    question: str,
    documents: list[str],
    top_k: int = 1,
    alpha: float = 0.85,
    graph_k: int = 3,
) -> dict:
    """Build a query-seeded PPR topology using document text only.

    The returned neighbor set is global: every target receives the same
    highest-PPR sources, including a selected source's own index.
    """
    if not isinstance(top_k, int) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if not isinstance(graph_k, int) or graph_k < 0:
        raise ValueError("graph_k must be a non-negative integer")
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and between 0 and 1")
    if any(not isinstance(document, str) for document in documents) or not isinstance(question, str):
        raise TypeError("question and documents must be strings")

    document_frequency: dict[str, int] = {}
    for document in documents:
        for token in set(_tokens(document)):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    vectors = _tfidf(documents, document_frequency, len(documents))
    graph, weighted = _knn(vectors, min(graph_k, max(0, len(documents) - 1)))
    seed = _query_seed(question, vectors, document_frequency)
    scores, iterations = _ppr(seed, weighted, alpha) if documents else ([], 0)
    selected = sorted(range(len(documents)), key=lambda index: (-scores[index], index))[: min(top_k, len(documents))]
    return {
        "neighbors": [selected[:] for _ in documents],
        "node_scores": scores,
        "method": "query_seeded_ppr_tfidf_cosine_knn",
        "config": {
            "top_k": top_k,
            "alpha": alpha,
            "graph_k": graph_k,
            "tolerance": _TOLERANCE,
            "iterations": iterations,
        },
        "diagnostic_graph": {
            "neighbors": graph,
            "weighted_neighbors": [[{"node": node, "weight": weight} for node, weight in row] for row in weighted],
            "seed": seed,
            "dangling_nodes": [index for index, row in enumerate(weighted) if not row],
            "document_only": True,
        },
    }


def stage_neighbors(n_documents: int, stages: Sequence[Sequence[int]]) -> list[list[int]]:
    """Return explicit staged cache reads.

    ``stages`` is ordered from earliest to latest and contains document IDs in
    each layer. A document in layer *i* reads exactly layer *i-1*; layer zero
    reads nothing. This is a separate staged engine description, not a claim
    that one PPR pass performed multi-hop propagation.
    """
    if not isinstance(n_documents, int) or n_documents < 0:
        raise ValueError("n_documents must be a non-negative integer")
    normalized: list[list[int]] = []
    owner: dict[int, int] = {}
    for layer, members in enumerate(stages):
        current = []
        for node in members:
            if not isinstance(node, int) or isinstance(node, bool) or not 0 <= node < n_documents:
                raise ValueError("stage document IDs must be valid document indices")
            if node not in current:
                current.append(node)
            owner.setdefault(node, layer)
        normalized.append(current)
    result = [[] for _ in range(n_documents)]
    for node, layer in owner.items():
        if layer:
            result[node] = normalized[layer - 1][:]
    return result
