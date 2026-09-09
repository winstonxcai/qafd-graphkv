"""Deterministic lexical query topologies for MoreHop-style passages.

This module intentionally has no dataset, answer, hop-label, or embedding
dependencies.  A topology edge is ``target -> source``.  Consequently the
reported edge count includes diagonal edges when a target is itself one of
the selected sources; ``diagonal_edge_count`` reports that part separately.
"""

from __future__ import annotations

from collections import Counter
from math import isfinite, log
import random
import re
from typing import Any, Sequence


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_K1 = 1.5
_B = 0.75


def _tokens(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("question and documents must contain strings")
    return _TOKEN_RE.findall(text.casefold())


def _validate_documents(documents: Sequence[str]) -> list[str]:
    if isinstance(documents, (str, bytes)):
        raise TypeError("documents must be a sequence of strings")
    result = list(documents)
    for document in result:
        if not isinstance(document, str):
            raise TypeError("question and documents must contain strings")
    return result


def _validate_k(top_k: int, document_count: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 1 or top_k > document_count:
        raise ValueError("top_k must be between 1 and len(documents)")


def _bm25_scores(question: str, documents: Sequence[str]) -> list[float]:
    query_tokens = _tokens(question)
    document_tokens = [_tokens(document) for document in documents]
    document_count = len(document_tokens)
    average_length = (
        sum(len(tokens) for tokens in document_tokens) / document_count
        if document_count
        else 0.0
    )
    document_frequency = Counter(
        token for tokens in document_tokens for token in set(tokens)
    )
    scores = []
    for tokens in document_tokens:
        counts = Counter(tokens)
        length = len(tokens)
        score = 0.0
        for token in query_tokens:
            frequency = counts[token]
            if not frequency:
                continue
            # The +1 form keeps IDF non-negative, including in a one-document
            # corpus, while remaining a standard BM25 lexical score.
            idf = log(
                1.0
                + (document_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + _K1 * (
                1.0 - _B + _B * length / average_length
            ) if average_length else frequency + _K1
            score += idf * frequency * (_K1 + 1.0) / denominator
        if not isfinite(score):
            raise ValueError("BM25 produced a non-finite score")
        scores.append(float(score))
    return scores


def _validate_source_indices(documents: Sequence[str], source_indices: Sequence[int]) -> list[int]:
    if isinstance(source_indices, (str, bytes)):
        raise TypeError("source_indices must be a sequence of integers")
    sources = list(source_indices)
    if not sources:
        raise ValueError("source_indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in sources):
        raise TypeError("source_indices must contain integers")
    if any(index < 0 or index >= len(documents) for index in sources):
        raise IndexError("source index is outside documents")
    if len(set(sources)) != len(sources):
        raise ValueError("source_indices must be unique")
    return sources


def build_source_topology(
    documents: list[str],
    source_indices: list[int],
    *,
    method: str = "source_control",
    node_scores: Sequence[float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize one explicit source set for every target document.

    This is the common constructor for random-source, reverse-rank, and
    released-last-k controls.  Those controls only need to provide a list of
    indices; this function never changes its order.
    """
    checked_documents = _validate_documents(documents)
    sources = _validate_source_indices(checked_documents, source_indices)
    if node_scores is None:
        scores = [0.0] * len(checked_documents)
    else:
        scores = [float(score) for score in node_scores]
        if len(scores) != len(checked_documents) or not all(isfinite(score) for score in scores):
            raise ValueError("node_scores must be finite and document-indexed")
    result_config = dict(config or {})
    result_config.update({"source_indices": list(sources), "edge_count_includes_diagonal": True})
    return {
        "neighbors": [list(sources) for _ in checked_documents],
        "node_scores": scores,
        "method": method,
        "config": result_config,
        "edge_count": len(checked_documents) * len(sources),
        "diagonal_edge_count": sum(index in sources for index in range(len(checked_documents))),
    }


def random_source_indices(documents: Sequence[str], top_k: int, seed: int = 0) -> list[int]:
    """Return a reproducible random source selection in shuffled order."""
    checked_documents = _validate_documents(documents)
    _validate_k(top_k, len(checked_documents))
    indices = list(range(len(checked_documents)))
    random.Random(seed).shuffle(indices)
    return indices[:top_k]


def reverse_rank_source_indices(source_indices: Sequence[int]) -> list[int]:
    """Return a reverse-rank control without changing document indices."""
    return list(reversed(source_indices))


def released_last_k_source_indices(documents: Sequence[str], top_k: int) -> list[int]:
    """Return the final ``top_k`` documents in their released order."""
    checked_documents = _validate_documents(documents)
    _validate_k(top_k, len(checked_documents))
    return list(range(len(checked_documents) - top_k, len(checked_documents)))


def build_query_topology(question: str, documents: list[str], top_k: int = 1) -> dict[str, Any]:
    """Build a query-only BM25 topology over the ordered document strings."""
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    checked_documents = _validate_documents(documents)
    _validate_k(top_k, len(checked_documents))
    scores = _bm25_scores(question, checked_documents)
    ranked_sources = sorted(range(len(checked_documents)), key=lambda index: (-scores[index], index))[:top_k]
    return build_source_topology(
        checked_documents,
        ranked_sources,
        method="bm25_query",
        node_scores=scores,
        config={"top_k": top_k, "k1": _K1, "b": _B, "ranking": "score descending, index ascending"},
    )
