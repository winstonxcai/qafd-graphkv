"""GPU gates for variable-k CSA and dense-control top-token equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import igraph as ig

from src.csa.graph import qafd_prior_matrix
from src.eval.csa_benchmark import parse_passage, read_retrieval, request_generation


def csa_payload(question, passages, graph_scores, backend="gathered_sdpa"):
    return {
        "mode": "csa",
        "question": question,
        "passages": passages,
        "graph_scores": graph_scores,
        "beta": 0.5,
        "top_b": min(4, len(passages) - 1),
        "pooling": "normalized_token_mean",
        "backend": backend,
        "max_new_tokens": 8,
    }


def validate_trace(result: dict, passage_count: int) -> None:
    traces = result["routing_trace"]
    if not traces:
        raise ValueError("smoke response has no routing trace")
    for layer, trace in enumerate(traces):
        if trace["layer"] != layer or len(trace["selected"]) != passage_count:
            raise ValueError("routing trace is incomplete or out of order")
        for target, sources in enumerate(trace["selected"]):
            if any(source >= target for source in sources):
                raise ValueError("routing selected a non-causal passage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--port", type=int, default=8980)
    parser.add_argument("--real-prompts", type=int, default=10)
    args = parser.parse_args()

    synthetic_passages = [
        {"title": "Alpha", "text": "Alpha links to Beta."},
        {"title": "Beta", "text": "Beta links to Gamma."},
        {"title": "Gamma", "text": "Gamma is the final answer."},
    ]
    synthetic = request_generation(
        args.port,
        csa_payload(
            "What is the final linked name?",
            synthetic_passages,
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ),
    )
    validate_trace(synthetic, 3)

    _metadata, retrieval_rows = read_retrieval(args.retrieval)
    graph = ig.Graph.Read_Pickle(args.graph)
    content_to_vertex = {
        vertex["content"]: vertex.index
        for vertex in graph.vs
        if str(vertex["name"]).startswith("chunk-") and vertex["content"]
    }
    comparisons = []
    for row in retrieval_rows[: args.real_prompts]:
        documents = row["docs"][:5]
        passages = [parse_passage(document) for document in documents]
        vertices = [content_to_vertex[document] for document in documents]
        graph_scores, _paths = qafd_prior_matrix(
            graph, vertices, row["entity_scores"]
        )
        base = {
            "question": row["question"],
            "passages": passages,
            "graph_scores": graph_scores,
            "beta": 0.0,
            "top_b": 4,
            "pooling": "normalized_token_mean",
            "max_new_tokens": 8,
        }
        vanilla = request_generation(args.port, {**base, "mode": "vanilla", "backend": "gathered_sdpa"})
        gathered = request_generation(args.port, {**base, "mode": "csa", "backend": "gathered_sdpa"})
        reference = request_generation(args.port, {**base, "mode": "csa", "backend": "dense_reference"})
        validate_trace(gathered, 5)
        validate_trace(reference, 5)
        token_ids = [vanilla["first_token_id"], gathered["first_token_id"], reference["first_token_id"]]
        if len(set(token_ids)) != 1:
            raise ValueError(f"B=4 top-token mismatch for QID {row['qid']}: {token_ids}")
        comparisons.append({"qid": row["qid"], "first_token_id": token_ids[0]})
    print(json.dumps({"three_passage_smoke": "passed", "dense_top_token_checks": comparisons}, indent=2))


if __name__ == "__main__":
    main()
