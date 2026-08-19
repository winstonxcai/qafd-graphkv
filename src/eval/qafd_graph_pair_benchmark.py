"""Benchmark one graph-specific QAFD+GraphKV method or its matched control."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import igraph as ig
import requests

from src.eval.test4_benchmark import (
    PassageGraph,
    build_suffix,
    parse_passage,
    score_prediction,
)


QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "has", "have", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "whose", "why", "with",
}


def query_overlap(question: str, passage: str) -> int:
    """Count content-word overlap without consulting answers or gold passages."""
    question_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", question.lower())
        if term not in QUESTION_STOPWORDS
    }
    passage_terms = set(re.findall(r"[a-z0-9]+", passage.lower()))
    return len(question_terms & passage_terms)


def choose_center_and_neighbors(
    adjacency: list[set[int]],
    scores: list[float],
    center_rule: str,
    max_neighbors: int,
    passages: list[str] | None = None,
    question: str | None = None,
) -> tuple[int, list[int]]:
    if len(scores) < 2:
        raise ValueError("Graph-specific generation requires at least two passages")
    if max_neighbors < 1:
        raise ValueError("max_neighbors must be positive")

    edges = [
        (i, j)
        for i, neighbors in enumerate(adjacency)
        for j in neighbors
        if i < j
    ]
    if center_rule in {"best_edge", "bridge_target"} and edges:
        first, second = max(
            edges,
            key=lambda edge: (
                scores[edge[0]] + scores[edge[1]],
                max(scores[edge[0]], scores[edge[1]]),
                -min(edge),
            ),
        )
        if center_rule == "bridge_target":
            if passages is None or question is None:
                raise ValueError(
                    "bridge_target requires the question and passage texts"
                )
            first_key = (query_overlap(question, passages[first]), scores[first], -first)
            second_key = (
                query_overlap(question, passages[second]), scores[second], -second
            )
            # The endpoint with less direct query overlap is more likely to be
            # the second-hop target. It becomes the passage that attends to all
            # neighbor caches in GraphKV's directed center/neighbor operation.
            center = first if first_key < second_key else second
        else:
            center = first if scores[first] >= scores[second] else second
        required_neighbor = second if center == first else first
    elif center_rule == "weighted_degree":
        center = max(
            range(len(scores)),
            key=lambda node: (
                sum(scores[n] for n in adjacency[node]),
                len(adjacency[node]),
                scores[node],
                -node,
            ),
        )
        required_neighbor = None
    elif center_rule == "top_seed":
        center = max(range(len(scores)), key=lambda node: (scores[node], -node))
        required_neighbor = None
    else:
        center = max(range(len(scores)), key=lambda node: (scores[node], -node))
        required_neighbor = None

    ranked_graph_neighbors = sorted(
        adjacency[center], key=lambda node: (-scores[node], node)
    )
    neighbors = []
    if required_neighbor is not None:
        neighbors.append(required_neighbor)
    neighbors.extend(n for n in ranked_graph_neighbors if n not in neighbors)

    # GraphKV's graph path requires at least one neighbor. Preserve all 250
    # questions by falling back to the strongest retrieved non-center passage.
    if not neighbors:
        neighbors = sorted(
            (node for node in range(len(scores)) if node != center),
            key=lambda node: (-scores[node], node),
        )
    return center, neighbors[:max_neighbors]


def connected_components(adjacency: list[set[int]]) -> list[set[int]]:
    components = []
    unseen = set(range(len(adjacency)))
    while unseen:
        root = min(unseen)
        component = {root}
        frontier = [root]
        unseen.remove(root)
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        components.append(component)
    return components


def choose_component_stars(
    adjacency: list[set[int]],
    scores: list[float],
    max_stars: int,
    max_neighbors: int,
) -> list[tuple[int, list[int]]]:
    """Choose one high-scoring directed star from each of the best components."""
    if max_stars < 1:
        raise ValueError("max_stars must be positive")
    candidates = []
    for component in connected_components(adjacency):
        edges = [
            (left, right)
            for left in component
            for right in adjacency[left]
            if left < right and right in component
        ]
        if not edges:
            continue
        first, second = max(
            edges,
            key=lambda edge: (
                scores[edge[0]] + scores[edge[1]],
                max(scores[edge[0]], scores[edge[1]]),
                -min(edge),
            ),
        )
        center = first if scores[first] >= scores[second] else second
        required_neighbor = second if center == first else first
        neighbors = [required_neighbor]
        neighbors.extend(
            node
            for node in sorted(
                adjacency[center], key=lambda node: (-scores[node], node)
            )
            if node != required_neighbor
        )
        candidates.append(
            (
                scores[first] + scores[second],
                max(scores[first], scores[second]),
                -min(component),
                center,
                neighbors[:max_neighbors],
            )
        )
    candidates.sort(reverse=True)
    stars = [(center, neighbors) for _, _, _, center, neighbors in candidates[:max_stars]]
    if stars:
        return stars
    return [
        choose_center_and_neighbors(
            adjacency, scores, "top_seed", max_neighbors=max_neighbors
        )
    ]


def request_generation(url: str, payload: dict) -> tuple[str, float]:
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=1800)
    response.raise_for_status()
    return response.json()["generated"], time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", choices=["sequential", "qafd_graphkv"], required=True)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--hops", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument(
        "--center-rule",
        choices=["best_edge", "bridge_target", "weighted_degree", "top_seed"],
        default="best_edge",
    )
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--max-stars", type=int, default=1)
    parser.add_argument(
        "--prompt-style",
        choices=["default", "concise", "multihop"],
        default="concise",
    )
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args()

    retrieval = json.loads(args.results.read_text())["per_query"][: args.limit]
    question_rows = json.loads(args.questions.read_text())
    answers = {row["question"]: [row["answer"]] for row in question_rows}
    graph = ig.Graph.Read_Pickle(args.graph)
    passage_graph = PassageGraph(graph, args.hops)
    content_to_vertex = {
        vertex["content"]: vertex.index
        for vertex in graph.vs
        if vertex["name"].startswith("chunk-") and vertex["content"]
    }
    prefix = (
        "<|user|>\nYou are an intelligent AI assistant. Answer questions "
        "using only the reference documents.\n\n"
    )
    endpoint_name = (
        "generate_matched_sequential"
        if args.method == "sequential"
        else "generate_qafd_graphkv"
    )
    if args.max_stars > 1:
        endpoint_name += "_batch"
    endpoint = f"http://127.0.0.1:{args.port}/{endpoint_name}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.strategy_name}.jsonl"
    totals = {"em": 0.0, "f1": 0.0, "seconds": 0.0}

    with output_path.open("w") as handle:
        for qid, item in enumerate(retrieval):
            docs = item["docs"][: args.pool_k]
            scores = (item.get("doc_scores") or [args.pool_k - i for i in range(args.pool_k)])[: args.pool_k]
            vertices = [content_to_vertex.get(doc) for doc in docs]
            if any(vertex is None for vertex in vertices):
                raise ValueError(f"QID {qid} has passages missing from the graph")
            adjacency = passage_graph.adjacency(vertices)
            formatted = []
            for doc in docs:
                title, body = parse_passage(doc)
                formatted.append(f"- Title: {title}\n{body}\n")
            question = item["question"]
            if args.max_stars > 1:
                stars = choose_component_stars(
                    adjacency, scores, args.max_stars, args.max_neighbors
                )
            else:
                stars = [
                    choose_center_and_neighbors(
                        adjacency,
                        scores,
                        args.center_rule,
                        args.max_neighbors,
                        passages=docs,
                        question=item["question"],
                    )
                ]
            center_indices = [center for center, _ in stars]
            neighbor_index_groups = [neighbors for _, neighbors in stars]
            payload = {
                "prefix": prefix,
                "query": build_suffix(question, args.prompt_style),
            }
            if args.max_stars > 1:
                payload.update(
                    {
                        "centers": [formatted[index] for index in center_indices],
                        "neighbor_groups": [
                            [formatted[index] for index in neighbors]
                            for neighbors in neighbor_index_groups
                        ],
                    }
                )
            else:
                payload.update(
                    {
                        "center": formatted[center_indices[0]],
                        "neighbors": [
                            formatted[index] for index in neighbor_index_groups[0]
                        ],
                    }
                )
            generated, seconds = request_generation(endpoint, payload)
            em, f1 = score_prediction(generated, answers[question])
            totals["em"] += em
            totals["f1"] += f1
            totals["seconds"] += seconds
            handle.write(
                json.dumps(
                    {
                        "qid": qid,
                        "question": question,
                        "answers": answers[question],
                        "generated": generated,
                        "em": em,
                        "f1": f1,
                        "seconds": seconds,
                        "center_index": center_indices[0],
                        "neighbor_indices": neighbor_index_groups[0],
                        "center_indices": center_indices,
                        "neighbor_index_groups": neighbor_index_groups,
                    }
                )
                + "\n"
            )
            handle.flush()

    summary = [
        {
            "method": args.strategy_name,
            "questions": args.limit,
            "em": totals["em"] / args.limit,
            "f1": totals["f1"] / args.limit,
            "avg_seconds": totals["seconds"] / args.limit,
            "total_seconds": totals["seconds"],
        }
    ]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
