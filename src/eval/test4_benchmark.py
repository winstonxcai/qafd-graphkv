"""Test 4: compare GraphKV generation with QAFD-derived passage ordering.

The recursive rows use the project recursive inference server and propagate
KV caches over the h<=1 passage graph. The adaptive row uses
``T=min(diameter(G_P), 3)`` independently for each question.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import deque
from pathlib import Path

import igraph as ig
import requests


def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def score_prediction(prediction: str, answers: list[str]) -> tuple[float, float]:
    pred = normalize(prediction)
    gold = [normalize(a) for a in answers]
    em = float(any(a in pred for a in gold))
    pred_tokens = set(pred.split())
    f1 = 0.0
    for answer in gold:
        answer_tokens = set(answer.split())
        overlap = len(pred_tokens & answer_tokens)
        if overlap:
            precision = overlap / max(len(pred_tokens), 1)
            recall = overlap / max(len(answer_tokens), 1)
            f1 = max(f1, 2 * precision * recall / (precision + recall))
    return em, f1


def parse_passage(text: str) -> tuple[str, str]:
    title, _, body = text.partition("\n")
    return title, body or title


def build_suffix(question: str, style: str) -> str:
    if style == "concise":
        instruction = (
            "Answer using only the provided search documents. Resolve any "
            "intermediate entity needed by the question, then return only the "
            "shortest answer phrase with no explanation."
        )
    elif style == "multihop":
        instruction = (
            "Answer using only the provided search documents. Follow the "
            "question's entity links step by step, verify each hop against the "
            "documents, and give a concise final answer with no unrelated facts."
        )
    else:
        instruction = (
            "Please write a high-quality answer for the given question using "
            "only the provided search documents (some of which might be irrelevant)."
        )
    return f"{instruction} \n Question: {question} \n<|assistant|>\n"


def remap_adjacency(adjacency: list[set[int]], old_order: list[int]) -> list[set[int]]:
    """Remap adjacency indices after passages move from old to new order."""
    old_to_new = {old: new for new, old in enumerate(old_order)}
    return [
        {old_to_new[neighbor] for neighbor in adjacency[old]}
        for old in old_order
    ]


def graph_diameter(adjacency: list[set[int]]) -> int:
    """Return the largest finite shortest-path distance in an undirected graph.

    Test 4 defines adaptive recursion from the diameter of the retrieved
    passage graph. For a disconnected graph this is the maximum diameter of
    any connected component; an empty or singleton graph has diameter zero.
    """
    diameter = 0
    for start in range(len(adjacency)):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        diameter = max(diameter, max(distances.values(), default=0))
    return diameter


def select_graph_indices(
    adjacency: list[set[int]],
    scores: list[float],
    limit: int,
    rule: str,
) -> list[int]:
    """Select a compact graph-supported context from a larger retrieval pool."""
    if limit < 1:
        raise ValueError("Graph context limit must be positive")
    limit = min(limit, len(scores))
    score_order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    selected: list[int] = []

    if rule == "best_edge":
        edges = [
            (i, j)
            for i, neighbors in enumerate(adjacency)
            for j in neighbors
            if i < j
        ]
        if edges:
            first, second = max(
                edges,
                key=lambda edge: (
                    scores[edge[0]] + scores[edge[1]],
                    max(scores[edge[0]], scores[edge[1]]),
                    -min(edge),
                ),
            )
            supported = {first, second} | adjacency[first] | adjacency[second]
            selected.extend(i for i in score_order if i in supported)
    elif rule == "top_component":
        components = []
        unseen = set(range(len(scores)))
        while unseen:
            root = min(unseen)
            component = set()
            frontier = [root]
            unseen.remove(root)
            while frontier:
                node = frontier.pop()
                component.add(node)
                for neighbor in adjacency[node]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        frontier.append(neighbor)
            components.append(component)
        best_component = max(
            components,
            key=lambda component: (
                sum(sorted((scores[i] for i in component), reverse=True)[:2]),
                max(scores[i] for i in component),
                -min(component),
            ),
        )
        selected.extend(i for i in score_order if i in best_component)
    else:
        raise ValueError(f"Unknown graph selection rule: {rule}")

    selected_set = set(selected)
    selected.extend(i for i in score_order if i not in selected_set)
    return selected[:limit]


def resolve_order_key(method: str, sequential_order: str) -> str:
    if method != "sequential" or sequential_order == "retrieval":
        return method
    return sequential_order


class PassageGraph:
    def __init__(self, graph: ig.Graph, max_hops: int):
        self.graph = graph
        self.max_hops = max_hops
        self.entity_neighbors = {
            v.index: [n for n in graph.neighbors(v.index) if graph.vs[n]["name"].startswith("entity-")]
            for v in graph.vs
            if graph.vs[v.index]["name"].startswith("entity-")
        }

    def _passage_entities(self, passage: int) -> set[int]:
        return {
            node
            for node in self.graph.neighbors(passage)
            if self.graph.vs[node]["name"].startswith("entity-")
        }

    def _reachable_entities(self, source: int) -> set[int]:
        frontier = self._passage_entities(source)
        visited = set(frontier)
        for _ in range(self.max_hops):
            next_frontier = set()
            for entity in frontier:
                next_frontier.update(self.entity_neighbors.get(entity, []))
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def _bounded(self, source: int, target: int) -> bool:
        return bool(self._reachable_entities(source) & self._passage_entities(target))

    def adjacency(self, vertices: list[int]) -> list[set[int]]:
        adjacency = {i: set() for i in range(len(vertices))}
        reachable = [self._reachable_entities(vertex) for vertex in vertices]
        passage_entities = [self._passage_entities(vertex) for vertex in vertices]
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if reachable[i] & passage_entities[j]:
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        return [adjacency[i] for i in range(len(vertices))]

    def order(self, vertices: list[int]) -> list[int]:
        adjacency = self.adjacency(vertices)
        ordered: list[int] = []
        seen: set[int] = set()
        for root in range(len(vertices)):
            if root in seen:
                continue
            queue = deque([root])
            seen.add(root)
            while queue:
                current = queue.popleft()
                ordered.append(current)
                for neighbor in sorted(adjacency[current]):
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
        return [vertices[i] for i in ordered]


def request_generation(url: str, blocks: list[str], extra: dict | None = None) -> tuple[str, float]:
    start = time.perf_counter()
    response = requests.post(
        url,
        data=json.dumps({"blocks": blocks, **(extra or {})}),
        headers={"Content-Type": "application/json"},
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()["generated"], time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--recursive-port", type=int, default=8772)
    parser.add_argument("--include-recursive", action="store_true")
    parser.add_argument(
        "--method",
        choices=[
            "sequential",
            "block_rag",
            "graphkv_original",
            "qafd_h0",
            "qafd_h1",
            "qafd_h2",
            "qafd_recursive_h1_t2",
            "qafd_adaptive_recursive_h1",
        ],
    )
    parser.add_argument("--strategy-name")
    parser.add_argument(
        "--prompt-style",
        choices=["default", "concise", "multihop"],
        default="default",
    )
    parser.add_argument("--include-graph-links", action="store_true")
    parser.add_argument(
        "--sequential-order",
        choices=["retrieval", "qafd_h0", "qafd_h1", "qafd_h2"],
        default="retrieval",
    )
    parser.add_argument(
        "--graph-selection",
        choices=["none", "best_edge", "top_component"],
        default="none",
    )
    parser.add_argument("--pool-k", type=int)
    parser.add_argument("--context-k", type=int)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.strategy_name and not args.method:
        parser.error("--strategy-name requires a single --method")
    if args.graph_selection != "none" and args.method not in {"qafd_h0", "qafd_h1", "qafd_h2"}:
        parser.error("graph selection requires one of the QAFD ordering methods")
    if args.graph_selection != "none" and not args.context_k:
        parser.error("graph selection requires --context-k")
    if args.sequential_order != "retrieval" and args.method != "sequential":
        parser.error("--sequential-order is only valid with --method sequential")

    results = json.loads(Path(args.results).read_text())["per_query"][: args.limit]
    questions = json.loads(Path(args.questions).read_text())
    answers = {q["question"]: [q["answer"]] for q in questions}
    graph = ig.Graph.Read_Pickle(args.graph)
    content_to_vertex = {
        v["content"]: v.index
        for v in graph.vs
        if v["name"].startswith("chunk-") and v["content"]
    }
    prefix = "<|user|>\nYou are an intelligent AI assistant. Please answer questions based on the user instructions. Below are some reference documents that may help you in answering the user's question.\n\n"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = ["sequential", "block_rag", "graphkv_original", "qafd_h0", "qafd_h1", "qafd_h2"]
    if args.include_recursive:
        methods.extend(["qafd_recursive_h1_t2", "qafd_adaptive_recursive_h1"])
    if args.method:
        methods = [args.method]
    labels = {
        method: args.strategy_name if args.strategy_name and method == args.method else method
        for method in methods
    }
    handles = {
        method: (output_dir / f"{labels[method]}.jsonl").open("w")
        for method in methods
    }
    graph_cache = {h: PassageGraph(graph, h) for h in (0, 1, 2)}
    totals = {method: {"em": 0.0, "f1": 0.0, "seconds": 0.0} for method in methods}

    try:
        for qid, item in enumerate(results):
            pool_k = args.pool_k or args.k
            docs = item["docs"][:pool_k]
            scores = (item.get("doc_scores") or [float(pool_k - i) for i in range(pool_k)])[:pool_k]
            documents = []
            vertices = []
            for doc, score in zip(docs, scores):
                title, body = parse_passage(doc)
                documents.append({"title": title, "text": body, "score": float(score)})
                vertices.append(content_to_vertex.get(doc))
            if any(vertex is None for vertex in vertices):
                raise ValueError(f"qid={qid} has passages missing from graph")

            if args.graph_selection != "none":
                graph_h = int(args.method[-1])
                pool_adjacency = graph_cache[graph_h].adjacency(vertices)
                selected = select_graph_indices(
                    pool_adjacency,
                    scores,
                    args.context_k,
                    args.graph_selection,
                )
                documents = [documents[index] for index in selected]
                vertices = [vertices[index] for index in selected]

            question = item["question"]
            suffix = build_suffix(question, args.prompt_style)
            base_docs = sorted(documents, key=lambda doc: doc["score"])
            orders = {
                "sequential": documents,
                "block_rag": documents,
                "graphkv_original": base_docs,
            }
            vertex_orders = {}
            order_indices = {}
            adjacency_by_h = {}
            original_index = {vertex: index for index, vertex in enumerate(vertices)}
            for h in (0, 1, 2):
                vertex_order = graph_cache[h].order(vertices)
                vertex_orders[h] = vertex_order
                order_indices[h] = [original_index[vertex] for vertex in vertex_order]
                adjacency_by_h[h] = remap_adjacency(
                    graph_cache[h].adjacency(vertices), order_indices[h]
                )
                by_vertex = {vertex: doc for vertex, doc in zip(vertices, documents)}
                orders[f"qafd_h{h}"] = [by_vertex[vertex] for vertex in vertex_order]
            recursive_methods = {
                "qafd_recursive_h1_t2",
                "qafd_adaptive_recursive_h1",
            }
            if args.include_recursive or args.method in recursive_methods:
                for recursive_method in recursive_methods:
                    orders[recursive_method] = orders["qafd_h1"]

            # PassageGraph.adjacency returns indices in retrieval order. The
            # recursive endpoint receives passages in QAFD h<=1 order, so
            # remap every edge into that same order before propagation.
            h1_adjacency = adjacency_by_h[1]

            for method in methods:
                ordered_docs = orders[resolve_order_key(method, args.sequential_order)]
                contexts = []
                graph_h = (
                    int(method[-1])
                    if args.include_graph_links and method in {"qafd_h0", "qafd_h1", "qafd_h2"}
                    else None
                )
                for index, document in enumerate(ordered_docs):
                    link_note = ""
                    if graph_h is not None:
                        linked_titles = [
                            ordered_docs[neighbor]["title"]
                            for neighbor in sorted(adjacency_by_h[graph_h][index])
                        ]
                        links = "; ".join(linked_titles) if linked_titles else "none"
                        link_note = f"QAFD-linked passages: {links}\n"
                    contexts.append(
                        f"- Title: {document['title']}\n{link_note}{document['text']}\n"
                    )
                blocks = [prefix, ""] + contexts + [suffix]
                if method == "sequential":
                    endpoint = f"http://127.0.0.1:{args.port}/generate_vanilla"
                elif method in {
                    "qafd_recursive_h1_t2",
                    "qafd_adaptive_recursive_h1",
                }:
                    endpoint = f"http://127.0.0.1:{args.recursive_port}/generate_recursive"
                elif method == "block_rag":
                    endpoint = f"http://127.0.0.1:{args.port}/generate_block"
                else:
                    endpoint = f"http://127.0.0.1:{args.port}/generate_gapemp"
                extra = None
                if method in {
                    "qafd_recursive_h1_t2",
                    "qafd_adaptive_recursive_h1",
                }:
                    rounds = 2
                    if method == "qafd_adaptive_recursive_h1":
                        rounds = min(graph_diameter(h1_adjacency), 3)
                    extra = {
                        "neighbors": [sorted(n) for n in h1_adjacency],
                        "rounds": rounds,
                        "max_new_tokens": 128,
                    }
                generated, seconds = request_generation(endpoint, blocks, extra)
                em, f1 = score_prediction(generated, answers[question])
                totals[method]["em"] += em
                totals[method]["f1"] += f1
                totals[method]["seconds"] += seconds
                handles[method].write(json.dumps({"qid": qid, "question": question, "answers": answers[question], "generated": generated, "em": em, "f1": f1, "seconds": seconds}) + "\n")
                handles[method].flush()
    finally:
        for handle in handles.values():
            handle.close()

    summary = []
    for method, values in totals.items():
        summary.append({"method": labels[method], "questions": args.limit, "em": values["em"] / args.limit, "f1": values["f1"] / args.limit, "avg_seconds": values["seconds"] / args.limit, "total_seconds": values["seconds"]})
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
