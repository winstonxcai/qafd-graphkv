"""Test 4: compare GraphKV generation with QAFD-derived passage ordering.

The recursive row uses the project recursive inference server and propagates
KV caches for two rounds over the h<=1 passage graph.
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


class PassageGraph:
    def __init__(self, graph: ig.Graph, max_hops: int):
        self.graph = graph
        self.max_hops = max_hops
        self.entity_neighbors = {
            v.index: [n for n in graph.neighbors(v.index) if graph.vs[n]["name"].startswith("entity-")]
            for v in graph.vs
            if graph.vs[v.index]["name"].startswith("entity-")
        }

    def _bounded(self, source: int, target: int) -> bool:
        start = [n for n in self.graph.neighbors(source) if self.graph.vs[n]["name"].startswith("entity-")]
        target_entities = set(
            n for n in self.graph.neighbors(target) if self.graph.vs[n]["name"].startswith("entity-")
        )
        if set(start) & target_entities:
            return True
        frontier = set(start)
        visited = set(frontier)
        for _ in range(self.max_hops):
            next_frontier = set()
            for entity in frontier:
                next_frontier.update(self.entity_neighbors.get(entity, []))
            next_frontier -= visited
            if next_frontier & target_entities:
                return True
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return False

    def adjacency(self, vertices: list[int]) -> list[set[int]]:
        adjacency = {i: set() for i in range(len(vertices))}
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if self._bounded(vertices[i], vertices[j]):
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
    parser.add_argument("--method", choices=["sequential", "block_rag", "graphkv_original", "qafd_h0", "qafd_h1", "qafd_h2", "qafd_recursive_h1_t2"])
    parser.add_argument("--strategy-name")
    parser.add_argument(
        "--prompt-style",
        choices=["default", "concise", "multihop"],
        default="default",
    )
    parser.add_argument("--include-graph-links", action="store_true")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.strategy_name and not args.method:
        parser.error("--strategy-name requires a single --method")

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
        methods.append("qafd_recursive_h1_t2")
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
            docs = item["docs"][: args.k]
            scores = item.get("doc_scores") or [float(args.k - i) for i in range(args.k)]
            documents = []
            vertices = []
            for doc, score in zip(docs, scores):
                title, body = parse_passage(doc)
                documents.append({"title": title, "text": body, "score": float(score)})
                vertices.append(content_to_vertex.get(doc))
            if any(vertex is None for vertex in vertices):
                raise ValueError(f"qid={qid} has passages missing from graph")

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
            if args.include_recursive or args.method == "qafd_recursive_h1_t2":
                orders["qafd_recursive_h1_t2"] = orders["qafd_h1"]

            # PassageGraph.adjacency returns indices in retrieval order. The
            # recursive endpoint receives passages in QAFD h<=1 order, so
            # remap every edge into that same order before propagation.
            h1_adjacency = adjacency_by_h[1]

            for method in methods:
                ordered_docs = orders[method]
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
                elif method == "qafd_recursive_h1_t2":
                    endpoint = f"http://127.0.0.1:{args.recursive_port}/generate_recursive"
                elif method == "block_rag":
                    endpoint = f"http://127.0.0.1:{args.port}/generate_block"
                else:
                    endpoint = f"http://127.0.0.1:{args.port}/generate_gapemp"
                extra = None
                if method == "qafd_recursive_h1_t2":
                    extra = {"neighbors": [sorted(n) for n in h1_adjacency], "rounds": 2, "max_new_tokens": 128}
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
