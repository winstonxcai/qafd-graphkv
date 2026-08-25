"""Resumable matched evaluation for joint-prefill soft-graph CSA."""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import igraph as ig
import requests

from src.csa.attention import BACKENDS
from src.csa.graph import qafd_prior_matrix
from src.csa.routing import POOLING_METHODS


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def extract_answer(text: str) -> str:
    marker = re.search(r"the answer is\s*:?", text, flags=re.IGNORECASE)
    if marker:
        lines = text[marker.end() :].strip().splitlines()
        return lines[0].strip() if lines else ""
    return text.strip().splitlines()[0] if text.strip() else ""


def score_prediction(prediction: str, answers: list[str]) -> tuple[float, float]:
    pred = normalize(extract_answer(prediction))
    gold = [normalize(answer) for answer in answers]
    pred_set = set(pred.split())
    accuracy = float(any(pred_set & set(answer.split()) for answer in gold))
    pred_tokens = Counter(pred.split())
    best_f1 = 0.0
    for answer in gold:
        answer_tokens = Counter(answer.split())
        overlap = sum((pred_tokens & answer_tokens).values())
        if overlap:
            precision = overlap / max(sum(pred_tokens.values()), 1)
            recall = overlap / max(sum(answer_tokens.values()), 1)
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    return accuracy, best_f1


def parse_passage(document: str) -> dict:
    title, separator, body = document.partition("\n")
    return {"title": title, "text": body if separator else title}


def read_retrieval(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    rows = payload.get("per_query")
    if not isinstance(rows, list):
        raise ValueError("retrieval artifact is missing per_query")
    qids = [int(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("retrieval artifact contains duplicate QIDs")
    return payload.get("metadata", {}), rows


def configuration(args) -> dict:
    return {
        "mode": args.mode,
        "beta": args.beta,
        "top_b": args.top_b,
        "pooling": args.pooling,
        "backend": args.backend,
        "k": 5,
        "graph_hops": 1,
        "graph_decay_lambda": 1.0,
        "question_first": True,
        "max_new_tokens": args.max_new_tokens,
    }


def load_completed(path: Path, config: dict, expected_qids: list[int]) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) > len(expected_qids):
        raise ValueError("prediction file has too many rows")
    for index, row in enumerate(rows):
        if row.get("qid") != expected_qids[index]:
            raise ValueError("prediction file is not a contiguous expected-QID prefix")
        if row.get("configuration") != config:
            raise ValueError("prediction configuration mismatch")
        if len(row.get("passages", [])) != 5:
            raise ValueError("prediction row does not contain five passages")
        if config["mode"] == "csa" and not row.get("routing_trace"):
            raise ValueError("CSA prediction row has no routing trace")
    return rows


def request_generation(port: int, payload: dict) -> dict:
    response = requests.post(
        f"http://127.0.0.1:{port}/generate", json=payload, timeout=1800
    )
    response.raise_for_status()
    result = response.json()
    if result.get("ret") != 0:
        raise RuntimeError(result.get("message", "CSA server failed"))
    return result


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strategy-name", required=True)
    parser.add_argument("--port", type=int, default=8980)
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--mode", choices=["csa", "vanilla"], default="csa")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--top-b", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--pooling", choices=sorted(POOLING_METHODS), default="normalized_token_mean")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="gathered_sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()
    if args.beta < 0:
        parser.error("beta must be non-negative")

    metadata, all_rows = read_retrieval(args.retrieval)
    by_qid = {int(row["qid"]): row for row in all_rows}
    expected_qids = list(range(args.start, args.start + args.limit))
    missing = [qid for qid in expected_qids if qid not in by_qid]
    if missing:
        raise ValueError(f"retrieval artifact is missing QIDs {missing[:5]}")
    retrieval_rows = [by_qid[qid] for qid in expected_qids]

    graph = ig.Graph.Read_Pickle(args.graph)
    content_to_vertex = {
        vertex["content"]: vertex.index
        for vertex in graph.vs
        if str(vertex["name"]).startswith("chunk-") and vertex["content"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.strategy_name}.jsonl"
    config = configuration(args)
    completed = load_completed(output_path, config, expected_qids)

    with output_path.open("a", encoding="utf-8") as handle:
        for row in retrieval_rows[len(completed) :]:
            qid = int(row["qid"])
            documents = row["docs"][:5]
            if len(documents) != 5:
                raise ValueError(f"QID {qid} has fewer than five passages")
            scores = (row.get("doc_scores") or [5 - index for index in range(5)])[:5]
            vertices = [content_to_vertex.get(document) for document in documents]
            if any(vertex is None for vertex in vertices):
                raise ValueError(f"QID {qid} has a passage missing from the QAFD graph")
            if not row.get("entity_scores"):
                raise ValueError(f"QID {qid} has no entity flow trace")
            graph_matrix, graph_paths = qafd_prior_matrix(
                graph, vertices, row["entity_scores"]
            )
            passages = []
            for index, (document, score, vertex) in enumerate(
                zip(documents, scores, vertices)
            ):
                passage = parse_passage(document)
                passage.update(
                    {
                        "index": index,
                        "vertex_id": str(graph.vs[vertex]["name"]),
                        "qafd_score": float(score),
                    }
                )
                passages.append(passage)
            answers = list(row.get("gold_answers") or [row.get("answer", "")])
            payload = {
                "mode": args.mode,
                "question": row["question"],
                "passages": passages,
                "graph_scores": graph_matrix,
                "beta": args.beta,
                "top_b": args.top_b,
                "pooling": args.pooling,
                "backend": args.backend,
                "max_new_tokens": args.max_new_tokens,
            }
            if args.warmup and not completed and qid == expected_qids[0]:
                warmup_payload = {**payload, "max_new_tokens": 1}
                request_generation(args.port, warmup_payload)
            generated = request_generation(args.port, payload)
            accuracy, f1 = score_prediction(generated["generated"], answers)
            prediction = {
                "qid": qid,
                "question": row["question"],
                "answers": answers,
                "configuration": config,
                "passages": passages,
                "graph_scores": graph_matrix,
                "graph_paths": graph_paths,
                "entity_score_source": {
                    "artifact": str(args.retrieval.resolve()),
                    "artifact_metadata": metadata,
                    "available_entities": len(row["entity_scores"]),
                },
                "entity_scores": row["entity_scores"],
                "generated": generated["generated"],
                "first_token_id": generated["first_token_id"],
                "accuracy": accuracy,
                "f1": f1,
                "seconds": generated["model_seconds"],
                "timing": generated["timing"],
                "peak_vram_bytes": generated["peak_vram_bytes"],
                "token_spans": generated["token_spans"],
                "prompt_hash": generated["prompt_hash"],
                "model_revision": generated["model_revision"],
                "attention_implementation": generated["attention_implementation"],
                "routing_trace": generated["routing_trace"],
                "routing_summary": generated["routing_summary"],
            }
            handle.write(json.dumps(prediction) + "\n")
            handle.flush()
            completed.append(prediction)

    if [row["qid"] for row in completed] != expected_qids:
        raise ValueError("completed predictions do not match the requested QIDs")
    summaries = [row.get("routing_summary") for row in completed]
    summaries = [row for row in summaries if row]
    summary = {
        "method": args.strategy_name,
        "configuration": config,
        "questions": len(completed),
        "qid_start": expected_qids[0],
        "qid_end": expected_qids[-1],
        "accuracy": mean(row["accuracy"] for row in completed),
        "f1": mean(row["f1"] for row in completed),
        "avg_seconds": mean(row["seconds"] for row in completed),
        "avg_tokenization_seconds": mean(
            row["timing"]["tokenization_seconds"] for row in completed
        ),
        "avg_prefill_seconds": mean(
            row["timing"]["prefill_seconds"] for row in completed
        ),
        "avg_routing_gpu_seconds": mean(
            row["timing"]["routing_gpu_seconds"] for row in completed
        ),
        "avg_decode_seconds": mean(
            row["timing"]["decode_seconds"] for row in completed
        ),
        "total_seconds": sum(row["seconds"] for row in completed),
        "max_peak_vram_bytes": max(row["peak_vram_bytes"] for row in completed),
        "avg_qk_pair_ratio": mean(row["qk_pair_ratio"] for row in summaries),
        "avg_routing_churn": mean(row["mean_layer_churn"] for row in summaries),
        "avg_graph_prior_agreement": mean(
            row["graph_prior_agreement"] for row in summaries
        ),
        "model_revision": completed[0]["model_revision"],
        "attention_implementation": completed[0]["attention_implementation"],
        "predictions": str(output_path.resolve()),
    }
    if args.mode == "vanilla":
        summary["avg_qk_pair_ratio"] = 1.0
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
