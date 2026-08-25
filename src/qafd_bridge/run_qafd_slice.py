"""Produce a stable-QID QAFD retrieval slice without editing upstream code."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def collect_entity_scores(graph, passage_vertices, node_scores, max_hops=1):
    frontier = {
        neighbor
        for passage in passage_vertices
        for neighbor in graph.neighbors(passage)
        if str(graph.vs[neighbor]["name"]).startswith("entity-")
    }
    visited = set(frontier)
    for _ in range(max_hops):
        next_frontier = {
            neighbor
            for entity in frontier
            for neighbor in graph.neighbors(entity)
            if str(graph.vs[neighbor]["name"]).startswith("entity-")
        } - visited
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return {
        str(graph.vs[entity]["name"]): max(float(node_scores[entity]), 0.0)
        for entity in sorted(visited)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qafd-root", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--questions", type=int, default=450)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--entity-hops", type=int, default=1)
    parser.add_argument("--save-results", required=True)
    args = parser.parse_args()
    if args.start < 0 or args.questions < 1:
        parser.error("start must be non-negative and questions must be positive")
    if args.retrieval_top_k != 5:
        parser.error("CSA v1 retrieval requires exactly five passages")
    if args.entity_hops != 1:
        parser.error("CSA v1 prior requires entity-hops=1")

    root = os.path.abspath(args.qafd_root)
    data_root = Path(root) / "data" / "multihop"
    source_questions = json.loads((data_root / f"{args.dataset}.json").read_text())
    sliced = source_questions[args.start : args.start + args.questions]
    if len(sliced) != args.questions:
        raise ValueError("requested QAFD slice extends past the dataset")
    questions = [row["question"] for row in sliced]
    if len(questions) != len(set(questions)):
        raise ValueError("QAFD slice contains duplicate question strings")
    qid_by_question = {
        row["question"]: args.start + index for index, row in enumerate(sliced)
    }
    gold_by_question = {}
    for row in sliced:
        answer = row.get("answer") or row.get("gold_ans") or row.get("reference") or ""
        answers = [answer] if isinstance(answer, str) else list(answer)
        answers.extend(row.get("answer_aliases", []))
        gold_by_question[row["question"]] = list(dict.fromkeys(answers))

    os.chdir(root)
    sys.path.insert(0, root)
    for package in ["src", "src.retrievers", "src.passage_entity", "src.utils"]:
        module = types.ModuleType(package)
        module.__path__ = [os.path.join(root, *package.split("."))]
        module.__package__ = package
        sys.modules.setdefault(package, module)
    src = os.path.join(root, "src")
    _load_module("src.retrievers.base", os.path.join(src, "retrievers", "base.py"))
    utils_module = sys.modules["src.utils"]
    logging_module = _load_module(
        "src.utils.logging", os.path.join(src, "utils", "logging.py")
    )
    utils_module.logger = logging_module.logger
    _load_module(
        "src.retrievers.flow_diffusion",
        os.path.join(src, "retrievers", "flow_diffusion.py"),
    )
    from src.passage_entity.benchmark_runner import main as benchmark_main
    from src.passage_entity.graph_adapter import IGraphQAFD
    from src.passage_entity.retriever import PassageEntityRetriever
    from src.passage_entity.utils import QuerySolution

    entity_scores_by_question = {}
    last_flow = {"scores": None}
    original_run = IGraphQAFD.run
    original_graph_search = PassageEntityRetriever._graph_search

    def traced_run(self, *run_args, **run_kwargs):
        scores = original_run(self, *run_args, **run_kwargs)
        last_flow["scores"] = scores
        return scores

    def traced_graph_search(self, query, *search_args, **search_kwargs):
        last_flow["scores"] = None
        sorted_ids, sorted_scores = original_graph_search(
            self, query, *search_args, **search_kwargs
        )
        node_scores = last_flow["scores"]
        if node_scores is None:
            entity_scores_by_question[query] = {}
        else:
            selected = sorted_ids[: args.retrieval_top_k]
            passages = [self.passage_node_idxs[int(index)] for index in selected]
            entity_scores_by_question[query] = collect_entity_scores(
                self.graph, passages, node_scores, args.entity_hops
            )
        return sorted_ids, sorted_scores

    IGraphQAFD.run = traced_run
    PassageEntityRetriever._graph_search = traced_graph_search

    def full_to_dict(self):
        return {
            "qid": qid_by_question[self.question],
            "question": self.question,
            "answer": None,
            "gold_answers": gold_by_question[self.question],
            "docs": self.docs,
            "doc_scores": self.doc_scores.tolist() if self.doc_scores is not None else None,
            "entity_scores": entity_scores_by_question.get(self.question, {}),
            "gold_docs": self.gold_docs,
        }

    QuerySolution.to_dict = full_to_dict
    result_dir = os.path.join(
        root,
        "kg",
        "multihop",
        f"gpt-4o-mini_nvidia-nv-embed-v2_{args.dataset}",
    )
    result_path = os.path.join(result_dir, f"results_{args.dataset}.json")
    backup_path = result_path + ".csa_slice_backup"
    had_result = os.path.exists(result_path)
    if os.path.exists(result_path):
        shutil.copy2(result_path, backup_path)

    with tempfile.TemporaryDirectory(prefix="qafd_csa_slice_") as temporary:
        temporary_path = Path(temporary)
        (temporary_path / f"{args.dataset}.json").write_text(json.dumps(sliced))
        corpus_source = data_root / f"{args.dataset}_corpus.json"
        os.symlink(corpus_source, temporary_path / corpus_source.name)
        sys.argv = [
            "benchmark_runner",
            "--task", "multihop",
            "--dataset", args.dataset,
            "--data_dir", str(temporary_path),
            "--embedding_model", "nvidia-nv-embed-v2",
            "--llm_model", "gpt-4o-mini",
            "--num_queries", str(args.questions),
            "--qafd_alpha", "2.0",
            "--qafd_epsilon", "0.01",
            "--qafd_max_iterations", "500",
            "--qafd_step_size", "0.2",
            "--qafd_weight_scheme", "original",
            "--linking_top_k", "5",
            "--passage_node_weight", "0.05",
            "--retrieval_top_k", str(args.retrieval_top_k),
            "--skip_qa",
        ]
        try:
            benchmark_main()
            payload = json.loads(Path(result_path).read_text())
            payload["metadata"] = {
                "dataset": args.dataset,
                "qid_start": args.start,
                "qid_end": args.start + args.questions - 1,
                "questions": args.questions,
                "retrieval_top_k": args.retrieval_top_k,
                "entity_hops": args.entity_hops,
                "qafd_deterministic_seed": 0,
            }
            rows = payload.get("per_query", [])
            if [row.get("qid") for row in rows] != list(
                range(args.start, args.start + args.questions)
            ):
                raise ValueError("QAFD output QIDs do not match requested slice")
            missing_flow = [row["qid"] for row in rows if not row.get("entity_scores")]
            if missing_flow:
                raise ValueError(f"QAFD flow trace missing for QIDs {missing_flow[:5]}")
            destination = Path(args.save_results)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(payload))
        finally:
            if os.path.exists(backup_path):
                shutil.move(backup_path, result_path)
            elif not had_result and os.path.exists(result_path):
                os.unlink(result_path)


if __name__ == "__main__":
    main()
