"""Run QAFD retrieval while preserving all selected passages for Step 9.

QAFD's benchmark JSON serializer truncates every query to five passages. This
wrapper changes only that output serialization at runtime; it does not modify
the QAFD submodule or its retrieval algorithm.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import types


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qafd-root", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--retrieval-top-k", type=int, default=20)
    parser.add_argument("--save-results", required=True)
    args = parser.parse_args()

    root = os.path.abspath(args.qafd_root)
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
    from src.passage_entity.utils import QuerySolution

    def full_to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs,
            "doc_scores": self.doc_scores.tolist() if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }

    QuerySolution.to_dict = full_to_dict
    sys.argv = [
        "benchmark_runner",
        "--task", "multihop",
        "--dataset", args.dataset,
        "--data_dir", os.path.join(root, "data", "multihop"),
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
    result_path = os.path.join(
        root,
        "kg",
        "multihop",
        "gpt-4o-mini_nvidia-nv-embed-v2_hotpotqa",
        "results_hotpotqa.json",
    )
    backup_path = result_path + ".qafd_bridge_backup"
    if os.path.exists(result_path):
        shutil.copy2(result_path, backup_path)
    try:
        benchmark_main()
        shutil.copy2(result_path, args.save_results)
    finally:
        if os.path.exists(backup_path):
            shutil.move(backup_path, result_path)


if __name__ == "__main__":
    main()
