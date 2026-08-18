"""Run QAFD retrieval while preserving all selected passages for Step 9.

QAFD's benchmark JSON serializer truncates every query to five passages. This
wrapper changes only that output serialization at runtime; it does not modify
the QAFD submodule or its retrieval algorithm.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
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
    args = parser.parse_args()

    root = os.path.abspath(args.qafd_root)
    sys.path.insert(0, root)
    for package in ["src", "src.retrievers", "src.passage_entity"]:
        module = types.ModuleType(package)
        module.__path__ = [os.path.join(root, *package.split("."))]
        module.__package__ = package
        sys.modules.setdefault(package, module)

    src = os.path.join(root, "src")
    _load_module("src.retrievers.base", os.path.join(src, "retrievers", "base.py"))
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
        "--num_queries", str(args.questions),
        "--retrieval_top_k", str(args.retrieval_top_k),
        "--skip_qa",
    ]
    benchmark_main()


if __name__ == "__main__":
    main()
