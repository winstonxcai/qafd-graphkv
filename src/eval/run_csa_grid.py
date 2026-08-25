"""Distribute the unique joint-prefill CSA development configurations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def beta_label(beta: float) -> str:
    return f"{beta:g}".replace(".", "p")


def unique_grid() -> list[tuple[str, float, int]]:
    configurations = []
    for top_b in (1, 2, 3):
        for beta in BETAS:
            name = f"csa_norm_b{top_b}_beta{beta_label(beta)}"
            configurations.append((name, beta, top_b))
    # B=4 always selects every causally available passage, so one model run is
    # shared by all five beta-specific score-trace aliases.
    configurations.append(("csa_norm_b4_shared", 0.0, 4))
    return configurations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8980)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--plain-beta", type=float)
    parser.add_argument("--plain-b", type=int, choices=[1, 2, 3, 4])
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        parser.error("worker-index must be in [0, worker-count)")

    if (args.plain_beta is None) != (args.plain_b is None):
        parser.error("plain-beta and plain-b must be supplied together")
    if args.plain_beta is not None:
        configurations = [
            (
                f"csa_plain_b{args.plain_b}_beta{beta_label(args.plain_beta)}",
                args.plain_beta,
                args.plain_b,
                "plain_mean",
            )
        ]
    else:
        configurations = [(*values, "normalized_token_mean") for values in unique_grid()]

    selected = configurations[args.worker_index :: args.worker_count]
    for local_index, (name, beta, top_b, pooling) in enumerate(selected):
        destination = args.output_dir / name
        command = [
            sys.executable,
            "-m",
            "src.eval.csa_benchmark",
            "--retrieval",
            str(args.retrieval),
            "--graph",
            str(args.graph),
            "--output-dir",
            str(destination),
            "--strategy-name",
            name,
            "--port",
            str(args.port),
            "--start",
            str(args.start),
            "--limit",
            str(args.limit),
            "--beta",
            str(beta),
            "--top-b",
            str(top_b),
            "--pooling",
            pooling,
            "--backend",
            "gathered_sdpa",
            "--max-new-tokens",
            str(args.max_new_tokens),
        ]
        if local_index == 0:
            command.append("--warmup")
        print(f"running {name}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
