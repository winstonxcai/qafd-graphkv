"""Validate, compare, select, and freeze CSA development results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.eval.run_csa_grid import BETAS, beta_label


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strategy(root: Path, name: str) -> tuple[dict, list[dict]]:
    directory = root / name
    summary = json.loads((directory / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in Path(summary["predictions"]).read_text().splitlines()
        if line.strip()
    ]
    expected = list(range(500, 700))
    if [row["qid"] for row in rows] != expected:
        raise ValueError(f"{name} does not contain exactly QIDs 500-699")
    if summary["questions"] != 200:
        raise ValueError(f"{name} summary does not contain 200 questions")
    if any(row.get("qid", 700) >= 700 for row in rows):
        raise ValueError(f"{name} accessed a reserved final QID")
    return summary, rows


def paired_interval(candidate: list[dict], baseline: list[dict], samples=10000) -> tuple[float, float, float]:
    if [row["qid"] for row in candidate] != [row["qid"] for row in baseline]:
        raise ValueError("paired result QIDs do not align")
    differences = [
        float(left["accuracy"]) - float(right["accuracy"])
        for left, right in zip(candidate, baseline)
    ]
    observed = sum(differences) / len(differences)
    generator = random.Random(0)
    draws = []
    for _ in range(samples):
        draws.append(
            sum(differences[generator.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    draws.sort()
    return observed, draws[int(0.025 * samples)], draws[int(0.975 * samples)]


def materialize_b4_alias(root: Path, beta: float, shared_summary: dict, shared_rows: list[dict]) -> str:
    """Create beta-specific score traces without rerunning identical B=4 inference."""
    name = f"csa_norm_b4_beta{beta_label(beta)}"
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    prediction_path = directory / f"{name}.jsonl"
    with prediction_path.open("w") as handle:
        for original in shared_rows:
            row = dict(original)
            row["configuration"] = {**row["configuration"], "beta": beta}
            traces = []
            for original_trace in row["routing_trace"]:
                trace = dict(original_trace)
                combined = []
                for llm_row, graph_row in zip(
                    trace["llm_standardized"], trace["graph_standardized"]
                ):
                    combined.append([
                        None if llm is None else llm + beta * graph
                        for llm, graph in zip(llm_row, graph_row)
                    ])
                trace["combined_scores"] = combined
                traces.append(trace)
            row["routing_trace"] = traces
            row["shared_inference_source"] = shared_summary["predictions"]
            handle.write(json.dumps(row) + "\n")
    alias = {
        **shared_summary,
        "method": name,
        "strategy_description": (
            "Layerwise causal CSA, normalized_token_mean, B=4, "
            f"beta={beta:g}, h<=1 QAFD soft prior, gathered_sdpa; shared inference"
        ),
        "configuration": {**shared_summary["configuration"], "beta": beta},
        "predictions": str(prediction_path.resolve()),
        "shared_inference_source": shared_summary["predictions"],
    }
    (directory / "summary.json").write_text(json.dumps(alias, indent=2) + "\n")
    return name


def normalized_results(root: Path, baseline_rows: list[dict]) -> list[dict]:
    result = []
    for top_b in (1, 2, 3):
        for beta in BETAS:
            name = f"csa_norm_b{top_b}_beta{beta_label(beta)}"
            summary, rows = load_strategy(root, name)
            delta, low, high = paired_interval(rows, baseline_rows)
            result.append({**summary, "beta": beta, "top_b": top_b, "pooling": "normalized_token_mean", "delta_accuracy": delta, "ci_low": low, "ci_high": high, "shared_inference": False})
    shared_summary, shared_rows = load_strategy(root, "csa_norm_b4_shared")
    for beta in BETAS:
        name = materialize_b4_alias(root, beta, shared_summary, shared_rows)
        alias_summary, alias_rows = load_strategy(root, name)
        delta, low, high = paired_interval(alias_rows, baseline_rows)
        result.append({**alias_summary, "beta": beta, "top_b": 4, "pooling": "normalized_token_mean", "delta_accuracy": delta, "ci_low": low, "ci_high": high, "shared_inference": True})
    return result


def select_winner(rows: list[dict]) -> dict:
    best_accuracy = max(row["accuracy"] for row in rows)
    eligible = [row for row in rows if row["accuracy"] >= best_accuracy - 0.01]
    return min(
        eligible,
        key=lambda row: (
            row["avg_seconds"],
            -row["f1"],
            row["top_b"],
            row["beta"],
            row["pooling"] != "normalized_token_mean",
            row["method"],
        ),
    )


FIELDS = [
    "method", "strategy_description", "started_at", "completed_at", "pooling",
    "beta", "top_b", "accuracy", "f1", "avg_seconds",
    "avg_tokenization_seconds", "avg_prefill_seconds", "avg_routing_gpu_seconds",
    "avg_decode_seconds", "avg_qk_pair_ratio", "avg_routing_churn",
    "avg_graph_prior_agreement", "delta_accuracy", "ci_low", "ci_high",
    "max_peak_vram_bytes", "shared_inference", "predictions",
    "configuration_json", "source_artifacts_json",
]


def write_outputs(rows: list[dict], destination: Path, winner: dict) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "csa_dev_grid.csv"
    for row in rows:
        row["configuration_json"] = json.dumps(row.get("configuration", {}), sort_keys=True)
        row["source_artifacts_json"] = json.dumps(row.get("source_artifacts", {}), sort_keys=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# Joint-Prefill Soft-Graph CSA Development Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Only HotpotQA QIDs 500-699 are included. QIDs 700-949 remain prediction-free.",
        "",
        "| Method | Pooling | beta | B | Accuracy | F1 | Avg latency (s) | QK ratio | Delta vs dense | 95% paired CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {row['pooling']} | {row['beta']:g} | {row['top_b']} | "
            f"{row['accuracy']:.4f} | {row['f1']:.4f} | {row['avg_seconds']:.4f} | "
            f"{row.get('avg_qk_pair_ratio', 1.0):.4f} | {row['delta_accuracy']:+.4f} | "
            f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    markdown.extend([
        "",
        f"Selected configuration: `{winner['method']}` (accuracy={winner['accuracy']:.4f}, average latency={winner['avg_seconds']:.4f}s).",
        "",
        "Selection rule: within 0.01 of maximum development accuracy, then lowest average latency, higher F1, lower B, lower beta, normalized-token pooling, and strategy name.",
    ])
    dense = next(row for row in rows if row["method"] == "dense_vanilla_sdpa")
    promising = winner["accuracy"] > dense["accuracy"] or (
        winner["accuracy"] >= dense["accuracy"] - 0.01
        and winner["avg_seconds"] < dense["avg_seconds"]
    )
    markdown.extend([
        "",
        f"Prototype criterion: **{'promising' if promising else 'not yet promising'}**. It must beat dense accuracy or remain within 0.01 while reducing measured latency.",
    ])
    (destination / "csa_dev_grid.md").write_text("\n".join(markdown) + "\n")


def source_hashes(repository: Path) -> dict[str, str]:
    sources = [
        repository / "src/csa/attention.py",
        repository / "src/csa/routing.py",
        repository / "src/csa/prompt.py",
        repository / "src/csa/graph.py",
        repository / "src/eval/csa_server.py",
        repository / "src/eval/csa_benchmark.py",
        repository / "src/eval/csa_results.py",
        repository / "src/eval/run_csa_grid.py",
        repository / "src/eval/csa_smoke.py",
        repository / "src/qafd_bridge/run_qafd_slice.py",
        repository / "src/graph/sparsify.py",
    ]
    return {str(path.relative_to(repository)): sha256(path) for path in sources}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--include-plain", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    baseline_summary, baseline_rows = load_strategy(args.input_dir, "dense_vanilla_sdpa")
    flash_summary, flash_rows = load_strategy(args.input_dir, "dense_vanilla_flash2")
    flash_delta, flash_low, flash_high = paired_interval(flash_rows, baseline_rows)
    control_rows = [
        {**baseline_summary, "pooling": "not_applicable", "beta": 0.0, "top_b": 4, "delta_accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0, "shared_inference": False},
        {**flash_summary, "pooling": "not_applicable", "beta": 0.0, "top_b": 4, "delta_accuracy": flash_delta, "ci_low": flash_low, "ci_high": flash_high, "shared_inference": False},
    ]
    candidate_rows = normalized_results(args.input_dir, baseline_rows)
    normalized_winner = select_winner(candidate_rows)
    provisional = {
        "beta": normalized_winner["beta"],
        "top_b": normalized_winner["top_b"],
        "method": normalized_winner["method"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "csa_normalized_winner.json").write_text(json.dumps(provisional, indent=2) + "\n")

    if args.include_plain:
        name = f"csa_plain_b{normalized_winner['top_b']}_beta{beta_label(normalized_winner['beta'])}"
        plain_summary, plain_rows = load_strategy(args.input_dir, name)
        delta, low, high = paired_interval(plain_rows, baseline_rows)
        candidate_rows.append({**plain_summary, "beta": normalized_winner["beta"], "top_b": normalized_winner["top_b"], "pooling": "plain_mean", "delta_accuracy": delta, "ci_low": low, "ci_high": high, "shared_inference": False})
    winner = select_winner(candidate_rows)
    write_outputs(control_rows + candidate_rows, args.output_dir, winner)

    if args.freeze:
        if not args.include_plain:
            parser.error("--freeze requires --include-plain")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.repository, text=True
        ).strip()
        frozen_sources = list(source_hashes(args.repository.resolve()))
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *frozen_sources],
            cwd=args.repository,
            text=True,
        ).strip()
        if dirty:
            raise ValueError("refusing to freeze with uncommitted CSA source files")
        manifest = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "model_revision": winner.get("model_revision") or "ldsjmdy/Tulu3-Block-FT",
            "prompt_template_sha256": sha256(args.repository / "src/csa/prompt.py"),
            "retrieval_path": str(args.retrieval.resolve()),
            "retrieval_sha256": sha256(args.retrieval),
            "scorer": "GraphKV-style normalized token-intersection accuracy; diagnostic token F1",
            "development_qids": [500, 699],
            "reserved_prediction_free_qids": [700, 949],
            "selected_configuration": winner,
            "dense_vanilla": baseline_summary,
            "source_sha256": source_hashes(args.repository.resolve()),
            "development_summary_csv": str((args.output_dir / "csa_dev_grid.csv").resolve()),
        }
        (args.output_dir / "csa_frozen_configuration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(winner, indent=2), flush=True)


if __name__ == "__main__":
    main()
