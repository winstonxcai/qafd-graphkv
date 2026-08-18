"""CLI for Step 9 topology tables from QAFD benchmark output."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from statistics import mean, median

from .topology import analyze_record


METRICS = [
    "selected_passages",
    "mapped_passages",
    "selected_entities",
    "edges",
    "components",
    "largest_component",
    "average_degree",
    "edge_density",
    "diameter",
    "average_shortest_path",
    "isolated_passages",
    "unreachable_pairs",
]


def _p90(values):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(0.9 * len(values)))]


def analyze_file(results_path, graph_path, output_dir, ks=(5, 8, 10, 15, 20), max_hops=4):
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    graph = pickle.loads(Path(graph_path).read_bytes())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid, query in enumerate(results["per_query"]):
        passages = [
            {"text": text, "qafd_score": score}
            for text, score in zip(query["docs"], query["doc_scores"])
        ]
        for k in ks:
            row = analyze_record(graph, passages, k, max_hops=max_hops).to_dict()
            row.update({"qid": str(qid), "k": k, "question": query["question"]})
            rows.append(row)

    with (output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = []
    for k in ks:
        group = [row for row in rows if row["k"] == k]
        summary = {"k": k, "questions": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[f"avg_{metric}"] = mean(values) if values else None
            summary[f"median_{metric}"] = median(values) if values else None
            summary[f"p90_{metric}"] = _p90(values)
        summary["fully_connected_percent"] = 100 * mean(
            row["components"] == 1 and row["mapped_passages"] == row["selected_passages"]
            for row in group
        )
        summary["isolated_question_percent"] = 100 * mean(
            row["isolated_passages"] > 0 for row in group
        )
        summary_rows.append(summary)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary_rows[0]) if summary_rows else ["k"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    hop_rows = []
    for k in ks:
        counts = {}
        for row in rows:
            if row["k"] != k:
                continue
            for hop, count in row["hop_histogram"].items():
                counts[hop] = counts.get(hop, 0) + count
        hop_rows.extend({"k": k, "entity_hops": hop, "edge_count": count} for hop, count in sorted(counts.items()))
    with (output_dir / "hop_histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "entity_hops", "edge_count"])
        writer.writeheader()
        writer.writerows(hop_rows)
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-hops", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(analyze_file(args.results, args.graph, args.output_dir, max_hops=args.max_hops), indent=2))


if __name__ == "__main__":
    main()
