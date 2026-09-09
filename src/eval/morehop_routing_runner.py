"""Resumable single-model runner for one-round MoreHopQA routing methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.eval.graphkv_faithful_morehop import load_rows, prompt_hash, sha256, upstream_blocks
from src.eval.morehop_scoring import score_both
from src.graph.morehop_controls import graph_statistics

METHODS = ("released_last1", "released_last3", "full", "query_bm25_k1", "query_bm25_k3", "target_bm25_k1", "target_bm25_k3", "target_bm25_k1_noself", "ppr_k1", "random_k1_seed42", "reversed_rank_k1")
MAX_NEW_TOKENS = 256


def _run_gpu_parity(rows, model, tokenizer, emb, methods):
    """Compare Full/last-k against real upstream gapemp capture boundaries."""
    if not torch.cuda.is_available():
        raise RuntimeError("--parity-only requires a CUDA GPU")
    from src.eval.graphkv_engine_parity import (
        UpstreamCaptureProxy, cache_comparison, capture_upstream_cache,
        teacher_forced_comparison, tokenize_inputs, greedy_generate as generate_ids,
    )
    from src.eval.morehop_routing_engine import build_one_round_cache
    from pcw import gapemp_appr

    def capture_last_k(blocks, k):
        proxy = UpstreamCaptureProxy(model, tokenizer.eos_token_id)
        prefix, middle, *contexts, query = blocks
        gapemp_appr(tokenizer, proxy, emb, prefix, middle, query, contexts,
                    "ldsjmdy/Tulu3-Block-FT", 1, 1, k)
        if proxy.decode_cache is None or proxy.decode_input_ids is None or proxy.first_logits is None:
            raise RuntimeError("failed to capture upstream gapemp_appr decode boundary")
        return proxy.decode_cache, proxy.decode_input_ids, proxy.first_logits

    if set(methods) != {"full", "released_last1", "released_last3"}:
        raise ValueError("parity requires Full, last1, last3")
    failures, records = [], []
    for row in rows:
        blocks = upstream_blocks(row)
        query_ids = tokenize_inputs(tokenizer, blocks)["query_ids"]
        for method in methods:
            if method == "full":
                reference, reference_ids, reference_logits = capture_upstream_cache(model, tokenizer, emb, blocks)
            elif method in ("released_last1", "released_last3"):
                k = int(method.rsplit("last", 1)[1])
                reference, reference_ids, reference_logits = capture_last_k(blocks, k)
            else:
                continue
            neighbors = _topology(method, row["question"], row["documents"])["neighbors"]
            candidate, _ = build_one_round_cache(model, tokenizer, emb, blocks, neighbors)
            cache = cache_comparison(reference, candidate)
            forced = teacher_forced_comparison(model, reference, candidate, query_ids,
                                               tokenizer.eos_token_id, 8, reference_logits)
            reference_output = generate_ids(model, reference, query_ids, tokenizer.eos_token_id, MAX_NEW_TOKENS)
            candidate_output = generate_ids(model, candidate, query_ids, tokenizer.eos_token_id, MAX_NEW_TOKENS)
            record = {"_id": row["_id"], "method": method, "cache": cache,
                      "teacher_forced": forced, "input_ids_match": bool(torch.equal(reference_ids, query_ids)),
                      "complete_output_match": reference_output == candidate_output,
                      "reference_output_ids": reference_output, "candidate_output_ids": candidate_output}
            records.append(record)
            if not torch.equal(reference_ids, query_ids) or not cache["shapes_match"] or \
                    cache["relative_rms_error"] > 0.01 or not forced["steps"][0]["top_token_match"] or \
                    max(x["relative_logit_rms_error"] for x in forced["steps"]) > .01 or \
                    forced["captured_upstream_first_to_candidate_rms"] > .01 or reference_output != candidate_output:
                failures.append({"_id": row["_id"], "method": method})
            print(json.dumps({"parity_progress": len(records), "_id": row["_id"], "method": method,
                              "cache_rms": cache["relative_rms_error"], "output_match": reference_output == candidate_output}), flush=True)
            del reference, candidate
            torch.cuda.empty_cache()
    return {"status": "FAIL" if failures else "PASS", "methods": methods,
            "questions": len(rows), "records": records, "failures": failures}


def _topology(method, question, documents):
    n = len(documents)
    if method.startswith("released_last"):
        k = int(method.rsplit("last", 1)[1])
        return {"neighbors": [list(range(max(0, n-k), n)) for _ in documents], "node_scores": {}, "method": method, "config": {"k": k}}
    if method == "full":
        return {"neighbors": [list(range(n)) for _ in documents], "node_scores": {}, "method": method, "config": {}}
    if method.startswith("query_bm25"):
        from src.graph.morehop_query_topology import build_query_topology
        return build_query_topology(question, documents, int(method.rsplit("k", 1)[1]))
    if method == "target_bm25_k1_noself":
        from src.graph.morehop_query_topology import build_target_conditioned_topology
        return build_target_conditioned_topology(question, documents, 1, exclude_self=True)
    if method.startswith("target_bm25"):
        from src.graph.morehop_query_topology import build_target_conditioned_topology
        return build_target_conditioned_topology(question, documents, int(method.rsplit("k", 1)[1]))
    if method == "ppr_k1":
        from src.graph.morehop_path_topology import build_path_topology
        return build_path_topology(question, documents, 1, alpha=.85, graph_k=3)
    if method == "random_k1_seed42":
        seed = int(hashlib.sha256(("42:"+question).encode()).hexdigest(), 16)
        source = random.Random(seed).randrange(n)
        return {"neighbors": [[source] for _ in documents], "node_scores": {}, "method": method, "config": {"seed": 42, "k": 1, "sampling": "one global source; SHA256 question seed"}}
    if method == "reversed_rank_k1":
        return {"neighbors": [[0] for _ in documents], "node_scores": {}, "method": method, "config": {"k": 1, "ranking": "first_in_released_order"}}
    raise ValueError(f"unknown method {method}")


def _resume(path, expected_ids, config_hash, prompt_hashes, model_hash, code_hash):
    if not path.exists(): return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for i, row in enumerate(rows):
        if (
            i >= len(expected_ids)
            or row.get("_id") != expected_ids[i]
            or row.get("config_hash") != config_hash
            or row.get("prompt_hash") != prompt_hashes[i]
            or row.get("model_hash") != model_hash
            or row.get("code_hash") != code_hash
        ):
            raise ValueError(f"mismatched resume ID/config in {path}")
    return rows


def _source_hash(model_path: str) -> str:
    """Stable provenance hash for a local model directory or model ID."""
    path = Path(model_path)
    if not path.is_dir():
        raise ValueError("model must be the frozen local snapshot")
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(bytes.fromhex(sha256(child)))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--qid-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--parity-only", action="store_true", help="run actual GPU parity and do not write experiments")
    parser.add_argument("--parity-report", type=Path)
    args = parser.parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(METHODS):
        raise ValueError("empty, duplicate or unknown methods")
    all_rows = load_rows(args.dataset)
    frozen = json.loads(args.qid_manifest.read_text())
    ids = frozen.get("selected_question_ids")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)): raise ValueError("invalid frozen QID manifest")
    if frozen.get("dataset_sha256") != sha256(args.dataset):
        raise ValueError("dataset hash differs from frozen manifest")
    if Path(args.model).name != "62eecbdfa4e12821a487e91ed2ef847a85426efe":
        raise ValueError("unexpected model revision")
    if not torch.cuda.is_available():
        raise RuntimeError("A800 CUDA allocation required")
    if "A800" not in torch.cuda.get_device_name():
        raise RuntimeError("experiment is frozen to A800")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    by_id = {row["_id"]: row for row in all_rows}
    rows = [by_id[qid] for qid in ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_hash = _source_hash(args.model)
    root = Path(__file__).resolve().parents[2]
    source_paths = ["src/eval/morehop_routing_runner.py", "src/eval/morehop_routing_engine.py",
                    "src/eval/graphkv_engine_parity.py", "src/eval/graphkv_faithful_morehop.py",
                    "src/eval/morehop_scoring.py", "src/graph/morehop_query_topology.py",
                    "src/graph/morehop_path_topology.py", "src/graph/morehop_controls.py", "third_party/GraphKV/pcw.py"]
    source_hashes = {p: sha256(root/p) for p in source_paths}
    code_hash = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()
    provenance = {"code_hash": code_hash, "source_hashes": source_hashes, "model_hash": model_hash,
                  "model_revision": Path(args.model).name, "dataset_sha256": sha256(args.dataset),
                  "manifest_sha256": sha256(args.qid_manifest), "gpu": torch.cuda.get_device_name(),
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "max_new_tokens": MAX_NEW_TOKENS,
                  "created_at": datetime.now(timezone.utc).isoformat(), "selected_question_ids": ids}
    if not args.parity_only:
        if not args.parity_report:
            raise ValueError("a matching executable parity report is required")
        gate = json.loads(args.parity_report.read_text())
        if gate.get("status") != "PASS" or gate.get("code_hash") != code_hash or gate.get("model_hash") != model_hash or gate.get("questions", 0) < 10 or set(gate.get("methods", [])) != {"full", "released_last1", "released_last3"}:
            raise ValueError("parity gate missing or does not match this engine/model")
    (args.output_dir / ("parity_manifest.json" if args.parity_only else "run_manifest.json")).write_text(json.dumps(provenance, indent=2)+"\n")
    prompt_hashes = [prompt_hash(upstream_blocks(row)) for row in rows]
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from src.eval.morehop_routing_engine import build_one_round_cache, greedy_generate
    tokenizer = None
    model = None
    emb = None
    if args.parity_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        model.eval()
        emb = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
        report = {**provenance, **_run_gpu_parity(rows, model, tokenizer, emb, methods)}
        (args.output_dir/"parity.json").write_text(json.dumps(report, indent=2)+"\n")
        if report["status"] != "PASS":
            raise RuntimeError("GPU parity failed; inspect parity.json")
        print("PARITY PASS", flush=True)
        return
    for method in methods:
        path = args.output_dir / f"{method}.jsonl"
        topo_configs = [{"method": method, **_topology(method, row["question"], row["documents"])} for row in rows]
        config_hash = hashlib.sha256(json.dumps(topo_configs, sort_keys=True).encode()).hexdigest()
        completed = _resume(path, ids, config_hash, prompt_hashes, model_hash, code_hash)
        if len(completed) >= len(rows): continue
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        config = AutoConfig.from_pretrained(args.model)
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
        model.eval()
        if emb is None:
            emb = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
        # Warm the same method before recording timings.
        warm_blocks = upstream_blocks(rows[0])
        warm_topo = _topology(method, rows[0]["question"], rows[0]["documents"])
        warm_cache, _ = build_one_round_cache(model, tokenizer, emb, warm_blocks, warm_topo["neighbors"])
        greedy_generate(model, tokenizer, warm_cache, warm_blocks[-1], 2)
        del warm_cache
        torch.cuda.synchronize()
        handle = path.open("a", encoding="utf-8")
        try:
            for row in rows[len(completed):]:
                blocks = upstream_blocks(row); topo = _topology(method, row["question"], row["documents"])
                topo["neighbors"] = [sorted(x) for x in topo["neighbors"]]
                torch.cuda.reset_peak_memory_stats()
                if torch.cuda.is_available(): torch.cuda.synchronize()
                started = time.perf_counter()
                cache, meta = build_one_round_cache(model, tokenizer, emb, blocks, topo["neighbors"])
                prediction = greedy_generate(model, tokenizer, cache, tokenizer(blocks[-1], return_tensors="pt", add_special_tokens=False).input_ids, MAX_NEW_TOKENS)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                latency = time.perf_counter() - started
                edges = [(source, target) for target, sources in enumerate(topo["neighbors"]) for source in sources]
                scores = score_both(prediction, row["answers"])
                result = {"_id": row["_id"], "question": row["question"], "documents": row["documents"], "prompt_hash": prompt_hash(blocks), "source_config": topo, "config_hash": config_hash, "code_hash": code_hash, "model": args.model, "model_hash": model_hash, "neighbors": topo["neighbors"], "edge_count": len(edges), "diagonal_edges": sum(a == b for a,b in edges), "cross_edges": sum(a != b for a,b in edges), "source_token_exposure": meta["source_token_exposure"], "prediction": prediction, "generated": prediction, **scores, "latency_seconds": latency, "max_new_tokens": MAX_NEW_TOKENS}
                result.update({"answers": row["answers"], "no_of_hops": row["no_of_hops"], "method": method,
                               "graph_statistics": graph_statistics(topo["neighbors"]), "token_metadata": meta,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "timestamp": datetime.now(timezone.utc).isoformat()})
                handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
                print(f"{method} {len(completed)+1}/{len(rows)} {row['_id']}", flush=True)
                completed.append(result)
                del cache
        finally: handle.close()


if __name__ == "__main__": main()
