"""Resumable single-model runner for one-round MoreHopQA routing methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from src.eval.graphkv_faithful_morehop import load_rows, prompt_hash, sha256, upstream_blocks
from src.eval.morehop_scoring import score_both

METHODS = ("released_last1", "released_last3", "full", "query_bm25_k1", "query_bm25_k3", "ppr_k1", "random_k1_seed42", "reversed_rank_k1")
MAX_NEW_TOKENS = 256


def _run_gpu_parity(rows, model, tokenizer, emb, methods):
    """Compare Full/last-k against real upstream gapemp capture boundaries."""
    if not torch.cuda.is_available():
        raise RuntimeError("--parity-only requires a CUDA GPU")
    from src.eval.graphkv_engine_parity import (
        UpstreamCaptureProxy, cache_comparison, capture_upstream_cache,
        teacher_forced_comparison, tokenize_inputs,
    )
    from pcw import gapemp_appr

    def capture_last_k(blocks, k):
        proxy = UpstreamCaptureProxy(model, tokenizer.eos_token_id)
        prefix, middle, *contexts, query = blocks
        gapemp_appr(tokenizer, proxy, emb, prefix, middle, query, contexts,
                    "ldsjmdy/Tulu3-Block-FT", 1, 1, k)
        if proxy.decode_cache is None or proxy.decode_input_ids is None or proxy.first_logits is None:
            raise RuntimeError("failed to capture upstream gapemp_appr decode boundary")
        return proxy.decode_cache, proxy.decode_input_ids, proxy.first_logits

    failures = []
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
                                               tokenizer.eos_token_id, 1, reference_logits)
            if not torch.equal(reference_ids, query_ids) or not cache["shapes_match"] or \
                    cache["relative_rms_error"] > 0.01 or not forced["steps"][0]["top_token_match"]:
                failures.append({"_id": row["_id"], "method": method, "cache": cache,
                                 "input_ids_match": bool(torch.equal(reference_ids, query_ids)),
                                 "first_token_match": forced["steps"][0]["top_token_match"]})
    if failures:
        raise RuntimeError("GPU parity gate failed: " + json.dumps(failures[:3], sort_keys=True))
    return {"status": "PASS", "methods": [m for m in methods if m in ("full", "released_last1", "released_last3")], "questions": len(rows)}


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
    if method == "ppr_k1":
        from src.graph.morehop_path_topology import build_path_topology
        return build_path_topology(question, documents, 1, alpha=.85, graph_k=3)
    if method == "random_k1_seed42":
        rng = random.Random(42)
        return {"neighbors": [[rng.randrange(n)] if n else [] for _ in documents], "node_scores": {}, "method": method, "config": {"seed": 42, "k": 1}}
    if method == "reversed_rank_k1":
        return {"neighbors": [[n - 1 - i] for i in range(n)], "node_scores": {}, "method": method, "config": {"k": 1, "ranking": "reverse_released_order"}}
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
        return hashlib.sha256(model_path.encode()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode())
            digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--qid-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--parity-only", action="store_true", help="run actual GPU parity and do not write experiments")
    args = parser.parse_args()
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if set(methods) - set(METHODS): raise ValueError(f"unknown methods: {set(methods)-set(METHODS)}")
    all_rows = load_rows(args.dataset)
    frozen = json.loads(args.qid_manifest.read_text())
    ids = frozen.get("selected_question_ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)): raise ValueError("invalid frozen QID manifest")
    by_id = {row["_id"]: row for row in all_rows}
    rows = [by_id[qid] for qid in ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_hash = _source_hash(args.model)
    code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("morehop_routing_engine.py").read_bytes()).hexdigest()
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
        print(json.dumps(_run_gpu_parity(rows, model, tokenizer, emb, methods), sort_keys=True))
        return
    for method in methods:
        path = args.output_dir / f"{method}.jsonl"
        topo_configs = [_topology(method, row["question"], row["documents"])["config"] for row in rows]
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
        handle = path.open("a", encoding="utf-8")
        try:
            for row in rows[len(completed):]:
                blocks = upstream_blocks(row); topo = _topology(method, row["question"], row["documents"])
                if torch.cuda.is_available(): torch.cuda.synchronize()
                started = time.perf_counter()
                cache, meta = build_one_round_cache(model, tokenizer, emb, blocks, topo["neighbors"])
                prediction = greedy_generate(model, tokenizer, cache, tokenizer(blocks[-1], return_tensors="pt", add_special_tokens=False).input_ids, MAX_NEW_TOKENS)
                if torch.cuda.is_available(): torch.cuda.synchronize()
                latency = time.perf_counter() - started
                edges = [(source, target) for target, sources in enumerate(topo["neighbors"]) for source in sources]
                scores = score_both(prediction, row["answers"])
                result = {"_id": row["_id"], "question": row["question"], "documents": row["documents"], "prompt_hash": prompt_hash(blocks), "source_config": topo, "config_hash": config_hash, "code_hash": code_hash, "model": args.model, "model_hash": model_hash, "neighbors": topo["neighbors"], "edge_count": len(edges), "diagonal_edges": sum(a == b for a,b in edges), "cross_edges": sum(a != b for a,b in edges), "source_token_exposure": meta["source_token_exposure"], "prediction": prediction, "generated": prediction, **scores, "latency_seconds": latency, "max_new_tokens": MAX_NEW_TOKENS}
                handle.write(json.dumps(result, ensure_ascii=False) + "\n"); handle.flush()
        finally: handle.close()


if __name__ == "__main__": main()
