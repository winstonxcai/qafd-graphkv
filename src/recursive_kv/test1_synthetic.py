"""Test 1: synthetic recursive KV propagation on a three-node chain.

This deliberately keeps the graph tiny and reuses GraphKV's cache/RoPE
helpers.  The assertion is that information from P2 reaches P0 only after
two propagation rounds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers import AutoConfig


GRAPHKV_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "GraphKV")
)
sys.path.insert(0, GRAPHKV_ROOT)
from pcw import (  # noqa: E402
    apply_pkv_rerotary_position_embeddings,
    apply_pkv_rotary_position_embeddings,
    concact_pkv,
    cut_pkv,
)


def clone_cache(cache: DynamicCache) -> DynamicCache:
    result = DynamicCache()
    result.key_cache = [tensor.clone() for tensor in cache.key_cache]
    result.value_cache = [tensor.clone() for tensor in cache.value_cache]
    return result


def independent_prefill(model, tokenizer, text: str) -> DynamicCache:
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.inference_mode():
        output = model(tokens.to(model.device), use_cache=True)
    return clone_cache(output.past_key_values)


def concat_sources(caches: list[DynamicCache]) -> DynamicCache:
    merged = clone_cache(caches[0])
    for cache in caches[1:]:
        merged = concact_pkv(cache, merged)
    return merged


def propagate_one_round(model, tokenizer, emb, passages, previous, neighbors):
    next_caches = []
    lengths = [
        tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.shape[1]
        for text in passages
    ]

    for node, text in enumerate(passages):
        source = concat_sources([previous[j] for j in neighbors[node]])
        source_len = source.key_cache[0].shape[-2]
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.inference_mode():
            output = model(
                tokens.to(model.device),
                past_key_values=source,
                use_cache=True,
            )
        current = clone_cache(output.past_key_values)
        target_positions = torch.arange(
            source_len,
            source_len + lengths[node],
            device=model.device,
            dtype=torch.long,
        )
        current = cut_pkv(current, target_positions)
        current = apply_pkv_rerotary_position_embeddings(current, emb, target_positions)
        next_caches.append(current)
    return next_caches


def cache_distance(left: DynamicCache, right: DynamicCache) -> float:
    return max(
        torch.max(torch.abs(a.float() - b.float())).item()
        for a, b in zip(left.key_cache, right.key_cache)
    )


def run_case(model, tokenizer, emb, p2: str):
    passages = [
        "P0 is a shared introductory passage.",
        "P1 is a shared connecting passage.",
        p2,
    ]
    neighbors = {0: [1], 1: [0, 2], 2: [1]}
    initial = [independent_prefill(model, tokenizer, text) for text in passages]
    round1 = propagate_one_round(model, tokenizer, emb, passages, initial, neighbors)
    round2 = propagate_one_round(model, tokenizer, emb, passages, round1, neighbors)
    return {
        "initial": initial,
        "round1": round1,
        "round2": round2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument("--output", default="artifacts/results/test1_synthetic.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    emb = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
    emb.eval()

    red = run_case(model, tokenizer, emb, "The secret value is RED.")
    blue = run_case(model, tokenizer, emb, "The secret value is BLUE.")
    result = {
        "round1_p0_distance": cache_distance(red["round1"][0], blue["round1"][0]),
        "round2_p0_distance": cache_distance(red["round2"][0], blue["round2"][0]),
        "round1_expected": "approximately equal",
        "round2_expected": "different",
    }
    if not result["round1_p0_distance"] < 1e-4:
        raise AssertionError(result)
    if not result["round2_p0_distance"] > 1e-4:
        raise AssertionError(result)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
