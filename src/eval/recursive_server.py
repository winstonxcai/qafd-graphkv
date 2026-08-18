"""Minimal recursive KV generation endpoint for Test 4."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from flask import Flask, request
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

GRAPHKV_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "GraphKV"))
sys.path.insert(0, GRAPHKV_ROOT)
from pcw import apply_pkv_rotary_position_embeddings, concact_pkv  # noqa: E402
from src.recursive_kv.propagate import clone_cache, propagate_one_round  # noqa: E402

app = Flask(__name__)


def prefill(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.inference_mode():
        output = model(tokens.to(model.device), use_cache=True)
    return clone_cache(output.past_key_values)


def generate_recursive(model, tokenizer, emb, blocks, neighbors, rounds):
    prefix, _middle, *contexts, query = blocks
    passages = contexts
    caches = [prefill(model, tokenizer, passage) for passage in passages]
    for _ in range(rounds):
        caches = propagate_one_round(model, tokenizer, emb, passages, caches, neighbors)

    merged = clone_cache(caches[0])
    for cache in caches[1:]:
        merged = concact_pkv(merged, cache)
    prefix_cache = prefill(model, tokenizer, prefix)
    merged = concact_pkv(prefix_cache, merged)
    total_len = merged.key_cache[0].shape[-2]
    positions = torch.arange(total_len, device=model.device, dtype=torch.long)
    merged = apply_pkv_rotary_position_embeddings(merged, emb, positions)

    query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    generated = query_ids
    answer = []
    with torch.inference_mode():
        for _ in range(1024):
            output = model(generated, past_key_values=merged, use_cache=True)
            merged = output.past_key_values
            token = torch.argmax(output.logits[:, -1, :], dim=-1).unsqueeze(-1)
            if token.item() == tokenizer.eos_token_id:
                break
            answer.append(token.item())
            generated = token
    return tokenizer.decode(answer, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    config = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
    )
    model.eval()
    emb = LlamaRotaryEmbedding(config=config).to(device=model.device, dtype=torch.float32)
    emb.eval()

    @app.post("/generate_recursive")
    def endpoint():
        form = request.get_json()
        generated = generate_recursive(model, tokenizer, emb, form["blocks"], form["neighbors"], form.get("rounds", 2))
        return {"ret": 0, "generated": generated, "message": ""}

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
