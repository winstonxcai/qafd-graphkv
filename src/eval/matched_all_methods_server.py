"""Single-model server for the matched all-methods Test 4 comparison.

This project-local server keeps the upstream GraphKV submodule untouched while
making the decoding budget explicit and shared across the comparison methods.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from flask import Flask, request
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

GRAPHKV_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "GraphKV")
)
sys.path.insert(0, GRAPHKV_ROOT)

from pcw import (  # noqa: E402
    apply_pkv_rerotary_position_embeddings,
    apply_pkv_rotary_position_embeddings,
    concact_pkv,
    cut_pkv,
    flatten_pkv,
    stack_pkv,
)
from server.block_generate_server import block_generate  # noqa: E402
from src.recursive_kv.propagate import clone_cache, propagate_one_round  # noqa: E402


app = Flask(__name__)
MAX_NEW_TOKENS = 256


def limited_gapemp(tokenizer, model, emb, prefix, middle, query, contexts):
    """GraphKV gapemp with the shared matched 256-token generation cap."""
    with torch.no_grad():
        prefix_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
        middle_ids = tokenizer(middle, return_tensors="pt", add_special_tokens=False).input_ids
        query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids
        len_prefix, len_middle, len_query = prefix_ids.shape[1], middle_ids.shape[1], query_ids.shape[1]
        context_ids = tokenizer(
            contexts,
            return_tensors="pt",
            truncation=True,
            max_length=8192 - len_prefix - len_query - len_middle - MAX_NEW_TOKENS,
            padding=True,
            add_special_tokens=False,
        ).input_ids
        context_mask = context_ids != tokenizer.pad_token_id
        batch_size = context_ids.shape[0]
        flat_len = context_mask.reshape(-1).sum().item()

        context_outputs = model(context_ids.to(model.device), use_cache=True)
        context_pkv = apply_pkv_rerotary_position_embeddings(context_outputs.past_key_values, emb)
        context_clone = type(context_pkv)()
        context_clone.key_cache = [tensor.clone().detach() for tensor in context_pkv.key_cache]
        context_clone.value_cache = [tensor.clone().detach() for tensor in context_pkv.value_cache]
        context_pkv = flatten_pkv(context_pkv, context_mask.reshape(-1))
        context_pkv = apply_pkv_rotary_position_embeddings(context_pkv, emb)
        context_pkv = stack_pkv(context_pkv, batch_size)

        second = model(context_ids.to(model.device), past_key_values=context_pkv, use_cache=True)
        positions = torch.arange(context_ids.shape[-1], dtype=torch.int64) + flat_len
        context_pkv = cut_pkv(second.past_key_values, positions)
        context_pkv = apply_pkv_rerotary_position_embeddings(context_pkv, emb, positions)
        context_pkv = flatten_pkv(context_pkv, context_mask.reshape(-1))
        context_pkv = concact_pkv(context_clone, context_pkv)

        prefix_outputs = model(prefix_ids.to(model.device), use_cache=True)
        prefix_pkv = apply_pkv_rerotary_position_embeddings(prefix_outputs.past_key_values, emb)
        past = apply_pkv_rotary_position_embeddings(concact_pkv(prefix_pkv, context_pkv), emb)

        generated = query_ids.to(model.device)
        answer_ids = generated
        for _ in range(MAX_NEW_TOKENS):
            outputs = model(generated, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            generated = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
            answer_ids = torch.cat((answer_ids, generated), dim=1)
            if generated.item() == tokenizer.eos_token_id:
                break
        return tokenizer.decode(answer_ids[0, query_ids.shape[1]:], skip_special_tokens=True)


def prefill(model, tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.inference_mode():
        output = model(tokens.to(model.device), use_cache=True)
    return clone_cache(output.past_key_values)


def recursive_generate(model, tokenizer, emb, blocks, neighbors, rounds):
    prefix, _middle, *passages, query = blocks
    caches = [prefill(model, tokenizer, passage) for passage in passages]
    for _ in range(rounds):
        caches = propagate_one_round(model, tokenizer, emb, passages, caches, neighbors)
    merged = clone_cache(caches[0])
    for cache in caches[1:]:
        merged = concact_pkv(merged, cache)
    merged = concact_pkv(prefill(model, tokenizer, prefix), merged)
    total_len = merged.key_cache[0].shape[-2]
    positions = torch.arange(total_len, device=model.device, dtype=torch.long)
    merged = apply_pkv_rotary_position_embeddings(merged, emb, positions)

    query_ids = tokenizer(query, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    query_length = query_ids.shape[1]
    generated = query_ids
    answer_ids = generated
    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            output = model(generated, past_key_values=merged, use_cache=True)
            merged = output.past_key_values
            generated = torch.argmax(output.logits[:, -1, :], dim=-1).unsqueeze(-1)
            answer_ids = torch.cat((answer_ids, generated), dim=1)
            if generated.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(answer_ids[0, query_length:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
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
    generation_config = GenerationConfig(
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    @app.post("/generate_vanilla")
    def generate_vanilla():
        form = request.get_json()
        prompt = "".join(form["blocks"])
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        length = input_ids.shape[1]
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, generation_config=generation_config)[0]
        return {"ret": 0, "generated": tokenizer.decode(output[length:], skip_special_tokens=True), "message": ""}

    @app.post("/generate_gapemp")
    def generate_gapemp():
        form = request.get_json()
        blocks = form["blocks"]
        generated = limited_gapemp(
            tokenizer, model, emb, blocks[0], blocks[1], blocks[-1], blocks[2:-1]
        )
        return {"ret": 0, "generated": generated, "message": ""}

    @app.post("/generate_block")
    def generate_block():
        form = request.get_json()
        blocks = list(form["blocks"])
        del blocks[1]
        generated = block_generate(
            blocks=blocks[:-1],
            instruction=blocks[-1],
            generation_config=generation_config,
            model=model,
            emb=emb,
            tokenizer=tokenizer,
            num_local_attention_blocks=form.get("num_local_attention_blocks", 10000),
        )
        return {"ret": 0, "generated": generated, "message": ""}

    @app.post("/generate_recursive")
    def generate_recursive():
        form = request.get_json()
        generated = recursive_generate(
            model, tokenizer, emb, form["blocks"], form["neighbors"], form.get("rounds", 2)
        )
        return {"ret": 0, "generated": generated, "message": ""}

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
