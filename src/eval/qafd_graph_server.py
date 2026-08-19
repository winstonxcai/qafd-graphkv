"""Serve matched sequential and graph-specific QAFD+GraphKV generation."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from flask import Flask, request
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

GRAPHKV_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "GraphKV")
)
sys.path.insert(0, GRAPHKV_ROOT)
from pcw_parallel import gapemp_graph  # noqa: E402


app = Flask(__name__)
CENTER_INSTRUCTION = (
    "\nNow you will read the center paper and answer a related question: \n"
)


def sequential_prompt(prefix: str, center: str, neighbors: list[str], query: str) -> str:
    return prefix + "".join(neighbors) + CENTER_INSTRUCTION + center + query


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument("--port", type=int, default=8870)
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
    emb = LlamaRotaryEmbedding(config=config).to(
        device=model.device, dtype=torch.float32
    )
    emb.eval()

    @app.post("/generate_qafd_graphkv")
    def generate_qafd_graphkv():
        form = request.get_json()
        generated = gapemp_graph(
            tokenizer,
            model,
            emb,
            form["prefix"],
            form["center"],
            form["neighbors"],
            form["query"],
            args.model,
            1,
            1,
            None,
        )
        return {"ret": 0, "generated": generated, "message": ""}

    @app.post("/generate_matched_sequential")
    def generate_matched_sequential():
        form = request.get_json()
        prompt = sequential_prompt(
            form["prefix"], form["center"], form["neighbors"], form["query"]
        )
        input_ids = tokenizer(
            prompt, truncation=False, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)
        context_length = input_ids.shape[-1]
        with torch.inference_mode():
            response = model.generate(
                input_ids=input_ids,
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                eos_token_id=tokenizer.eos_token_id,
            )[0]
        generated = tokenizer.decode(
            response[context_length:], skip_special_tokens=True
        )
        return {"ret": 0, "generated": generated, "message": ""}

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
