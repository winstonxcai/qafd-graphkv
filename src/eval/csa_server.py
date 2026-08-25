"""Single-model server for joint-prefill soft-graph CSA."""

from __future__ import annotations

import argparse
import hashlib
import time

import torch
from flask import Flask, request
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.csa.attention import (
    CSAState,
    activate_csa,
    deactivate_csa,
    install_csa_attention,
)
from src.csa.prompt import build_question_first_prompt
from src.csa.routing import routing_summary


app = Flask(__name__)
MODEL = None
TOKENIZER = None
CSA_MODULES = None
MODEL_REVISION = None
ATTENTION_IMPLEMENTATION = None


def _encode_segment(text: str, max_tokens: int | None = None) -> list[int]:
    ids = TOKENIZER(
        text,
        add_special_tokens=False,
        truncation=max_tokens is not None,
        max_length=max_tokens,
    ).input_ids
    if not ids:
        raise ValueError("prompt segment tokenized to zero tokens")
    return ids


def tokenize_segments(question: str, passages: list[dict]) -> tuple:
    if not 3 <= len(passages) <= 5:
        raise ValueError("CSA interface requires three to five passages")
    prompt = build_question_first_prompt(question, passages)
    prefix_ids = _encode_segment(prompt.prefix)
    passage_ids = [_encode_segment(block, 512) for block in prompt.passages]
    suffix_ids = _encode_segment(prompt.suffix)
    ids = list(prefix_ids)
    spans = []
    for block in passage_ids:
        start = len(ids)
        ids.extend(block)
        spans.append((start, len(ids)))
    suffix_start = len(ids)
    ids.extend(suffix_ids)
    digest = hashlib.sha256(
        ",".join(map(str, ids)).encode("ascii")
    ).hexdigest()
    return (
        torch.tensor([ids], dtype=torch.long, device=MODEL.device),
        len(prefix_ids),
        spans,
        suffix_start,
        [len(block) for block in passage_ids],
        digest,
    )


def _greedy_generate(input_ids, max_new_tokens: int, state: CSAState | None):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_started = time.perf_counter()
    if state is not None:
        activate_csa(CSA_MODULES, state)
    try:
        with torch.inference_mode():
            outputs = MODEL(input_ids=input_ids, use_cache=True)
    finally:
        if state is not None:
            deactivate_csa(CSA_MODULES)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - prefill_started

    past = outputs.past_key_values
    token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
    first_token_id = int(token.item())
    answer = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_started = time.perf_counter()
    for _ in range(max_new_tokens):
        if token.item() == TOKENIZER.eos_token_id:
            break
        answer.append(token.item())
        with torch.inference_mode():
            outputs = MODEL(input_ids=token, past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    routing_gpu_seconds = 0.0
    if state is not None and state.routing_events:
        routing_gpu_seconds = sum(
            start.elapsed_time(end) for start, end in state.routing_events
        ) / 1000.0
    return TOKENIZER.decode(answer, skip_special_tokens=True), first_token_id, {
        "prefill_seconds": prefill_seconds,
        "routing_gpu_seconds": routing_gpu_seconds,
        "decode_seconds": decode_seconds,
    }


def generate(payload: dict) -> dict:
    mode = payload.get("mode", "csa")
    if mode not in {"csa", "vanilla"}:
        raise ValueError("mode must be csa or vanilla")
    max_new_tokens = int(payload.get("max_new_tokens", 256))
    if not 1 <= max_new_tokens <= 256:
        raise ValueError("max_new_tokens must be in [1, 256]")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenization_started = started
    input_ids, prefix_end, spans, suffix_start, passage_lengths, prompt_hash = (
        tokenize_segments(payload["question"], payload["passages"])
    )
    tokenization_seconds = time.perf_counter() - tokenization_started
    state = None
    graph_scores = payload.get("graph_scores")
    if mode == "csa":
        graph = torch.tensor(graph_scores, dtype=torch.float32, device=MODEL.device)
        passage_count = len(spans)
        if graph.shape != (passage_count, passage_count) or not torch.isfinite(graph).all():
            raise ValueError("graph_scores must be a finite square passage matrix")
        state = CSAState(
            passage_spans=spans,
            prefix_end=prefix_end,
            suffix_start=suffix_start,
            graph_scores=graph,
            beta=float(payload["beta"]),
            top_b=int(payload["top_b"]),
            pooling=payload["pooling"],
            backend=payload["backend"],
            total_tokens=input_ids.shape[1],
        )
    generated, first_token_id, timing = _greedy_generate(
        input_ids, max_new_tokens, state
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak_vram = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )

    traces = [] if state is None else sorted(state.traces, key=lambda row: row["layer"])
    if state is not None and len(traces) != len(MODEL.model.layers):
        raise ValueError(
            f"incomplete routing trace: {len(traces)} of {len(MODEL.model.layers)} layers"
        )
    summary = (
        None
        if state is None
        else routing_summary(traces, len(spans), graph_scores)
    )
    return {
        "generated": generated,
        "first_token_id": first_token_id,
        "model_seconds": seconds,
        "timing": {"tokenization_seconds": tokenization_seconds, **timing},
        "peak_vram_bytes": peak_vram,
        "token_spans": {
            "prefix": [0, prefix_end],
            "passages": [list(span) for span in spans],
            "suffix": [suffix_start, input_ids.shape[1]],
            "total_tokens": input_ids.shape[1],
            "passage_token_lengths": passage_lengths,
        },
        "prompt_hash": prompt_hash,
        "model_revision": MODEL_REVISION,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "routing_trace": traces,
        "routing_summary": summary,
    }


@app.post("/generate")
def endpoint():
    try:
        return {"ret": 0, **generate(request.get_json())}
    except Exception as error:
        app.logger.exception("CSA generation failed")
        return {"ret": 1, "message": f"{type(error).__name__}: {error}"}, 500


def main() -> None:
    global MODEL, TOKENIZER, CSA_MODULES, MODEL_REVISION, ATTENTION_IMPLEMENTATION
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ldsjmdy/Tulu3-Block-FT")
    parser.add_argument("--port", type=int, default=8980)
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "flash_attention_2"], default="sdpa"
    )
    args = parser.parse_args()
    TOKENIZER = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    TOKENIZER.pad_token = TOKENIZER.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
    )
    MODEL.eval()
    CSA_MODULES = install_csa_attention(MODEL)
    MODEL_REVISION = getattr(MODEL.config, "_commit_hash", None) or args.model
    ATTENTION_IMPLEMENTATION = args.attn_implementation
    app.run(host="127.0.0.1", port=args.port, threaded=False)


if __name__ == "__main__":
    main()
