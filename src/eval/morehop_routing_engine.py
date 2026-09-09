"""One-round, per-target GraphKV source-index routing."""

from __future__ import annotations

import os
import sys
import hashlib
from collections import defaultdict
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

_GRAPHKV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "GraphKV"))
if _GRAPHKV not in sys.path:
    sys.path.insert(0, _GRAPHKV)

from pcw import (  # noqa: E402
    apply_pkv_rerotary_position_embeddings,
    apply_pkv_rotary_position_embeddings,
    concact_pkv,
    cut_pkv,
    flatten_pkv,
    stack_pkv,
)
from src.eval.graphkv_engine_parity import tokenize_inputs  # noqa: E402


def clone_cache(cache: DynamicCache) -> DynamicCache:
    result = type(cache)()
    result.key_cache = [tensor.clone().detach() for tensor in cache.key_cache]
    result.value_cache = [tensor.clone().detach() for tensor in cache.value_cache]
    return result


def _concat(caches: list[DynamicCache]) -> DynamicCache:
    if not caches:
        raise ValueError("cannot concatenate an empty cache list")
    result = clone_cache(caches[0])
    for cache in caches[1:]:
        result = concact_pkv(result, clone_cache(cache))
    return result


def _split_cache(cache: DynamicCache, lengths: list[int]) -> list[DynamicCache]:
    """Split flattened cache by document token spans, never by layer index."""
    result, cursor = [], 0
    for length in lengths:
        positions = torch.arange(cursor, cursor + length, device=cache.key_cache[0].device)
        result.append(cut_pkv(clone_cache(cache), positions))
        cursor += length
    if cursor != cache.key_cache[0].shape[-2]:
        raise ValueError("document token spans do not cover the raw cache")
    return result


def _encode_raw_documents(model, tokenizer, emb, blocks: list[str]):
    ids = tokenize_inputs(tokenizer, blocks)
    context_ids = ids["context_ids"]
    mask = (context_ids != tokenizer.pad_token_id).reshape(-1)
    lengths = [int(row.sum().item()) for row in (context_ids != tokenizer.pad_token_id)]
    with torch.inference_mode():
        outputs = model(context_ids.to(model.device), use_cache=True)
        raw = apply_pkv_rerotary_position_embeddings(outputs.past_key_values, emb)
        raw = flatten_pkv(raw, mask)
    return _split_cache(raw, lengths), ids, lengths


def _updated_for_group(model, tokenizer, emb, input_ids, target_indices, raw_documents, source_set):
    """Run one homogeneous batched second pass for targets sharing sources."""
    context_ids = input_ids["context_ids"]
    target_ids = context_ids[target_indices].to(model.device)
    target_mask = (context_ids[target_indices] != tokenizer.pad_token_id).reshape(-1)
    source = _concat([raw_documents[index] for index in source_set]) if source_set else None
    source_len = source.key_cache[0].shape[-2] if source is not None else 0
    with torch.inference_mode():
        if source is None:
            outputs = model(target_ids, use_cache=True)
        else:
            # Raw caches are canonical; the second model call needs source RoPE.
            rotated_source = apply_pkv_rotary_position_embeddings(source, emb)
            outputs = model(target_ids, past_key_values=stack_pkv(rotated_source, len(target_indices)), use_cache=True)
    positions = torch.arange(target_ids.shape[-1], dtype=torch.long, device=model.device) + source_len
    updated = cut_pkv(outputs.past_key_values, positions)
    updated = apply_pkv_rerotary_position_embeddings(updated, emb, positions)
    updated = flatten_pkv(updated, target_mask)
    result, cursor = {}, 0
    for index in target_indices:
        length = int((context_ids[index] != tokenizer.pad_token_id).sum().item())
        positions = torch.arange(cursor, cursor + length, device=model.device)
        result[index] = cut_pkv(clone_cache(updated), positions)
        cursor += length
    return result


def build_one_round_cache(model, tokenizer, emb, blocks: list[str], neighbors: list[list[int]]) -> tuple[DynamicCache, dict[str, Any]]:
    """Build a final one-round cache from explicit upstream prompt blocks."""
    if len(blocks) < 4:
        raise ValueError("blocks must contain prefix, middle, a document, and query")
    documents = blocks[2:-1]
    if not documents or len(neighbors) != len(documents):
        raise ValueError("neighbors must have one entry per document")
    for target, sources in enumerate(neighbors):
        if len(sources) != len(set(sources)):
            raise ValueError("duplicate source index")
        for source in sources:
            if type(source) is not int or not 0 <= source < len(documents):
                raise ValueError(f"invalid source index {source} for target {target}")
    # Routing selects a set; preserve released order inside every source cache.
    neighbors = [sorted(sources) for sources in neighbors]

    raw_documents, ids, lengths = _encode_raw_documents(model, tokenizer, emb, blocks)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for target, sources in enumerate(neighbors):
        groups[tuple(sources)].append(target)
    updated_by_target = {}
    for source_set, targets in groups.items():
        updated_by_target.update(_updated_for_group(model, tokenizer, emb, ids, targets, raw_documents, source_set))

    context = _concat(raw_documents + [updated_by_target[index] for index in range(len(documents))])
    with torch.inference_mode():
        prefix_outputs = model(ids["prefix_ids"].to(model.device), use_cache=True)
        prefix_cache = apply_pkv_rerotary_position_embeddings(prefix_outputs.past_key_values, emb)
        # Apply final RoPE only after prefix + raw + updated are concatenated.
        final = apply_pkv_rotary_position_embeddings(concact_pkv(prefix_cache, context), emb)
    metadata = {
        "input_ids_sha256": {key: hashlib.sha256(value.numpy().tobytes()).hexdigest() for key, value in ids.items()},
        "document_token_counts": lengths,
        "raw_document_token_count": sum(lengths),
        "updated_document_token_count": sum(lengths),
        "source_token_exposure": sum(sum(lengths[source] for source in sources) for sources in neighbors),
        "final_cache_length": int(final.key_cache[0].shape[-2]),
        "homogeneous_source_groups": sum(len(targets) > 1 for targets in groups.values()),
    }
    return final, metadata


def greedy_generate(model, tokenizer, cache, query_ids: torch.Tensor | str, max_new_tokens: int = 256) -> str:
    if isinstance(query_ids, str):
        query_ids = tokenizer(query_ids, return_tensors="pt", add_special_tokens=False).input_ids
    current, past, generated = query_ids.to(model.device), clone_cache(cache), []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            output = model(current, past_key_values=past, use_cache=True)
            past = output.past_key_values
            token = int(torch.argmax(output.logits[:, -1, :], dim=-1).item())
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            current = torch.tensor([[token]], device=model.device)
    return tokenizer.decode(generated, skip_special_tokens=True)
