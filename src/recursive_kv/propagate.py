"""Reusable recursive KV propagation built on GraphKV cache utilities."""

from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

from pcw import apply_pkv_rerotary_position_embeddings, concact_pkv, cut_pkv


def clone_cache(cache: DynamicCache) -> DynamicCache:
    result = DynamicCache()
    result.key_cache = [tensor.clone() for tensor in cache.key_cache]
    result.value_cache = [tensor.clone() for tensor in cache.value_cache]
    return result


def concat_sources(caches: list[DynamicCache]) -> DynamicCache:
    merged = clone_cache(caches[0])
    for cache in caches[1:]:
        merged = concact_pkv(cache, merged)
    return merged


def propagate_one_round(model, tokenizer, emb, passages, previous, neighbors):
    """Propagate each node's neighbor caches into its own passage once."""
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
            output = model(tokens.to(model.device), past_key_values=source, use_cache=True)
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
