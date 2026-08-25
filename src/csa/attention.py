"""Project-owned Llama attention wrapper for joint-prefill CSA."""

from __future__ import annotations

import types
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from src.csa.routing import (
    LayerRouting,
    nullable_matrix,
    passage_index_scores,
    select_routes,
)


BACKENDS = {"dense_reference", "gathered_sdpa"}


@dataclass
class CSAState:
    passage_spans: list[tuple[int, int]]
    prefix_end: int
    suffix_start: int
    graph_scores: torch.Tensor
    beta: float
    top_b: int
    pooling: str
    backend: str
    total_tokens: int
    traces: list[dict] = field(default_factory=list)
    routing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list, repr=False
    )

    def validate(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"unsupported CSA backend: {self.backend}")
        if len(self.passage_spans) != self.graph_scores.shape[0]:
            raise ValueError("passage spans and graph matrix do not align")
        if self.graph_scores.shape != (
            len(self.passage_spans),
            len(self.passage_spans),
        ):
            raise ValueError("graph score matrix must be square")
        cursor = self.prefix_end
        for start, end in self.passage_spans:
            if start != cursor or end <= start:
                raise ValueError("passage spans must be non-empty and contiguous")
            cursor = end
        if cursor != self.suffix_start or not self.suffix_start < self.total_tokens:
            raise ValueError("suffix span does not follow passages")


def build_csa_mask(
    total_tokens: int,
    prefix_end: int,
    passage_spans: list[tuple[int, int]],
    suffix_start: int,
    selected: list[list[int]],
    device,
) -> torch.Tensor:
    """Construct the semantic CSA mask; True entries are attendable."""
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)
    positions = torch.arange(total_tokens, device=device)
    mask[:prefix_end, :prefix_end] = positions[:prefix_end].unsqueeze(0) <= positions[
        :prefix_end
    ].unsqueeze(1)
    for target, (start, end) in enumerate(passage_spans):
        mask[start:end, :prefix_end] = True
        for source in selected[target]:
            source_start, source_end = passage_spans[source]
            mask[start:end, source_start:source_end] = True
        length = end - start
        mask[start:end, start:end] = torch.ones(
            (length, length), dtype=torch.bool, device=device
        ).tril()
    global_queries = positions[suffix_start:].unsqueeze(1)
    mask[suffix_start:, :] = positions.unsqueeze(0) <= global_queries
    return mask


def _sdpa(query, key, value, mask, scaling):
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
    )


def dense_reference_attention(query, key, value, mask, scaling):
    return _sdpa(query, key, value, mask, scaling)


def gathered_passage_attention(
    query,
    key,
    value,
    prefix_end: int,
    passage_spans: list[tuple[int, int]],
    suffix_start: int,
    selected: list[list[int]],
    scaling: float,
):
    """Compute attention by gathering only the selected source blocks."""
    output = torch.empty_like(query)
    device = query.device

    prefix_mask = torch.ones(
        (prefix_end, prefix_end), dtype=torch.bool, device=device
    ).tril()
    output[:, :, :prefix_end, :] = _sdpa(
        query[:, :, :prefix_end, :],
        key[:, :, :prefix_end, :],
        value[:, :, :prefix_end, :],
        prefix_mask,
        scaling,
    )

    for target, (start, end) in enumerate(passage_spans):
        indices = list(range(prefix_end))
        for source in sorted(selected[target]):
            source_start, source_end = passage_spans[source]
            indices.extend(range(source_start, source_end))
        prior_length = len(indices)
        indices.extend(range(start, end))
        index = torch.tensor(indices, dtype=torch.long, device=device)
        target_length = end - start
        local_mask = torch.ones(
            (target_length, len(indices)), dtype=torch.bool, device=device
        )
        local_mask[:, prior_length:] = torch.ones(
            (target_length, target_length), dtype=torch.bool, device=device
        ).tril()
        output[:, :, start:end, :] = _sdpa(
            query[:, :, start:end, :],
            key.index_select(2, index),
            value.index_select(2, index),
            local_mask,
            scaling,
        )

    suffix_length = query.shape[-2] - suffix_start
    suffix_queries = torch.arange(
        suffix_start, query.shape[-2], device=device
    ).unsqueeze(1)
    all_keys = torch.arange(query.shape[-2], device=device).unsqueeze(0)
    suffix_mask = all_keys <= suffix_queries
    if suffix_length:
        output[:, :, suffix_start:, :] = _sdpa(
            query[:, :, suffix_start:, :],
            key,
            value,
            suffix_mask,
            scaling,
        )
    return output


def csa_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    state: CSAState | None = getattr(self, "_csa_state", None)
    if state is None or hidden_states.shape[1] != state.total_tokens:
        return self._csa_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
    if hidden_states.shape[0] != 1:
        raise ValueError("CSA v1 supports batch size one")

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    route_start = route_end = None
    if hidden_states.is_cuda:
        route_start = torch.cuda.Event(enable_timing=True)
        route_end = torch.cuda.Event(enable_timing=True)
        route_start.record()
    llm_scores = passage_index_scores(
        query_states,
        key_states,
        state.passage_spans,
        self.num_key_value_groups,
        state.pooling,
    )
    graph_scores = state.graph_scores.to(device=hidden_states.device, dtype=torch.float32)
    selected, llm_z, graph_z, combined = select_routes(
        llm_scores, graph_scores, state.beta, state.top_b
    )
    semantic_mask = build_csa_mask(
        state.total_tokens,
        state.prefix_end,
        state.passage_spans,
        state.suffix_start,
        selected,
        hidden_states.device,
    )
    if route_end is not None:
        route_end.record()
        state.routing_events.append((route_start, route_end))

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )
    expanded_keys = repeat_kv(key_states, self.num_key_value_groups)
    expanded_values = repeat_kv(value_states, self.num_key_value_groups)
    if state.backend == "dense_reference":
        attn_output = dense_reference_attention(
            query_states,
            expanded_keys,
            expanded_values,
            semantic_mask,
            self.scaling,
        )
    else:
        attn_output = gathered_passage_attention(
            query_states,
            expanded_keys,
            expanded_values,
            state.prefix_end,
            state.passage_spans,
            state.suffix_start,
            selected,
            self.scaling,
        )

    attended_pairs = int(semantic_mask.sum().item())
    dense_pairs = state.total_tokens * (state.total_tokens + 1) // 2
    trace = LayerRouting(
        layer=int(self.layer_idx),
        llm_scores=nullable_matrix(llm_scores),
        llm_standardized=nullable_matrix(llm_z),
        graph_standardized=nullable_matrix(graph_z),
        combined_scores=nullable_matrix(combined),
        selected=[list(values) for values in selected],
        attended_qk_pairs=attended_pairs,
        dense_qk_pairs=dense_pairs,
    )
    state.traces.append(trace.to_dict())

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), None


def install_csa_attention(model) -> list:
    """Patch project-owned runtime wrappers onto all Llama attention layers."""
    modules = []
    for layer in model.model.layers:
        attention = layer.self_attn
        if not hasattr(attention, "_csa_original_forward"):
            attention._csa_original_forward = attention.forward
            attention.forward = types.MethodType(csa_attention_forward, attention)
        attention._csa_state = None
        modules.append(attention)
    return modules


def activate_csa(modules: list, state: CSAState) -> None:
    state.validate()
    if state.traces:
        raise ValueError("CSA state trace must be empty at activation")
    for module in modules:
        module._csa_state = state


def deactivate_csa(modules: list) -> None:
    for module in modules:
        module._csa_state = None
