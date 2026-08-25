"""Training-free passage routing from layer-local Q/K projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


POOLING_METHODS = {"normalized_token_mean", "plain_mean"}


@dataclass(frozen=True)
class LayerRouting:
    layer: int
    llm_scores: list[list[float | None]]
    llm_standardized: list[list[float | None]]
    graph_standardized: list[list[float | None]]
    combined_scores: list[list[float | None]]
    selected: list[list[int]]
    attended_qk_pairs: int
    dense_qk_pairs: int

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "llm_scores": self.llm_scores,
            "llm_standardized": self.llm_standardized,
            "graph_standardized": self.graph_standardized,
            "combined_scores": self.combined_scores,
            "selected": self.selected,
            "attended_qk_pairs": self.attended_qk_pairs,
            "dense_qk_pairs": self.dense_qk_pairs,
        }


def _pool_tokens(states: torch.Tensor, start: int, end: int, method: str) -> torch.Tensor:
    if method not in POOLING_METHODS:
        raise ValueError(f"unsupported pooling method: {method}")
    if not 0 <= start < end <= states.shape[-2]:
        raise ValueError(f"invalid token span [{start}, {end})")
    tokens = states[..., start:end, :].float()
    if method == "normalized_token_mean":
        tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-12)
    return F.normalize(tokens.mean(dim=-2), p=2, dim=-1, eps=1e-12)


def passage_index_scores(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    passage_spans: list[tuple[int, int]],
    num_key_value_groups: int,
    pooling: str,
) -> torch.Tensor:
    """Return a causal passage score matrix from pre-RoPE Q/K tensors.

    Output entries on and above the diagonal are NaN because only earlier
    passages are eligible under decoder-causal routing.
    """
    if query_states.shape[0] != 1 or key_states.shape[0] != 1:
        raise ValueError("CSA v1 supports batch size one")
    query_heads = query_states.shape[1]
    key_value_heads = key_states.shape[1]
    if query_heads != key_value_heads * num_key_value_groups:
        raise ValueError("GQA head counts do not match num_key_value_groups")

    pooled_queries = [
        _pool_tokens(query_states, start, end, pooling)[0]
        for start, end in passage_spans
    ]
    pooled_keys = [
        _pool_tokens(key_states, start, end, pooling)[0]
        for start, end in passage_spans
    ]
    kv_for_query_head = torch.arange(query_heads, device=query_states.device)
    kv_for_query_head = torch.div(
        kv_for_query_head, num_key_value_groups, rounding_mode="floor"
    )
    scores = torch.full(
        (len(passage_spans), len(passage_spans)),
        torch.nan,
        dtype=torch.float32,
        device=query_states.device,
    )
    for target in range(len(passage_spans)):
        target_queries = pooled_queries[target]
        for source in range(target):
            mapped_keys = pooled_keys[source][kv_for_query_head]
            scores[target, source] = (target_queries * mapped_keys).sum(dim=-1).mean()
    return scores


def standardize(values: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Population-standardize a one-dimensional candidate vector."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("standardize expects a non-empty one-dimensional tensor")
    if values.numel() == 1:
        return torch.zeros_like(values, dtype=torch.float32)
    values = values.float()
    deviation = values.std(unbiased=False)
    if deviation <= epsilon:
        return torch.zeros_like(values)
    return (values - values.mean()) / (deviation + epsilon)


def select_routes(
    llm_scores: torch.Tensor,
    graph_scores: torch.Tensor,
    beta: float,
    top_b: int,
    epsilon: float = 1e-6,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select stable causal TopB sources for every target passage."""
    if llm_scores.shape != graph_scores.shape or llm_scores.ndim != 2:
        raise ValueError("LLM and graph score matrices must have equal square shape")
    if llm_scores.shape[0] != llm_scores.shape[1]:
        raise ValueError("passage score matrices must be square")
    if top_b < 1:
        raise ValueError("top_b must be positive")
    if beta < 0:
        raise ValueError("beta must be non-negative")

    count = llm_scores.shape[0]
    llm_z = torch.full_like(llm_scores, torch.nan, dtype=torch.float32)
    graph_z = torch.full_like(llm_scores, torch.nan, dtype=torch.float32)
    combined = torch.full_like(llm_scores, torch.nan, dtype=torch.float32)
    selected: list[list[int]] = [[] for _ in range(count)]
    for target in range(count):
        if target == 0:
            continue
        candidates = torch.arange(target, device=llm_scores.device)
        llm_values = llm_scores[target, candidates]
        graph_values = graph_scores[target, candidates]
        if not torch.isfinite(llm_values).all() or not torch.isfinite(graph_values).all():
            raise ValueError("eligible passage scores must be finite")
        llm_values_z = standardize(llm_values, epsilon)
        graph_values_z = standardize(graph_values, epsilon)
        combined_values = llm_values_z + float(beta) * graph_values_z
        llm_z[target, candidates] = llm_values_z
        graph_z[target, candidates] = graph_values_z
        combined[target, candidates] = combined_values
        # Candidate indices are ascending; stable descending sort therefore
        # resolves exact ties toward the lexicographically smaller index.
        order = torch.argsort(combined_values, descending=True, stable=True)
        keep = order[: min(top_b, target)]
        selected[target] = candidates[keep].tolist()
    return selected, llm_z, graph_z, combined


def nullable_matrix(values: torch.Tensor) -> list[list[float | None]]:
    result = []
    for row in values.detach().cpu().tolist():
        result.append([float(value) if value == value else None for value in row])
    return result


def routing_summary(
    traces: list[dict], passage_count: int, graph_scores: list[list[float]]
) -> dict:
    frequency = [[0 for _ in range(passage_count)] for _ in range(passage_count)]
    selected_edges = 0
    graph_supported = 0
    churn_values = []
    previous = None
    for trace in traces:
        current = [set(values) for values in trace["selected"]]
        for target, sources in enumerate(current):
            for source in sources:
                frequency[target][source] += 1
                selected_edges += 1
                graph_supported += int(graph_scores[target][source] > 0)
        if previous is not None:
            for target in range(passage_count):
                union = current[target] | previous[target]
                if union:
                    churn_values.append(
                        1.0 - len(current[target] & previous[target]) / len(union)
                    )
        previous = current
    attended = sum(int(trace["attended_qk_pairs"]) for trace in traces)
    dense = sum(int(trace["dense_qk_pairs"]) for trace in traces)
    return {
        "selection_frequency": frequency,
        "mean_layer_churn": (
            sum(churn_values) / len(churn_values) if churn_values else 0.0
        ),
        "graph_prior_agreement": (
            graph_supported / selected_edges if selected_edges else 0.0
        ),
        "attended_qk_pairs": attended,
        "dense_qk_pairs": dense,
        "qk_pair_ratio": attended / dense if dense else 0.0,
    }
