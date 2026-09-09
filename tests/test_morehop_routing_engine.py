from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers")
from transformers.cache_utils import DynamicCache

import src.eval.morehop_routing_engine as engine
from src.eval.graphkv_engine_parity import tokenize_inputs
from src.eval.morehop_routing_runner import METHODS


class RecordingTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def __call__(self, value, **kwargs):
        self.calls.append((value, kwargs))
        if isinstance(value, list):
            widths = [len(item) for item in value]
            width = max(widths)
            return SimpleNamespace(input_ids=torch.tensor([
                [i + 1] * width for i in range(len(value))
            ]))
        width = {"P": 2, "M": 3, "Q": 5}.get(value, len(value))
        return SimpleNamespace(input_ids=torch.ones((1, width), dtype=torch.long))


def _cache(length, value):
    cache = DynamicCache()
    tensor = torch.full((1, 1, length, 1), value, dtype=torch.float32)
    cache.key_cache = [tensor.clone()]
    cache.value_cache = [tensor.clone() + 100]
    return cache


def test_tokenize_inputs_preserves_blocks_and_upstream_budget():
    tokenizer = RecordingTokenizer()
    blocks = ["P", "M", "doc one\n", "doc two\n", "Q"]
    ids = tokenize_inputs(tokenizer, blocks)
    assert tokenizer.calls[3][0] == ["doc one\n", "doc two\n"]
    assert tokenizer.calls[3][1]["max_length"] == 8192 - 2 - 3 - 5 - 256
    assert tokenizer.calls[3][1]["truncation"] is True
    assert set(ids) == {"prefix_ids", "middle_ids", "query_ids", "context_ids"}


def test_raw_cache_split_uses_document_token_spans_not_layers():
    raw = _cache(6, 7)
    raw.key_cache[0][0, 0, :, 0] = torch.arange(6)
    pieces = engine._split_cache(raw, [2, 1, 3])
    assert [piece.key_cache[0].shape[-2] for piece in pieces] == [2, 1, 3]
    assert [piece.key_cache[0][0, 0, :, 0].tolist() for piece in pieces] == [[0, 1], [2], [3, 4, 5]]


def test_source_rope_is_applied_before_second_model_call(monkeypatch):
    events = []
    source = _cache(2, 1)
    context_ids = torch.tensor([[4, 5], [6, 0]])

    def rotate(cache, emb, *positions):
        events.append("rotate")
        return cache

    class Model:
        device = torch.device("cpu")

        def __call__(self, input_ids, **kwargs):
            if "past_key_values" in kwargs:
                events.append("model_with_source")
                assert events[-2:] == ["rotate", "model_with_source"]
            return SimpleNamespace(past_key_values=_cache(input_ids.shape[-1], 2))

    monkeypatch.setattr(engine, "apply_pkv_rotary_position_embeddings", rotate)
    monkeypatch.setattr(engine, "apply_pkv_rerotary_position_embeddings", lambda cache, emb, *pos: cache)
    monkeypatch.setattr(engine, "cut_pkv", lambda cache, positions: cache)
    monkeypatch.setattr(engine, "flatten_pkv", lambda cache, mask: cache)
    monkeypatch.setattr(engine, "stack_pkv", lambda cache, batch: cache)
    tokenizer = SimpleNamespace(pad_token_id=0)
    result = engine._updated_for_group(
        Model(), tokenizer, object(), {"context_ids": context_ids}, [0], [source], (0,)
    )
    assert 0 in result
    assert events[:2] == ["rotate", "model_with_source"]


def test_initial_methods_are_frozen():
    assert METHODS == (
        "released_last1", "released_last3", "full", "query_bm25_k1",
        "query_bm25_k3", "ppr_k1", "random_k1_seed42", "reversed_rank_k1",
    )
