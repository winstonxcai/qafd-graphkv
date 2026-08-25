import pytest
import torch

pytest.importorskip("transformers")

from src.csa.attention import (
    build_csa_mask,
    dense_reference_attention,
    gathered_passage_attention,
)


def test_mask_enforces_causal_passage_policy_and_dense_answer_access():
    spans = [(2, 4), (4, 6), (6, 8)]
    selected = [[], [0], [0]]
    mask = build_csa_mask(10, 2, spans, 8, selected, "cpu")
    assert mask[3, :4].all()  # prefix and causal self
    assert not mask[4, 5]  # no future self token
    assert mask[4, 2:4].all()  # selected earlier passage
    assert not mask[6, 4:6].any()  # unselected earlier passage is hidden
    assert not mask[2, 4:].any()  # no future passage access
    assert mask[9, :10].all()  # answer cue's final token has dense causal access


def test_gathered_backend_matches_dense_reference():
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 2, 10, 4, generator=generator)
    key = torch.randn(1, 2, 10, 4, generator=generator)
    value = torch.randn(1, 2, 10, 4, generator=generator)
    spans = [(2, 4), (4, 7), (7, 9)]
    selected = [[], [0], [0]]
    mask = build_csa_mask(10, 2, spans, 9, selected, "cpu")
    dense = dense_reference_attention(query, key, value, mask, 0.5)
    gathered = gathered_passage_attention(
        query, key, value, 2, spans, 9, selected, 0.5
    )
    assert torch.allclose(dense, gathered, atol=1e-6, rtol=1e-5)


def test_b4_policy_is_global_causal_for_five_passages():
    spans = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    selected = [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]
    mask = build_csa_mask(8, 2, spans, 7, selected, "cpu")
    assert torch.equal(mask, torch.ones(8, 8, dtype=torch.bool).tril())
