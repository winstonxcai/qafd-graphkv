import torch

from src.csa.routing import _pool_tokens, passage_index_scores, select_routes, standardize


def test_pooling_methods_and_final_normalization():
    states = torch.tensor([[[[3.0, 0.0], [0.0, 1.0]]]])
    normalized = _pool_tokens(states, 0, 2, "normalized_token_mean")
    plain = _pool_tokens(states, 0, 2, "plain_mean")
    assert torch.allclose(normalized.norm(dim=-1), torch.ones(1, 1))
    assert torch.allclose(plain.norm(dim=-1), torch.ones(1, 1))
    assert not torch.allclose(normalized, plain)


def test_population_standardization_and_constant_candidates():
    assert torch.allclose(standardize(torch.tensor([1.0])), torch.tensor([0.0]))
    assert torch.allclose(standardize(torch.tensor([2.0, 2.0])), torch.zeros(2))
    result = standardize(torch.tensor([1.0, 3.0]))
    assert torch.allclose(result, torch.tensor([-1.0, 1.0]), atol=2e-6)


def test_gqa_mapping_scores_corresponding_kv_head():
    # Four query heads, two KV heads, two query heads per KV head.
    query = torch.zeros(1, 4, 2, 2)
    key = torch.zeros(1, 2, 2, 2)
    query[:, :2, 1] = torch.tensor([1.0, 0.0])
    query[:, 2:, 1] = torch.tensor([0.0, 1.0])
    key[:, 0, 0] = torch.tensor([1.0, 0.0])
    key[:, 1, 0] = torch.tensor([0.0, 1.0])
    scores = passage_index_scores(query, key, [(0, 1), (1, 2)], 2, "plain_mean")
    assert torch.allclose(scores[1, 0], torch.tensor(1.0))
    assert torch.isnan(scores[0, 0])


def test_beta_zero_graph_invariance_and_stable_ties():
    llm = torch.full((4, 4), torch.nan)
    graph_a = torch.zeros(4, 4)
    graph_b = torch.zeros(4, 4)
    for target in range(1, 4):
        llm[target, :target] = 0.0
        graph_b[target, :target] = torch.arange(target, dtype=torch.float32)
    selected_a, *_ = select_routes(llm, graph_a, beta=0.0, top_b=2)
    selected_b, *_ = select_routes(llm, graph_b, beta=0.0, top_b=2)
    assert selected_a == selected_b
    assert selected_a == [[], [0], [0, 1], [0, 1]]


def test_missing_edges_are_zero_and_beta_can_change_route():
    llm = torch.full((3, 3), torch.nan)
    llm[1, 0] = 0
    llm[2, :2] = torch.tensor([1.0, 0.0])
    graph = torch.zeros(3, 3)
    graph[2, :2] = torch.tensor([0.0, 10.0])
    llm_only, *_ = select_routes(llm, graph, beta=0.0, top_b=1)
    graph_aided, *_ = select_routes(llm, graph, beta=2.0, top_b=1)
    assert llm_only[2] == [0]
    assert graph_aided[2] == [1]
