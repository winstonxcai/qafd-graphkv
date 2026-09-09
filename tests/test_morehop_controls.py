from src.graph.morehop_controls import degree_preserving_control, graph_statistics, reverse_neighbors


def test_star_cannot_be_rewired_with_labelled_degrees():
    graph = [[3] for _ in range(4)]
    out = degree_preserving_control(graph, 42)
    assert out['neighbors'] == graph
    assert not out['distinct_control']
    stats = graph_statistics(graph)
    assert stats['cross_edges'] == 3
    assert stats['diagonal_edges'] == 1
    assert stats['full_pair_ratio'] == .25
    assert stats['weak_component_count'] == 1
    assert stats['unreachable_ordered_pair_fraction'] == .75


def test_swaps_preserve_labelled_degrees_and_diagonal():
    graph = [[0, 1], [2], [3], [4], [0]]
    out = degree_preserving_control(graph, 12, 25)
    assert out == degree_preserving_control(graph, 12, 25)
    assert out['accepted_swaps'] > 0
    before, after = graph_statistics(graph), graph_statistics(out['neighbors'])
    for key in ['sources_per_target', 'targets_per_source', 'cross_edges', 'diagonal_edges']:
        assert before[key] == after[key]
    assert 0 in out['neighbors'][0]


def test_reverse_twice_and_isolation():
    graph = [[1], [], [2]]
    assert reverse_neighbors(reverse_neighbors(graph)) == graph
    stats = graph_statistics(graph)
    assert stats['isolated_nodes'] == 1
    assert stats['weak_component_count'] == 2
    assert stats['finite_directed_diameter'] == 1
