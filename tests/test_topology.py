from src.qafd_bridge.topology import analyze_passages


class Vertex:
    def __init__(self, name):
        self.data = {"name": name}

    def __getitem__(self, key):
        return self.data[key]


class Vertices:
    def __init__(self, names):
        self.items = [Vertex(name) for name in names]

    def __getitem__(self, index):
        return self.items[index]


class Graph:
    def __init__(self):
        self.vs = Vertices(["chunk-p0", "chunk-p1", "entity-e0", "entity-e1"])
        self.adjacency = {0: [2], 1: [3], 2: [0, 3], 3: [1, 2]}

    def neighbors(self, node):
        return self.adjacency[node]


def test_bounded_entity_topology():
    result = analyze_passages(Graph(), [0, 1], max_hops=1)
    assert result.edges == 1
    assert result.components == 1
    assert result.hop_histogram == {1: 1}
    assert result.diameter == 1
