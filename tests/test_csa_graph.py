from src.csa.graph import qafd_prior_matrix
class Vertex:
    def __init__(self, name, content):
        self.values = {"name": name, "content": content}

    def __getitem__(self, key):
        return self.values[key]


class FakeGraph:
    def __init__(self, vertices, edges):
        self.vs = [Vertex(vertex["name"], vertex["content"]) for vertex in vertices]
        self.edges = [set() for _ in vertices]
        for left, right in edges:
            self.edges[left].add(right)
            self.edges[right].add(left)

    def neighbors(self, vertex):
        return sorted(self.edges[vertex])

    def vcount(self):
        return len(self.vs)


def test_qafd_prior_is_symmetric_and_zero_for_missing_paths():
    graph = FakeGraph(
        [
            {"name": "chunk-0", "content": "p0"},
            {"name": "chunk-1", "content": "p1"},
            {"name": "chunk-2", "content": "p2"},
            {"name": "entity-a", "content": "a"},
        ],
        [(0, 3), (1, 3)],
    )
    matrix, paths = qafd_prior_matrix(graph, [0, 1, 2], {"entity-a": 1.0})
    assert matrix[0][1] == matrix[1][0] > 0
    assert matrix[0][2] == matrix[2][0] == 0
    assert len(paths) == 1
