from src.qafd_bridge.serialize import Passage, QAFDRecord, edge_from_path, read_jsonl, write_jsonl


def test_round_trip(tmp_path):
    record = QAFDRecord(
        qid="q1",
        question="Which passage is relevant?",
        answer="answer",
        passages=(
            Passage("p0", "first", 0.8),
            Passage("p1", "second", 0.2),
        ),
        edges=(edge_from_path(0, 1, ["entity-A", "entity-B"]),),
        diameter=1,
    )
    output = tmp_path / "questions.jsonl"
    assert write_jsonl([record], output) == 1
    assert read_jsonl(output)[0]["edges"][0]["entity_hops"] == 1


def test_rejects_invalid_edge(tmp_path):
    record = QAFDRecord(
        qid="q1",
        question="question",
        answer="answer",
        passages=(Passage("p0", "text", 1.0),),
        edges=(),
        diameter=0,
    )
    record = record.__class__(
        **{**record.__dict__, "edges": (edge_from_path(0, 0, ["a", "b"]),)}
    )
    try:
        write_jsonl([record], tmp_path / "bad.jsonl")
    except ValueError as error:
        assert "self-edges" in str(error)
    else:
        raise AssertionError("invalid self-edge was accepted")
