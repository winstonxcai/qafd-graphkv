import sys
import types

sys.modules.setdefault("igraph", types.SimpleNamespace(Graph=object))

from src.eval.ensemble_predictions import combine_rows


def prediction(qid, generated, seconds):
    return {
        "qid": qid,
        "question": "Who?",
        "answers": ["Alice"],
        "generated": generated,
        "seconds": seconds,
    }


def test_combine_rows_uses_candidate_union_and_serial_latency():
    combined = combine_rows(
        [[prediction(0, "Bob", 1.0)], [prediction(0, "Alice", 2.0)]]
    )

    assert combined[0]["em"] == 1.0
    assert combined[0]["seconds"] == 3.0
    assert "Alternative candidate" in combined[0]["generated"]
