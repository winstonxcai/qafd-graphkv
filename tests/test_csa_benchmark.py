import pytest

pytest.importorskip("igraph")

from src.eval.csa_benchmark import validate_prediction


def valid_row():
    matrix = [[None] * 5 for _ in range(5)]
    return {
        "qid": 500,
        "configuration": {"mode": "csa"},
        "passages": [{} for _ in range(5)],
        "graph_scores": [[0.0] * 5 for _ in range(5)],
        "token_spans": {
            "prefix": [0, 2],
            "passages": [[2, 3], [3, 5], [5, 6], [6, 7], [7, 8]],
            "passage_token_lengths": [1, 2, 1, 1, 1],
            "suffix": [8, 10],
            "total_tokens": 10,
        },
        "prompt_hash": "0" * 64,
        "routing_trace": [
            {
                "layer": 0,
                "selected": [[], [0], [0], [0], [0]],
                "llm_scores": matrix,
                "llm_standardized": matrix,
                "graph_standardized": matrix,
                "combined_scores": matrix,
            }
        ],
    }


def test_complete_prediction_row_is_accepted():
    validate_prediction(valid_row(), {"mode": "csa"}, 500)


def test_passage_span_misalignment_is_rejected():
    row = valid_row()
    row["token_spans"]["passages"][2][0] = 4
    with pytest.raises(ValueError, match="misaligned"):
        validate_prediction(row, {"mode": "csa"}, 500)
