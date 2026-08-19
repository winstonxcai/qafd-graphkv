import json
from pathlib import Path

import pytest

from src.eval.log_paired_attempt import (
    read_summary,
    validate_alignment,
    validate_summary,
)


def test_paired_summary_requires_fixed_question_count(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps([{"questions": 249}]))

    with pytest.raises(ValueError, match="expected 250"):
        read_summary(path)


def test_paired_predictions_require_identical_question_order():
    sequential = [
        {"qid": i, "question": f"q{i}", "answers": [f"a{i}"]}
        for i in range(250)
    ]
    qafd = [dict(row) for row in sequential]
    qafd[10]["question"] = "different"

    with pytest.raises(ValueError, match="Question mismatch at QID 10"):
        validate_alignment(sequential, qafd)


def test_paired_predictions_require_identical_selected_context():
    sequential = [
        {
            "qid": i,
            "question": f"q{i}",
            "answers": [f"a{i}"],
            "center_indices": [0, 2],
            "neighbor_index_groups": [[1], [3]],
        }
        for i in range(250)
    ]
    qafd = [dict(row) for row in sequential]
    qafd[12] = {**qafd[12], "center_indices": [0, 4]}

    with pytest.raises(ValueError, match="center_indices mismatch at QID 12"):
        validate_alignment(sequential, qafd)


def test_summary_must_recompute_from_raw_predictions():
    predictions = [
        {"em": 1.0, "f1": 0.5, "seconds": 0.25} for _ in range(250)
    ]
    summary = {"em": 1.0, "f1": 0.4, "avg_seconds": 0.25}

    with pytest.raises(ValueError, match="Summary f1"):
        validate_summary(summary, predictions)
