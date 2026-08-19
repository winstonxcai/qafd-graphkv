import json
from pathlib import Path

import pytest

from src.eval.log_paired_attempt import read_summary, validate_alignment


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
