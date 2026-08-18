import json
from pathlib import Path

import pytest

from src.eval.log_attempt import existing_attempts, read_summary


def test_read_summary_requires_250_questions(tmp_path: Path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps([{"questions": 50}]))

    with pytest.raises(ValueError, match="expected 250"):
        read_summary(summary)


def test_existing_attempts_reads_integer_ids(tmp_path: Path):
    ledger = tmp_path / "results.csv"
    ledger.write_text(
        "current_time,attempt_number,strategy_name,short_strategy_description,EM,F1,avg_latency\n"
        "2026-08-18T00:00:00+08:00,7,test,description,0.1,0.2,1.0\n"
    )

    assert existing_attempts(ledger) == {7}
