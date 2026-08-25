from src.eval.csa_results import paired_interval, select_winner
from src.eval.run_csa_grid import unique_grid


def candidate(name, accuracy, seconds, f1=0.0, top_b=2, beta=0.5, pooling="normalized_token_mean"):
    return {
        "method": name,
        "accuracy": accuracy,
        "avg_seconds": seconds,
        "f1": f1,
        "top_b": top_b,
        "beta": beta,
        "pooling": pooling,
    }


def test_unique_grid_deduplicates_b4():
    configurations = unique_grid()
    assert len(configurations) == 16
    assert sum(top_b == 4 for _name, _beta, top_b in configurations) == 1


def test_selection_uses_accuracy_band_then_latency():
    rows = [
        candidate("maximum", 0.60, 3.0),
        candidate("fast-eligible", 0.59, 1.0),
        candidate("too-inaccurate", 0.589, 0.5),
    ]
    assert select_winner(rows)["method"] == "fast-eligible"


def test_paired_interval_uses_aligned_accuracy_differences():
    candidate_rows = [{"qid": qid, "accuracy": 1.0} for qid in range(10)]
    baseline_rows = [{"qid": qid, "accuracy": 0.0} for qid in range(10)]
    observed, low, high = paired_interval(candidate_rows, baseline_rows, samples=100)
    assert (observed, low, high) == (1.0, 1.0, 1.0)
