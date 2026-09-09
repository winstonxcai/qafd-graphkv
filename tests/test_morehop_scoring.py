from src.eval.morehop_scoring import (
    containment,
    paper_compat_prediction,
    score_both,
    strict_final_prediction,
)


def test_paper_compat_falls_back_to_complete_response():
    generated = "The reasoning mentions Paris but has no final marker."
    assert paper_compat_prediction(generated) == generated
    assert score_both(generated, "Paris") == {
        "paper_compat_accuracy": 1.0,
        "strict_final_accuracy": 0.0,
        "has_explicit_answer": False,
    }


def test_both_scorers_use_final_answer_when_present():
    generated = "Answer: London\nMore thought. Answer: Paris<|eot_id|>ignored"
    assert strict_final_prediction(generated) == "Paris"
    assert paper_compat_prediction(generated) == "Paris"
    assert containment("Paris", ["Paris", "City of Paris"]) == 1.0
