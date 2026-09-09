import json

from src.eval.morehop_controlled import proportional_allocation, select_questions


def test_proportional_allocation_sums_to_requested_total():
    allocation = proportional_allocation({1: 444, 2: 416, 3: 154, 4: 13, 5: 91}, 100)
    assert allocation == {1: 40, 2: 37, 3: 14, 4: 1, 5: 8}


def test_selection_is_deterministic_and_preserves_dataset_order():
    rows = [
        {"_id": f"q{index}", "no_of_hops": 1 if index < 6 else 2}
        for index in range(10)
    ]
    first = select_questions(rows, 5, "seed")
    second = select_questions(rows, 5, "seed")
    assert first == second
    selected_indices = [int(row["_id"][1:]) for row in first]
    assert selected_indices == sorted(selected_indices)
    assert len(json.dumps(first)) > 0
