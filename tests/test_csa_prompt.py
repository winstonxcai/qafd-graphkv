from src.csa.prompt import build_question_first_prompt


def test_question_first_segmented_prompt():
    passages = [
        {"title": f"Title {index}", "text": f"Body {index}"} for index in range(5)
    ]
    prompt = build_question_first_prompt("Who won?", passages)
    assert prompt.text.index("Who won?") < prompt.text.index("[Passage 1]")
    assert prompt.text.index("[Passage 5]") < prompt.text.index("The answer is:")
    assert len(prompt.passages) == 5
    assert len(prompt.digest) == 64
