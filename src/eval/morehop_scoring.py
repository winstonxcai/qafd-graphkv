"""Explicit, non-interchangeable MoreHopQA scoring contracts."""

from __future__ import annotations

import re
import string


_SPECIAL_ENDINGS = ("<|im_end|>", "<|eot_id|>")


def normalize_answer(text: str) -> str:
    text = re.sub(r"\b(a|an|the)\b", " ", text.lower())
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def _strip_special_endings(text: str) -> str:
    for marker in _SPECIAL_ENDINGS:
        text = text.split(marker)[0]
    return text


def paper_compat_prediction(generated: str) -> str:
    """Match the public ``rag_eval.py`` control flow.

    When an ``Answer:`` marker exists, the final marker is used. Otherwise the
    complete generated response remains eligible for answer containment.
    """

    matches = list(re.finditer(r"answer:\s*(.*)", generated, re.IGNORECASE))
    if matches:
        generated = matches[-1].group(1)
    return _strip_special_endings(generated)


def strict_final_prediction(generated: str) -> str | None:
    """Return text after the final ``Answer:`` marker, or ``None`` if absent."""

    matches = list(re.finditer(r"answer:\s*(.*)", generated, re.IGNORECASE))
    if not matches:
        return None
    return _strip_special_endings(matches[-1].group(1))


def containment(prediction: str | None, answers: str | list[str]) -> float:
    if prediction is None:
        return 0.0
    candidates = [answers] if isinstance(answers, str) else answers
    normalized_prediction = normalize_answer(prediction)
    return float(
        any(
            normalized_answer and normalized_answer in normalized_prediction
            for normalized_answer in map(normalize_answer, candidates)
        )
    )


def score_both(generated: str, answers: str | list[str]) -> dict[str, float | bool]:
    strict_prediction = strict_final_prediction(generated)
    return {
        "paper_compat_accuracy": containment(
            paper_compat_prediction(generated), answers
        ),
        "strict_final_accuracy": containment(strict_prediction, answers),
        "has_explicit_answer": strict_prediction is not None,
    }
