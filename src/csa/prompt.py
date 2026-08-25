"""Question-first segmented prompts for matched CSA evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class SegmentedPrompt:
    prefix: str
    passages: tuple[str, ...]
    suffix: str

    @property
    def text(self) -> str:
        return self.prefix + "".join(self.passages) + self.suffix

    @property
    def digest(self) -> str:
        return sha256(self.text.encode("utf-8")).hexdigest()


def build_question_first_prompt(question: str, passages: list[dict]) -> SegmentedPrompt:
    prefix = (
        "<|user|>\nYou are an intelligent AI assistant. Answer the question using "
        "only the reference documents. Resolve intermediate entities before choosing "
        "the final answer.\n"
        f"Question: {question}\n\nReference documents:\n"
    )
    blocks = tuple(
        f"\n[Passage {index + 1}]\nTitle: {passage['title']}\n{passage['text']}\n"
        for index, passage in enumerate(passages)
    )
    suffix = (
        "\nReturn only the shortest supported answer phrase with no explanation, "
        "prefixed exactly with 'The answer is:'.\n<|assistant|>\n"
    )
    return SegmentedPrompt(prefix=prefix, passages=blocks, suffix=suffix)
