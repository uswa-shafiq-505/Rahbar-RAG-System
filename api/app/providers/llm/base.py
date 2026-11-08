"""The LLM provider interface (section 3.5). Swapping providers is one file."""

from __future__ import annotations

from typing import Any, Protocol

from ...schemas import Answer, Passage


class LLMProvider(Protocol):
    name: str

    def generate(
        self,
        prompt_or_question: str,
        passages: list[Passage] | None = None,
        **kwargs: Any,
    ) -> Answer:
        """Answer the question or prompt from the passages, as an `Answer`."""
        ...
