"""Timing wrappers: retrieval latency, generation latency and token counts (section 4.8)
without touching Member 2's retriever or provider code."""

from __future__ import annotations

import time
from typing import Any

from ..routes import metrics as m
from ..schemas import Answer, Passage


class TimedRetriever:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = inner.name

    def retrieve(self, question: str, k: int) -> list[Passage]:
        start = time.perf_counter()
        try:
            return self._inner.retrieve(question, k)  # type: ignore[no-any-return]
        finally:
            m.RETRIEVAL_SECONDS.observe(time.perf_counter() - start)


class TimedProvider:
    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = inner.name

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        start = time.perf_counter()
        try:
            answer: Answer = self._inner.generate(question, passages)
        finally:
            m.GENERATION_SECONDS.observe(time.perf_counter() - start)
        # Approximation (about 4 characters per token). A provider that knows the real
        # counts can call metrics.TOKENS directly instead.
        m.TOKENS.labels(direction="in").inc(
            (len(question) + sum(len(p.text) for p in passages)) // 4)
        m.TOKENS.labels(direction="out").inc(len(answer.answer) // 4)
        return answer
