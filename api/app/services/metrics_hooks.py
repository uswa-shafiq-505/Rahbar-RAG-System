"""Tiny helpers so services do not import FastAPI routes directly."""

from __future__ import annotations

from ..routes import metrics as m


def cache_result(result: str) -> None:
    m.CACHE_LOOKUPS.labels(result=result).inc()
    hits = m.CACHE_LOOKUPS.labels(result="hit")._value.get()  # noqa: SLF001
    misses = m.CACHE_LOOKUPS.labels(result="miss")._value.get()  # noqa: SLF001
    m.CACHE_HIT_RATIO.set(hits / (hits + misses) if hits + misses else 0.0)


def abstained() -> None:
    m.ABSTENTIONS.inc()


def fallback() -> None:
    """Member 2: call this from the validation-failure fallback path."""
    m.FALLBACKS.inc()
