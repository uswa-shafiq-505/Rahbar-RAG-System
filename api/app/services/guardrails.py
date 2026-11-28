"""Retrieval-stage prompt-injection guardrails (Assignment 03 Part D4)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")

INJECTION_PATTERNS: list[tuple[str, str]] = [
    (
        "instruction override (ignore instructions)",
        r"\bignore\s+(?:(?:previous|above|all|prior)\s+)+instructions?\b",
    ),
    (
        "instruction override (disregard instructions/prompt)",
        r"\bdisregard\b.*\b(?:instructions?|prompt)\b",
    ),
    (
        "role hijacking (you are now / act as)",
        r"\b(?:you are now|act as (?:if|though))\b",
    ),
    (
        "chat-role delimiter injection (system:/assistant:)",
        r"\b(?:system|assistant):\s*",
    ),
    (
        "Llama prompt template marker ([INST] / <<SYS>>)",
        r"\[INST\]|<<SYS>>",
    ),
]

_COMPILED_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (reason, pat, re.compile(pat, re.IGNORECASE)) for reason, pat in INJECTION_PATTERNS
]


def detect_injection(chunk_text: str) -> dict[str, str] | None:
    """Return {reason, matched_pattern} if injection-like text detected, else None."""
    if not chunk_text:
        return None
    for reason, pat, regex in _COMPILED_PATTERNS:
        if regex.search(chunk_text):
            return {"reason": reason, "matched_pattern": pat}
    return None


def _extract_text_and_id(chunk: Any) -> tuple[str, str]:
    if isinstance(chunk, dict):
        text = str(chunk.get("text") if chunk.get("text") is not None else chunk.get("content") or "")
        cid = str(chunk.get("chunk_id") if chunk.get("chunk_id") is not None else chunk.get("id") or "")
        return text, cid
    return str(getattr(chunk, "text", "") or ""), str(getattr(chunk, "chunk_id", "") or "")


def check_retrieved_chunks(
    chunks: Sequence[T],
) -> tuple[list[dict[str, Any]], list[T]]:
    """Flag and exclude retrieved chunks matching prompt-injection patterns.

    Returns (actions, cleaned_chunks).
    """
    actions: list[dict[str, Any]] = []
    cleaned: list[T] = []
    for chunk in chunks:
        text, chunk_id = _extract_text_and_id(chunk)
        match = detect_injection(text)
        if match is not None:
            actions.append(
                {
                    "rail": "retrieval",
                    "action": "flagged_and_excluded",
                    "reason": match["reason"],
                    "chunk_id": chunk_id,
                }
            )
        else:
            cleaned.append(chunk)
    return actions, cleaned
