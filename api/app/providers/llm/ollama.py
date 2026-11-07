"""Ollama provider: ollama-host or ollama-incluster (section 3.5)."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ...config import Settings
from ...schemas import Answer, Passage
from .base import LLMProvider

log = logging.getLogger(__name__)


def _canned_answer(reason: str) -> Answer:
    return Answer(
        reasoning=reason,
        answer="The model could not produce a valid answer. Please try rephrasing your question.",
        citations=[],
        answered_from_context=False,
        confidence="low",
    )


def _build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        blocks.append(f"[{i}] {p.source} | {p.section} | page {p.page}\n{p.text}")
    evidence = "\n\n".join(blocks)
    return (
        "Answer the user question using ONLY the provided evidence passages. "
        "Output JSON matching the Answer schema with fields: reasoning, answer, citations, answered_from_context, confidence.\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Question: {question}"
    )


class OllamaProvider(LLMProvider):
    name: str = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        prompt: str,
        temperature: float | list[Passage] | None = None,
    ) -> Answer:
        """Call Ollama /api/generate with JSON format, validate Answer, and retry once on failure."""
        try:
            if isinstance(temperature, list):
                temp = float(self.settings.temperature)
                prompt_text = _build_prompt(prompt, temperature)
            elif temperature is not None:
                temp = float(temperature)
                prompt_text = prompt
            else:
                temp = float(self.settings.temperature)
                prompt_text = prompt

            url = f"{self.settings.ollama_url.rstrip('/')}/api/generate"
            payload: dict[str, Any] = {
                "model": self.settings.llm_model,
                "prompt": prompt_text,
                "stream": False,
                "format": "json",
                "options": {"temperature": temp},
            }

            for attempt in range(1, 3):
                try:
                    with httpx.Client(timeout=120.0) as client:
                        resp = client.post(url, json=payload)

                    # 4xx (e.g. model not found) -> no retry, canned answer
                    if 400 <= resp.status_code < 500:
                        err_text = resp.text.strip() or f"status {resp.status_code}"
                        log.warning("ollama 4xx client error: %s", err_text)
                        return _canned_answer(f"model unavailable: {err_text}")

                    # 5xx server error -> retry once
                    if resp.status_code >= 500:
                        log.warning("ollama 5xx server error (attempt %d/2): %d", attempt, resp.status_code)
                        if attempt == 1:
                            continue
                        return _canned_answer("Model failed to produce valid JSON twice.")

                    data = resp.json()
                    raw_response = data.get("response")
                    if not isinstance(raw_response, str):
                        log.warning("ollama response missing string 'response' (attempt %d/2)", attempt)
                        if attempt == 1:
                            continue
                        return _canned_answer("Model failed to produce valid JSON twice.")

                    parsed = json.loads(raw_response)
                    return Answer.model_validate(parsed)

                except (httpx.TimeoutException, httpx.RequestError) as exc:
                    log.warning("ollama request error (attempt %d/2): %s", attempt, exc)
                    if attempt == 1:
                        continue
                    return _canned_answer(f"Model request failed: {exc}")
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    log.warning("ollama output validation error (attempt %d/2): %s", attempt, exc)
                    if attempt == 1:
                        continue
                    return _canned_answer("Model failed to produce valid JSON twice.")

            return _canned_answer("Model failed to produce valid JSON twice.")

        except Exception as exc:
            log.error("unexpected error in OllamaProvider.generate: %s", exc)
            return _canned_answer(f"Unexpected error: {exc}")
