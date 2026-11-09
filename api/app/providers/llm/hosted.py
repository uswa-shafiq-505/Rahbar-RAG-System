"""Hosted free-tier provider (Groq, Google AI Studio, OpenRouter).

The key arrives as LLM_API_KEY from a Kubernetes Secret created on the command
line. It never appears in a file you commit, and never in a log line.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ...config import Settings
from ...schemas import Answer, Passage
from .base import LLMProvider
from .ollama import _canned_answer, _coerce_answer_dict

log = logging.getLogger(__name__)


class HostedProvider(LLMProvider):
    name: str = "hosted"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.hosted_base_url = settings.hosted_base_url
        self.llm_api_key = settings.llm_api_key
        self.llm_model = settings.llm_model

    def generate(
        self,
        prompt_or_question: str,
        passages: list[Passage] | None = None,
        **kwargs: Any,
    ) -> Answer:
        """Call OpenAI-compatible hosted chat completion endpoint and return Answer."""
        if not self.llm_api_key:
            return _canned_answer("LLM_API_KEY not set")

        prompt = prompt_or_question
        retrieved_passages = passages or []
        url = f"{self.hosted_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(1, 3):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, headers=headers, json=body)
                if resp.status_code >= 400:
                    log.warning(
                        "hosted provider HTTP %d on attempt %d: %s",
                        resp.status_code,
                        attempt,
                        resp.text,
                    )
                    if attempt == 1:
                        continue
                    return _canned_answer(f"Hosted provider HTTP error {resp.status_code}")

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    coerced = _coerce_answer_dict(parsed, retrieved_passages)
                    return Answer.model_validate(coerced)
                if attempt == 1:
                    continue
                return _canned_answer("Hosted provider output was not a JSON object")
            except Exception as exc:
                log.warning("hosted provider error on attempt %d: %s", attempt, exc)
                if attempt == 1:
                    continue
                return _canned_answer(f"Hosted provider failure: {exc}")

        return _canned_answer("Hosted provider failed after retry")
