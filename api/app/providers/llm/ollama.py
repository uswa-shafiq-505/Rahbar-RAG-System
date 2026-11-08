"""Ollama provider: ollama-host or ollama-incluster (section 3.5)."""

from __future__ import annotations

import json
import logging
import re
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


def _coerce_answer_dict(raw: dict[str, Any], retrieved_passages: list[Passage] | None = None) -> dict[str, Any]:
    passages = retrieved_passages or []
    coerced: dict[str, Any] = {}

    # 1. Missing reasoning -> set to ""
    coerced["reasoning"] = str(raw.get("reasoning") or "")

    # 2. Missing answer -> set to "" (then treat as abstention)
    raw_answer = raw.get("answer")
    answer_str = str(raw_answer) if raw_answer is not None else ""
    coerced["answer"] = answer_str

    # 3. citations coercion
    raw_citations = raw.get("citations")
    coerced_citations: list[dict[str, Any]] = []

    if isinstance(raw_citations, list):
        for c in raw_citations:
            if isinstance(c, int):
                # 1-indexed rank
                rank = c
                if 1 <= rank <= len(passages):
                    p = passages[rank - 1]
                    coerced_citations.append({
                        "source": p.source,
                        "section": p.section,
                        "page": p.page,
                    })
            elif isinstance(c, str):
                parts = [part.strip() for part in c.split("|")]
                src = parts[0] if len(parts) > 0 else ""
                sec = parts[1] if len(parts) > 1 else ""
                page_val: int | None = None
                if len(parts) > 2:
                    m = re.search(r"\d+", parts[2])
                    if m:
                        page_val = int(m.group(0))
                coerced_citations.append({
                    "source": src,
                    "section": sec,
                    "page": page_val,
                })
            elif isinstance(c, dict):
                src = str(c.get("source") or "")
                sec = str(c.get("section") or "")
                raw_p = c.get("page")
                page_val = None
                if raw_p is not None:
                    try:
                        page_val = int(raw_p)
                    except (ValueError, TypeError):
                        page_val = None
                coerced_citations.append({
                    "source": src,
                    "section": sec,
                    "page": page_val,
                })

    coerced["citations"] = coerced_citations

    # 4. confidence is float/number
    raw_conf = raw.get("confidence")
    if isinstance(raw_conf, (int, float)):
        if raw_conf >= 0.75:
            coerced["confidence"] = "high"
        elif raw_conf >= 0.4:
            coerced["confidence"] = "medium"
        else:
            coerced["confidence"] = "low"
    elif isinstance(raw_conf, str):
        c_str = raw_conf.strip().lower()
        if c_str in ("high", "medium", "low"):
            coerced["confidence"] = c_str
        else:
            try:
                num = float(c_str)
                if num >= 0.75:
                    coerced["confidence"] = "high"
                elif num >= 0.4:
                    coerced["confidence"] = "medium"
                else:
                    coerced["confidence"] = "low"
            except ValueError:
                coerced["confidence"] = "low"
    else:
        coerced["confidence"] = "low"

    # 5. answered_from_context
    raw_afc = raw.get("answered_from_context")
    if "could not find" in answer_str.lower():
        coerced["answered_from_context"] = False
    elif not answer_str.strip():
        coerced["answered_from_context"] = False
    elif raw_afc is not None:
        coerced["answered_from_context"] = bool(raw_afc)
    else:
        coerced["answered_from_context"] = True

    return coerced


class OllamaProvider(LLMProvider):
    name: str = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        prompt: str,
        temperature: float | list[Passage] | None = None,
        passages: list[Passage] | None = None,
    ) -> Answer:
        """Call Ollama /api/generate with JSON format, validate Answer, and retry once on failure."""
        try:
            retrieved_passages: list[Passage] = []
            if isinstance(temperature, list):
                temp = float(self.settings.temperature)
                retrieved_passages = temperature
                prompt_text = _build_prompt(prompt, temperature)
            elif temperature is not None:
                temp = float(temperature)
                prompt_text = prompt
                if passages:
                    retrieved_passages = passages
            else:
                temp = float(self.settings.temperature)
                prompt_text = prompt
                if passages:
                    retrieved_passages = passages

            if not retrieved_passages and prompt_text:
                for m in re.finditer(r"\[(\d+)\]\s*source=([^|\n]+?)\s*\|\s*section=([^|\n]+?)\s*\|\s*page=([^\n]+)", prompt_text):
                    pg_str = m.group(4).strip()
                    pg_num = int(pg_str) if pg_str.isdigit() else None
                    retrieved_passages.append(
                        Passage(
                            chunk_id=f"p{m.group(1)}",
                            text="",
                            source=m.group(2).strip(),
                            section=m.group(3).strip(),
                            page=pg_num,
                            score=1.0,
                        )
                    )

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
                    if isinstance(parsed, dict):
                        coerced = _coerce_answer_dict(parsed, retrieved_passages)
                        return Answer.model_validate(coerced)
                    raise ValueError("Ollama response JSON is not an object")

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
