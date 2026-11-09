"""Embed a question with the pinned model (section 3.2 step 3).

Shared by the semantic cache and (if you want) ChromaRetriever, so a question is embedded
once per request: results are memoised. Ollama's /api/embeddings endpoint is used.
"""

from __future__ import annotations

import math
from functools import lru_cache

import httpx

from ..config import Settings


@lru_cache(maxsize=512)
def _embed(url: str, model: str, text: str) -> tuple[float, ...]:
    r = httpx.post(f"{url}/api/embeddings", json={"model": model, "prompt": text}, timeout=30.0)
    r.raise_for_status()
    vec = r.json()["embedding"]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return tuple(v / norm for v in vec)


def embed_query(text: str, settings: Settings) -> list[float]:
    """Unit-length embedding of `text`, from settings.embed_model."""
    return list(_embed(settings.ollama_url, settings.embed_model, text))
