"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import logging
import time

import httpx

from .models import Chunk

log = logging.getLogger(__name__)


def embed(
    chunks: list[Chunk],
    model: str,
    ollama_url: str = "",
) -> list[list[float]]:
    """Embed chunks using Ollama HTTP API (Stage 4).

    Endpoint: POST {ollama_url}/api/embeddings
    Request: {"model": model, "prompt": chunk.text}
    Response: {"embedding": [float, ...]}
    """
    if not chunks:
        return []

    # Handle invocation from run.py or callers passing IndexerSettings
    if hasattr(model, "embed_model") and hasattr(model, "ollama_url"):
        actual_model: str = getattr(model, "embed_model")
        actual_url: str = getattr(model, "ollama_url")
    else:
        actual_model = str(model)
        actual_url = str(ollama_url)

    if not actual_url:
        raise RuntimeError("embed: ollama_url must not be empty")

    endpoint = f"{actual_url.rstrip('/')}/api/embeddings"
    total = len(chunks)
    vectors: list[list[float]] = []
    expected_dim: int | None = None

    with httpx.Client(timeout=60.0) as client:
        for idx, item in enumerate(chunks, 1):
            text = item.text if isinstance(item, Chunk) else str(item)
            payload = {"model": actual_model, "prompt": text}
            vec: list[float] | None = None

            for attempt in range(1, 4):
                try:
                    resp = client.post(endpoint, json=payload)
                    # 4xx: client error (e.g. model not found) — do NOT retry
                    if 400 <= resp.status_code < 500:
                        raise RuntimeError(
                            f"embed: Ollama client error {resp.status_code} for model '{actual_model}': {resp.text}"
                        )
                    # 5xx: server error — retry up to 3 attempts
                    if resp.status_code >= 500:
                        if attempt < 3:
                            time.sleep(2)
                            continue
                        raise RuntimeError(
                            f"embed: Ollama server error {resp.status_code} for model '{actual_model}' after 3 attempts: {resp.text}"
                        )
                    if not resp.is_success:
                        raise RuntimeError(
                            f"embed: Ollama returned unexpected status {resp.status_code} for model '{actual_model}': {resp.text}"
                        )

                    try:
                        data = resp.json()
                    except Exception as exc:
                        raise RuntimeError(
                            f"embed: malformed JSON response from Ollama on chunk {idx}: {exc}"
                        ) from exc

                    if (
                        not isinstance(data, dict)
                        or "embedding" not in data
                        or not isinstance(data["embedding"], list)
                    ):
                        raise RuntimeError(
                            f"embed: malformed response from Ollama on chunk {idx}: missing 'embedding' list"
                        )

                    try:
                        vec = [float(x) for x in data["embedding"]]
                    except (ValueError, TypeError) as exc:
                        raise RuntimeError(
                            f"embed: non-numeric values in embedding vector on chunk {idx}: {exc}"
                        ) from exc

                    break
                except (httpx.HTTPError, ConnectionError) as exc:
                    if attempt < 3:
                        time.sleep(2)
                        continue
                    raise RuntimeError(
                        f"embed: Ollama connection error on chunk {idx} after 3 attempts: {exc}"
                    ) from exc

            if vec is None:
                raise RuntimeError(f"embed: failed to get embedding for chunk {idx}")

            # Dimension validation
            if expected_dim is None:
                if "nomic-embed-text" in actual_model:
                    if len(vec) != 768:
                        raise RuntimeError(
                            f"embed: model {actual_model} returned {len(vec)}-dim vectors, expected 768. "
                            f"Check EMBED_MODEL matches the collection metadata."
                        )
                    expected_dim = 768
                else:
                    if len(vec) == 0:
                        raise RuntimeError(
                            f"embed: model '{actual_model}' returned empty embedding vector"
                        )
                    expected_dim = len(vec)
            else:
                if len(vec) != expected_dim:
                    raise RuntimeError(
                        f"embed: vector {idx} dimension {len(vec)} does not match expected {expected_dim}"
                    )

            vectors.append(vec)

            # Log progress every 25 chunks
            if idx % 25 == 0:
                pct = round(idx / total * 100)
                log.info(f"embedding: {idx}/{total} chunks ({pct}%)")

    return vectors
