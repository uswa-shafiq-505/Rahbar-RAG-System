"""Stage 4: Embed. One vector per chunk, from the pinned model only."""

from __future__ import annotations

import logging
import time

import httpx

from .models import Chunk

log = logging.getLogger(__name__)


def _try_embed_text(client: httpx.Client, endpoint: str, model: str, text: str, idx: int) -> list[float] | None:
    """Embed one text with context-length recovery. Returns None if cannot embed even at min length."""
    MIN_CHARS = 200
    payload = {"model": model, "prompt": text}

    for shrink_attempt in range(4):  # allow 3 halvings
        for attempt in range(1, 4):  # retry transport/5xx errors
            try:
                resp = client.post(endpoint, json=payload)

                # Special: 500 with context-length error → shrink and retry immediately
                if resp.status_code == 500 and "context length" in resp.text.lower():
                    if len(text) <= MIN_CHARS:
                        log.error(
                            "embed: chunk %d cannot fit even at %d chars — giving up", idx, len(text)
                        )
                        return None
                    new_len = max(len(text) // 2, MIN_CHARS)
                    log.warning(
                        "embed: chunk %d context overflow (%d chars), shrinking to %d",
                        idx, len(text), new_len,
                    )
                    text = text[:new_len]
                    payload["prompt"] = text
                    break  # break inner retry loop, go to next shrink_attempt

                if 400 <= resp.status_code < 500:
                    raise RuntimeError(
                        f"embed: Ollama client error {resp.status_code} for model '{model}': {resp.text}"
                    )

                if resp.status_code >= 500:
                    if attempt < 3:
                        time.sleep(2)
                        continue
                    raise RuntimeError(
                        f"embed: Ollama server error {resp.status_code} for model '{model}' "
                        f"after 3 attempts: {resp.text}"
                    )

                if not resp.is_success:
                    raise RuntimeError(
                        f"embed: unexpected status {resp.status_code} for model '{model}': {resp.text}"
                    )

                try:
                    data = resp.json()
                except Exception as exc:
                    raise RuntimeError(f"embed: malformed JSON on chunk {idx}: {exc}") from exc

                if not isinstance(data, dict) or not isinstance(data.get("embedding"), list):
                    raise RuntimeError(
                        f"embed: malformed response on chunk {idx}: missing 'embedding' list"
                    )

                try:
                    return [float(x) for x in data["embedding"]]
                except (ValueError, TypeError) as exc:
                    raise RuntimeError(
                        f"embed: non-numeric values in vector on chunk {idx}: {exc}"
                    ) from exc

            except (httpx.HTTPError, ConnectionError) as exc:
                if attempt < 3:
                    time.sleep(2)
                    continue
                raise RuntimeError(
                    f"embed: Ollama connection error on chunk {idx} after 3 attempts: {exc}"
                ) from exc
        else:
            # inner for loop completed without break and without return — shouldn't happen
            continue

        # if we broke out of inner loop due to shrink, continue outer loop
        continue

    return None


def embed(
    chunks: list[Chunk],
    model: str,
    ollama_url: str = "",
) -> list[list[float]]:
    """Embed chunks using Ollama HTTP API (Stage 4). One vector per chunk."""
    if not chunks:
        return []

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
    MAX_CHARS = 2000  # conservative; context-length recovery handles the rest

    with httpx.Client(timeout=90.0) as client:
        for idx, item in enumerate(chunks, 1):
            text = item.text if isinstance(item, Chunk) else str(item)
            if len(text) > MAX_CHARS:
                log.warning("embed: truncating chunk %d from %d to %d chars", idx, len(text), MAX_CHARS)
                text = text[:MAX_CHARS]

            vec = _try_embed_text(client, endpoint, actual_model, text, idx)
            if vec is None:
                raise RuntimeError(
                    f"embed: chunk {idx} cannot be embedded even after shrinking to min length"
                )

            # Dimension validation
            if expected_dim is None:
                if "nomic-embed-text" in actual_model and len(vec) != 768:
                    raise RuntimeError(
                        f"embed: model {actual_model} returned {len(vec)}-dim vectors, expected 768."
                    )
                expected_dim = len(vec) if "nomic-embed-text" not in actual_model else 768
                if expected_dim == 0:
                    raise RuntimeError(f"embed: model '{actual_model}' returned empty vector")
            elif len(vec) != expected_dim:
                raise RuntimeError(
                    f"embed: vector {idx} dim {len(vec)} != expected {expected_dim}"
                )

            vectors.append(vec)

            if idx % 25 == 0:
                pct = round(idx / total * 100)
                log.info(f"embedding: {idx}/{total} chunks ({pct}%)")

    return vectors