"""Stage 5: Store. The only code in the whole system that writes to Chroma."""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from .config import IndexerSettings
from .models import Chunk

log = logging.getLogger(__name__)


def get_client(settings: IndexerSettings) -> Any:
    """Return a Chroma HTTP client configured from settings."""
    return chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )


def store(
    chunks: list[Chunk],
    vectors: list[list[float]],
    chroma_host: str = "chroma",
    chroma_port: int = 8000,
    collection_name: str = "rahbar",
    embed_model: str = "nomic-embed-text",
    embed_dim: int = 768,
    index_version: str = "v1",
) -> int:
    """Write chunks and vectors to Chroma collection with embedding-pin guardrail (Stage 5).

    Returns the number of chunks actually stored.
    """
    # Sanity checks BEFORE any Chroma call
    if len(chunks) != len(vectors):
        raise RuntimeError(
            f"store: chunk count ({len(chunks)}) does not match vector count ({len(vectors)})"
        )
    if not chunks:
        return 0

    # Support invocation with IndexerSettings from run.py or explicit parameters
    if hasattr(chroma_host, "chroma_host") and hasattr(chroma_host, "chroma_port"):
        settings = chroma_host
        host = settings.chroma_host
        port = settings.chroma_port
        col_name = settings.collection  # type: ignore[attr-defined]
        model = settings.embed_model  # type: ignore[attr-defined]
        dim = len(vectors[0]) if vectors else 768
        version = getattr(settings, "index_version", "v1")
    else:
        host = str(chroma_host)
        port = int(chroma_port)
        col_name = str(collection_name)
        model = str(embed_model)
        dim = int(embed_dim)
        version = str(index_version)

    total = len(chunks)
    log.info(
        "store: starting write to collection '%s' at %s:%d (model=%s, dim=%d, chunks=%d)",
        col_name,
        host,
        port,
        model,
        dim,
        total,
    )

    try:
        client = chromadb.HttpClient(host=host, port=port)
        collection = client.get_or_create_collection(name=col_name)
        old = collection.metadata or {}

        # Collection handling & embedding-pin guardrail (§4.4)
        if not old or "embed_model" not in old:
            # On very first store (empty collection), set metadata
            collection.modify(
                metadata={
                    "embed_model": model,
                    "embed_dim": dim,
                    "index_version": version,
                }
            )
        else:
            old_model = old.get("embed_model")
            old_dim = old.get("embed_dim")
            if old_model != model or (old_dim is not None and int(old_dim) != dim):
                raise RuntimeError(
                    f"store: embedding model mismatch. Collection was built with "
                    f"embed_model={old_model} embed_dim={old_dim}, "
                    f"but the indexer is now using embed_model={model} embed_dim={dim}. "
                    f"Delete the collection and re-index, or revert EMBED_MODEL."
                )
            if old.get("index_version") != version:
                updated_meta = dict(old)
                updated_meta["index_version"] = version
                collection.modify(metadata=updated_meta)

        # Upsert in batches of 500
        batch_size = 500
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            b_chunks = chunks[start:end]
            b_vectors = vectors[start:end]
            collection.upsert(
                ids=[c.id for c in b_chunks],
                documents=[c.text for c in b_chunks],
                embeddings=b_vectors,  # type: ignore[arg-type]
                metadatas=[c.metadata() for c in b_chunks],
            )
            done = end
            if done % 500 == 0:
                pct = round(done / total * 100)
                log.info(f"store: {done}/{total} chunks stored ({pct}%)")

        final_count = collection.count()
        if final_count == 0 and total > 0:
            raise RuntimeError(
                f"store: collection '{col_name}' has 0 chunks after storing {total} chunks"
            )

        log.info(f"store: collection '{col_name}' now has {final_count} chunks")
        log.info("store: successfully stored %d chunks (collection count: %d)", total, final_count)
        return total

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"store: failed to write to Chroma at {host}:{port} "
            f"(collection={col_name}): {exc}"
        ) from exc
