"""Chroma-backed retriever (sections 3.2 and 4.4)."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..config import Settings
from ..schemas import Passage
from ..services.embedding import embed_query
from .base import Retriever

log = logging.getLogger(__name__)

# Alias for schema reference as named in retrieval contract
RetrievedPassage = Passage


def _asks_about_old_version(question: str) -> bool:
    """Return True if question explicitly refers to an older / superseded version."""
    q = question.lower()
    if "before 2026" in q:
        return True
    pattern = r"\b(201\d|202[0-5]|old|older|previous|previously|earlier|superseded|prior|past)\b"
    return bool(re.search(pattern, q))


class ChromaRetriever(Retriever):
    name: str = "chroma"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name: str = settings.collection

    def retrieve(self, question: str, top_k: int = 5, k: int | None = None) -> list[RetrievedPassage]:
        """Embed question, query Chroma with superseded filter, and return top passages."""
        if k is not None:
            top_k = k

        if top_k <= 0:
            log.info("retrieve: question=%r top_k=%d hits=0", question[:60], top_k)
            return []

        # 1. Embed query vector using pinned model and Ollama URL
        query_vec = embed_query(question, self.settings)

        # 2. Build Chroma query arguments with metadata filter (unless asking for old version)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": top_k,
        }
        if not _asks_about_old_version(question):
            query_kwargs["where"] = {"superseded": False}

        # 3. Query Chroma using a short-lived client connection
        try:
            import chromadb

            client = chromadb.HttpClient(
                host=self.settings.chroma_host,
                port=self.settings.chroma_port,
            )
            collection = client.get_collection(name=self.collection_name)
            res = collection.query(**query_kwargs)
        except Exception as exc:
            raise RuntimeError(f"retrieve: chroma query failed: {exc}") from exc

        # 4. Handle empty results
        if not res or not res.get("ids") or not res["ids"][0]:
            log.info("retrieve: question=%r top_k=%d hits=0", question[:60], top_k)
            return []

        passages: list[RetrievedPassage] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0] if res.get("documents") else []
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        dists = res.get("distances", [[]])[0] if res.get("distances") else []

        for i, chunk_id in enumerate(ids):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            doc_text = docs[i] if i < len(docs) and docs[i] else ""
            dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 0.0
            log.info("chroma hit %d: dist=%.3f", i, dist)
            score = max(0.0, min(1.0, 1.0 - (dist / 500.0)))
            page_raw = meta.get("page")
            page_val: int | None = int(page_raw) if page_raw is not None else None

            passages.append(
                Passage(
                    chunk_id=str(chunk_id),
                    text=str(doc_text),
                    source=str(meta.get("source", "")),
                    section=str(meta.get("section", "")),
                    page=page_val,
                    score=score,
                )
            )

        passages.sort(key=lambda p: p.score, reverse=True)
        log.info("retrieve: question=%r top_k=%d hits=%d", question[:60], top_k, len(passages))
        return passages
