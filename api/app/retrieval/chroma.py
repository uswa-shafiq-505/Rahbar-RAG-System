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

    def retrieve(self, question: str, k: int = 5) -> list[Passage]:
        """Embed question, query Chroma with superseded filter, and return top passages."""
        if k <= 0:
            log.info("retrieve: question=%r k=%d hits=0", question[:60], k)
            return []

        # 1. Embed query vector using pinned model and Ollama URL
        query_vec = embed_query(question, self.settings)

        # 2. Build Chroma query arguments with metadata filter (unless asking for old version)
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vec],
            "n_results": k,
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
        if (
            not res
            or not res.get("ids")
            or not res.get("documents")
            or not res.get("metadatas")
            or not res.get("distances")
            or res["ids"] is None
            or res["documents"] is None
            or res["metadatas"] is None
            or res["distances"] is None
            or not res["ids"][0]
        ):
            log.info("retrieve: question=%r k=%d hits=0", question[:60], k)
            return []

        passages: list[Passage] = []
        ids_list = res["ids"][0]
        docs_list = res["documents"][0]
        metas_list = res["metadatas"][0]
        dists_list = res["distances"][0]

        for i, chunk_id in enumerate(ids_list):
            meta = metas_list[i] if i < len(metas_list) and metas_list[i] else {}
            doc_text = docs_list[i] if i < len(docs_list) and docs_list[i] else ""
            dist = float(dists_list[i]) if i < len(dists_list) and dists_list[i] is not None else 0.0
            log.info("chroma hit %d: dist=%.3f", i, dist)
            score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
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
        log.info("retrieve: question=%r k=%d hits=%d", question[:60], k, len(passages))
        return passages
