"""Indexer configuration, from the same ConfigMap the API reads (section 4.1)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexerSettings:
    embed_model: str        # EMBED_MODEL: must equal the API's value (section 4.4)
    chroma_host: str
    chroma_port: int
    collection: str
    corpus_dir: str         # where the documents and _index.json sit inside the Job's pod
    chunk_size: int
    chunk_overlap: int
    ollama_url: str
    verify_enabled: bool = True
    verify_near_dup_threshold: float = 0.95
    quarantine_dir: str = "/data/quarantine"
    lake_dir: str = "/data/lake"
    verify_report_path: str = "/data/results"


def get_settings() -> IndexerSettings:
    return IndexerSettings(
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chroma_host=os.environ.get("CHROMA_HOST", "chroma"),
        chroma_port=int(os.environ.get("CHROMA_PORT", 8000)),
        collection=os.environ.get("CHROMA_COLLECTION", "rahbar"),
        corpus_dir=os.environ.get("CORPUS_DIR", "/corpus"),
        chunk_size=int(os.environ.get("CHUNK_SIZE", 400)),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", 50)),
        ollama_url=os.environ.get("OLLAMA_URL", "http://host.minikube.internal:11434"),
        verify_enabled=os.environ.get("VERIFY_ENABLED", "true").lower() in {"1", "true", "yes"},
        verify_near_dup_threshold=float(os.environ.get("VERIFY_NEAR_DUP_THRESHOLD", 0.95)),
        quarantine_dir=os.environ.get("QUARANTINE_DIR", "/data/quarantine"),
        lake_dir=os.environ.get("LAKE_DIR", "/data/lake"),
        verify_report_path=os.environ.get("VERIFY_REPORT_PATH", "/data/results"),
    )
