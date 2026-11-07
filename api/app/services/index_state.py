"""What state is the vector index in? Used by /readyz and by the cache (sections 4.3, 4.4).

This module only READS Chroma and Redis. The indexer Job is the only writer.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..config import Settings

log = logging.getLogger("rahbar.index_state")

OK = "ok"


class IndexState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any = None
        self._redis: Any = None
        self._last_summary = ""
        self._fp: tuple[float, str | None] = (0.0, None)

    # -- connections (created lazily, so importing this module needs no cluster) ----------
    def _chroma(self) -> Any:
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.HttpClient(
                host=self.settings.chroma_host, port=self.settings.chroma_port,
                settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    def redis(self) -> Any:
        if self._redis is None and self.settings.redis_host:
            import redis

            self._redis = redis.Redis(
                host=self.settings.redis_host, port=self.settings.redis_port,
                socket_connect_timeout=1, socket_timeout=1, decode_responses=True)
        return self._redis

    @property
    def index_required(self) -> bool:
        return self.settings.retriever == "chroma" or self.settings.require_index

    # -- readiness -------------------------------------------------------------------------
    def check_chroma(self) -> str:
        """Reachable, collection exists, non-empty, embedding model matches the pin."""
        name = self.settings.collection
        try:
            self._chroma().heartbeat()
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            return f"chroma unreachable at {self.settings.chroma_host}:" \
                   f"{self.settings.chroma_port} ({type(exc).__name__})"
        try:
            collection = self._chroma().get_collection(name)
        except Exception:  # noqa: BLE001
            return f"collection '{name}' does not exist yet (has the indexer Job run?)"
        if collection.count() == 0:
            return f"collection '{name}' is empty (indexer Job not finished?)"
        stored = (collection.metadata or {}).get("embed_model")
        if stored != self.settings.embed_model:
            # Section 4.4: refuse to serve rather than return silent nonsense.
            return (f"embedding model mismatch: index was built with '{stored}' but this "
                    f"API has EMBED_MODEL='{self.settings.embed_model}'. Re-run the indexer "
                    f"or fix the ConfigMap.")
        return OK

    def check_redis(self) -> str:
        if not self.settings.redis_host:
            return "skipped (cache disabled: REDIS_HOST empty)"
        try:
            self.redis().ping()
        except Exception as exc:  # noqa: BLE001
            return f"redis unreachable at {self.settings.redis_host}:" \
                   f"{self.settings.redis_port} ({type(exc).__name__})"
        return OK

    def readiness(self) -> tuple[bool, dict[str, str]]:
        checks = {
            "chroma": self.check_chroma() if self.index_required
            else "skipped (RETRIEVER=stub, REQUIRE_INDEX=false)",
            "redis": self.check_redis(),
        }
        ready = all(v == OK or v.startswith("skipped") for v in checks.values())
        # Log only when the verdict changes, so a probe every 10 s does not flood stdout.
        summary = "|".join(f"{k}={v}" for k, v in checks.items())
        if summary != self._last_summary:
            self._last_summary = summary
            level = logging.INFO if ready else logging.ERROR
            log.log(level, "readiness changed", extra={"ready": ready, "checks": checks})
        return ready, checks

    # -- cache namespace ---------------------------------------------------------------------
    def fingerprint(self) -> str | None:
        """Identifies the CURRENT index build. Changes whenever the indexer re-builds it
        (new collection id, new chunk count or a new index_version in the metadata).
        None means "unknown", and the cache is bypassed. Memoised for 15 seconds."""
        now = time.monotonic()
        if now - self._fp[0] < 15:
            return self._fp[1]
        fp: str | None
        try:
            collection = self._chroma().get_collection(self.settings.collection)
            version = (collection.metadata or {}).get("index_version", "")
            fp = f"{collection.id}-{collection.count()}-{version}"
        except Exception:  # noqa: BLE001
            fp = None
        self._fp = (now, fp)
        return fp
