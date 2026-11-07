"""Semantic cache on Redis (section 3.3, ADR 0002).

A hit = cosine similarity >= CACHE_THRESHOLD (from the ConfigMap) between the new
question's embedding and a cached question's embedding. Vectors are unit length, so cosine
is a dot product.

Rules, all enforced here:
  * never cache follow-ups, abstentions, fallbacks or anything that is not a clean answer
    (a cached mistake is served to everyone);
  * invalidate on re-index: every key is namespaced by the index fingerprint, and keys of
    older fingerprints are deleted as soon as a pod notices the index changed.

Storage: one Redis LIST per index fingerprint, newest last, trimmed to CACHE_MAX_ENTRIES and
expiring after CACHE_TTL_SECONDS. No Redis modules needed (plain redis image).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..config import Settings
from ..schemas import AskResponse
from .index_state import IndexState

log = logging.getLogger("rahbar.cache")

PREFIX = "rahbar:cache:"


def never_cache(response: AskResponse, follow_up: bool = False) -> str | None:
    """Return the reason a response must NOT be cached, or None if it is safe to cache."""
    if follow_up:
        return "follow-up question"
    if not response.answered_from_context:
        return "abstention or fallback"
    if not response.citations:
        return "no citations"
    cited = {(c.source, c.section) for c in response.citations}
    retrieved = {(r.source, r.section) for r in response.retrieved}
    if not cited <= retrieved:
        return "citation does not match a retrieved passage"
    return None


class SemanticCache:
    def __init__(self, settings: Settings, state: IndexState) -> None:
        self.settings = settings
        self.state = state
        self._seen_fp: str | None = None

    @property
    def enabled(self) -> bool:
        # The dummy index (week 1) has no real embedding model, so there is nothing to embed
        # questions with: the cache stays off until EMBED_MODEL is the real model.
        return bool(self.settings.redis_host) and not self.settings.embed_model.startswith("dummy")

    def _key(self, fp: str) -> str:
        return f"{PREFIX}{fp}"

    def _invalidate_stale(self, r: Any, fp: str) -> None:
        """Re-index detected: drop every cache list that belongs to an older index."""
        if fp == self._seen_fp:
            return
        stale = [k for k in r.scan_iter(f"{PREFIX}*") if k != self._key(fp)]
        if stale:
            r.delete(*stale)
            log.info("cache invalidated after re-index",
                     extra={"deleted_keys": len(stale), "new_fingerprint": fp})
        self._seen_fp = fp

    def lookup(self, vec: list[float]) -> tuple[AskResponse | None, float]:
        """Return (cached response or None, best similarity seen)."""
        fp = self.state.fingerprint()
        r = self.state.redis()
        if not self.enabled or fp is None or r is None:
            return None, 0.0
        self._invalidate_stale(r, fp)
        best, best_sim = None, 0.0
        for raw in r.lrange(self._key(fp), 0, -1):
            entry = json.loads(raw)
            sim = sum(a * b for a, b in zip(vec, entry["v"], strict=False))
            if sim > best_sim:
                best, best_sim = entry, sim
        if best is not None and best_sim >= self.settings.cache_threshold:
            log.info("cache hit", extra={"similarity": round(best_sim, 4),
                                         "cached_question": best["q"]})
            return AskResponse(**best["r"]), best_sim
        return None, best_sim

    def store(self, question: str, vec: list[float], response: AskResponse) -> None:
        fp = self.state.fingerprint()
        r = self.state.redis()
        if not self.enabled or fp is None or r is None:
            return
        entry = {"q": question, "t": int(time.time()),
                 "v": [round(x, 5) for x in vec], "r": response.model_dump(mode="json")}
        key = self._key(fp)
        pipe = r.pipeline()
        pipe.rpush(key, json.dumps(entry))
        pipe.ltrim(key, -self.settings.cache_max_entries, -1)
        pipe.expire(key, self.settings.cache_ttl_seconds)
        pipe.execute()
