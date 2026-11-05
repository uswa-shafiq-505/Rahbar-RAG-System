"""Wraps answer_question() with the semantic cache. Kept separate so the answer logic
(abstention, validation, citations) stays in services/answer.py and the cache can be
tested alone. Any cache or embedding failure degrades to "no cache", never to a 500."""

from __future__ import annotations

import logging

from fastapi import Request

from ..schemas import AskResponse
from . import metrics_hooks
from .answer import answer_question
from .cache import never_cache
from .embedding import embed_query

log = logging.getLogger("rahbar.cache_layer")


def answer_with_cache(question: str, request: Request, follow_up: bool = False) -> tuple[AskResponse, str]:
    """Return (response, "HIT" | "MISS" | "BYPASS")."""
    st = request.app.state
    cache = st.cache
    vec: list[float] | None = None
    if cache.enabled and not follow_up:
        try:
            vec = embed_query(question, st.settings)
            hit, sim = cache.lookup(vec)
            if hit is not None:
                metrics_hooks.cache_result("hit")
                hit.cached = True
                from ..logjson import request_id_var

                hit.request_id = request_id_var.get()
                return hit, "HIT"
            metrics_hooks.cache_result("miss")
        except Exception:  # noqa: BLE001
            log.warning("cache lookup failed; continuing without cache", exc_info=True)
            vec = None
    response = answer_question(question, st.retriever, st.provider, st.settings)
    if not response.answered_from_context:
        metrics_hooks.abstained()
    if vec is not None:
        reason = never_cache(response, follow_up)
        if reason is None:
            try:
                cache.store(question, vec, response)
            except Exception:  # noqa: BLE001
                log.warning("cache store failed", exc_info=True)
        else:
            log.info("not cached", extra={"reason": reason})
    return response, "MISS" if vec is not None else "BYPASS"
