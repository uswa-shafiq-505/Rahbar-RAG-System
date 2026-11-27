"""POST /api/ask. The Ingress sends /api/* here unchanged, so the prefix stays."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Request, Response

from ..config import get_settings
from ..logjson import request_id_var
from ..schemas import AskRequest, AskResponse
from ..services import trace_sink
from ..services.answer import (
    build_trace_record,
    trace_emitted_var,
    trace_id_var,
)
from ..services.cache_layer import answer_with_cache

router = APIRouter(prefix="/api")
log = logging.getLogger("rahbar.ask")


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request, response: Response) -> AskResponse:
    t0 = time.perf_counter()
    request_id = request.headers.get("X-Request-ID") or request_id_var.get() or uuid.uuid4().hex
    trace_id = str(uuid.uuid4())
    tok_tid = trace_id_var.set(trace_id)
    tok_emitted = trace_emitted_var.set(False)
    settings = getattr(request.app.state, "settings", None) or get_settings()
    retriever = getattr(request.app.state, "retriever", None)

    try:
        bypass = request.headers.get("x-cache-bypass", "") in {"1", "true", "yes"}
        result, cache_status = answer_with_cache(body.question, request, bypass=bypass)
        response.headers["X-Cache"] = cache_status

        if cache_status == "HIT" and not trace_emitted_var.get():
            total_ms = (time.perf_counter() - t0) * 1000.0
            try:
                trace_sink.write_trace(
                    build_trace_record(
                        trace_id=trace_id,
                        request_id=request_id,
                        question=body.question,
                        settings=settings,
                        retriever=retriever,
                        retrieved=result.retrieved,
                        filters_applied={},
                        reranked_order=[r.chunk_id for r in result.retrieved],
                        prompt=None,
                        response_text=result.answer,
                        citations=result.citations,
                        latency_ms={
                            "embed": total_ms,
                            "retrieve": 0.0,
                            "rerank": 0.0,
                            "generate": 0.0,
                            "validate": 0.0,
                            "total": total_ms,
                        },
                        cache="hit",
                        abstained=not result.answered_from_context,
                    ),
                    settings,
                )
            except Exception:  # noqa: BLE001
                log.warning("cache-hit trace write failed; continuing", exc_info=True)

        log.info(
            "ask handled",
            extra={"trace_id": trace_id, "request_id": request_id, "cache": cache_status},
        )
        return result
    except Exception as exc:
        if not trace_emitted_var.get():
            total_ms = (time.perf_counter() - t0) * 1000.0
            try:
                trace_sink.write_trace(
                    build_trace_record(
                        trace_id=trace_id,
                        request_id=request_id,
                        question=body.question,
                        settings=settings,
                        retriever=retriever,
                        retrieved=[],
                        filters_applied={},
                        reranked_order=[],
                        prompt=None,
                        response_text="",
                        citations=[],
                        latency_ms={
                            "embed": 0.0,
                            "retrieve": 0.0,
                            "rerank": 0.0,
                            "generate": 0.0,
                            "validate": 0.0,
                            "total": total_ms,
                        },
                        cache="miss",
                        abstained=True,
                        stage="error",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    settings,
                )
            except Exception:  # noqa: BLE001
                log.warning("error trace write failed", exc_info=True)
        raise
    finally:
        trace_id_var.reset(tok_tid)
        trace_emitted_var.reset(tok_emitted)
