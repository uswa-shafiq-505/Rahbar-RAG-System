"""Unit tests for Part D1 trace wiring in answer.py and routes/ask.py.

No network, no real LLM: mocks trace_sink.write_trace and uses stub retriever/provider.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.providers.llm.base import LLMProvider
from app.providers.llm.stub import StubProvider
from app.retrieval.base import Retriever
from app.retrieval.stub import StubRetriever
from app.schemas import Answer, AskResponse, Citation, Passage, RetrievedRef
from app.services import trace_sink
from app.services.answer import answer_question

REQUIRED_TRACE_KEYS = {
    "abstained",
    "cache",
    "citations",
    "collection",
    "embed_model",
    "embed_model_version",
    "filters_applied",
    "guardrail_actions",
    "index_version",
    "latency_ms",
    "model",
    "model_params",
    "prompt_hash",
    "prompt_text",
    "request_id",
    "reranked_order",
    "response",
    "retrieved",
    "rewritten_query",
    "trace_id",
    "ts",
    "user_query",
}


class EmptyRetriever(Retriever):
    name = "empty-stub"

    def retrieve(self, question: str, k: int = 5) -> list[Passage]:
        return []


class FailingProvider(LLMProvider):
    name = "failing-stub"

    def generate(self, question: str, passages: list[Passage]) -> Answer:
        raise RuntimeError("simulated LLM failure")


def test_trace_happy_path_emits_all_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trace_sink,
        "write_trace",
        lambda rec, settings=None: captured.append(rec) or str(rec["trace_id"]),
    )

    resp = answer_question(
        "What does the stub handbook say?",
        retriever=StubRetriever(),
        provider=StubProvider(),
        settings=Settings(),
        request_id="req-happy-1",
        trace_id="trace-happy-1",
    )

    assert resp.answered_from_context is True
    assert len(captured) == 1
    rec = captured[0]
    assert REQUIRED_TRACE_KEYS.issubset(rec.keys())
    assert rec["trace_id"] == "trace-happy-1"
    assert rec["request_id"] == "req-happy-1"
    assert rec["cache"] == "miss"
    assert rec["abstained"] is False
    assert rec["citations"]
    assert set(rec["latency_ms"].keys()) == {"embed", "retrieve", "rerank", "generate", "validate", "total"}
    assert rec["guardrail_actions"] == []


def test_trace_abstention_path_sets_abstained_and_empty_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trace_sink,
        "write_trace",
        lambda rec, settings=None: captured.append(rec) or str(rec["trace_id"]),
    )

    resp = answer_question(
        "What are the hostel timings?",
        retriever=EmptyRetriever(),
        provider=StubProvider(),
        settings=Settings(),
        trace_id="trace-abstain-1",
    )

    assert resp.answered_from_context is False
    assert resp.citations == []
    assert len(captured) == 1
    rec = captured[0]
    assert rec["abstained"] is True
    assert rec["citations"] == []
    assert rec["retrieved"] == []


def test_trace_error_path_emits_stage_error_and_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trace_sink,
        "write_trace",
        lambda rec, settings=None: captured.append(rec) or str(rec["trace_id"]),
    )

    with pytest.raises(RuntimeError, match="simulated LLM failure"):
        answer_question(
            "Trigger an error",
            retriever=StubRetriever(),
            provider=FailingProvider(),
            settings=Settings(),
            trace_id="trace-err-1",
        )

    assert len(captured) == 1
    rec = captured[0]
    assert rec["trace_id"] == "trace-err-1"
    assert rec["stage"] == "error"
    assert "simulated LLM failure" in rec["error"]


def test_trace_cache_hit_path_sets_hit_and_zero_generate_ms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trace_sink,
        "write_trace",
        lambda rec, settings=None: captured.append(rec) or str(rec["trace_id"]),
    )
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.setenv("LLM_PROVIDER", "stub")

    from app.main import app
    from app.services import cache_layer

    cached_resp = AskResponse(
        request_id="orig",
        answer="Cached attendance requirement is 80% [1].",
        citations=[Citation(source="stub-handbook.pdf", section="1.1 About the stub", page=1)],
        answered_from_context=True,
        confidence="high",
        retrieved=[
            RetrievedRef(
                chunk_id="stub-1",
                source="stub-handbook.pdf",
                section="1.1 About the stub",
                page=1,
                score=0.91,
            )
        ],
        provider="stub",
        cached=True,
    )

    class FakeHitCache:
        enabled = True

        def lookup(self, _vec: list[float]) -> tuple[AskResponse, float]:
            return cached_resp.model_copy(), 0.99

    monkeypatch.setattr(cache_layer, "embed_query", lambda _q, _s: [1.0, 0.0, 0.0])

    with TestClient(app) as client:
        app.state.cache = FakeHitCache()
        r = client.post(
            "/api/ask",
            json={"question": "What is the minimum attendance?"},
            headers={"X-Request-ID": "req-cache-1"},
        )

    assert r.status_code == 200
    assert r.headers["X-Cache"] == "HIT"
    assert len(captured) == 1
    rec = captured[0]
    assert rec["request_id"] == "req-cache-1"
    assert rec["cache"] == "hit"
    assert rec["latency_ms"]["generate"] == 0.0
