"""Tests for the platform layer: readiness, the embedding pin and the semantic cache.
No cluster, no model, no network: Chroma and Redis are faked."""

from __future__ import annotations

from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.schemas import AskResponse, Citation, RetrievedRef
from app.services.cache import SemanticCache, never_cache
from app.services.index_state import IndexState


class FakeCollection:
    def __init__(self, n: int, model: str | None, cid: str = "c1") -> None:
        self._n, self.id = n, cid
        self.metadata = {"embed_model": model} if model else {}

    def count(self) -> int:
        return self._n


class FakeChroma:
    def __init__(self, collection: FakeCollection | None, up: bool = True) -> None:
        self.collection, self.up = collection, up

    def heartbeat(self) -> int:
        if not self.up:
            raise ConnectionError("down")
        return 1

    def get_collection(self, name: str) -> FakeCollection:
        if self.collection is None:
            raise ValueError("not found")
        return self.collection


def state_with(chroma: FakeChroma, **kw: Any) -> IndexState:
    settings = Settings(retriever="chroma", embed_model="nomic-embed-text", **kw)
    st = IndexState(settings)
    st._client = chroma  # noqa: SLF001
    return st


# ---- readiness and the embedding pin -------------------------------------------------------
def test_ready_when_collection_is_non_empty_and_pin_matches() -> None:
    ready, checks = state_with(FakeChroma(FakeCollection(10, "nomic-embed-text"))).readiness()
    assert ready and checks["chroma"] == "ok"


@pytest.mark.parametrize("chroma,needle", [
    (FakeChroma(None, up=False), "unreachable"),
    (FakeChroma(None), "does not exist"),
    (FakeChroma(FakeCollection(0, "nomic-embed-text")), "empty"),
    (FakeChroma(FakeCollection(5, "dummy-768")), "mismatch"),
])
def test_not_ready_names_the_failed_dependency(chroma: FakeChroma, needle: str) -> None:
    ready, checks = state_with(chroma).readiness()
    assert not ready and needle in checks["chroma"]


def test_mismatch_message_names_both_models() -> None:
    _, checks = state_with(FakeChroma(FakeCollection(5, "dummy-768"))).readiness()
    assert "dummy-768" in checks["chroma"] and "nomic-embed-text" in checks["chroma"]


def test_unreachable_redis_makes_pod_not_ready() -> None:
    st = state_with(FakeChroma(FakeCollection(5, "nomic-embed-text")), redis_host="nowhere.invalid")
    ready, checks = st.readiness()
    assert not ready and "redis unreachable" in checks["redis"]


def test_readyz_endpoint_returns_503_on_pin_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.setenv("REQUIRE_INDEX", "true")
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr(IndexState, "_chroma",
                        lambda self: FakeChroma(FakeCollection(3, "dummy-768")))
    from app.main import app

    with TestClient(app) as c:
        r = c.get("/readyz")
        assert r.status_code == 503
        assert "mismatch" in r.json()["failed"]["chroma"]
        assert c.get("/healthz").status_code == 200   # liveness must not care


def test_readyz_is_ready_with_stubs_and_no_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.delenv("REQUIRE_INDEX", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    from app.main import app

    with TestClient(app) as c:
        assert c.get("/readyz").status_code == 200


# ---- semantic cache --------------------------------------------------------------------------
def _resp(answered: bool = True, cited: bool = True) -> AskResponse:
    ref = RetrievedRef(chunk_id="a", source="s.pdf", section="1", page=1, score=0.9)
    return AskResponse(
        request_id="r", answer="x", answered_from_context=answered, confidence="high",
        citations=[Citation(source="s.pdf", section="1", page=1)] if cited else [],
        retrieved=[ref], provider="stub")


class FixedState:
    """IndexState stand-in with a fake Redis and a controllable fingerprint."""

    def __init__(self) -> None:
        self.r = fakeredis.FakeRedis(decode_responses=True)
        self.fp: str | None = "idx-1"

    def redis(self) -> Any:
        return self.r

    def fingerprint(self) -> str | None:
        return self.fp


def make_cache(**kw: Any) -> tuple[SemanticCache, FixedState]:
    st = FixedState()
    return SemanticCache(Settings(redis_host="x", cache_threshold=0.9, **kw), st), st  # type: ignore[arg-type]


def test_hit_above_threshold_and_miss_below() -> None:
    cache, _ = make_cache()
    cache.store("q1", [1.0, 0.0], _resp())
    hit, sim = cache.lookup([0.99, 0.141])        # cosine about 0.99
    assert hit is not None and sim >= 0.9
    miss, sim2 = cache.lookup([0.0, 1.0])         # orthogonal question
    assert miss is None and sim2 < 0.9


def test_reindex_invalidates_the_cache() -> None:
    cache, st = make_cache()
    cache.store("q1", [1.0, 0.0], _resp())
    assert cache.lookup([1.0, 0.0])[0] is not None
    st.fp = "idx-2"                                # the indexer Job rebuilt the collection
    assert cache.lookup([1.0, 0.0])[0] is None
    assert st.r.keys("rahbar:cache:*") == []       # old entries were deleted, not just hidden


def test_unknown_fingerprint_bypasses_the_cache() -> None:
    cache, st = make_cache()
    st.fp = None
    cache.store("q1", [1.0, 0.0], _resp())
    assert st.r.keys("*") == []


def test_disabled_without_redis_host() -> None:
    cache = SemanticCache(Settings(), None)  # type: ignore[arg-type]
    assert not cache.enabled


def test_disabled_while_the_dummy_index_is_in_use() -> None:
    cache = SemanticCache(Settings(redis_host="x", embed_model="dummy-768"), None)  # type: ignore[arg-type]
    assert not cache.enabled


@pytest.mark.parametrize("response,follow_up,expected", [
    (_resp(), False, None),
    (_resp(), True, "follow-up"),
    (_resp(answered=False), False, "abstention"),
    (_resp(cited=False), False, "no citations"),
])
def test_never_cache_rules(response: AskResponse, follow_up: bool, expected: str | None) -> None:
    reason = never_cache(response, follow_up)
    assert (reason is None) if expected is None else (expected in (reason or ""))


def test_citation_not_in_retrieved_is_never_cached() -> None:
    bad = _resp()
    bad.citations = [Citation(source="invented.pdf", section="9", page=1)]
    assert never_cache(bad) is not None


def test_ask_sets_x_cache_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/ask", json={"question": "What is the attendance rule?"})
        assert r.headers["X-Cache"] == "BYPASS"
        assert r.json()["cached"] is False


def test_second_identical_question_is_served_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.setattr("app.services.cache_layer.embed_query", lambda q, s: [1.0, 0.0])
    from app.main import app

    with TestClient(app) as c:
        cache, _ = make_cache()
        app.state.cache = cache
        first = c.post("/api/ask", json={"question": "What does the stub say?"})
        second = c.post("/api/ask", json={"question": "What does the stub say?"},
                        headers={"X-Request-ID": "again"})
        assert first.headers["X-Cache"] == "MISS" and first.json()["cached"] is False
        assert second.headers["X-Cache"] == "HIT" and second.json()["cached"] is True
        assert second.json()["request_id"] == "again"
        assert 'rahbar_cache_lookups_total{result="hit"}' in c.get("/metrics").text
