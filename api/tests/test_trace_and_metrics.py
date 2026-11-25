"""Tests for the RAGOps additions: trace sink, extra metrics, cache bypass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.services import trace_sink


def test_write_trace_goes_to_stdout_and_a_per_pod_file(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    settings = Settings(trace_dir=str(tmp_path), pod_name="api-abc", collection="rahbar")
    trace_id = trace_sink.write_trace({"query": "attendance?", "cache": "miss"}, settings)
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["event"] == "trace" and out["trace_id"] == trace_id and out["pod"] == "api-abc"
    path = tmp_path / "traces-api-abc.jsonl"
    saved = json.loads(path.read_text().strip())
    assert saved["trace_id"] == trace_id and saved["query"] == "attendance?"


def test_two_pods_write_two_files(tmp_path: Path) -> None:
    for pod in ("api-1", "api-2"):
        trace_sink.write_trace({"query": "q"}, Settings(trace_dir=str(tmp_path), pod_name=pod))
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == [
        "traces-api-1.jsonl", "traces-api-2.jsonl"]


def test_unwritable_trace_dir_never_raises(capsys: pytest.CaptureFixture[str]) -> None:
    blocker = Path("/proc/definitely/not/writable")
    trace_sink.write_trace({"query": "q"}, Settings(trace_dir=str(blocker), pod_name="x"))
    assert '"event": "trace"' in capsys.readouterr().out


def test_no_trace_dir_means_stdout_only(tmp_path: Path) -> None:
    trace_sink.write_trace({"query": "q"}, Settings(trace_dir="", pod_name="x"))
    assert list(tmp_path.iterdir()) == []


def test_base_fields_name_collection_and_model() -> None:
    f = trace_sink.base_fields(Settings(collection="rahbar_v2", embed_model="nomic-embed-text"))
    assert f["collection"] == "rahbar_v2" and f["embed_model"] == "nomic-embed-text"


def test_top1_similarity_and_injection_metrics_are_exposed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    from app.main import app
    from app.services import metrics_hooks

    with TestClient(app) as c:
        c.post("/api/ask", json={"question": "What is the attendance rule?"})
        metrics_hooks.injection_flagged()
        text = c.get("/metrics").text
    assert "rahbar_top1_similarity_bucket" in text
    assert "rahbar_injection_attempts_total" in text


def test_cache_bypass_header_skips_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVER", "stub")
    monkeypatch.delenv("REDIS_HOST", raising=False)
    from app.main import app

    with TestClient(app) as c:
        r = c.post("/api/ask", json={"question": "What is the attendance rule?"},
                   headers={"X-Cache-Bypass": "1"})
        assert r.headers["X-Cache"] == "BYPASS"
