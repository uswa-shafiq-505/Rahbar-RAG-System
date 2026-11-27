"""Trace tool (D2), trace report (D3) and Batch B generator."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import importlib.util

import make_batch_b
import trace_report

_spec = importlib.util.spec_from_file_location(
    "trace_tool", Path(__file__).resolve().parents[1] / "trace.py")
assert _spec and _spec.loader
trace_tool = importlib.util.module_from_spec(_spec)       # scripts/trace.py, loaded by path
_spec.loader.exec_module(trace_tool)


def tid(i: int) -> str:
    return f"{i}" + "f" * 31


def make_trace(i: int, minutes_ago: int, **kw: object) -> dict[str, object]:
    ts = (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat()
    rec: dict[str, object] = {
        "event": "trace", "trace_id": tid(i), "ts": ts, "pod": "api-1",
        "query": f"question {i}", "collection": "rahbar", "embed_model": "nomic-embed-text",
        "index_version": "v1", "cache": "miss", "abstained": False,
        "filters": {"superseded": False},
        "retrieved": [{"chunk_id": f"c{i}", "doc_id": "rules", "doc_version": "2026",
                       "score": 0.6 + i / 100}],
        "context": [{"chunk_id": f"c{i}", "text": "Attendance must be 80 percent."}],
        "prompt": "SYSTEM: use only the context.\nQUESTION: ...", "response": "80 percent",
        "citations": [{"source": "rules", "section": "1"}], "guardrail_actions": [],
        "latency_ms": {"retrieve": 40 + i, "generate": 900 + 10 * i, "total": 1000 + 10 * i},
    }
    rec.update(kw)
    return rec


@pytest.fixture()
def trace_dir(tmp_path: Path) -> Path:
    records = [make_trace(i, minutes_ago=10 * i) for i in range(1, 6)]
    records[1]["cache"] = "hit"
    records[2]["abstained"] = True
    records[3]["guardrail_actions"] = [{"rail": "injection", "action": "removed",
                                        "detail": "instruction in chunk"}]
    records.append(make_trace(9, minutes_ago=60 * 24 * 3))           # outside a 24h window
    (tmp_path / "traces-api-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records[:3]) + "\nnot json\n")
    (tmp_path / "traces-api-2.jsonl").write_text("\n".join(json.dumps(r) for r in records[3:]))
    return tmp_path


def test_trace_tool_prints_context_and_prompt(trace_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert trace_tool.main([tid(1), "--dir", str(trace_dir)]) == 0
    out = capsys.readouterr().out
    assert "Attendance must be 80 percent." in out and "SYSTEM: use only the context." in out
    assert "collection: rahbar" in out and "nomic-embed-text" in out


def test_trace_tool_accepts_an_id_prefix(trace_dir: Path) -> None:
    assert trace_tool.main([tid(2)[:12], "--dir", str(trace_dir)]) == 0


def test_trace_tool_unknown_and_ambiguous_ids(trace_dir: Path, tmp_path: Path) -> None:
    assert trace_tool.main(["deadbeef", "--dir", str(trace_dir)]) == 1
    twin_dir = tmp_path / "twins"
    twin_dir.mkdir()
    (twin_dir / "t.jsonl").write_text("\n".join(json.dumps(make_trace(i, 1, trace_id=f"ab{i}"))
                                               for i in (1, 2)))
    assert trace_tool.main(["ab", "--dir", str(twin_dir)]) == 1          # prefix not unique
    assert trace_tool.main(["ab1", "--dir", str(twin_dir)]) == 0


def test_trace_tool_says_so_when_only_a_hash_was_stored(tmp_path: Path,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    rec = make_trace(7, 1)
    rec.pop("prompt"), rec.pop("context")
    rec["prompt_hash"] = "abc123"
    (tmp_path / "t.jsonl").write_text(json.dumps(rec))
    assert trace_tool.main([tid(7), "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "only the hash was stored" in out and "did not store the context" in out


def test_trace_tool_reads_kubectl_log_lines(monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    import io
    lines = '{"msg": "an ordinary log line"}\n' + json.dumps(make_trace(3, 1)) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(lines))
    assert trace_tool.main([tid(3), "--stdin"]) == 0
    assert "question 3" in capsys.readouterr().out


def test_report_window_and_numbers(trace_dir: Path, capsys: pytest.CaptureFixture[str],
                                   tmp_path: Path) -> None:
    out_json = tmp_path / "r.json"
    assert trace_report.main(["--dir", str(trace_dir), "--since", "24h",
                              "--json", str(out_json)]) == 0
    s = json.loads(out_json.read_text())
    assert s["queries"] == 5                                   # the 3-day-old trace is excluded
    assert s["abstention_rate"] == pytest.approx(1 / 5)
    assert s["cache_hit_rate"] == pytest.approx(1 / 5)         # 1 hit, 4 misses
    assert s["injection_attempts_flagged"] == 1
    assert s["top1_similarity"]["count"] == 5
    assert s["latency_ms"]["total"]["p95"] >= s["latency_ms"]["total"]["p50"]
    assert "cache hit rate" in capsys.readouterr().out


def test_report_on_an_empty_window_does_not_crash(trace_dir: Path) -> None:
    assert trace_report.main(["--dir", str(trace_dir), "--since", "1m"]) == 0


def test_percentile_is_nearest_rank() -> None:
    assert trace_report.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10
    assert trace_report.percentile([], 0.5) is None


def test_batch_b_has_every_planted_problem(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    assert make_batch_b.main(["--base-dir", str(root / "eval" / "fixtures"),
                              "--out", str(tmp_path)]) == 0
    manifest = json.loads((tmp_path / "MANIFEST-batch-b.json").read_text())
    kinds = [p["problem"] for p in manifest["planted"]]
    assert manifest["files"] >= 10
    assert kinds.count("exact duplicate") >= 2
    for needed in ("near-duplicate", "newer version of an existing document",
                   "missing metadata", "injected instruction"):
        assert needed in kinds
    assert any("truncated" in k or "corrupt" in k for k in kinds)
    dups = [p for p in manifest["planted"] if p["problem"] == "exact duplicate"]
    assert len({p["sha256"] for p in dups}) == 2          # two different originals


def test_batch_b_is_deterministic(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        make_batch_b.main(["--base-dir", str(root / "eval" / "fixtures"), "--out", str(d)])
    assert (a / "MANIFEST-batch-b.json").read_text() == (b / "MANIFEST-batch-b.json").read_text()


def test_render_fills_variables_and_refuses_missing_ones() -> None:
    import render
    assert render.render('value: "${COLLECTION}"', {"COLLECTION": "rahbar_v2"}) == 'value: "rahbar_v2"'
    with pytest.raises(KeyError):
        render.render("a: ${NOPE}", {})


def test_every_templated_manifest_renders_and_parses() -> None:
    import render
    import yaml
    env = {"COLLECTION": "c", "CHUNK_SIZE": "300", "CHUNK_OVERLAP": "50", "EMBED_MODEL": "m"}
    root = Path(__file__).resolve().parents[2] / "k8s"
    for name in ("candidate.yaml", "candidate-ingest-job.yaml", "eval-job-candidate.yaml"):
        docs = list(yaml.safe_load_all(render.render((root / name).read_text(), env)))
        assert docs and "${" not in json.dumps(docs)
