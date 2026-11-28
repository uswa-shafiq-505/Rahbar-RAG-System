"""Unit tests for eval/coverage.py (Assignment 03 Part E2).

Uses stub embedding functions: zero network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.coverage import evaluate_coverage, main

REQUIRED_TOP_KEYS = {
    "ts",
    "embed_model",
    "n_live",
    "n_golden",
    "threshold",
    "overall_coverage",
    "in_domain_coverage",
    "out_domain_coverage",
    "uncovered",
    "uncovered_topics",
    "per_live_query",
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _vec_embed(text: str) -> list[float]:
    t = text.lower()
    if "attendance" in t:
        return [1.0, 0.0, 0.0]
    if "hostel" in t:
        return [0.0, 1.0, 0.0]
    if "parking" in t:
        return [0.0, 0.0, 1.0]
    if "partial" in t:
        return [0.6, 0.8, 0.0]  # cos with [1, 0, 0] == 0.60
    return [0.0, 0.0, 1.0]


def test_all_live_queries_covered_exits_zero(tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [{"id": "g1", "question": "What is the attendance rule?"}],
    )
    live = _write_jsonl(
        tmp_path / "live.jsonl",
        [
            {"id": "l1", "query": "What is attendance percentage?", "domain": "in"},
            {"id": "l2", "query": "Can attendance be excused?", "domain": "in"},
        ],
    )
    code = main(
        ["--live", str(live), "--golden", str(golden), "--out", str(tmp_path / "out")],
        embed_fn=_vec_embed,
    )
    assert code == 0


def test_half_live_queries_uncovered_exits_one(tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [{"id": "g1", "question": "What is the attendance rule?"}],
    )
    live_rows = [
        {"id": f"l{i}", "query": f"attendance query {i}", "domain": "in"} for i in range(5)
    ] + [
        {"id": f"l{i}", "query": f"hostel query {i}", "domain": "out"} for i in range(5, 10)
    ]
    live = _write_jsonl(tmp_path / "live.jsonl", live_rows)

    report = evaluate_coverage(live, golden, embed_fn=_vec_embed, threshold=0.75)
    assert report["overall_coverage"] == pytest.approx(0.5)
    assert report["in_domain_coverage"] == pytest.approx(1.0)
    assert report["out_domain_coverage"] == pytest.approx(0.0)

    code = main(
        ["--live", str(live), "--golden", str(golden), "--out", str(tmp_path / "out")],
        embed_fn=_vec_embed,
    )
    assert code == 1


def test_threshold_override_changes_verdict(tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [{"id": "g1", "question": "What is the attendance rule?"}],
    )
    live = _write_jsonl(
        tmp_path / "live.jsonl",
        [
            {"id": "l1", "query": "partial overlap rule", "domain": "in"},  # sim == 0.60
        ],
    )

    fail_code = main(
        [
            "--live", str(live),
            "--golden", str(golden),
            "--out", str(tmp_path / "out1"),
            "--threshold", "0.75",
        ],
        embed_fn=_vec_embed,
    )
    pass_code = main(
        [
            "--live", str(live),
            "--golden", str(golden),
            "--out", str(tmp_path / "out2"),
            "--threshold", "0.50",
        ],
        embed_fn=_vec_embed,
    )
    assert fail_code == 1
    assert pass_code == 0


def test_uncovered_queries_grouped_by_topic(tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [{"id": "g1", "question": "What is the attendance rule?"}],
    )
    live = _write_jsonl(
        tmp_path / "live.jsonl",
        [
            {"id": "l1", "query": "What are hostel room fees?", "domain": "out", "topic": "hostel"},
            {"id": "l2", "query": "What are hostel curfew hours?", "domain": "out", "topic": "hostel"},
            {"id": "l3", "query": "Where is student parking lot?", "domain": "out"},
        ],
    )
    report = evaluate_coverage(live, golden, embed_fn=_vec_embed, threshold=0.75)
    topics = {item["topic"]: item["count"] for item in report["uncovered_topics"]}
    assert topics == {"hostel": 2, "where is student": 1}
    assert len(report["uncovered"]) == 3


def test_output_json_has_all_required_keys(tmp_path: Path) -> None:
    golden = _write_jsonl(
        tmp_path / "golden.jsonl",
        [{"id": "g1", "question": "attendance rule"}],
    )
    live = _write_jsonl(
        tmp_path / "live.jsonl",
        [
            {"id": "l1", "query": "attendance requirement", "domain": "in"},
            {"id": "l2", "query": "parking permit cost", "domain": "out"},
        ],
    )
    out_dir = tmp_path / "results"
    main(
        ["--live", str(live), "--golden", str(golden), "--out", str(out_dir), "--no-exit-code"],
        embed_fn=_vec_embed,
    )
    files = list(out_dir.glob("coverage_*.json"))
    assert len(files) == 1
    loaded = json.loads(files[0].read_text(encoding="utf-8"))

    assert REQUIRED_TOP_KEYS.issubset(loaded.keys())
    assert loaded["n_live"] == 2
    assert loaded["n_golden"] == 1
    assert {"id", "query", "best_match_golden_id", "best_sim", "covered"}.issubset(
        loaded["per_live_query"][0].keys()
    )
    assert {"id", "query", "best_sim", "suggested_topic"}.issubset(
        loaded["uncovered"][0].keys()
    )
