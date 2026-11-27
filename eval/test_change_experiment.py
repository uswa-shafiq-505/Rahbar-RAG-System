"""Unit tests for eval/change_experiment.py (Assignment 03 Part C1).

Zero network calls; validates 3-level deltas, gate RELEASE/REJECT verdicts, and JSON shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.change_experiment import DEFAULT_GATE_CONFIG, compare_all_levels, write_results

REQUIRED_TOP_KEYS = {
    "ts",
    "change_type",
    "baseline",
    "candidate",
    "metrics",
    "verdict",
    "gate_result",
}


def _baseline() -> dict[str, dict[str, float]]:
    return {
        "module": {"embedding_gap": 0.30},
        "component": {
            "recall_at_3": 0.80,
            "recall_at_5": 0.85,
            "mrr": 0.78,
            "ndcg_at_5": 0.82,
        },
        "end_to_end": {
            "faithfulness": 0.85,
            "abstention_correctness": 1.0,
            "citation_validity": 0.96,
            "latency_p50": 1500.0,
            "latency_p95": 4200.0,
        },
    }


def test_delta_computed_correctly_across_all_three_levels() -> None:
    base = _baseline()
    cand = {
        "module": {"embedding_gap": 0.34},
        "component": {
            "recall_at_3": 0.85,
            "recall_at_5": 0.90,
            "mrr": 0.81,
            "ndcg_at_5": 0.86,
        },
        "end_to_end": {
            "faithfulness": 0.90,
            "abstention_correctness": 1.0,
            "citation_validity": 0.98,
            "latency_p50": 1200.0,
            "latency_p95": 3800.0,
        },
    }
    report = compare_all_levels("chunk_size", base, cand, gate_config_path=DEFAULT_GATE_CONFIG)

    assert report["metrics"]["module"]["delta"]["embedding_gap"] == pytest.approx(0.04)
    assert report["metrics"]["component"]["delta"]["recall_at_3"] == pytest.approx(0.05)
    assert report["metrics"]["component"]["delta"]["recall_at_5"] == pytest.approx(0.05)
    assert report["metrics"]["component"]["delta"]["mrr"] == pytest.approx(0.03)
    assert report["metrics"]["component"]["delta"]["ndcg_at_5"] == pytest.approx(0.04)
    assert report["metrics"]["end_to_end"]["delta"]["faithfulness"] == pytest.approx(0.05)
    assert report["metrics"]["end_to_end"]["delta"]["latency_p95"] == pytest.approx(-400.0)


def test_verdict_release_when_candidate_passes_gate() -> None:
    base = _baseline()
    cand = {
        "module": {"embedding_gap": 0.31},
        "component": {
            "recall_at_3": 0.80,
            "recall_at_5": 0.84,  # drop 0.01 <= 0.02 max_recall_at_5_drop
            "mrr": 0.79,
            "ndcg_at_5": 0.83,
        },
        "end_to_end": {
            "faithfulness": 0.85,
            "abstention_correctness": 1.0,
            "citation_validity": 0.97,
            "latency_p50": 1400.0,
            "latency_p95": 5000.0,  # 5.0s <= 60.0s
        },
    }
    report = compare_all_levels("chunk_size", base, cand, gate_config_path=DEFAULT_GATE_CONFIG)
    assert report["verdict"] == "RELEASE"
    assert report["gate_result"]["pass"] is True
    assert report["gate_result"]["failures"] == []


def test_verdict_reject_when_candidate_fails_gate() -> None:
    base = _baseline()
    cand = {
        "module": {"embedding_gap": 0.25},
        "component": {
            "recall_at_3": 0.60,
            "recall_at_5": 0.70,  # drop 0.15 > 0.02
            "mrr": 0.55,
            "ndcg_at_5": 0.65,
        },
        "end_to_end": {
            "faithfulness": 0.72,  # < 0.80
            "abstention_correctness": 0.80,  # < baseline 1.0
            "citation_validity": 0.90,  # < 0.95
            "latency_p50": 2000.0,
            "latency_p95": 65000.0,  # 65.0s > 60.0s
        },
    }
    report = compare_all_levels("chunk_size", base, cand, gate_config_path=DEFAULT_GATE_CONFIG)
    assert report["verdict"] == "REJECT"
    assert report["gate_result"]["pass"] is False
    assert len(report["gate_result"]["failures"]) >= 4


def test_output_json_has_all_required_keys(tmp_path: Path) -> None:
    base = _baseline()
    report = compare_all_levels("top_k", base, base, gate_config_path=DEFAULT_GATE_CONFIG)
    out_file = write_results(report, out_dir=tmp_path / "results")
    loaded = json.loads(out_file.read_text(encoding="utf-8"))

    assert REQUIRED_TOP_KEYS.issubset(loaded.keys())
    assert set(loaded["metrics"].keys()) == {"module", "component", "end_to_end"}
    for level in ("module", "component", "end_to_end"):
        assert set(loaded["metrics"][level].keys()) == {"baseline", "candidate", "delta"}
    assert set(loaded["gate_result"].keys()) == {"pass", "failures"}
    assert loaded["verdict"] in {"RELEASE", "REJECT"}
