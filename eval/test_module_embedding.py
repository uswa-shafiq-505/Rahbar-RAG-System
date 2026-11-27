"""Unit tests for eval/module_embedding.py (Assignment 03 Part B2).

Runs completely offline using stub embedding functions and temporary output dirs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from eval.module_embedding import (
    DEFAULT_PAIRS_PATH,
    evaluate_embeddings,
    stub_embed_384,
    write_results,
)

REQUIRED_TOP_KEYS = {
    "ts",
    "embed_model",
    "paraphrase",
    "unrelated",
    "gap",
    "embed_ms",
    "determinism_max_abs_diff",
    "determinism_pass",
}


def test_paraphrase_sim_exceeds_unrelated_sim() -> None:
    report = evaluate_embeddings(
        pairs_path=DEFAULT_PAIRS_PATH,
        embed_fn=stub_embed_384,
        embed_model="stub-384",
        n_timing=5,
        check_subprocess=False,
    )
    assert len(report["paraphrase"]["pairs"]) == 15
    assert len(report["unrelated"]["pairs"]) == 15
    assert report["paraphrase"]["mean"] > report["unrelated"]["mean"]


def test_gap_computed_correctly() -> None:
    report = evaluate_embeddings(
        pairs_path=DEFAULT_PAIRS_PATH,
        embed_fn=stub_embed_384,
        embed_model="stub-384",
        n_timing=5,
        check_subprocess=False,
    )
    expected_gap = round(report["paraphrase"]["mean"] - report["unrelated"]["mean"], 6)
    assert report["gap"] == pytest.approx(expected_gap, abs=1e-9)
    assert report["gap"] > 0.0


def test_determinism_detects_nondeterministic_stub() -> None:
    calls = 0

    def flaky_embed(text: str) -> list[float]:
        nonlocal calls
        calls += 1
        base = stub_embed_384(text)
        # Perturb first coordinate on every call so repeat embeds differ by > 1e-6.
        base[0] = base[0] + (0.05 * calls)
        return base

    report = evaluate_embeddings(
        pairs_path=DEFAULT_PAIRS_PATH,
        embed_fn=flaky_embed,
        embed_model="flaky-stub",
        n_timing=5,
        check_subprocess=False,
    )
    assert report["determinism_max_abs_diff"] > 1e-6
    assert report["determinism_pass"] is False


def test_timestamped_filename_format_is_valid(tmp_path: Path) -> None:
    report = evaluate_embeddings(
        pairs_path=DEFAULT_PAIRS_PATH,
        embed_fn=stub_embed_384,
        embed_model="stub-384",
        n_timing=5,
        check_subprocess=False,
    )
    out_file = write_results(report, out_dir=tmp_path)
    assert out_file.exists()
    assert re.fullmatch(r"module_embedding_\d{8}T\d{6}Z\.json", out_file.name) is not None


def test_output_json_has_all_required_keys(tmp_path: Path) -> None:
    report = evaluate_embeddings(
        pairs_path=DEFAULT_PAIRS_PATH,
        embed_fn=stub_embed_384,
        embed_model="stub-384",
        n_timing=5,
        check_subprocess=False,
    )
    out_file = write_results(report, out_dir=tmp_path)
    loaded = json.loads(out_file.read_text(encoding="utf-8"))

    assert REQUIRED_TOP_KEYS.issubset(loaded.keys())
    assert {"mean", "pairs"}.issubset(loaded["paraphrase"].keys())
    assert {"mean", "pairs"}.issubset(loaded["unrelated"].keys())
    assert {"id", "a_short", "b_short", "sim"}.issubset(loaded["paraphrase"]["pairs"][0].keys())
    assert {"mean", "p50", "p95", "n"}.issubset(loaded["embed_ms"].keys())
    assert loaded["embed_ms"]["n"] == 5
    assert loaded["determinism_pass"] is True
    assert loaded["determinism_max_abs_diff"] == pytest.approx(0.0, abs=1e-12)

