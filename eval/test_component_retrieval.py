"""Unit tests for eval/component_retrieval.py (Assignment 03 Part B3).

Uses injected fake Chroma client and stub embed_fn: no network, no live Chroma.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from eval.component_retrieval import (
    add_typos,
    evaluate_retrieval,
    mrr,
    ndcg_at_k,
    recall_at_k,
    write_results,
)

REQUIRED_TOP_KEYS = {
    "ts",
    "embed_model",
    "collection",
    "n_questions",
    "n_answerable",
    "n_must_abstain",
    "metrics",
    "noise_test",
    "per_question",
}

REQUIRED_NOISE_KEYS = {
    "recall_at_3_clean",
    "recall_at_3_noisy",
    "delta_r3",
    "recall_at_5_clean",
    "recall_at_5_noisy",
    "delta_r5",
    "mrr_clean",
    "mrr_noisy",
    "delta_mrr",
    "ndcg_at_5_clean",
    "ndcg_at_5_noisy",
    "delta_ndcg",
}


class FakeCollection:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self._responses = responses
        self.queries_seen: list[list[float]] = []

    def query(self, query_embeddings: list[list[float]], n_results: int = 5) -> dict[str, Any]:
        vec = query_embeddings[0]
        self.queries_seen.append(vec)
        key = f"{vec[0]:.2f}:{vec[1]:.2f}"
        if key in self._responses:
            return self._responses[key]
        return next(iter(self._responses.values()))


class FakeChromaClient:
    def __init__(self, collection: FakeCollection) -> None:
        self._collection = collection

    def get_collection(self, name: str) -> FakeCollection:
        return self._collection


def _write_golden(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_recall_at_k_perfect_and_zero() -> None:
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    assert recall_at_k(retrieved, {"c1"}, 3) == 1.0
    assert recall_at_k(retrieved, {"c4"}, 5) == 1.0
    assert recall_at_k(retrieved, {"c4"}, 3) == 0.0
    assert recall_at_k(retrieved, {"c99"}, 5) == 0.0


def test_mrr_ranks_one_and_three() -> None:
    assert mrr(["c1", "c2", "c3"], {"c1"}) == pytest.approx(1.0)
    assert mrr(["c1", "c2", "c3"], {"c3"}) == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert mrr(["c1", "c2", "c3"], {"c99"}) == 0.0


def test_ndcg_at_5_positions_one_and_three() -> None:
    assert ndcg_at_k(["c1", "c2", "c3", "c4", "c5"], {"c1"}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["c1", "c2", "c3", "c4", "c5"], {"c3"}, 5) == pytest.approx(0.5)


def test_missing_ground_truth_question_is_skipped(tmp_path: Path) -> None:
    golden = _write_golden(
        tmp_path,
        [
            {
                "id": "q1",
                "type": "answerable",
                "question": "Valid question about attendance",
                "ground_truth": {"source": "rules.pdf", "section": "1", "page": 4},
                "must_abstain": False,
            },
            {
                "id": "q2",
                "type": "answerable",
                "question": "Broken question without ground truth",
                "ground_truth": None,
                "must_abstain": False,
            },
            {
                "id": "q3",
                "type": "unanswerable",
                "question": "What are the hostel curfew rules?",
                "ground_truth": None,
                "must_abstain": True,
            },
        ],
    )
    col = FakeCollection(
        {
            "1.00:0.00": {
                "ids": [["c1", "c2", "c3"]],
                "metadatas": [[
                    {"source": "rules.pdf", "page": 4},
                    {"source": "other.pdf", "page": 1},
                    {"source": "other.pdf", "page": 2},
                ]],
                "distances": [[0.2, 0.8, 1.0]],
            }
        }
    )
    report = evaluate_retrieval(
        golden_path=golden,
        chroma_client=FakeChromaClient(col),
        collection_name="rahbar",
        embed_fn=lambda _q: [1.0, 0.0],
        embed_model="stub",
    )
    assert report["n_answerable"] == 1
    assert report["n_must_abstain"] == 1
    assert report["n_questions"] == 2
    assert [row["id"] for row in report["per_question"]] == ["q1", "q3"]
    assert report["metrics"]["recall_at_3"] == 1.0
    assert report["per_question"][1]["recall_at_3"] is None
    assert report["per_question"][1]["top1_score"] == pytest.approx(0.9)


def test_noise_test_typos_are_deterministic_with_seed_42(tmp_path: Path) -> None:
    q = "What is the minimum attendance percentage required for undergraduate courses?"
    assert add_typos(q, random.Random(42)) == add_typos(q, random.Random(42))
    assert add_typos(q, random.Random(42)) != q

    golden = _write_golden(
        tmp_path,
        [
            {
                "id": "q1",
                "type": "answerable",
                "question": q,
                "ground_truth": {"source": "rules.pdf", "section": "1", "page": 5},
                "must_abstain": False,
            }
        ],
    )
    col = FakeCollection(
        {
            "1.00:0.00": {
                "ids": [["c1"]],
                "metadatas": [[{"source": "rules.pdf", "page": 5}]],
                "distances": [[0.1]],
            }
        }
    )
    seen_queries: list[str] = []

    def tracking_embed(text: str) -> list[float]:
        seen_queries.append(text)
        return [1.0, 0.0]

    r1 = evaluate_retrieval(golden, FakeChromaClient(col), "rahbar", tracking_embed)
    first_run_queries = list(seen_queries)
    seen_queries.clear()
    r2 = evaluate_retrieval(golden, FakeChromaClient(col), "rahbar", tracking_embed)
    second_run_queries = list(seen_queries)

    assert first_run_queries == second_run_queries
    assert len(first_run_queries) == 2
    assert first_run_queries[0] != first_run_queries[1]
    assert r1["noise_test"] == r2["noise_test"]


def test_output_json_has_all_required_keys(tmp_path: Path) -> None:
    golden = _write_golden(
        tmp_path,
        [
            {
                "id": "q1",
                "type": "answerable",
                "question": "Minimum CGPA for undergraduate warning?",
                "ground_truth": {"source": "rules.pdf", "section": "5.4", "page": 25},
                "must_abstain": False,
            },
            {
                "id": "q2",
                "type": "unanswerable",
                "question": "Student parking permit fee?",
                "ground_truth": None,
                "must_abstain": True,
            },
        ],
    )
    col = FakeCollection(
        {
            "1.00:0.00": {
                "ids": [["c_other", "c_hit"]],
                "metadatas": [[
                    {"source": "other.pdf", "page": 1},
                    {"source": "rules.pdf", "page": 24},  # page within +/- 1
                ]],
                "distances": [[0.4, 0.5]],
            }
        }
    )
    report = evaluate_retrieval(
        golden_path=golden,
        chroma_client=FakeChromaClient(col),
        collection_name="rahbar",
        embed_fn=lambda _q: [1.0, 0.0],
        embed_model="stub-model",
    )
    out_file = write_results(report, out_dir=tmp_path / "results")
    loaded = json.loads(out_file.read_text(encoding="utf-8"))

    assert REQUIRED_TOP_KEYS.issubset(loaded.keys())
    assert {"recall_at_3", "recall_at_5", "mrr", "ndcg_at_5"} == set(loaded["metrics"].keys())
    assert REQUIRED_NOISE_KEYS == set(loaded["noise_test"].keys())
    assert loaded["metrics"]["mrr"] == pytest.approx(0.5)
    assert loaded["metrics"]["recall_at_3"] == 1.0
