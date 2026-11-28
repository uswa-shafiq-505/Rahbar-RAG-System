"""Unit tests for eval/e2e_answers.py (Assignment 03 Part B4).

Uses stub HTTP clients and stub judge callables: zero network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.e2e_answers import (
    compute_judge_human_agreement,
    evaluate_e2e,
    is_answer_faithful,
    write_results,
)

REQUIRED_TOP_KEYS = {
    "ts",
    "n_questions",
    "n_answerable",
    "n_must_abstain",
    "faithfulness",
    "abstention_correctness",
    "citation_validity",
    "latency_p50",
    "latency_p95",
    "judge_human_agreement",
    "per_question",
}

REQUIRED_ROW_KEYS = {
    "id",
    "question",
    "answer",
    "citations",
    "abstained",
    "latency_ms",
    "judge_faithful",
    "judge_reason",
}


class FakeResponse:
    def __init__(self, status_code: int, body: dict[str, Any]) -> None:
        self.status_code = status_code
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._body


class FakeAskClient:
    def __init__(self, by_question: dict[str, dict[str, Any]]) -> None:
        self._by_question = by_question

    def post(self, url: str, json: dict[str, Any], **_kw: Any) -> FakeResponse:
        q = str(json["question"])
        return FakeResponse(200, self._by_question[q])

    def close(self) -> None:
        pass


def _write_golden(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    p = tmp_path / "golden.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_faithfulness_all_cited_vs_uncited_claim() -> None:
    assert is_answer_faithful("Students need 80% attendance to sit finals [1]. Minimum CGPA is 2.00 [2].") is True
    assert is_answer_faithful("Students need 80% attendance to sit finals. Minimum CGPA is 2.00 [2].") is False


def test_abstention_correctness_when_must_abstain_declines(tmp_path: Path) -> None:
    q = "What is the monthly hostel fee?"
    golden = _write_golden(
        tmp_path,
        [{"id": "q1", "type": "unanswerable", "question": q, "must_abstain": True}],
    )
    client = FakeAskClient(
        {
            q: {
                "answer": "I could not find this in the indexed corpus.",
                "citations": [],
                "answered_from_context": False,
                "retrieved": [],
            }
        }
    )
    report = evaluate_e2e(
        golden_path=golden,
        http_client=client,
        labels_path=None,
        skip_judge=True,
    )
    assert report["abstention_correctness"] == 1.0
    assert report["n_must_abstain"] == 1


def test_citation_validity_matching_vs_outside_retrieved(tmp_path: Path) -> None:
    q1 = "What is the CGPA threshold?"
    q2 = "What is the tuition fee?"
    golden = _write_golden(
        tmp_path,
        [
            {"id": "q1", "type": "answerable", "question": q1, "must_abstain": False},
            {"id": "q2", "type": "answerable", "question": q2, "must_abstain": False},
        ],
    )

    client_valid = FakeAskClient(
        {
            q1: {
                "answer": "Minimum CGPA is 2.00 [1].",
                "citations": [{"source": "rules.pdf", "section": "5.4", "page": 25}],
                "answered_from_context": True,
                "retrieved": [{"source": "rules.pdf", "section": "5.4", "page": 25, "score": 0.9}],
            },
            q2: {
                "answer": "Tuition is Rs. 12,000 per credit [1].",
                "citations": [{"source": "fees.html", "section": "Tuition", "page": 1}],
                "answered_from_context": True,
                "retrieved": [{"source": "fees.html", "section": "Tuition", "page": 1, "score": 0.9}],
            },
        }
    )
    res_valid = evaluate_e2e(golden, http_client=client_valid, labels_path=None, skip_judge=True)
    assert res_valid["citation_validity"] == 1.0

    client_invalid = FakeAskClient(
        {
            q1: {
                "answer": "Minimum CGPA is 2.00 [1].",
                "citations": [{"source": "fabricated.pdf", "section": "9.9", "page": 99}],
                "answered_from_context": True,
                "retrieved": [{"source": "rules.pdf", "section": "5.4", "page": 25, "score": 0.9}],
            },
            q2: {
                "answer": "Tuition is Rs. 12,000 per credit [1].",
                "citations": [{"source": "fabricated.pdf", "section": "1.1", "page": 7}],
                "answered_from_context": True,
                "retrieved": [{"source": "fees.html", "section": "Tuition", "page": 1, "score": 0.9}],
            },
        }
    )
    res_invalid = evaluate_e2e(golden, http_client=client_invalid, labels_path=None, skip_judge=True)
    assert res_invalid["citation_validity"] == 0.0


def test_judge_human_agreement_computed() -> None:
    rows = [
        {"id": "q001", "judge_faithful": True},
        {"id": "q002", "judge_faithful": False},
        {"id": "q003", "judge_faithful": True},
        {"id": "q004", "judge_faithful": False},
    ]
    labels = {"q001": True, "q002": False, "q003": True, "q004": True}
    assert compute_judge_human_agreement(rows, labels) == pytest.approx(0.75)


def test_output_json_has_all_required_keys(tmp_path: Path) -> None:
    q1 = "What is the minimum CGPA?"
    q2 = "What is the hostel fee?"
    golden = _write_golden(
        tmp_path,
        [
            {"id": "q001", "type": "answerable", "question": q1, "must_abstain": False},
            {"id": "q002", "type": "unanswerable", "question": q2, "must_abstain": True},
        ],
    )
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"question_id": "q001", "human_faithful": True, "human_reason": "supported"})
        + "\n",
        encoding="utf-8",
    )
    client = FakeAskClient(
        {
            q1: {
                "answer": "Minimum CGPA is 2.00 [1].",
                "citations": [{"source": "rules.pdf", "section": "5.4", "page": 25}],
                "answered_from_context": True,
                "retrieved": [{"source": "rules.pdf", "section": "5.4", "page": 25, "score": 0.91}],
            },
            q2: {
                "answer": "I could not find this in the indexed corpus.",
                "citations": [],
                "answered_from_context": False,
                "retrieved": [],
            },
        }
    )
    report = evaluate_e2e(
        golden_path=golden,
        http_client=client,
        judge_fn=lambda _q, _ev, _ans: (True, "supported by evidence"),
        labels_path=labels,
    )
    out_file = write_results(report, out_dir=tmp_path / "results")
    loaded = json.loads(out_file.read_text(encoding="utf-8"))

    assert REQUIRED_TOP_KEYS.issubset(loaded.keys())
    assert len(loaded["per_question"]) == 2
    assert REQUIRED_ROW_KEYS.issubset(loaded["per_question"][0].keys())
    assert loaded["judge_human_agreement"] == pytest.approx(1.0)
    assert loaded["faithfulness"] == 1.0
    assert loaded["abstention_correctness"] == 1.0
    assert loaded["citation_validity"] == 1.0
