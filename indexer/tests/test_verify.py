"""Unit tests for Stage 1.5 verification (Assignment 03 Part A1).

Uses stub Document objects, stub extractor, and stub embed_fn: no network, no real PDFs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.models import Document, Page
from pipeline.verify import (
    DocumentMeta,
    find_superseded_docs,
    run_verification_batch,
    verify_document,
)

CLEAN_TEXT = (
    "Undergraduate students must maintain a minimum CGPA of 2.00 across regular "
    "semesters to remain in good academic standing and avoid academic warning."
)


def _make_doc(
    source: str = "academic-rules-2026.pdf",
    text: str = CLEAN_TEXT,
    title: str | None = "Academic Rules Handbook",
    doc_version: str | None = "2026-06",
    page_count: int | None = 1,
    content_hash: str | None = None,
    pages: list[Page] | None = None,
) -> Document:
    doc_pages = pages if pages is not None else [Page(source=source, page=1, text=text)]
    return Document(
        source=source,
        doc_version=doc_version,
        superseded=False,
        pages=doc_pages,
        content_hash=content_hash or hashlib.sha256(text.encode("utf-8")).hexdigest(),
        title=title,
        page_count=page_count,
    )


def _stub_embed(text: str) -> list[float]:
    if "fee" in text.lower() or "tuition" in text.lower():
        return [0.0, 1.0, 0.0]
    return [1.0, 0.0, 0.0]


def test_quality_empty_text_fails() -> None:
    doc = _make_doc(text="   ")
    res = verify_document(doc, existing_docs={})
    assert res.passed is False
    assert res.checks["quality"].passed is False
    assert "empty" in res.checks["quality"].reason.lower()
    assert res.quarantine_reason is not None


def test_quality_garbled_non_printable_fails() -> None:
    garbled = ("A" * 50) + ("\x00" * 50)
    doc = _make_doc(text=garbled)
    res = verify_document(doc, existing_docs={})
    assert res.passed is False
    assert res.checks["quality"].passed is False
    assert "non-printable" in res.checks["quality"].reason.lower()


def test_completeness_missing_doc_version_fails() -> None:
    doc = _make_doc(doc_version=None)
    res = verify_document(doc, existing_docs={})
    assert res.passed is False
    assert res.checks["completeness"].passed is False
    assert "doc_version" in res.checks["completeness"].reason


def test_completeness_truncated_multipage_doc_fails() -> None:
    short_text = "Short extract under one hundred characters for a three-page file."
    assert len(short_text) == 65
    pages = [
        Page(source="truncated.pdf", page=1, text=short_text),
        Page(source="truncated.pdf", page=2, text=""),
        Page(source="truncated.pdf", page=3, text=""),
    ]
    doc = _make_doc(source="truncated.pdf", text=short_text, page_count=3, pages=pages)
    res = verify_document(doc, existing_docs={})
    assert res.passed is False
    assert res.checks["completeness"].passed is False
    assert "truncated" in res.checks["completeness"].reason.lower()


def test_recency_existing_newer_version_fails() -> None:
    existing = {
        "academic-rules-handbook": DocumentMeta(
            doc_id="academic-rules-handbook",
            content_hash="old-other-hash",
            doc_version="2026-06",
            title="Academic Rules Handbook",
            source="academic-rules-2026.pdf",
        )
    }
    older_doc = _make_doc(source="academic-rules-2023.pdf", doc_version="2023-08")
    res = verify_document(older_doc, existing_docs=existing)
    assert res.passed is False
    assert res.checks["recency"].passed is False
    assert res.quarantine_reason is not None
    assert "2026-06" in res.quarantine_reason
    assert "superseded_by:" in res.checks["recency"].reason


def test_recency_reverse_check_marks_older_doc_for_archive() -> None:
    existing = {
        "academic-rules-handbook": DocumentMeta(
            doc_id="academic-rules-handbook",
            content_hash="old-hash",
            doc_version="2023-08",
            title="Academic Rules Handbook",
            source="academic-rules-2023.pdf",
        )
    }
    newer_doc = _make_doc(source="academic-rules-2026.pdf", doc_version="2026-06")
    assert find_superseded_docs(newer_doc, existing) == ["academic-rules-handbook"]
    res = verify_document(newer_doc, existing_docs=existing)
    assert res.passed is True
    assert res.docs_to_archive == ["academic-rules-handbook"]


def test_uniqueness_exact_duplicate_fails() -> None:
    doc1 = _make_doc(source="doc-a.pdf", title="Policy A", content_hash="sha-same")
    existing = {
        "policy-a": DocumentMeta(
            doc_id="policy-a",
            content_hash="sha-same",
            doc_version="2026-06",
            title="Policy A",
            source="doc-a.pdf",
        )
    }
    doc2 = _make_doc(source="doc-b.pdf", title="Policy B", content_hash="sha-same")
    res = verify_document(doc2, existing_docs=existing)
    assert res.passed is False
    assert res.checks["uniqueness"].passed is False
    assert "exact duplicate" in res.checks["uniqueness"].reason.lower()
    assert doc1.source == "doc-a.pdf"


def test_uniqueness_near_duplicate_fails() -> None:
    existing = {
        "policy-a": DocumentMeta(
            doc_id="policy-a",
            content_hash="hash-a",
            doc_version="2026-06",
            title="Policy A",
            source="doc-a.pdf",
            embedding=[1.0, 0.0, 0.0],
        )
    }
    doc2 = _make_doc(
        source="doc-b.pdf",
        title="Policy B",
        text=CLEAN_TEXT + " Additional sentence to alter hash.",
        content_hash="hash-b",
    )
    # Near-identical vector: cos([1, 0, 0], [0.99, 0.141, 0]) ~= 0.9900
    res = verify_document(
        doc2,
        existing_docs=existing,
        near_dup_threshold=0.95,
        embed_fn=lambda _t: [0.99, 0.141, 0.0],
    )
    assert res.passed is False
    assert res.checks["uniqueness"].passed is False
    assert "near-duplicate" in res.checks["uniqueness"].reason.lower()


def test_happy_path_clean_doc_passes_all_four() -> None:
    existing = {
        "fee-structure": DocumentMeta(
            doc_id="fee-structure",
            content_hash="different-hash",
            doc_version="2026-06",
            title="Fee Structure",
            source="fees.html",
            embedding=[0.0, 1.0, 0.0],
        )
    }
    doc = _make_doc()
    res = verify_document(doc, existing_docs=existing, embed_fn=_stub_embed)
    assert res.passed is True
    assert res.quarantine_reason is None
    assert all(outcome.passed for outcome in res.checks.values())
    assert set(res.checks.keys()) == {"quality", "completeness", "recency", "uniqueness"}


def test_run_verification_batch_moves_failures_and_writes_report(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    q_dir = tmp_path / "quarantine"
    report = tmp_path / "lake" / "verify_report.jsonl"
    in_dir.mkdir()

    (in_dir / "01_rules_2026.txt").write_text(CLEAN_TEXT, encoding="utf-8")
    (in_dir / "02_fees_2026.txt").write_text(
        "Tuition fee per credit hour for Fall 2026 semester is Rs. 12,000 across programs.",
        encoding="utf-8",
    )
    (in_dir / "03_empty_2026.txt").write_text("", encoding="utf-8")

    def stub_extract(p: Path) -> Document:
        text = p.read_text(encoding="utf-8")
        return Document(
            source=p.name,
            doc_version="2026-06",
            superseded=False,
            pages=[Page(source=p.name, page=1, text=text)],
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            title=p.stem,
            page_count=1,
        )

    results = run_verification_batch(
        input_dir=in_dir,
        quarantine_dir=q_dir,
        report_path=report,
        embed_fn=_stub_embed,
        extract_fn=stub_extract,
    )

    assert len(results) == 3
    assert sum(1 for r in results if r.passed) == 2
    assert sum(1 for r in results if not r.passed) == 1

    report_lines = report.read_text(encoding="utf-8").splitlines()
    assert len(report_lines) == 3
    parsed = [json.loads(line) for line in report_lines]
    assert [row["passed"] for row in parsed] == [True, True, False]

    quarantined_files = list(q_dir.iterdir())
    assert len(quarantined_files) == 1
    assert quarantined_files[0].name == "03_empty_2026.txt"
    assert not (in_dir / "03_empty_2026.txt").exists()

