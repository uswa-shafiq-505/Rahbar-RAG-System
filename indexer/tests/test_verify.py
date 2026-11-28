"""Unit tests for Stage 1.5 verification (Assignment 03 Part A1).

Uses stub Document objects, stub extractor, and stub embed_fn: no network, no real PDFs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pipeline.config import IndexerSettings
from pipeline.extract import _extract_metadata, extract
from pipeline.models import Chunk, Document, Page
from pipeline.verify import (
    DocumentMeta,
    derive_doc_id,
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


def test_doc_id_derived_from_filename_not_title(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pipeline.extract as extract_mod

    class FakePage:
        def extract_text(self) -> str:
            return "2026\n\n" + CLEAN_TEXT

    class FakePdfReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage()]

    monkeypatch.setattr(extract_mod, "PdfReader", FakePdfReader)

    pdf_like = tmp_path / "handbook-2026.pdf"
    pdf_like.write_bytes(b"%PDF-1.4 fake")
    meta = _extract_metadata(pdf_like)
    assert meta["title"] == "2026"
    assert meta["doc_id"] == "handbook-2026"
    assert derive_doc_id("handbook-2026.pdf", title="2026") == "handbook-2026"

    doc = _make_doc(source="handbook-2026.pdf", title="2026", doc_version="2026-06")
    res = verify_document(doc, existing_docs={})
    assert res.doc_id == "handbook-2026"


def test_handbook_not_duplicate_of_itself() -> None:
    doc = _make_doc(
        source="handbook-2026.pdf",
        title="2026",
        doc_version="2026-06",
    )
    res = verify_document(doc, existing_docs={})
    assert res.passed is True
    assert res.quarantine_reason is None
    assert res.checks["uniqueness"].passed is True
    assert "duplicate_of" not in res.checks["uniqueness"].details


def test_existing_meta_not_populated_with_current_doc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pipeline.run as run_mod

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("doc1_2026-06.txt", "doc2_2026-06.txt", "doc3_2026-06.txt"):
        (corpus / name).write_text(f"{name}\n\n{CLEAN_TEXT} unique={name}", encoding="utf-8")

    snapshots: list[tuple[str, set[str]]] = []
    orig_verify = run_mod.verify_document

    def spy_verify(
        doc: Document, existing_docs: dict[str, DocumentMeta], *args: Any, **kwargs: Any
    ) -> Any:
        snapshots.append((doc.source, set(existing_docs.keys())))
        return orig_verify(doc, existing_docs, *args, **kwargs)

    monkeypatch.setattr(run_mod, "verify_document", spy_verify)

    cfg = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar",
        corpus_dir=str(corpus),
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://localhost:11434",
        verify_enabled=True,
        verify_near_dup_threshold=0.95,
        quarantine_dir=str(tmp_path / "quarantine"),
        lake_dir=str(tmp_path / "lake"),
        verify_report_path=str(tmp_path / "results"),
    )

    class EmptyChroma:
        def get_collection(self, name: str) -> Any:
            raise ValueError(name)

    run_mod.run_pipeline(
        cfg,
        chroma_client=EmptyChroma(),
        extract_fn=extract,
        embed_fn=lambda items, _s: [[1.0, 0.0, 0.0] for _ in items],
        store_fn=lambda chunks, _v, _s: len(chunks),
    )

    assert len(snapshots) == 3
    assert snapshots[0] == ("doc1_2026-06.txt", set())
    assert snapshots[1] == ("doc2_2026-06.txt", {"doc1_2026-06"})
    assert snapshots[2] == ("doc3_2026-06.txt", {"doc1_2026-06", "doc2_2026-06"})


def test_a2_second_run_zero_chunks(tmp_path: Path) -> None:
    import pipeline.run as run_mod

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    q_dir = tmp_path / "quarantine"
    metadatas: list[dict[str, Any]] = []

    for i in range(1, 4):
        fname = f"policy_{i}_2026-06.txt"
        fpath = corpus / fname
        fpath.write_text(f"Title {i}\n\n{CLEAN_TEXT} policy={i}", encoding="utf-8")
        metadatas.append(
            {
                "source": fname,
                "title": f"Title {i}",
                "doc_version": "2026-06",
                "content_hash": hashlib.sha256(fpath.read_bytes()).hexdigest(),
            }
        )

    class PopulatedCollection:
        def get(self, include: list[str] | None = None) -> dict[str, Any]:
            return {"metadatas": list(metadatas)}

    class PopulatedChroma:
        def get_collection(self, name: str) -> PopulatedCollection:
            return PopulatedCollection()

    stored: list[Chunk] = []
    cfg = IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar",
        corpus_dir=str(corpus),
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://localhost:11434",
        verify_enabled=True,
        verify_near_dup_threshold=0.95,
        quarantine_dir=str(q_dir),
        lake_dir=str(tmp_path / "lake"),
        verify_report_path=str(tmp_path / "results"),
    )

    count = run_mod.run_pipeline(
        cfg,
        chroma_client=PopulatedChroma(),
        extract_fn=extract,
        embed_fn=lambda items, _s: [[1.0, 0.0, 0.0] for _ in items],
        store_fn=lambda chunks, _v, _s: stored.extend(chunks) or len(chunks),
    )

    assert count == 0
    assert stored == []
    assert list(q_dir.iterdir()) == []
    assert len(list(corpus.iterdir())) == 3

