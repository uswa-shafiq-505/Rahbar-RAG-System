"""Unit tests for wired verification in pipeline.run (Assignment 03 Part A1).

Uses mocked Chroma client, stub embed_fn, and stub/local extract on tmp_path:
no network, no real PDFs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline.config import IndexerSettings
from pipeline.extract import extract
from pipeline.models import Chunk
from pipeline.run import run_pipeline

VALID_BODY = (
    "Academic Rules Policy Title\n\n"
    "Undergraduate students must maintain a cumulative grade point average of at least 2.00 "
    "across all semesters to avoid academic warning and remain in good academic standing."
)


class FakeCollection:
    def __init__(self, metadatas: list[dict[str, Any]] | None = None) -> None:
        self._metadatas = metadatas or []

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        return {"metadatas": list(self._metadatas)}


class FakeChromaClient:
    def __init__(self, metadatas: list[dict[str, Any]] | None = None) -> None:
        self.collection = FakeCollection(metadatas) if metadatas is not None else None

    def get_collection(self, name: str) -> FakeCollection:
        if self.collection is None:
            raise ValueError(f"collection {name!r} does not exist")
        return self.collection


def _settings(tmp_path: Path, verify_enabled: bool = True) -> IndexerSettings:
    return IndexerSettings(
        embed_model="nomic-embed-text",
        chroma_host="chroma",
        chroma_port=8000,
        collection="rahbar",
        corpus_dir=str(tmp_path / "corpus"),
        chunk_size=400,
        chunk_overlap=50,
        ollama_url="http://localhost:11434",
        verify_enabled=verify_enabled,
        verify_near_dup_threshold=0.95,
        quarantine_dir=str(tmp_path / "quarantine"),
        lake_dir=str(tmp_path / "lake"),
        verify_report_path=str(tmp_path / "results"),
    )


def _stub_embed(items: list[Chunk], _settings: IndexerSettings) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in items]


def test_clean_doc_passes_and_is_chunked_embedded_stored(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    corpus = Path(cfg.corpus_dir)
    corpus.mkdir()
    doc_path = corpus / "rules_2026-06.txt"
    doc_path.write_text(VALID_BODY, encoding="utf-8")

    stored_chunks: list[Chunk] = []

    def fake_store(chunks: list[Chunk], vectors: list[list[float]], _cfg: IndexerSettings) -> int:
        assert len(chunks) == len(vectors)
        stored_chunks.extend(chunks)
        return len(chunks)

    count = run_pipeline(
        cfg,
        chroma_client=FakeChromaClient(),
        extract_fn=extract,
        embed_fn=_stub_embed,
        store_fn=fake_store,
    )

    assert count == len(stored_chunks)
    assert count > 0
    assert doc_path.exists()
    assert list(Path(cfg.quarantine_dir).iterdir()) == []


def test_empty_text_doc_is_quarantined_and_zero_chunks_stored(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    corpus = Path(cfg.corpus_dir)
    corpus.mkdir()
    bad_path = corpus / "empty_2026-06.txt"
    bad_path.write_text("   \n\n  ", encoding="utf-8")

    stored_chunks: list[Chunk] = []

    def fake_store(chunks: list[Chunk], _vectors: list[list[float]], _cfg: IndexerSettings) -> int:
        stored_chunks.extend(chunks)
        return len(chunks)

    count = run_pipeline(
        cfg,
        chroma_client=FakeChromaClient(),
        extract_fn=extract,
        embed_fn=_stub_embed,
        store_fn=fake_store,
    )

    assert count == 0
    assert stored_chunks == []
    assert not bad_path.exists()
    assert (Path(cfg.quarantine_dir) / "empty_2026-06.txt").exists()


def test_exact_duplicate_is_quarantined(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    corpus = Path(cfg.corpus_dir)
    corpus.mkdir()
    dup_path = corpus / "dup_rules_2026-06.txt"
    dup_path.write_text(VALID_BODY, encoding="utf-8")
    existing_hash = hashlib.sha256(dup_path.read_bytes()).hexdigest()

    client = FakeChromaClient(
        metadatas=[
            {
                "source": "orig_rules_2026-06.txt",
                "title": "Academic Rules Policy Title",
                "doc_version": "2026-06",
                "content_hash": existing_hash,
            }
        ]
    )

    stored_chunks: list[Chunk] = []
    count = run_pipeline(
        cfg,
        chroma_client=client,
        extract_fn=extract,
        embed_fn=_stub_embed,
        store_fn=lambda c, _v, _s: stored_chunks.extend(c) or len(c),
    )

    assert count == 0
    assert stored_chunks == []
    assert (Path(cfg.quarantine_dir) / "dup_rules_2026-06.txt").exists()


def test_superseded_doc_is_quarantined(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    corpus = Path(cfg.corpus_dir)
    corpus.mkdir()
    old_path = corpus / "rules_2023-08.txt"
    old_path.write_text(VALID_BODY, encoding="utf-8")

    client = FakeChromaClient(
        metadatas=[
            {
                "source": "rules_2026-06.txt",
                "title": "Academic Rules Policy Title",
                "doc_version": "2026-06",
                "content_hash": "different-newer-hash",
            }
        ]
    )

    stored_chunks: list[Chunk] = []
    count = run_pipeline(
        cfg,
        chroma_client=client,
        extract_fn=extract,
        embed_fn=_stub_embed,
        store_fn=lambda c, _v, _s: stored_chunks.extend(c) or len(c),
    )

    assert count == 0
    assert stored_chunks == []
    assert (Path(cfg.quarantine_dir) / "rules_2023-08.txt").exists()


def test_verify_disabled_processes_all_docs_backward_compat(tmp_path: Path) -> None:
    cfg = _settings(tmp_path, verify_enabled=False)
    corpus = Path(cfg.corpus_dir)
    corpus.mkdir()
    (corpus / "doc1_2026-06.txt").write_text(VALID_BODY, encoding="utf-8")
    (corpus / "doc2_2026-06.txt").write_text(VALID_BODY, encoding="utf-8")

    stored_chunks: list[Chunk] = []
    count = run_pipeline(
        cfg,
        extract_fn=extract,
        embed_fn=_stub_embed,
        store_fn=lambda c, _v, _s: stored_chunks.extend(c) or len(c),
    )

    assert count == len(stored_chunks)
    assert count >= 2
    assert not Path(cfg.quarantine_dir).exists()
    assert not Path(cfg.verify_report_path).exists()


def test_report_file_has_exact_n_jsonl_lines(tmp_path: Path) -> None:
    cfg = _settings(tmp_path)
    corpus = Path(cfg.corpus_dir)
    corpus.mkdir()
    (corpus / "01_valid_2026-06.txt").write_text(VALID_BODY, encoding="utf-8")
    (corpus / "02_valid_2026-06.txt").write_text(
        "Fee Structure Policy\n\n"
        "Tuition fee for undergraduate programs is Rs. 12,000 per credit hour, payable before "
        "each semester begins, alongside a refundable security deposit of Rs. 20,000.",
        encoding="utf-8",
    )
    (corpus / "03_empty_2026-06.txt").write_text("", encoding="utf-8")

    run_pipeline(
        cfg,
        chroma_client=FakeChromaClient(),
        extract_fn=extract,
        embed_fn=_stub_embed,
        store_fn=lambda c, _v, _s: len(c),
    )

    reports = list(Path(cfg.verify_report_path).glob("verify-*.jsonl"))
    assert len(reports) == 1
    lines = reports[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [row["passed"] for row in parsed] == [True, True, False]
