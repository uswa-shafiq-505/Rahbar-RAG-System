"""Stage 1.5: Verification between extraction and chunking (Assignment 03 Part A1).

Checks each extracted Document for:
  - quality: minimum extracted length, non-printable byte ratio, extraction errors
  - completeness: required metadata {source, title, doc_version, page_count} and truncation
  - recency: detects whether a newer version of the same doc_id already exists (or vice versa)
  - uniqueness: exact duplicate by SHA-256 and near-duplicate by embedding cosine similarity

Self-contained and unit-testable: never imports pipeline.store, pipeline.embed, or
pipeline.extract, and never hardcodes runtime paths.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import Document, Page

_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


class CheckOutcome(BaseModel):
    passed: bool
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


class DocumentMeta(BaseModel):
    """Lightweight metadata for previously verified/indexed documents."""

    doc_id: str
    content_hash: str
    doc_version: str | None = None
    title: str | None = None
    source: str | None = None
    embedding: list[float] | None = None


class VerifyResult(BaseModel):
    doc_id: str
    source: str
    passed: bool
    checks: dict[str, CheckOutcome]  # keys: quality, completeness, recency, uniqueness
    quarantine_reason: str | None = None
    docs_to_archive: list[str] = Field(default_factory=list)


def parse_doc_version_from_filename(filename: str) -> str | None:
    """Extract a version token (e.g. '2026-06', '2023-24', '2026') from a filename."""
    stem = Path(filename).stem.lower()
    m_month = re.search(
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_](20\d{2})",
        stem,
    )
    if m_month:
        return f"{m_month.group(2)}-{_MONTHS[m_month.group(1)]}"
    m_ym = re.search(r"(20\d{2})[-_](\d{2})\b", stem)
    if m_ym:
        return f"{m_ym.group(1)}-{m_ym.group(2)}"
    m_y = re.search(r"(20\d{2})\b", stem)
    if m_y:
        return m_y.group(1)
    return None


def derive_doc_id(source: str, title: str | None = None) -> str:
    """Derive a stable logical doc_id across versions of the same document."""
    if title and title.strip():
        slug = re.sub(r"[^a-z0-9]+", "-", title.strip().lower()).strip("-")
        if slug:
            return slug
    stem = Path(source).stem.lower()
    stem = re.sub(
        r"[-_](?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[-_])?20\d{2}(?:[-_]\d{2})?",
        "",
        stem,
    )
    stem = re.sub(r"[-_]v\d+$", "", stem)
    return re.sub(r"[^a-z0-9]+", "-", stem).strip("-") or source


def _version_key(version: str) -> tuple[tuple[int, ...], str]:
    nums = tuple(int(n) for n in re.findall(r"\d+", version))
    return (nums, version)


def _compare_versions(v1: str, v2: str) -> int:
    """Return -1 if v1 < v2, 0 if equal, 1 if v1 > v2."""
    k1, k2 = _version_key(v1), _version_key(v2)
    if k1 < k2:
        return -1
    if k1 > k2:
        return 1
    return 0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _compute_hash(doc: Document, full_text: str) -> str:
    if doc.content_hash:
        return doc.content_hash
    return hashlib.sha256(full_text.encode("utf-8")).hexdigest()


def _matching_existing(
    doc_id: str, doc: Document, existing_docs: dict[str, DocumentMeta]
) -> list[tuple[str, DocumentMeta]]:
    matches: list[tuple[str, DocumentMeta]] = []
    for key, meta in existing_docs.items():
        if key in {doc_id, doc.source} or meta.doc_id == doc_id:
            matches.append((key, meta))
            continue
        if doc.title and meta.title and doc.title.strip().lower() == meta.title.strip().lower():
            matches.append((key, meta))
            continue
        if meta.source and derive_doc_id(meta.source, meta.title) == doc_id:
            matches.append((key, meta))
    return matches


def find_superseded_docs(
    doc: Document,
    existing_docs: dict[str, DocumentMeta],
) -> list[str]:
    """Return doc_ids in existing_docs that are strictly older than `doc`."""
    if not doc.doc_version:
        return []
    doc_id = derive_doc_id(doc.source, doc.title)
    to_archive: list[str] = []
    for key, meta in _matching_existing(doc_id, doc, existing_docs):
        if meta.doc_version and _compare_versions(doc.doc_version, meta.doc_version) > 0:
            to_archive.append(meta.doc_id or key)
    return to_archive


def _check_quality(doc: Document, full_text: str, extraction_error: str | None) -> CheckOutcome:
    err = extraction_error or getattr(doc, "extraction_error", None)
    if err:
        return CheckOutcome(
            passed=False,
            reason=f"extraction error: {err}",
            details={"error": str(err)},
        )

    stripped = full_text.strip()
    if not stripped:
        return CheckOutcome(
            passed=False,
            reason="empty extracted text (0 chars)",
            details={"char_count": 0},
        )

    non_printable = sum(
        1 for ch in full_text if ch == "\ufffd" or not (ch.isprintable() or ch in "\n\r\t")
    )
    ratio = non_printable / len(full_text) if full_text else 0.0
    if ratio > 0.30:
        return CheckOutcome(
            passed=False,
            reason=f"garbled text: {ratio:.1%} non-printable bytes (> 30%)",
            details={"non_printable_ratio": round(ratio, 4), "char_count": len(stripped)},
        )

    if len(stripped) < 50:
        return CheckOutcome(
            passed=False,
            reason=f"extracted text too short ({len(stripped)} < 50 chars)",
            details={"char_count": len(stripped)},
        )

    return CheckOutcome(
        passed=True,
        reason="text length and printability within thresholds",
        details={"char_count": len(stripped), "non_printable_ratio": round(ratio, 4)},
    )


def _check_completeness(doc: Document, full_text: str) -> CheckOutcome:
    missing: list[str] = []
    if not doc.source or not doc.source.strip():
        missing.append("source")
    if not doc.title or not doc.title.strip():
        missing.append("title")
    if not doc.doc_version or not doc.doc_version.strip():
        missing.append("doc_version")
    if doc.page_count is None or doc.page_count < 1:
        missing.append("page_count")

    if missing:
        return CheckOutcome(
            passed=False,
            reason=f"missing required metadata: {', '.join(missing)}",
            details={"missing": missing},
        )

    text_len = len(full_text.strip())
    assert doc.page_count is not None
    if doc.page_count > 1 and text_len < 100:
        return CheckOutcome(
            passed=False,
            reason=f"truncated text: {text_len} chars across {doc.page_count} pages (< 100)",
            details={"char_count": text_len, "page_count": doc.page_count},
        )

    return CheckOutcome(
        passed=True,
        reason="required metadata present and text not truncated",
        details={"page_count": doc.page_count, "char_count": text_len},
    )


def _check_recency(
    doc: Document,
    doc_id: str,
    existing_docs: dict[str, DocumentMeta],
) -> tuple[CheckOutcome, list[str]]:
    if not doc.doc_version:
        return (
            CheckOutcome(
                passed=True,
                reason="no doc_version on incoming doc; recency not applicable",
            ),
            [],
        )

    matches = _matching_existing(doc_id, doc, existing_docs)
    for key, meta in matches:
        if meta.doc_version and _compare_versions(doc.doc_version, meta.doc_version) < 0:
            newer_id = meta.doc_id or key
            return (
                CheckOutcome(
                    passed=False,
                    reason=f"superseded_by:{newer_id} (newer version {meta.doc_version})",
                    details={"newer_id": newer_id, "newer_version": meta.doc_version},
                ),
                [],
            )

    to_archive = find_superseded_docs(doc, existing_docs)
    if to_archive:
        return (
            CheckOutcome(
                passed=True,
                reason=f"newer than existing version(s): {', '.join(to_archive)}",
                details={"docs_to_archive": to_archive},
            ),
            to_archive,
        )

    return (
        CheckOutcome(passed=True, reason="no newer version found in existing index"),
        [],
    )


def _check_uniqueness(
    doc: Document,
    doc_id: str,
    doc_hash: str,
    full_text: str,
    existing_docs: dict[str, DocumentMeta],
    near_dup_threshold: float,
    embed_fn: Callable[[str], list[float]] | None,
    doc_embedding: list[float] | None,
) -> CheckOutcome:
    for key, meta in existing_docs.items():
        if meta.content_hash and meta.content_hash == doc_hash:
            return CheckOutcome(
                passed=False,
                reason=f"exact duplicate of {meta.doc_id or key} (sha256={doc_hash[:12]})",
                details={"duplicate_of": meta.doc_id or key, "sha256": doc_hash},
            )

    has_embeddings = any(meta.embedding is not None for meta in existing_docs.values())
    if has_embeddings:
        vec = doc_embedding
        if vec is None and embed_fn is not None and full_text.strip():
            vec = embed_fn(full_text)
        if vec is not None:
            for key, meta in existing_docs.items():
                if meta.embedding is None:
                    continue
                # Skip older versions of the same logical doc_id so upgrades can proceed.
                is_same_doc = (meta.doc_id == doc_id) or (
                    meta.source is not None and derive_doc_id(meta.source, meta.title) == doc_id
                )
                if (
                    is_same_doc
                    and doc.doc_version
                    and meta.doc_version
                    and _compare_versions(doc.doc_version, meta.doc_version) > 0
                ):
                    continue
                sim = _cosine_similarity(vec, meta.embedding)
                if sim >= near_dup_threshold:
                    return CheckOutcome(
                        passed=False,
                        reason=f"near-duplicate of {meta.doc_id or key} (cosine={sim:.4f})",
                        details={"duplicate_of": meta.doc_id or key, "cosine": round(sim, 4)},
                    )

    return CheckOutcome(
        passed=True,
        reason="unique document (no exact or near duplicate)",
        details={"sha256": doc_hash},
    )


def verify_document(
    doc: Document,
    existing_docs: dict[str, DocumentMeta],  # doc_id -> hash + version + title
    near_dup_threshold: float = 0.95,
    embed_fn: Callable[[str], list[float]] | None = None,
    doc_embedding: list[float] | None = None,
    extraction_error: str | None = None,
) -> VerifyResult:
    """Run the four A1 checks. Return pass/fail with per-check reasons."""
    full_text = "\n".join(p.text for p in doc.pages)
    doc_id = derive_doc_id(doc.source, doc.title)
    doc_hash = _compute_hash(doc, full_text)

    quality = _check_quality(doc, full_text, extraction_error)
    completeness = _check_completeness(doc, full_text)
    recency, docs_to_archive = _check_recency(doc, doc_id, existing_docs)
    uniqueness = _check_uniqueness(
        doc=doc,
        doc_id=doc_id,
        doc_hash=doc_hash,
        full_text=full_text,
        existing_docs=existing_docs,
        near_dup_threshold=near_dup_threshold,
        embed_fn=embed_fn,
        doc_embedding=doc_embedding,
    )

    checks = {
        "quality": quality,
        "completeness": completeness,
        "recency": recency,
        "uniqueness": uniqueness,
    }
    failed = [f"{name}: {outcome.reason}" for name, outcome in checks.items() if not outcome.passed]
    passed = len(failed) == 0
    quarantine_reason = "; ".join(failed) if failed else None

    return VerifyResult(
        doc_id=doc_id,
        source=doc.source,
        passed=passed,
        checks=checks,
        quarantine_reason=quarantine_reason,
        docs_to_archive=docs_to_archive,
    )


def _default_load_document(path: Path) -> tuple[Document, str | None]:
    """Minimal self-contained file loader when no extract_fn is injected."""
    raw_bytes = path.read_bytes()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [
                Page(source=path.name, page=i + 1, text=(pg.extract_text() or "").strip())
                for i, pg in enumerate(reader.pages)
            ]
            raw_meta: dict[Any, Any] = dict(reader.metadata) if reader.metadata else {}
            title = str(raw_meta.get("/Title") or "").strip() or path.stem
            return (
                Document(
                    source=path.name,
                    doc_version=parse_doc_version_from_filename(path.name),
                    superseded=False,
                    pages=pages,
                    content_hash=content_hash,
                    title=title,
                    page_count=len(pages),
                ),
                None,
            )
        except Exception as exc:  # noqa: BLE001
            return (
                Document(
                    source=path.name,
                    doc_version=parse_doc_version_from_filename(path.name),
                    superseded=False,
                    pages=[],
                    content_hash=content_hash,
                    title=path.stem,
                    page_count=0,
                ),
                str(exc),
            )

    text = raw_bytes.decode("utf-8", errors="replace")
    return (
        Document(
            source=path.name,
            doc_version=parse_doc_version_from_filename(path.name),
            superseded=False,
            pages=[Page(source=path.name, page=1, text=text)],
            content_hash=content_hash,
            title=path.stem,
            page_count=1,
        ),
        None,
    )


def run_verification_batch(
    input_dir: Path,
    quarantine_dir: Path,
    report_path: Path,
    embed_fn: Callable[[str], list[float]],  # injected, not imported
    extract_fn: Callable[[Path], Document] | None = None,
    existing_docs: dict[str, DocumentMeta] | None = None,
    near_dup_threshold: float = 0.95,
) -> list[VerifyResult]:
    """For every doc in input_dir, call verify_document, move failures to
    quarantine_dir, append one JSONL line per doc to report_path."""
    known: dict[str, DocumentMeta] = dict(existing_docs) if existing_docs is not None else {}
    results: list[VerifyResult] = []

    files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.name != "_index.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as report_fh:
        for file_path in files:
            extraction_error: str | None = None
            if extract_fn is not None:
                try:
                    doc = extract_fn(file_path)
                except Exception as exc:  # noqa: BLE001
                    doc = Document(source=file_path.name, pages=[])
                    extraction_error = str(exc)
            else:
                doc, extraction_error = _default_load_document(file_path)

            full_text = "\n".join(p.text for p in doc.pages)
            doc_vec: list[float] | None = None
            if extraction_error is None and full_text.strip():
                doc_vec = embed_fn(full_text)

            result = verify_document(
                doc=doc,
                existing_docs=known,
                near_dup_threshold=near_dup_threshold,
                embed_fn=embed_fn,
                doc_embedding=doc_vec,
                extraction_error=extraction_error,
            )

            if not result.passed:
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file_path), str(quarantine_dir / file_path.name))
            else:
                doc_hash = _compute_hash(doc, full_text)
                known[result.doc_id] = DocumentMeta(
                    doc_id=result.doc_id,
                    content_hash=doc_hash,
                    doc_version=doc.doc_version,
                    title=doc.title,
                    source=doc.source,
                    embedding=doc_vec,
                )

            report_fh.write(result.model_dump_json() + "\n")
            results.append(result)

    return results
