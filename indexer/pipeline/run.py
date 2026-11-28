"""The indexer Job's entry point. It runs once and exits (section 4.2).

    python -m pipeline.run --dummy    # week 1: three fake chunks, no model needed
    python -m pipeline.run            # the real pipeline: extract, verify, clean, chunk, embed, store

Exit code 0 means the index was written; anything else fails the Job, and
Kubernetes retries it up to the Job's backoffLimit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import IndexerSettings, get_settings
from .models import Chunk, Document
from .verify import DocumentMeta, derive_doc_id, verify_document


def log(msg: str, **fields: object) -> None:
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "logger": "rahbar.indexer", "msg": msg, **fields}), flush=True)


def _utc_now_iso() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _doc_meta(doc: Document, doc_id: str | None = None) -> DocumentMeta:
    full_text = "\n".join(p.text for p in doc.pages)
    content_hash = doc.content_hash or hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    resolved_id = doc_id or derive_doc_id(doc.source, doc.title)
    return DocumentMeta(
        doc_id=resolved_id,
        content_hash=content_hash,
        doc_version=doc.doc_version,
        title=doc.title,
        source=doc.source,
    )


def load_existing_meta(chroma_client: Any, collection_name: str) -> dict[str, DocumentMeta]:
    """Load previously indexed document metadata ONLY from Chroma; return {} if empty/absent."""
    if chroma_client is None:
        return {}
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:  # noqa: BLE001
        return {}
    try:
        result = collection.get(include=["metadatas"])
    except Exception:  # noqa: BLE001
        return {}
    metadatas = (result or {}).get("metadatas") or []
    existing: dict[str, DocumentMeta] = {}
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        source = str(meta.get("source") or "")
        title = str(meta["title"]) if meta.get("title") is not None else None
        doc_id = derive_doc_id(source) if source else ""
        if not doc_id:
            continue
        content_hash = str(meta.get("content_hash") or "")
        doc_version = str(meta["doc_version"]) if meta.get("doc_version") else None
        existing[doc_id] = DocumentMeta(
            doc_id=doc_id,
            content_hash=content_hash,
            doc_version=doc_version,
            title=title,
            source=source or None,
            in_chroma=True,
        )
    return existing


def _front_matter(path: Path) -> dict[str, str]:
    """Read `key: value` lines between leading '---' markers (used by eval/fixtures)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def discover(corpus_dir: str) -> list[tuple[Path, str, bool]]:
    """List (path, doc_version, superseded) for every document to index.

    corpus/fetch.py writes _index.json next to the files it downloads; use it when
    present. Otherwise read Markdown front matter (the CI fixtures work this way).
    """
    root = Path(corpus_dir)
    index = root / "_index.json"
    if index.exists():
        entries = json.loads(index.read_text(encoding="utf-8"))
        return [(root / e["file"], str(e.get("doc_version", "unknown")),
                 bool(e.get("superseded", False))) for e in entries]
    found = []
    for path in sorted(root.glob("*")):
        if path.suffix.lower() not in {".pdf", ".md", ".html", ".txt"}:
            continue
        if path.name.lower().startswith("readme"):
            continue
        meta = _front_matter(path) if path.suffix.lower() == ".md" else {}
        found.append((path, meta.get("doc_version", "unknown"),
                      meta.get("superseded", "false").lower() == "true"))
    return found


def run_pipeline(
    settings: IndexerSettings,
    *,
    chroma_client: Any | None = None,
    extract_fn: Callable[[str, str, bool], Document] | None = None,
    embed_fn: Callable[[list[Chunk], IndexerSettings], list[list[float]]] | None = None,
    store_fn: Callable[[list[Chunk], list[list[float]], IndexerSettings], int] | None = None,
) -> int:
    from .chunk import chunk
    from .clean import clean
    from .embed import embed
    from .extract import extract
    from .store import get_client, store

    def _default_embed(chunks: list[Chunk], cfg: IndexerSettings) -> list[list[float]]:
        return embed(chunks, cfg)  # type: ignore[arg-type]

    def _default_store(
        chunks: list[Chunk], vecs: list[list[float]], cfg: IndexerSettings
    ) -> int:
        return store(chunks, vecs, cfg)  # type: ignore[arg-type]

    active_extract = extract_fn or extract
    active_embed = embed_fn or _default_embed
    active_store = store_fn or _default_store

    documents = discover(settings.corpus_dir)
    if not documents:
        raise SystemExit(f"no documents found in {settings.corpus_dir}")

    if not settings.verify_enabled:
        all_chunks: list[Chunk] = []
        for path, doc_version, superseded in documents:
            doc = clean(active_extract(str(path), doc_version, superseded))
            pieces = chunk(doc, settings.chunk_size, settings.chunk_overlap)
            log("chunked", source=path.name, chunks=len(pieces))
            all_chunks.extend(pieces)
        vectors = active_embed(all_chunks, settings)
        return active_store(all_chunks, vectors, settings)

    if chroma_client is None:
        try:
            chroma_client = get_client(settings)
        except Exception:  # noqa: BLE001
            chroma_client = None

    existing_meta = load_existing_meta(chroma_client, settings.collection)
    Path(settings.lake_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.quarantine_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(settings.verify_report_path) / f"verify-{_utc_now_iso()}.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    passed, skipped, quarantined, chunks_stored = 0, 0, 0, 0
    for path, doc_version, superseded in documents:
        doc = active_extract(str(path), doc_version, superseded)
        vr = verify_document(doc, existing_meta, settings.verify_near_dup_threshold)
        _append_jsonl(report_path, vr.model_dump())
        if not vr.passed:
            # Legit prior-quarantine / failed-doc handling: move to quarantine and skip indexing.
            if path.exists():
                shutil.move(str(path), str(Path(settings.quarantine_dir) / path.name))
            quarantined += 1
            continue
        if vr.skipped:
            skipped += 1
            continue

        cleaned = clean(doc)
        cleaned.content_hash = doc.content_hash
        cleaned.title = doc.title
        cleaned.page_count = doc.page_count

        pieces = chunk(cleaned, settings.chunk_size, settings.chunk_overlap)
        for piece in pieces:
            piece.content_hash = doc.content_hash
            piece.title = doc.title
            piece.page_count = doc.page_count
        log("chunked", source=path.name, chunks=len(pieces))

        if pieces:
            vectors = active_embed(pieces, settings)
            stored = active_store(pieces, vectors, settings)
            chunks_stored += stored

        # Record in existing_meta ONLY after verification and chunk/embed/store succeed.
        passed += 1
        existing_meta[vr.doc_id] = _doc_meta(doc, vr.doc_id)

    log(
        "verify_summary",
        total=len(documents),
        passed=passed,
        skipped=skipped,
        quarantined=quarantined,
        chunks_stored=chunks_stored,
        report=str(report_path),
    )
    return chunks_stored


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the RAHBAR index, then exit.")
    parser.add_argument("--dummy", action="store_true", help="write three fake chunks")
    args = parser.parse_args()
    settings = get_settings()
    log("indexer starting", dummy=args.dummy, embed_model=settings.embed_model,
        collection=settings.collection, chroma=f"{settings.chroma_host}:{settings.chroma_port}",
        pod=os.environ.get("HOSTNAME", "-"))
    started = time.perf_counter()
    try:
        if args.dummy:
            from .dummy import write_dummy_index
            from .store import get_client

            count = write_dummy_index(settings, get_client(settings))
        else:
            count = run_pipeline(settings)
    except NotImplementedError as exc:
        log("indexer stage not implemented yet", error=str(exc))
        return 2
    log("indexer finished", chunks=count, seconds=round(time.perf_counter() - started, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
