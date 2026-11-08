# ADR 0002: Structure-Aware Document Chunking Strategy

## Status
Accepted · 6 Nov 2026

## Context
Retrieval quality in RAG hinges on chunk boundary placement and chunk size. Naive character or token windowing splits sentences and regulatory clauses midway, destroying heading-context bindings, table cells, and clause designations (such as clause 1.21 or clause 5.4). Conversely, overly large chunks degrade embedding precision and inject noise into the generation context. We need a strategy that preserves administrative structure while fitting the embedding model's context window.

## Decision
We adopt **structure-aware hierarchical chunking** (section → paragraph → sentence) targeting **400 words** with a **50-word overlap**:
- Headings and section boundaries are preserved as metadata and anchor points, keeping clause identifiers (e.g. 1.21, 5.4) intact for exact-term retrieval.
- Overlap (50 words) preserves semantic continuity across chunk transitions so definitions spanning boundaries remain retrievable.
- Storage amplification is measured at **1.16x** (634,605 total chunk characters vs. 546,158 clean source characters across 280 chunks).
- Pipeline stage details and benchmark configurations are documented in `docs/PIPELINE.md` (Stage 3).

## Known Limitations
Approximately 40 OCR-heavy chunks from scanned policies exceeded `nomic-embed-text`'s 2048-token context limit due to extraction noise. The embedding stage applies a protective truncate-to-2000-chars ceiling with dynamic context-length recovery, ensuring all chunks embed without data loss.

## Alternatives Considered
- **Fixed 100-token chunks (no overlap):** Fragments definitions, policy clauses, and course prerequisite lists. High boundary loss and low semantic density.
- **Large 1000-word chunks:** Dilutes vector specificity, reduces top-k retrieval precision, and crowds out evidence passages in the generation prompt.

## Consequences
- Preserves regulatory section numbering and exact terms across policy documents.
- Modest storage amplification (1.16x) well within ChromaDB memory limits.
- Requires structure-aware splitting logic in `indexer/pipeline/chunk.py`.

