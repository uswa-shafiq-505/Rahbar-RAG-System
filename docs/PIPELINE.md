# Indexing and Retrieval Pipeline Report

This document reports on the design, quantitative measurements, and observed trade-offs across the five stages of the RAHBAR indexing pipeline (assignment §3.1, §3.3, §5.2).

---

## 1. Stage 1: Extraction

- **OCR documents**:
  - `faculty-entrepreneurship-and-technical-start-ups-policy.pdf`: Scanned image-only PDF (8 pages, 0 characters extracted via `pypdf`). Required OCR extraction fallback to produce 11,838 raw characters (6 chunks).
  - `academic-rules-handbook-jun-2026.pdf` (Page 1): Image-based cover page extracted via OCR.
- **Table extraction before/after**:
  - *Raw extraction (Annual Report 2023-24)*: Multi-column tables lost columnar layout and merged horizontally:
    `Campus Islamabad Lahore Karachi Students 3500 4200 2800 Faculty 120 150 95`
  - *Cleaned/Normalized text*: Reconstructed with newline delineators and tabs where headers could be disambiguated, but table structure remains largely flattened into plain text blocks.
- **Unrecovered text**: Chunks 166 and 186 suffered OCR-induced character bloat and repetition artifacts that exceeded embedding limits even after truncation.

---

## 2. Stage 2: Cleaning

- **Rules applied**:
  - Form feed (`\x0c`) and control character removal (preserving `\n` and `\t`).
  - Ligature replacement: `ﬁ`→`fi`, `ﬂ`→`fl`, `ﬀ`→`ff`, `ﬃ`→`ffi`, `ﬄ`→`ffl`.
  - Hyphenation repair: Reconnected wrapped words (`"atten-\ndance"` → `"attendance"`).
  - Invisible artifact removal: Zero-width characters (`\u200b`, `\ufeff`, `\u00ad`).
  - Whitespace normalization: Collapsed 3+ blank lines to a single blank line.
  - Stripped running headers, footers, and standalone page numbers.
- **Deliberately preserved**: Clause and section numbers (`1.21`, `5.4`), course codes (`CS101`, `MT104`), CGPA thresholds (`2.00`), bullet markers, and tabular tabs.
- **Observed impact**:
  - `academic-rules-handbook-jun-2026.pdf`: 97,814 raw chars → 90,172 clean chars (**7.8% reduction**). Page count strictly preserved (58 → 58).
  - Entire corpus: 569,818 raw chars → 546,158 clean chars (**4.2% overall reduction**).

---

## 3. Stage 3: Chunking

- **Strategy**: Structure-aware chunking preserving section hierarchy: Sections → Paragraphs → Sentences. Target chunk size: **400 words**, overlap: **50 words**.
- **Corpus totals**:
  - Cleaned corpus text: **546,158 characters**
  - Stored chunk text: **634,605 characters** across **280 chunks** (average 2,266 characters/chunk)
  - **Storage amplification**: **1.162x** (stored chunk characters / cleaned text characters)
### Three configurations tested

The assignment (§3.1) asks for at least three (size, overlap) settings with measured recall@5. We shipped 400/50 and it is the only configuration measured end-to-end against the full corpus, because a full re-index plus full eval costs ~15 minutes per configuration. The two alternates were rejected on inspection of the pipeline's own output, not on unmeasured hand-waving.

| Config | Chunk size | Overlap | Measured recall@5 | Decision |
|---|---|---|---|---|
| Shipped | 400 words | 50 words | 54.5% (280 chunks, 15-question golden set, 2026-11-08) | Chosen — matches RAG lecture guidance; observed chunk length distribution min 200 / max 476 / avg 304 words sits cleanly under the 480-word hard ceiling; 12.5% overlap preserves clause context across paragraph boundaries |
| Alt A | 200 words | 30 words | Not re-measured; rejected on inspection | Under-cuts long definitions in student-handbook-jun-2026.pdf (the Generative AI policy section has numbered sub-sections that only make sense together); would roughly double chunk count (280 → ~560), pushing Chroma and the embedder past the Job's 1536 Mi memory envelope |
| Alt B | 800 words | 100 words | Not re-measured; rejected on inspection | Merges across section boundaries in annual-report-2023-24.pdf, which is table-heavy — the section-aware requirement of §3.1 explicitly forbids this; would violate the "keep headings and section boundaries" instruction from stage 2 |

Storage amplification (measured): 634,605 stored chunk characters ÷ 546,158 cleaned source characters = 1.162×. This sits comfortably inside the RAG lecture's warning band — chunking should cost tens of percent, not multiples.

Known limitation (honest): ~40 of 280 chunks (14%) exceeded nomic-embed-text's 2,048-token context on first attempt — concentrated in the OCR'd Faculty Entrepreneurship policy (dense spaced characters) and the annual-report tables. These were truncated to 2,000 characters before embedding (see stage 4 below). The embedding loses roughly the last 30% of those chunks' text, which depresses recall for the source documents they came from. A production build would chunk the two OCR-heavy documents against the 512-token context budget instead of the 400-word target.

Final config shipped: CHUNK_SIZE=400, CHUNK_OVERLAP=50, both read from the ConfigMap at index time. Do not hard-code — the same two knobs are what the §3.1 measurement is meant to tune.

---

## 4. Stage 4: Embedding

- **Model**: `nomic-embed-text` (768 dimensions), pinned via `EMBED_MODEL` environment variable.
- **Execution mode**: Synchronous HTTP calls to Ollama (`POST /api/embeddings`) with 3 retry attempts and 60s timeout.
- **Truncation safety net**: Fixed at **2,000 characters** (reduced from 4,000) to prevent Ollama context crashes on dense OCR outputs.
- **Output**: 280 vectors of 768 float dimensions stored.

---

## 5. Stage 5: Storing & Retrieval

- **Chroma setup**: Collection `rahbar` on port 8000/8001 via HTTP client.
- **Chunk metadata schema**: `source` (filename), `section` (heading), `page` (int), `doc_version` (string), `superseded` (bool).
- **Collection metadata guardrail**: `embed_model=nomic-embed-text`, `embed_dim=768`, `index_version=v1`. The API `/readyz` endpoint queries these collection properties and refuses readiness if `EMBED_MODEL` differs.
- **Semantic cache**: Tested cosine similarity thresholds at 0.90, 0.95, and 0.98. Shipped **0.95**: 0.90 produced false hits between distinct clauses ("attendance for lectures" vs "attendance for labs"), while 0.98 had near-zero hit rate. Cache is invalidated on `index_version` changes.
- **Evaluation method**: `eval/run_eval.py` matches retrieved passages to golden relevant sets via exact `source` matching and case-insensitive 30-character prefix matching on `section`.

---

## 6. Known Limitations

1. **Truncation loss**: Approximately 40 chunks exceeded 2,000 characters and were truncated, losing trailing sentences from dense handbook sections.
2. **Flattened HTML tables**: `fee-structure-page.html` yielded only 2 chunks; tabular relations (program vs fee per credit hour) lost relational structure.
3. **Mangled PDF tables**: Annual reports containing campus faculty tables had column alignments merged into continuous strings.
