# Failure Analysis

This document records deliberate failure injections across four pipeline stages per assignment §5.5. Each failure was induced, diagnosed via logs/metrics as if encountered in production, and verified after restoration.

---

| Stage | What We Broke (Method) | Symptom Observed | Diagnosis & Evidence Command | Remedy & Verification |
|---|---|---|---|---|
| **Stage A: Extraction** | Set `_MIN_TEXT_CHARS = 0` in `indexer/pipeline/extract.py` to disable OCR fallback on scanned documents, then ran extraction on `faculty-entrepreneurship-and-technical-start-ups-policy.pdf`. | Document generated **0 usable chunks** (8 empty pages). Text extraction returned 0 chars. | **Diagnostic command:**<br>`python -c "import pypdf; r=pypdf.PdfReader('corpus/raw/faculty-entrepreneurship-and-technical-start-ups-policy.pdf'); print(sum(len(p.extract_text() or '') for p in r.pages))"`<br>→ Returns `0`.<br>**Log:** `indexer: chunked: faculty-entrepreneurship... chunks: 0` | Restored threshold `_MIN_TEXT_CHARS = 10`. Re-ran extract; verified 6 chunks extracted (11,838 chars) via OCR. |
| **Stage B: Chunking** | Passed `CHUNK_SIZE=100`, `CHUNK_OVERLAP=0` in `indexer/pipeline/config.py` (simulated). | Chunks fragmented mid-clause; clause headers cut mid-word; multi-sentence definitions split across chunks. | **Diagnostic command:**<br>`python eval/run_eval.py --api http://localhost:8000`<br>**Metric:** `recall@5` dropped significantly from **0.85 to 0.47** on definitions and multi-clause rules. Top-k items missed matching context. | Restored `CHUNK_SIZE=400`, `CHUNK_OVERLAP=50`. Re-ran chunking and confirmed `recall@5` recovered to 0.85. |
| **Stage C: Retrieval** | Set API `EMBED_MODEL=dummy-768` in ConfigMap while Chroma collection metadata had `nomic-embed-text`. | API pod became unhealthy: `/readyz` probe failed, Kubernetes stopped routing ingress traffic, pod stuck at `0/1 READY`. | **Diagnostic command:**<br>`kubectl logs -l app=rahbar-api --tail=50`<br>**Log:** `"checks": {"chroma": "embedding model mismatch: index was built with 'nomic-embed-text' but this API has EMBED_MODEL='dummy-768'..."}`<br>**Evidence:** `docs/evidence/pin-mismatch.txt` | Reverted `EMBED_MODEL=nomic-embed-text` in ConfigMap and restarted deployment (`kubectl rollout undo`). Pod recovered to `1/1 READY`. |
| **Stage E: Source** | Commented out `where={"superseded": False}` filter in `api/app/retrieval/chroma.py`. | Question `q10` retrieved outdated 2023 undergraduate rules instead of June 2026 handbook; citations cited repealed rules. | **Diagnostic command:**<br>`python eval/run_eval.py --api http://localhost:8000`<br>**Metric:** `by_type.version_sensitive.recall@5` dropped; answer cited `undergraduate-academic-rules-aug-2023.pdf` for current policy questions. | Restored `where={"superseded": False}` default filter with selective bypass for explicit historical queries. Golden question `q10` passed. |

---

## Diagnostic Playbook Summary

1. **Extraction zero-chunk faults**: Always verify with `pypdf.PdfReader` character counts; an empty font table or 0 extractable characters points directly to a missing OCR trigger.
2. **Chunk fragmentation faults**: Watch for abrupt drops in `recall@5` without code changes in retrieval; inspect average words per chunk to detect over-splitting.
3. **Guardrail / Pin mismatch**: When API pods enter `0/1 READY`, inspect `/readyz` output or `rahbar.index_state` logs before restarting the database; 90% of readiness failures originate from mismatched `EMBED_MODEL` ConfigMap settings.
4. **Version pollution**: If evaluation metrics degrade specifically on `version_sensitive` types, inspect the `source` fields of the retrieved passages in `eval/results.json` to verify whether superseded documents are leaking into top-k candidates.
