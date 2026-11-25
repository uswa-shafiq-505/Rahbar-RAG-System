# Evaluation Golden Sets (RAHBAR / Assignment 03)

## Relationship Between `golden.yaml` and `golden.jsonl`

- `eval/golden.yaml` is the **Assignment 02** golden set (15 questions covering ordinary, version-sensitive, exact-term, must-abstain, and prompt-injection checks) used by `eval/run_eval.py`. It remains untouched to preserve A2 behavior and CI compatibility.
- `eval/golden.jsonl` is the **Assignment 03** source of truth (`>= 25` questions once Batch B is added), formatted as newline-delimited JSON with explicit provenance labels (`human` | `synthetic`), review status (`reviewed`), difficulty, and 1-based PDF page ground truth (`extract.py` / Chroma page numbering).

Both files coexist in `eval/`.

## Validation

Run the validator from the repository root:

```bash
python -m eval.validate_golden
```

It verifies JSON formatting, required schema keys, non-null `ground_truth.source` and `page` on `answerable` rows, `must_abstain=true` and `ground_truth=null` on `unanswerable` rows, and prints a count breakdown table.

## Current Question-Type Breakdown

| Type | Current Count | Target (A3 Final) | Notes |
| --- | ---: | ---: | --- |
| `answerable` | 15 | >= 15 | Covers all 7 extractable corpus docs (`academic-rules-handbook-jun-2026.pdf`, `prospectus-2026-academic-rules-and-regulations.pdf`, `student-handbook-jun-2026.pdf`, `fee-structure-page.html`, `grading-page.html`, `annual-report-2019-20.pdf`, `annual-report-2023-24.pdf`) |
| `unanswerable` | 5 | >= 5 | Out-of-corpus topics verified absent (library timings, hostel fees, parking permits/fees, parent portal, convocation fee/gown rental) |
| `hard` | 0 | >= 5 | Pending Batch B ingestion |
| **Total** | **20** | **>= 25** | All 20 initial drafts marked `label="synthetic"`, `reviewed=false` pending hand review |

# TODO: add >=5 hard questions from Batch B

## Corpus PDFs Without a Usable Text Layer (Skipped)

During `pypdf` inspection of `corpus/raw/`, the following PDFs lacked a usable text layer (Part A scanned-PDF verification issue) and were skipped when drafting questions:

- `corpus/raw/faculty-entrepreneurship-and-technical-start-ups-policy.pdf` — 8 pages, 0 extracted characters across all pages (fully image-only / scanned).
- `corpus/raw/undergraduate-academic-rules-aug-2023.pdf` — 26 pages, only 212 extracted characters across 2 pages (cover/front matter only; policy body pages are scanned images with no text layer).

