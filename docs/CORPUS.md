# Corpus

Source: FAST-NUCES public website (nu.edu.pk) · Documents: 9 · Pages: 279 (estimated)
Fetched: 2026-11-06 · Provenance manifest: [corpus/MANIFEST.md](../corpus/MANIFEST.md) · Fetch script: [corpus/fetch.py](../corpus/fetch.py)

Raw files live in `corpus/raw/` (gitignored). Re-download with `python corpus/fetch.py`, then verify with `python corpus/fetch.py --verify`.

---

## Required Properties (§2)

| # | Required property | Document(s) | Evidence / notes |
|---|---|---|---|
| 1 | ≥ 6 documents, ≥ 40 pages | All 9 files in `corpus/raw/` — see list below | 9 documents; page counts: handbook 58 pp, 2023 rules 26 pp, prospectus 5 pp, student handbook 64 pp, annual reports 35 + 86 pp, fee HTML 1 pp, grading HTML 1 pp, policy PDF 8 pp → ≥ 279 pages total |
| 2 | Two versions of one policy | `undergraduate-academic-rules-aug-2023.pdf` (superseded) vs `academic-rules-handbook-jun-2026.pdf` (current) | Both indexed with chunk metadata fields `doc_version` and `superseded: true/false`; the retriever's `_asks_about_old_version` check bypasses the superseded filter only when the question explicitly names the old rules |
| 3 | Scanned, image-only PDF | `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` | `pypdf` extracts **0 characters** across all 8 pages (`total_chars=0`); no font table present — confirmed image-only. The extract stage detects this and returns an empty Document so the pipeline skips it gracefully rather than crashing |
| 4 | Document with real tables | `fee-structure-page.html`, `grading-page.html`, `annual-report-2019-20.pdf`, `annual-report-2023-24.pdf` | Fee page contains HTML `<table>` rows of per-credit-hour tuition; grading page has the full A+ → F grade-point table; annual reports contain programme-enrolment and financial tables parsed by BeautifulSoup in the HTML extract path |
| 5 | Long definitions | `student-handbook-jun-2026.pdf` (64 pages) | Contains multi-page definitions of "academic dishonesty", the generative-AI acceptable-use policy (§4–§12, ≈ 15 pages), Code of Conduct, Financial Rules, and Disciplinary Conduct — rich source of boundary-testing definitions |
| 6 | Deliberately absent topic | Hostel allocation criteria and campus bus / shuttle routes | The document `Transport and Hostel Facilities.pdf` (linked from nu.edu.pk/Admissions) is explicitly **commented out** in `corpus/sources.yaml`. Golden questions **q13** and **q14** are `must_abstain` entries that verify the system declines to answer rather than hallucinating |
| 7 | Exact-term content | `academic-rules-handbook-jun-2026.pdf`, `fee-structure-page.html`, `grading-page.html` | Clause **1.21** (100 % attendance mandate), clause **5.4** (CGPA 2.00 warning threshold), grade-point values (A = 4.00, A– = 3.70 … F = 0.00), and per-credit-hour tuition fees are all present verbatim and tested by golden questions q11, q12, q06 |

---

## Document List

| File | Version | Superseded | Description |
|---|---|---|---|
| `undergraduate-academic-rules-aug-2023.pdf` | 2023-08 | **yes** | Old undergraduate rules — attendance, grading, repetition policies pre-2026 |
| `academic-rules-handbook-jun-2026.pdf` | 2026-06 | no | Current 58-page handbook — definitive source for all academic rules |
| `prospectus-2026-academic-rules-and-regulations.pdf` | 2026 | no | Compact 5-page prospectus summary with its own clause numbering |
| `student-handbook-jun-2026.pdf` | 2026-06 | no | 64-page student handbook — conduct, financial rules, generative-AI policy |
| `annual-report-2019-20.pdf` | 2019-20 | no | Annual report with enrolment and programme tables |
| `annual-report-2023-24.pdf` | 2023-24 | no | Annual report with enrolment and programme tables |
| `fee-structure-page.html` | 2026-27 | no | HTML fee-structure page with per-credit-hour tuition table |
| `grading-page.html` | 2026 | no | HTML grading page with full letter-grade / grade-point table |
| `faculty-entrepreneurship-and-technical-start-ups-policy.pdf` | 2022-05 | no | **Image-only** (scanned) — 8 pages, 0 extractable characters; pipeline skips gracefully |

---

## Outdated-Source Trap

Both the August 2023 rules and the June 2026 handbook are indexed. Every chunk carries `doc_version` and `superseded` metadata. The ChromaRetriever applies a `where: {superseded: false}` filter by default; it removes the filter only when a question contains explicit markers such as *"2023 rules"*, *"old rules"*, or *"August 2023"* — preventing the retriever from silently returning stale policy.

## What Is NOT in the Corpus

- **Hostel allocation and curfew rules** — covered only in the excluded `Transport and Hostel Facilities` prospectus.
- **Campus bus / shuttle timetables** — same excluded document.
- **Postgraduate / PhD regulations** — not fetched; only undergraduate rules are present.
- **Individual campus fee breakdowns** — only the central fee-structure page is included.
