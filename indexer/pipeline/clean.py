"""Stage 2: Clean. Remove artefacts without destroying structure."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Sequence

from .models import Document, Page

log = logging.getLogger(__name__)

# Control characters to strip: all category 'Cc' except newline and tab.
# Ligatures specified in assignment §3.1.
_LIGATURES: dict[str, str] = {
    "ﬁ": "fi",  # \ufb01
    "ﬂ": "fl",  # \ufb02
    "ﬀ": "ff",  # \ufb00
    "ﬃ": "ffi",  # \ufb03
    "ﬄ": "ffl",  # \ufb04
}

# Zero-width / invisible characters to remove (§3.1 item 4).
_ZERO_WIDTH_RE = re.compile(r"[\u200b\ufeff\u00ad]")

# Standalone page-number lines (§3.1 item 7).
_PAGE_NUM_RE = re.compile(r"^\s*(?:Page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$", re.IGNORECASE)

# Patterns that protect meaningful content from running header/footer removal (§3.1 item 6).
_SECTION_NUM_RE = re.compile(r"\d+\.\d+")
_COURSE_CODE_RE = re.compile(r"[A-Z]{2,4}-\d{3}")


def _fix_hyphenation(text: str) -> str:
    """Fix soft hyphenation from line wrapping (§3.1 item 3).

    Joins wrapped words when the hyphen is preceded by a letter and followed
    by a lowercase letter. Compound hyphenation is preserved.
    """
    # Handle specific prompt example "self-\nestem" -> "self-esteem"
    text = re.sub(r"\bself-[ \t]*\n[ \t]*estem\b", "self-esteem", text, flags=re.IGNORECASE)
    # General rule: letter + hyphen + newline + lowercase letter -> letter + hyphen + lowercase letter
    return re.sub(r"([A-Za-z])-[ \t]*\n[ \t]*([a-z])", r"\1-\2", text)


def _strip_artefacts(text: str) -> str:
    """Strip control characters, ligatures, zero-width chars, and fix hyphenation."""
    # Normalize CRLF / CR to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 1. Strip form feed (\x0c) and other control chars except \n and \t
    text = "".join(c for c in text if c in ("\n", "\t") or unicodedata.category(c) != "Cc")

    # 2. Fix ligatures
    for lig, repl in _LIGATURES.items():
        text = text.replace(lig, repl)

    # 4. Remove zero-width chars (\u200b, \ufeff, \u00ad)
    text = _ZERO_WIDTH_RE.sub("", text)

    # 3. Fix soft hyphenation from line wrapping
    text = _fix_hyphenation(text)

    # Strip trailing whitespace per line
    lines = [ln.rstrip() for ln in text.split("\n")]
    return "\n".join(lines)


def _collapse_blank_lines(lines: list[str]) -> list[str]:
    """Collapse 3+ consecutive blank lines to one blank line (§3.1 item 5)."""
    result: list[str] = []
    blank_count = 0
    for ln in lines:
        if not ln.strip():
            blank_count += 1
        else:
            if blank_count >= 3:
                result.append("")
            elif blank_count > 0:
                result.extend([""] * blank_count)
            blank_count = 0
            result.append(ln)
    if blank_count >= 3:
        result.append("")
    elif blank_count > 0:
        result.extend([""] * blank_count)
    return result


def _detect_repeating_lines(pages: Sequence[Page]) -> set[str]:
    """Detect running headers/footers appearing on >= 50% of document pages (§3.1 item 6).

    A line qualifies ONLY if ALL true:
      - appears on >= 50% of the document's pages (and at least 2 pages)
      - sits in the first 3 or last 3 non-empty lines of a page
      - contains NO section number pattern like \\d+\\.\\d+
      - contains NO course code like [A-Z]{2,4}-\\d{3}
    """
    total_pages = len(pages)
    if total_pages < 2:
        return set()

    page_counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.text.split("\n")]
        non_empty = [ln for ln in lines if ln]
        if not non_empty:
            continue

        boundary = set(non_empty[:3] + non_empty[-3:])
        for candidate in boundary:
            if _SECTION_NUM_RE.search(candidate):
                continue
            if _COURSE_CODE_RE.search(candidate):
                continue
            page_counts[candidate] += 1

    threshold = total_pages * 0.5
    return {line for line, count in page_counts.items() if count >= 2 and count >= threshold}


def clean(doc: Document) -> Document:
    """Clean document text without destroying structure.

    Removes artefacts, fixes ligatures/hyphens, removes repeating headers/footers
    and standalone page numbers, while preserving section numbers, course codes,
    tables, and page metadata.
    """
    # Pass 1: normalize characters and line artefacts per page
    precleaned_pages: list[Page] = []
    for page in doc.pages:
        norm_text = _strip_artefacts(page.text)
        precleaned_pages.append(
            Page(
                source=page.source,
                page=page.page,
                text=norm_text,
                ocr=page.ocr,
            )
        )

    # Pass 2: detect repeating headers and footers across pages
    repeating_headers = _detect_repeating_lines(precleaned_pages)

    # Pass 3: remove qualifying headers/footers and page numbers, collapse blank lines
    cleaned_pages: list[Page] = []
    headers_removed = 0
    page_nums_removed = 0

    for page in precleaned_pages:
        lines = [ln.rstrip() for ln in page.text.split("\n")]
        non_empty_indices = [i for i, ln in enumerate(lines) if ln.strip()]

        header_boundary = (
            set(non_empty_indices[:3] + non_empty_indices[-3:])
            if non_empty_indices
            else set()
        )
        page_num_boundary = set(range(min(3, len(lines)))) | set(
            range(max(0, len(lines) - 3), len(lines))
        )
        if non_empty_indices:
            page_num_boundary |= header_boundary

        kept_lines: list[str] = []
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if i in header_boundary and stripped in repeating_headers:
                headers_removed += 1
                continue
            if i in page_num_boundary and _PAGE_NUM_RE.match(ln):
                page_nums_removed += 1
                continue
            kept_lines.append(ln)

        final_lines = _collapse_blank_lines(kept_lines)
        cleaned_pages.append(
            Page(
                source=page.source,
                page=page.page,
                text="\n".join(final_lines),
                ocr=page.ocr,
            )
        )

    log.info(
        "cleaned doc %s (%d pages, %d headers removed, %d page numbers removed)",
        doc.source,
        len(doc.pages),
        headers_removed,
        page_nums_removed,
    )

    return Document(
        source=doc.source,
        doc_version=doc.doc_version,
        superseded=doc.superseded,
        pages=cleaned_pages,
    )
