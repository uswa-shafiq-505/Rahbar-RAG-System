"""Stage 3: Chunk. Structure-aware with overlap."""

from __future__ import annotations

import hashlib
import logging
import re

from .models import Chunk, Document

log = logging.getLogger(__name__)


def _is_heading(
    line: str,
    prev_blank: bool = False,
    next_blank: bool = False,
    next_is_heading: bool = False,
) -> bool:
    """Strict heading detection heuristic (assignment §3.1)."""
    stripped = line.strip()
    if not stripped or len(stripped) >= 100:
        return False
    if "|" in stripped or stripped.startswith(("-", "*", "•")):
        return False
    # Not more than 15% of the line is digits
    num_digits = sum(1 for c in stripped if c.isdigit())
    if num_digits / len(stripped) > 0.15:
        return False
    # Does not end with a period followed by lowercase word
    if re.search(r"\.\s+[a-z]", stripped):
        return False

    content = re.sub(r"^#+\s*", "", stripped).strip()
    if not content:
        return False

    # 4a: Numbered section like "5.4 ", "1.21 " (must have trailing space or dot)
    m_4a = bool(re.match(r"^\d+(\.\d+)+[\s.]", content))
    if m_4a:
        after_num = re.sub(r"^\d+(\.\d+)+[\s.]+", "", content).strip()
        words = after_num.split()
        if len(words) > 7 or stripped.endswith("."):
            m_4a = False
        if words and not words[0][0].isupper():
            m_4a = False

    # 4b: Literal CHAPTER, Section, PART
    m_4b = bool(re.match(r"^(CHAPTER|Section|PART)\s+[IVX0-9]+", content, re.I))

    # 4c: Line is ALL-CAPS (>= 80% uppercase), length 5 to 60, no lowercase words, at least 2 words
    alphas = [c for c in content if c.isalpha()]
    m_4c = False
    if alphas and 5 <= len(content) <= 60 and not any(c.islower() for c in content):
        if len(content.split()) >= 2 and sum(1 for c in alphas if c.isupper()) / len(alphas) >= 0.8:
            if not re.search(r"\b(AND|OR|OF|FOR|TO|IN|WITH|THE|BY|AT)$", content, re.I):
                m_4c = True

    # Markdown headings
    m_md = bool(re.match(r"^#{1,6}\s+", stripped))

    # Single-number headings like "1. Introduction"
    m_single = bool(re.match(r"^\d+\.\s+[A-Z]", content))
    if m_single:
        after_num = re.sub(r"^\d+\.\s+", "", content).strip()
        words = after_num.split()
        if len(words) > 7 or stripped.endswith("."):
            m_single = False

    matched_pattern = m_4a or m_4b or m_4c or m_md or m_single
    if not matched_pattern:
        return False

    if m_4c and not (next_blank or next_is_heading or len(content) < 60):
        return False

    # Rule 5: Followed by blank line, OR next line is heading, OR line is short (< 60 chars)
    if next_blank or next_is_heading or len(content) < 60:
        return True
    return False


def _clean_heading(line: str) -> str:
    """Return clean heading title stripped of markdown hashes and whitespace."""
    return re.sub(r"^#+\s*", "", line.strip()).strip()


def _split_into_blocks(page_text: str, max_block_words: int = 400) -> list[str]:
    """Split page text on 2+ newlines and isolate heading lines."""
    raw_blocks = re.split(r"\n{2,}", page_text.strip())
    blocks: list[str] = []
    for rb in raw_blocks:
        lines = [ln.strip() for ln in rb.split("\n") if ln.strip()]
        if not lines:
            continue
        curr_lines: list[str] = []
        for j, ln in enumerate(lines):
            pb = j == 0
            nb = j == len(lines) - 1
            if _is_heading(ln, prev_blank=pb, next_blank=nb):
                if curr_lines:
                    blocks.append("\n".join(curr_lines))
                    curr_lines = []
                blocks.append(ln)
            else:
                curr_lines.append(ln)
        if curr_lines:
            blocks.append("\n".join(curr_lines))

    # Decompose any block that exceeds max_block_words on sentence / word boundaries
    final_blocks: list[str] = []
    for b in blocks:
        if not b.strip():
            continue
        if len(b.split()) <= max_block_words:
            final_blocks.append(b)
        else:
            sentences = [s.strip() for s in re.split(r"(?<=[.?!])(?:\s+|\n+)", b) if s.strip()] or [b]
            for s in sentences:
                s_words = s.split()
                if len(s_words) <= max_block_words:
                    final_blocks.append(s)
                else:
                    for i in range(0, len(s_words), max_block_words):
                        final_blocks.append(" ".join(s_words[i : i + max_block_words]))
    return final_blocks


def _carry_overlap(buffer: list[tuple[str, int, str]], overlap: int) -> list[tuple[str, int, str]]:
    """Extract the last `overlap` words from the buffer to prepend to the next chunk."""
    if overlap <= 0 or not buffer:
        return []
    overlap_items: list[tuple[str, int, str]] = []
    accum_words = 0
    for text, page, sec in reversed(buffer):
        w = len(text.split())
        if accum_words + w <= overlap:
            overlap_items.append((text, page, sec))
            accum_words += w
        else:
            needed = overlap - accum_words
            if needed > 0:
                words = text.split()
                tail_words = words[-needed:]
                overlap_items.append((" ".join(tail_words), page, sec))
            break
    overlap_items.reverse()
    return overlap_items


def chunk(doc: Document, chunk_size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Structure-aware chunking with overlap (Stage 3).

    Hierarchy: section boundary -> paragraph boundary -> sentence boundary -> word boundary.
    """
    if not doc.pages:
        return []

    chunks: list[Chunk] = []
    buffer: list[tuple[str, int, str]] = []
    buffer_words = 0
    chunk_index = 0

    page1_words = len(doc.pages[0].text.split()) if doc.pages else 0
    current_section = "Cover" if page1_words < 20 else "Preamble"

    def emit_chunk(buf: list[tuple[str, int, str]], sec: str) -> None:
        nonlocal chunk_index
        if not buf:
            return
        text = "\n\n".join(t for t, p, s in buf).strip()
        words = text.split()
        if len(words) < 15:
            return
        start_page = buf[0][1]
        cid = hashlib.sha256(f"{doc.source}|{sec}|{start_page}|{chunk_index}".encode("utf-8")).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=cid,
                text=text,
                source=doc.source,
                section=sec,
                page=start_page,
                doc_version=doc.doc_version,
                superseded=doc.superseded,
            )
        )
        chunk_index += 1

    for p_idx, page in enumerate(doc.pages):
        raw_text = page.text
        if p_idx == 0 and raw_text.startswith("---"):
            m = re.match(r"^---\n.*?\n---\n?", raw_text, re.DOTALL)
            if m:
                raw_text = raw_text[m.end() :]

        if p_idx > 0 and current_section == "Cover":
            current_section = "Preamble"

        blocks = _split_into_blocks(raw_text, chunk_size)
        for block in blocks:
            if _is_heading(block, prev_blank=True, next_blank=True):
                if buffer_words >= chunk_size * 0.5:
                    emit_chunk(buffer, current_section)
                    buffer = _carry_overlap(buffer, overlap)
                    buffer_words = sum(len(t.split()) for t, p, s in buffer)
                current_section = _clean_heading(block)[:80]
                buffer.append((block, page.page, current_section))
                buffer_words += len(block.split())
                continue

            b_words = len(block.split())
            if buffer_words + b_words > chunk_size * 1.2:
                emit_chunk(buffer, current_section)
                buffer = _carry_overlap(buffer, overlap)
                buffer_words = sum(len(t.split()) for t, p, s in buffer)

            buffer.append((block, page.page, current_section))
            buffer_words += b_words

    if buffer:
        emit_chunk(buffer, current_section)

    log.info("chunked %s: %d chunks across %d sections", doc.source, len(chunks), len({c.section for c in chunks}))
    return chunks
