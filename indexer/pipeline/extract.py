"""Stage 1: Extract. PDF (or HTML, or Markdown) to text, page by page."""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path

from pypdf import PdfReader

from .models import Document, Page

log = logging.getLogger(__name__)

# Below this many non-whitespace characters, treat a page as image-only and OCR it.
_MIN_TEXT_CHARS = 20
_OCR_DPI = 200


class _HTMLTextParser(HTMLParser):
    """Strip tags, keep visible text, insert paragraph breaks at block elements."""

    _SKIP = {"script", "style", "noscript", "head", "nav", "footer"}
    _BREAK = {
        "p", "div", "br", "li", "tr", "td", "th", "table", "section", "article",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._pieces: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BREAK:
            self._pieces.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BREAK:
            self._pieces.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def text(self) -> str:
        raw = "".join(self._pieces)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.splitlines()]
        return "\n".join(ln for ln in lines if ln)


def _ocr_one_page(path: Path, page_number: int) -> str:
    """OCR a single PDF page. Returns '' if tesseract or poppler is missing."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception as exc:  # pragma: no cover
        log.warning("OCR unavailable (%s); page %d of %s stays empty", exc, page_number, path.name)
        return ""
    try:
        images = convert_from_path(
            str(path), dpi=_OCR_DPI, first_page=page_number, last_page=page_number
        )
    except Exception as exc:
        log.warning("PDF→image failed for %s page %d: %s", path.name, page_number, exc)
        return ""
    if not images:
        return ""
    try:
        return (pytesseract.image_to_string(images[0]) or "").strip()
    except Exception as exc:
        log.warning("tesseract failed on %s page %d: %s", path.name, page_number, exc)
        return ""


def _extract_pdf(path: Path) -> list[Page]:
    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as exc:
            log.warning("pypdf failed on %s page %d: %s", path.name, i + 1, exc)
            text = ""
        if len(text) < _MIN_TEXT_CHARS:
            ocr_text = _ocr_one_page(path, i + 1)
            pages.append(Page(source=path.name, page=i + 1, text=ocr_text, ocr=bool(ocr_text)))
        else:
            pages.append(Page(source=path.name, page=i + 1, text=text, ocr=False))
    return pages


def _extract_html(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parser = _HTMLTextParser()
    parser.feed(raw)
    return [Page(source=path.name, page=1, text=parser.text(), ocr=False)]


def _extract_plain(path: Path) -> list[Page]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Page(source=path.name, page=1, text=text, ocr=False)]


def extract(path: str, doc_version: str, superseded: bool) -> Document:
    """Return the document's text page by page.

    Detect pages with no text layer and OCR them; mark those pages ocr=True.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"extract: not a file: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf(p)
    elif suffix in {".html", ".htm"}:
        pages = _extract_html(p)
    elif suffix in {".md", ".markdown", ".txt"}:
        pages = _extract_plain(p)
    else:
        raise ValueError(f"extract: unsupported file type {suffix!r} for {path}")

    return Document(source=p.name, doc_version=doc_version, superseded=superseded, pages=pages)
    