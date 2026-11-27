"""The shapes passed between stages. Metadata fields are required (section 3.1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Page:
    """Text of one page, as extraction produced it."""

    source: str              # file name, e.g. Academic_Rules_Handbook_2026.pdf
    page: int                # 1-based
    text: str
    ocr: bool = False        # True when this page had no text layer and was OCR'd


@dataclass
class Document:
    source: str
    doc_version: str | None = None   # e.g. "2026-06"; from corpus/sources.yaml or filename
    superseded: bool = False         # True for an older version of a policy
    pages: list[Page] = field(default_factory=list)
    content_hash: str | None = None
    title: str | None = None
    page_count: int | None = None


@dataclass
class Chunk:
    id: str                  # stable: same document + same position gives the same id
    text: str
    source: str
    section: str             # the heading this chunk sits under, e.g. "5.4 Academic Warning"
    page: int
    doc_version: str | None = None
    superseded: bool = False
    content_hash: str | None = None
    title: str | None = None
    page_count: int | None = None

    def metadata(self) -> dict[str, str | int | bool]:
        """Chroma metadata values must be str, int, float or bool."""
        meta: dict[str, str | int | bool] = {
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "doc_version": self.doc_version or "",
            "superseded": self.superseded,
        }
        if self.content_hash is not None:
            meta["content_hash"] = self.content_hash
        if self.title is not None:
            meta["title"] = self.title
        if self.page_count is not None:
            meta["page_count"] = self.page_count
        return meta
