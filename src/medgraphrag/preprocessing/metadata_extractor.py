"""Stage 3: Metadata Extraction.

Extracts document- and chunk-level metadata that downstream stages rely on
for provenance and evidence grounding: source textbook title, chapter/section
headings (heuristically detected via font size / formatting cues in
PyMuPDF), and page ranges. Every chunk ultimately carries this metadata so
that generated answers can cite "Source: DeVita's Cancer Principles,
Chapter 12, p. 341" rather than an opaque chunk ID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import fitz  # PyMuPDF

from medgraphrag.utils.exceptions import PDFLoadError
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)

# Heuristic patterns for chapter/section headings common in medical textbooks.
_CHAPTER_PATTERN = re.compile(r"^\s*(Chapter|CHAPTER)\s+\d+[:.\s]", re.IGNORECASE)
_SECTION_PATTERN = re.compile(r"^\s*\d+(\.\d+)*\s+[A-Z][A-Za-z\s]{3,80}$")


@dataclass
class DocumentMetadata:
    """Document-level metadata for a source textbook."""

    source_file: str
    title: str
    author: str
    total_pages: int


@dataclass
class SectionHeading:
    """A detected chapter or section heading and the page it occurs on."""

    page_number: int
    heading_text: str
    level: str  # "chapter" or "section"


class MetadataExtractor:
    """Extracts document- and heading-level metadata from PDFs."""

    def extract_document_metadata(self, pdf_path: str) -> DocumentMetadata:
        """Read PDF metadata (title, author, page count) via PyMuPDF's built-in dict.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A populated ``DocumentMetadata`` instance. Falls back to the
            filename when embedded metadata fields are absent, which is
            common in scanned/converted textbook PDFs.

        Raises:
            PDFLoadError: If the PDF cannot be opened.
        """
        try:
            with fitz.open(pdf_path) as doc:
                meta = doc.metadata or {}
                title = meta.get("title") or _filename_to_title(pdf_path)
                author = meta.get("author") or "Unknown"
                total_pages = doc.page_count
        except Exception as exc:  # noqa: BLE001
            raise PDFLoadError(f"Failed to extract metadata from {pdf_path}: {exc}") from exc

        return DocumentMetadata(
            source_file=pdf_path,
            title=title.strip(),
            author=author.strip(),
            total_pages=total_pages,
        )

    def detect_headings(self, pdf_path: str) -> list[SectionHeading]:
        """Heuristically detect chapter/section headings using font size cues.

        Uses PyMuPDF's ``get_text("dict")`` to inspect span font sizes: lines
        rendered notably larger than the page's median font size, or that
        match common chapter/section regex patterns, are treated as headings.
        This is a heuristic (textbooks are not semantically tagged), so
        recall is prioritized over precision — false positives are filtered
        further downstream during chunking.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of detected ``SectionHeading`` objects across the document.
        """
        headings: list[SectionHeading] = []
        try:
            with fitz.open(pdf_path) as doc:
                for page_index in range(doc.page_count):
                    page = doc.load_page(page_index)
                    page_dict = page.get_text("dict")
                    font_sizes = [
                        span["size"]
                        for block in page_dict.get("blocks", [])
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    ]
                    if not font_sizes:
                        continue
                    median_size = sorted(font_sizes)[len(font_sizes) // 2]

                    for block in page_dict.get("blocks", []):
                        for line in block.get("lines", []):
                            spans = line.get("spans", [])
                            if not spans:
                                continue
                            line_text = "".join(s["text"] for s in spans).strip()
                            if not line_text:
                                continue
                            avg_size = sum(s["size"] for s in spans) / len(spans)

                            is_large_font = avg_size > median_size * 1.25
                            matches_pattern = bool(
                                _CHAPTER_PATTERN.match(line_text)
                                or _SECTION_PATTERN.match(line_text)
                            )

                            if is_large_font or matches_pattern:
                                level = "chapter" if _CHAPTER_PATTERN.match(line_text) else "section"
                                headings.append(
                                    SectionHeading(
                                        page_number=page_index + 1,
                                        heading_text=line_text,
                                        level=level,
                                    )
                                )
        except Exception as exc:  # noqa: BLE001
            raise PDFLoadError(f"Failed to detect headings in {pdf_path}: {exc}") from exc

        logger.info(f"Detected {len(headings)} candidate headings in {pdf_path}")
        return headings

    def get_heading_for_page(
        self, page_number: int, headings: list[SectionHeading]
    ) -> str | None:
        """Find the most recent heading preceding (or on) a given page.

        Args:
            page_number: Target page number (1-indexed).
            headings: Sorted or unsorted list of detected headings for the document.

        Returns:
            The nearest preceding heading's text, or ``None`` if no heading
            occurs at or before this page.
        """
        candidates = [h for h in headings if h.page_number <= page_number]
        if not candidates:
            return None
        return max(candidates, key=lambda h: h.page_number).heading_text


def _filename_to_title(pdf_path: str) -> str:
    """Derive a human-readable title from a filename as a metadata fallback."""
    from pathlib import Path

    stem = Path(pdf_path).stem
    return stem.replace("_", " ").replace("-", " ").title()
