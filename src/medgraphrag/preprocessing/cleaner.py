"""Stage 2: Text Cleaning.

Textbook PDF extraction commonly introduces artifacts: hyphenated line
breaks, repeated running headers/footers, page numbers, and irregular
whitespace. This module normalizes raw page text before chunking so that
downstream NER and embedding quality is not degraded by extraction noise.
"""

from __future__ import annotations

import re
from collections import Counter

from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)

_HYPHEN_LINEBREAK = re.compile(r"(\w+)-\n(\w+)")
_MULTI_NEWLINE = re.compile(r"\n{2,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d{1,4}\s*$")
_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

_FIGURE_TABLE_LINE = re.compile(r"^\s*(Figure|Fig\.|Table)\s+\d+[\.\-]?\d*\s*.*$", re.IGNORECASE)
_CHAPTER_HEADER_LINE = re.compile(r"^\s*(Chapter|CHAPTER)\s+\d+.*$")
_PAGE_PREFIX_LINE = re.compile(r"^\s*Page\s+\d+\s*$", re.IGNORECASE)
_BRACKETED_REFS = re.compile(r"\[\d+(?:,\s*\d+)*\]")


class TextCleaner:
    """Cleans raw extracted PDF text for a single source document."""

    def __init__(self, header_footer_min_frequency: float = 0.4) -> None:
        """Initialize the cleaner.

        Args:
            header_footer_min_frequency: A line appearing on at least this
                fraction of pages is treated as a repeated running
                header/footer and removed (e.g. "Chapter 4: Breast Cancer"
                repeated on every page).
        """
        self.header_footer_min_frequency = header_footer_min_frequency

    @staticmethod
    def _fix_hyphenation(text: str) -> str:
        """Rejoin words split across a line break by a hyphen, e.g. 'chemo-\\ntherapy'."""
        return _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    @staticmethod
    def _strip_page_numbers(text: str) -> str:
        """Remove lines that consist solely of a page number or 'Page X'."""
        lines = text.split("\n")
        return "\n".join(line for line in lines if not (_PAGE_NUMBER_LINE.match(line) or _PAGE_PREFIX_LINE.match(line)))

    @staticmethod
    def _strip_figures_tables_chapters(text: str) -> str:
        """Remove lines that are purely figure/table captions or chapter headers."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            if _FIGURE_TABLE_LINE.match(line) or _CHAPTER_HEADER_LINE.match(line):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    @staticmethod
    def _strip_inline_refs(text: str) -> str:
        """Remove bracketed citation numbers like [12, 13] from text."""
        return _BRACKETED_REFS.sub("", text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse repeated whitespace and normalize newlines."""
        text = _NON_PRINTABLE.sub("", text)
        text = _MULTI_SPACE.sub(" ", text)
        text = _MULTI_NEWLINE.sub("\n\n", text)
        return text.strip()

    def detect_running_headers_footers(self, page_texts: list[str]) -> set[str]:
        """Identify lines repeated across many pages (likely headers/footers).

        Args:
            page_texts: Raw text of every page in a document.

        Returns:
            Set of line strings considered running headers/footers.
        """
        if len(page_texts) < 3:
            return set()  # too few pages for frequency-based detection to be reliable

        line_counts: Counter[str] = Counter()
        for text in page_texts:
            first_last_lines = set()
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            if lines:
                first_last_lines.add(lines[0])
                first_last_lines.add(lines[-1])
            line_counts.update(first_last_lines)

        threshold = max(2, int(len(page_texts) * self.header_footer_min_frequency))
        repeated = {line for line, count in line_counts.items() if count >= threshold}
        if repeated:
            logger.debug(f"Detected {len(repeated)} repeated header/footer lines")
        return repeated

    def clean_page(self, text: str, running_lines: set[str] | None = None) -> str:
        """Apply the full cleaning pipeline to a single page's text.

        Args:
            text: Raw extracted text for one page.
            running_lines: Optional set of known header/footer lines to strip.

        Returns:
            Cleaned text.
        """
        if running_lines:
            lines = [ln for ln in text.split("\n") if ln.strip() not in running_lines]
            text = "\n".join(lines)

        text = self._fix_hyphenation(text)
        text = self._strip_page_numbers(text)
        text = self._strip_figures_tables_chapters(text)
        text = self._strip_inline_refs(text)
        text = self._normalize_whitespace(text)
        return text

    def clean_document(self, page_texts: list[str]) -> list[str]:
        """Clean all pages of a document, using cross-page header/footer detection.

        Args:
            page_texts: Raw text of every page in a document, in order.

        Returns:
            List of cleaned page texts (same length and order as input).
        """
        running_lines = self.detect_running_headers_footers(page_texts)
        cleaned = [self.clean_page(text, running_lines) for text in page_texts]
        logger.info(f"Cleaned {len(cleaned)} pages (removed {len(running_lines)} repeated lines)")
        return cleaned
