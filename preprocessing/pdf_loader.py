"""Stage 1: PDF Loading.

Loads oncology textbook PDFs using PyMuPDF (``fitz``), extracting raw text
per page along with basic structural metadata (page number, source file).
PyMuPDF is used over alternatives (e.g. pdfminer, PyPDF2) because it is
fast, has low memory overhead, and preserves reading order well for
textbook-style layouts — all relevant on an 8GB-RAM laptop processing
multi-hundred-page books.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from utils.exceptions import PDFLoadError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """Raw text and metadata for a single PDF page."""

    source_file: str
    page_number: int  # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


class PDFLoader:
    """Loads one or more PDF files and yields page-level text content."""

    def __init__(self, min_chars_per_page: int = 20) -> None:
        """Initialize the loader.

        Args:
            min_chars_per_page: Pages with fewer characters than this are
                skipped (typically blank pages, section dividers, or scanned
                images with no extractable text layer).
        """
        self.min_chars_per_page = min_chars_per_page

    def load_pdf(self, pdf_path: str | Path) -> list[PageContent]:
        """Extract text from every page of a single PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of ``PageContent`` objects, one per non-empty page.

        Raises:
            PDFLoadError: If the file does not exist, is not a valid PDF,
                or contains zero pages.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise PDFLoadError(f"PDF file not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise PDFLoadError(f"Expected a .pdf file, got: {path}")

        pages: list[PageContent] = []
        try:
            with fitz.open(path) as doc:
                if doc.page_count == 0:
                    raise PDFLoadError(f"PDF has zero pages: {path}")

                for page_index in range(doc.page_count):
                    page = doc.load_page(page_index)
                    text = page.get_text("text").strip()

                    if len(text) < self.min_chars_per_page:
                        logger.debug(
                            f"Skipping near-empty page {page_index + 1} in {path.name} "
                            f"({len(text)} chars)"
                        )
                        continue

                    pages.append(
                        PageContent(
                            source_file=path.name,
                            page_number=page_index + 1,
                            text=text,
                        )
                    )
        except fitz.FileDataError as exc:
            raise PDFLoadError(f"Corrupt or unreadable PDF: {path} ({exc})") from exc
        except Exception as exc:  # noqa: BLE001 — surface any PyMuPDF-internal failure clearly
            raise PDFLoadError(f"Unexpected error loading PDF {path}: {exc}") from exc

        logger.info(f"Loaded {len(pages)} non-empty pages from {path.name}")
        return pages

    def load_directory(self, directory: str | Path) -> dict[str, list[PageContent]]:
        """Load every PDF in a directory (non-recursive).

        Args:
            directory: Directory containing ``.pdf`` files (e.g. the six
                oncology textbooks).

        Returns:
            Mapping of filename -> list of ``PageContent``.

        Raises:
            PDFLoadError: If the directory does not exist or contains no PDFs.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            raise PDFLoadError(f"Directory not found: {dir_path}")

        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            raise PDFLoadError(f"No PDF files found in {dir_path}")

        results: dict[str, list[PageContent]] = {}
        for pdf_file in pdf_files:
            try:
                results[pdf_file.name] = self.load_pdf(pdf_file)
            except PDFLoadError as exc:
                logger.warning(f"Skipping {pdf_file.name}: {exc}")
                continue

        logger.info(f"Loaded {len(results)}/{len(pdf_files)} PDFs from {dir_path}")
        return results
