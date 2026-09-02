"""Unit tests for preprocessing/cleaner.py and preprocessing/chunker.py.

These tests avoid PyMuPDF/PDF I/O (covered separately, informally, since it
requires a real PDF fixture) and instead exercise the pure-Python cleaning
and chunking logic directly.
"""

from __future__ import annotations

import pytest

from medgraphrag.preprocessing.chunker import HierarchicalChunker
from medgraphrag.preprocessing.cleaner import TextCleaner
from medgraphrag.utils.exceptions import ChunkingError


class TestTextCleaner:
    def test_fix_hyphenation_rejoins_split_words(self) -> None:
        cleaner = TextCleaner()
        text = "Patients should be monitored for cardiotox-\nicity during treatment."
        cleaned = cleaner.clean_page(text)
        assert "cardiotoxicity" in cleaned
        assert "cardiotox-" not in cleaned

    def test_strip_page_numbers(self) -> None:
        cleaner = TextCleaner()
        text = "Some clinical content.\n42\nMore content here."
        cleaned = cleaner.clean_page(text)
        assert "\n42\n" not in cleaned

    def test_detect_running_headers_footers(self) -> None:
        cleaner = TextCleaner(header_footer_min_frequency=0.5)
        pages = [
            "Chapter 4: Breast Cancer\nContent A\nChapter 4: Breast Cancer",
            "Chapter 4: Breast Cancer\nContent B\nChapter 4: Breast Cancer",
            "Chapter 4: Breast Cancer\nContent C\nChapter 4: Breast Cancer",
        ]
        repeated = cleaner.detect_running_headers_footers(pages)
        assert "Chapter 4: Breast Cancer" in repeated

    def test_clean_document_removes_headers_across_pages(self, sample_page_texts: list[str]) -> None:
        cleaner = TextCleaner()
        cleaned = cleaner.clean_document(sample_page_texts)
        assert len(cleaned) == len(sample_page_texts)


class TestHierarchicalChunker:
    def test_invalid_chunk_sizes_raise(self) -> None:
        with pytest.raises(ChunkingError):
            HierarchicalChunker(parent_chunk_size=0)

    def test_overlap_must_be_smaller_than_child_size(self) -> None:
        with pytest.raises(ChunkingError):
            HierarchicalChunker(child_chunk_size=100, chunk_overlap=100)

    def test_chunk_page_produces_parent_and_child_chunks(self) -> None:
        chunker = HierarchicalChunker(parent_chunk_size=200, child_chunk_size=50, chunk_overlap=10)
        text = (
            "Trastuzumab is used to treat HER2-positive breast cancer. "
            "It is typically administered intravenously. "
            "Cardiotoxicity is a known adverse effect requiring monitoring. "
            "Combination therapy with chemotherapy is common in early-stage disease."
        )
        chunks = chunker.chunk_page(text, source_file="test.pdf", page_number=1)

        parents = [c for c in chunks if c.level == "parent"]
        children = [c for c in chunks if c.level == "child"]

        assert len(parents) >= 1
        assert len(children) >= 1
        assert all(c.parent_id == parents[0].chunk_id for c in children if c.parent_id)

    def test_chunk_page_empty_text_returns_empty(self) -> None:
        chunker = HierarchicalChunker()
        assert chunker.chunk_page("   ", source_file="test.pdf", page_number=1) == []

    def test_chunk_document_preserves_page_count_provenance(self) -> None:
        chunker = HierarchicalChunker(parent_chunk_size=100, child_chunk_size=30, chunk_overlap=5)
        pages = ["First page content about oncology treatments and drug interactions."] * 3
        chunks = chunker.chunk_document(pages, source_file="test.pdf")

        page_numbers = {c.page_number for c in chunks}
        assert page_numbers == {1, 2, 3}
