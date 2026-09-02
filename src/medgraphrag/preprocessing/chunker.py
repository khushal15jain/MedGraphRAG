"""Stage 4: Hierarchical Chunking.

Splits cleaned document text into a two-level hierarchy:
  - Parent chunks: larger, section-scale windows (default ~1024 "tokens",
    approximated as chars // 4) that preserve broader clinical context.
  - Child chunks: smaller windows (default ~256 tokens) nested within each
    parent, used for fine-grained dense/sparse retrieval.

This hierarchical structure is what "Hierarchical RAG" in the research
requirements refers to: retrieval happens at the child level for precision,
but the parent chunk can be attached as expanded context for the LLM,
trading off precision and context sufficiency — a well-established
technique for long-document RAG (e.g. "small-to-big" retrieval).

Chunking respects sentence boundaries where possible using a lightweight
regex-based splitter rather than a heavy sentence tokenizer, to avoid an
additional model load on top of spaCy in the entity-extraction stage.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from medgraphrag.utils.exceptions import ChunkingError
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Rough heuristic: ~4 characters per token for English biomedical text.
_CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A single chunk of text with hierarchy links and provenance metadata."""

    chunk_id: str
    text: str
    level: str  # "parent" or "child"
    parent_id: str | None
    source_file: str
    page_number: int
    heading: str | None
    token_estimate: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_estimate = max(1, len(self.text) // _CHARS_PER_TOKEN)


class HierarchicalChunker:
    """Produces parent/child chunk hierarchies from cleaned page text."""

    def __init__(
        self,
        parent_chunk_size: int = 3500,
        child_chunk_size: int = 300,
        chunk_overlap: int = 35,
    ) -> None:
        """Initialize the chunker.

        Args:
            parent_chunk_size: Target token count for parent chunks.
            child_chunk_size: Target token count for child chunks.
            chunk_overlap: Token overlap between consecutive child chunks,
                to avoid losing context at boundaries (e.g. a sentence
                describing a drug dosage split across two chunks).

        Raises:
            ChunkingError: If sizes are non-positive or overlap >= child size.
        """
        if parent_chunk_size <= 0 or child_chunk_size <= 0:
            raise ChunkingError("Chunk sizes must be positive integers")
        if chunk_overlap >= child_chunk_size:
            raise ChunkingError("chunk_overlap must be smaller than child_chunk_size")

        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using a lightweight regex heuristic."""
        sentences = _SENTENCE_SPLIT.split(text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _pack_sentences(self, sentences: list[str], target_tokens: int) -> list[str]:
        """Greedily pack sentences into windows near ``target_tokens`` size, respecting sections."""
        windows: list[str] = []
        current: list[str] = []
        current_tokens = 0
        
        # Recognize major clinical sections to force a chunk boundary
        section_pattern = re.compile(r"^\s*(Diagnosis|Treatment|Prognosis|Risk Factors|Complications|Etiology|Pathophysiology|Clinical Features|Management|Epidemiology)\s*[:]?\s*$", re.IGNORECASE)

        for sentence in sentences:
            is_section = bool(section_pattern.match(sentence))
            sentence_tokens = max(1, len(sentence) // _CHARS_PER_TOKEN)
            
            # Flush current chunk if it's too large or we hit a new section
            if current and (current_tokens + sentence_tokens > target_tokens or is_section):
                windows.append(" ".join(current))
                current = []
                current_tokens = 0
                
            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            windows.append(" ".join(current))
        return windows

    def _split_children(self, parent_text: str) -> list[str]:
        """Split a parent chunk's text into overlapping child windows."""
        sentences = self._split_sentences(parent_text)
        raw_children = self._pack_sentences(sentences, self.child_chunk_size)

        if self.chunk_overlap == 0 or len(raw_children) <= 1:
            return raw_children

        # Apply overlap by prepending a tail slice of the previous child.
        overlapped: list[str] = [raw_children[0]]
        overlap_chars = self.chunk_overlap * _CHARS_PER_TOKEN
        for i in range(1, len(raw_children)):
            prev_tail = raw_children[i - 1][-overlap_chars:]
            overlapped.append(f"{prev_tail} {raw_children[i]}")
        return overlapped

    def chunk_page(
        self,
        text: str,
        source_file: str,
        page_number: int,
        heading: str | None = None,
    ) -> list[Chunk]:
        """Chunk a single page's cleaned text into parent + child chunks.

        Args:
            text: Cleaned page text.
            source_file: Originating PDF filename (for provenance).
            page_number: Page number within the source document.
            heading: Nearest preceding chapter/section heading, if known.

        Returns:
            Flat list of ``Chunk`` objects: parents followed by their children.
        """
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        parent_texts = self._pack_sentences(sentences, self.parent_chunk_size)

        chunks: list[Chunk] = []
        for parent_text in parent_texts:
            parent_id = str(uuid.uuid4())
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    text=parent_text,
                    level="parent",
                    parent_id=None,
                    source_file=source_file,
                    page_number=page_number,
                    heading=heading,
                )
            )

            child_texts = self._split_children(parent_text)
            for child_text in child_texts:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=child_text,
                        level="child",
                        parent_id=parent_id,
                        source_file=source_file,
                        page_number=page_number,
                        heading=heading,
                    )
                )

        return chunks

    def chunk_document(
        self,
        page_texts: list[str],
        source_file: str,
        headings_by_page: dict[int, str] | None = None,
    ) -> list[Chunk]:
        """Chunk every page of a document, preserving page-level provenance.

        Args:
            page_texts: Cleaned text for each page, in order (index 0 = page 1).
            source_file: Originating PDF filename.
            headings_by_page: Optional mapping of page_number -> nearest heading.

        Returns:
            Flat list of all ``Chunk`` objects for the document.
        """
        headings_by_page = headings_by_page or {}
        all_chunks: list[Chunk] = []

        for i, text in enumerate(page_texts):
            page_number = i + 1
            heading = headings_by_page.get(page_number)
            page_chunks = self.chunk_page(text, source_file, page_number, heading)
            all_chunks.extend(page_chunks)

        n_parents = sum(1 for c in all_chunks if c.level == "parent")
        n_children = sum(1 for c in all_chunks if c.level == "child")
        logger.info(
            f"Chunked {source_file}: {n_parents} parent chunks, "
            f"{n_children} child chunks across {len(page_texts)} pages"
        )
        return all_chunks
