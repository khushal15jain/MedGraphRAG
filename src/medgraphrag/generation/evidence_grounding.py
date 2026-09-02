"""Stage 15: Evidence Grounding.

Post-hoc verification layer that checks whether the LLM's generated answer
is actually supported by the evidence passages it was given, addressing the
central risk of any RAG system: the LLM ignoring retrieved context and
hallucinating. This module implements two checks:

  1. Citation-validity check: every [Pn] citation in the answer must refer
     to an evidence passage that was actually provided.
  2. Lexical-overlap grounding score: for each sentence in the answer,
     measure token overlap against the cited passage's text as a cheap,
     model-free proxy for faithfulness (a full NLI-based check is
     performed later by RAGAS/DeepEval in the evaluation stage — this
     module provides a fast, always-available first-pass signal usable
     even without network access to a larger judge model).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)

_CITATION_PATTERN = re.compile(r"\[P(\d+)\]")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class GroundingReport:
    """Result of evidence-grounding verification for one generated answer."""

    total_citations: int
    valid_citations: int
    invalid_citation_numbers: list[int]
    avg_sentence_overlap: float  # mean lexical overlap of cited sentences with their passage
    uncited_sentence_ratio: float  # fraction of factual-looking sentences with no citation
    is_well_grounded: bool


class EvidenceGroundingChecker:
    """Verifies that generated answers are properly grounded in cited evidence."""

    def __init__(self, overlap_threshold: float = 0.3, min_valid_citation_ratio: float = 0.8) -> None:
        """Initialize grounding thresholds.

        Args:
            overlap_threshold: Minimum average token-overlap ratio (cited
                sentence vs. its passage) to be considered well-grounded.
            min_valid_citation_ratio: Minimum fraction of citations that
                must point to real passages for the answer to pass.
        """
        self.overlap_threshold = overlap_threshold
        self.min_valid_citation_ratio = min_valid_citation_ratio

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_TOKEN_PATTERN.findall(text.lower()))

    def _sentence_overlap(self, sentence: str, passage_text: str) -> float:
        """Jaccard-style token overlap between a sentence and its cited passage."""
        sent_tokens = self._tokenize(sentence)
        passage_tokens = self._tokenize(passage_text)
        if not sent_tokens:
            return 0.0
        return len(sent_tokens & passage_tokens) / len(sent_tokens)

    def check(self, answer: str, evidence_chunks: list[dict], text_key: str = "text") -> GroundingReport:
        """Run grounding verification on a generated answer.

        Args:
            answer: The LLM's generated answer text, expected to contain
                [Pn] citations.
            evidence_chunks: The evidence chunks provided to the LLM, in the
                same order used to build the prompt (index 0 -> [P1]).
            text_key: Dict key holding chunk text within ``evidence_chunks``.

        Returns:
            A ``GroundingReport`` summarizing citation validity and lexical grounding.
        """
        n_passages = len(evidence_chunks)
        citation_numbers = [int(m) for m in _CITATION_PATTERN.findall(answer)]
        total_citations = len(citation_numbers)

        invalid = [n for n in citation_numbers if n < 1 or n > n_passages]
        valid_citations = total_citations - len(invalid)

        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
        overlaps: list[float] = []
        uncited = 0

        for sentence in sentences:
            cited = _CITATION_PATTERN.findall(sentence)
            if not cited:
                uncited += 1
                continue
            for cite_num in cited:
                idx = int(cite_num) - 1
                if 0 <= idx < n_passages:
                    passage_text = evidence_chunks[idx].get(text_key, "")
                    overlaps.append(self._sentence_overlap(sentence, passage_text))

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        uncited_ratio = uncited / len(sentences) if sentences else 1.0
        citation_validity_ratio = valid_citations / total_citations if total_citations else 0.0

        is_well_grounded = (
            citation_validity_ratio >= self.min_valid_citation_ratio
            and avg_overlap >= self.overlap_threshold
        )

        report = GroundingReport(
            total_citations=total_citations,
            valid_citations=valid_citations,
            invalid_citation_numbers=invalid,
            avg_sentence_overlap=round(avg_overlap, 4),
            uncited_sentence_ratio=round(uncited_ratio, 4),
            is_well_grounded=is_well_grounded,
        )

        if not is_well_grounded:
            logger.warning(f"Answer failed grounding check: {report}")
        else:
            logger.info(f"Answer passed grounding check: overlap={avg_overlap:.3f}")

        return report
