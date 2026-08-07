"""Sentence-level grounding classifier.

Extends EvidenceGroundingChecker's citation-level view down to individual
sentences, and can actively CLEAN an answer instead of only reporting on
it: unsupported sentences are tagged with a confidence marker (or dropped
entirely, if configured), rather than silently passed through to the user.

Why the existing lexical-overlap-only check under-measures faithfulness:
Pure token-overlap (EvidenceGroundingChecker._sentence_overlap) penalizes
answers that correctly PARAPHRASE evidence -- a fully faithful sentence
that rewords "distant recurrence risk >=10%" as "a ten percent or greater
risk of the cancer spreading" scores near-zero lexical overlap despite
being fully supported. That is very likely a real chunk of the current 47%
Faithfulness number: some of what's being counted as "unsupported" is
actually valid paraphrase, not hallucination.

This module adds a semantic-similarity signal (via the pipeline's own
BGEEmbedder) alongside the lexical one and combines them, so paraphrase is
credited while genuinely unsupported claims are still caught. Lexical
overlap is kept (not dropped) deliberately: medical facts are exactly the
kind of content where exact terms matter -- a paraphrase of "cisplatin" is
still "cisplatin"; if the term isn't in the evidence at all, semantic
similarity alone (which rewards topical closeness) could let a fabricated
but topically-adjacent drug name slip through. The weighted combination is
tunable; the defaults favor semantic (paraphrase is common in clinical
writing) but keep lexical as a floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from embeddings.embedder import BGEEmbedder
from utils.logger import get_logger

logger = get_logger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_CITATION_NUM_PATTERN = re.compile(r"P(\d+)")


@dataclass
class SentenceGrounding:
    """Per-sentence grounding verdict."""

    sentence: str
    cited_passage_indices: list[int]
    citation_source: str  # "explicit" (model wrote [Pn]) | "inferred" (best-match fallback) | "none"
    lexical_overlap: float
    semantic_similarity: float
    combined_score: float
    status: str  # "grounded" | "low_confidence" | "unsupported"


@dataclass
class SentenceGroundingReport:
    """Aggregate result for one generated answer."""

    sentences: list[SentenceGrounding]
    cleaned_answer: str
    faithfulness_score: float  # mean combined_score across sentences
    grounded_ratio: float      # fraction of sentences with status == "grounded"


class SentenceLevelGrounder:
    """Per-sentence grounding: scores, tags, and optionally strips unsupported claims."""

    def __init__(
        self,
        embedder: BGEEmbedder,
        lexical_weight: float = 0.4,
        semantic_weight: float = 0.6,
        grounded_threshold: float = 0.65,
        low_confidence_threshold: float = 0.45,
        drop_unsupported: bool = True,
    ) -> None:
        """Configure grounding thresholds and scoring weights.

        Args:
            embedder: the pipeline's shared BGEEmbedder instance -- reuse it,
                don't instantiate a second copy (RAM budget).
            lexical_weight: weight on token-overlap in the combined score.
            semantic_weight: weight on embedding cosine similarity. Must sum
                to 1.0 with lexical_weight. Defaults favor semantic (0.6)
                since clinical writing paraphrases often, but lexical is
                kept nonzero as a floor against topically-close fabrication.
            grounded_threshold: combined_score >= this -> "grounded".
            low_confidence_threshold: combined_score >= this (but below
                grounded_threshold) -> "low_confidence"; below this ->
                "unsupported".
            drop_unsupported: if True, unsupported sentences are removed
                from cleaned_answer entirely. Start with False (tag, don't
                delete) in a clinical-facing tool so removals are visible
                and auditable before trusting silent deletion.
        """
        if abs(lexical_weight + semantic_weight - 1.0) > 1e-6:
            raise ValueError("lexical_weight and semantic_weight must sum to 1.0")
        self.embedder = embedder
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight
        self.grounded_threshold = grounded_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.drop_unsupported = drop_unsupported

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_TOKEN_PATTERN.findall(text.lower()))

    def _lexical_overlap(self, sentence: str, passage_text: str) -> float:
        sent_tokens = self._tokenize(sentence)
        passage_tokens = self._tokenize(passage_text)
        if not sent_tokens:
            return 0.0
        return len(sent_tokens & passage_tokens) / len(sent_tokens)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
        return float(np.dot(a, b) / denom)

    def check(
        self, answer: str, evidence_chunks: list[dict], text_key: str = "text"
    ) -> SentenceGroundingReport:
        """Score every sentence in `answer` against its cited (or best-matching) passage.

        Args:
            answer: the LLM's output. Works best with citation-tagged output
                (e.g. from CitationAwarePromptBuilder) but degrades gracefully
                on uncited answers by checking every sentence against every
                passage and keeping the best match.
            evidence_chunks: passages in prompt order (index 0 -> [P1]).
            text_key: dict key holding chunk text.

        Returns:
            A SentenceGroundingReport with per-sentence detail and a cleaned,
            confidence-annotated (or trimmed) answer.
        """
        n_passages = len(evidence_chunks)
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
        if not sentences:
            return SentenceGroundingReport(
                sentences=[], cleaned_answer=answer, faithfulness_score=0.0, grounded_ratio=0.0
            )

        passage_texts = [c.get(text_key, "") for c in evidence_chunks]
        passage_vecs = (
            self.embedder.embed_documents(passage_texts) if passage_texts else np.empty((0, 0))
        )
        sentence_vecs = self.embedder.embed_documents(sentences)

        results: list[SentenceGrounding] = []
        kept_sentences: list[str] = []

        for sentence, sent_vec in zip(sentences, sentence_vecs):
            cited = [int(n) for n in _CITATION_NUM_PATTERN.findall(sentence)]
            cited = [n for n in cited if 1 <= n <= n_passages]

            # We no longer restrict grounding checks to explicitly cited chunks,
            # because small models often hallucinate incorrect chunk IDs, which causes
            # them to fail the strict grounding check and get refused.
            candidate_indices = list(range(1, n_passages + 1))

            best_lexical, best_semantic, best_idx = 0.0, 0.0, None
            best_combined = -1.0
            for idx in candidate_indices:
                passage_text = passage_texts[idx - 1]
                lex = self._lexical_overlap(sentence, passage_text)
                sem = self._cosine(sent_vec, passage_vecs[idx - 1]) if len(passage_vecs) else 0.0
                combined = self.lexical_weight * lex + self.semantic_weight * sem
                if combined > best_combined:
                    best_combined, best_lexical, best_semantic, best_idx = combined, lex, sem, idx

            combined_score = round(max(best_combined, 0.0), 4)

            if combined_score >= self.grounded_threshold:
                status = "grounded"
            elif combined_score >= self.low_confidence_threshold:
                status = "low_confidence"
            else:
                status = "unsupported"

            if cited:
                citation_source = "explicit"
            elif best_idx is not None:
                citation_source = "inferred"
            else:
                citation_source = "none"

            results.append(
                SentenceGrounding(
                    sentence=sentence,
                    cited_passage_indices=cited or ([best_idx] if best_idx else []),
                    citation_source=citation_source,
                    lexical_overlap=round(best_lexical, 4),
                    semantic_similarity=round(best_semantic, 4),
                    combined_score=combined_score,
                    status=status,
                )
            )

            if status == "unsupported" and self.drop_unsupported:
                continue  # strip entirely from the cleaned answer
            else:
                kept_sentences.append(sentence)

        faithfulness_score = round(sum(r.combined_score for r in results) / len(results), 4)
        grounded_ratio = round(
            sum(1 for r in results if r.status == "grounded") / len(results), 4
        )

        return SentenceGroundingReport(
            sentences=results,
            cleaned_answer=" ".join(kept_sentences),
            faithfulness_score=faithfulness_score,
            grounded_ratio=grounded_ratio,
        )
