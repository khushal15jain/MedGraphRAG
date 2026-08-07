"""Explainability report builder.

Turns a SentenceGroundingReport plus the original ranked evidence chunks
into a structured, per-sentence explanation: which passage supported each
claim, which source file/page it came from, its retrieval rank, and a
confidence label -- the exact set of things the Explainability requirement
asked for (which evidence was used, why, which textbook/page, chunk IDs,
confidence, ranking).

This is intentionally a pure reporting layer with no new scoring logic --
it reuses SentenceLevelGrounder's per-sentence scores and the reranker's
own ordering rather than re-deriving relevance, so the "why selected"
explanation always stays consistent with what the reranker actually did
upstream. If that consistency ever breaks (e.g. explanation says P2 but
the reranker had ranked P2 low), that's a signal something upstream
changed, not a second source of truth to reconcile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from generator.sentence_grounder import SentenceGroundingReport


@dataclass
class EvidenceExplanation:
    """Everything traced back to one cited passage."""

    passage_tag: str            # "P1", "P2", ...
    chunk_id: str
    source_file: str
    page_number: str | int
    rerank_position: int        # 1-based position in the list given to the LLM
    used_by_sentences: list[str] = field(default_factory=list)
    confidence: float = 0.0     # highest combined_score among sentences citing it


@dataclass
class ExplainabilityReport:
    """Full explanation trace for one generated answer."""

    answer: str
    evidence_explanations: list[EvidenceExplanation]
    unsupported_sentences: list[str]
    low_confidence_sentences: list[str]
    explainability_score: float  # fraction of sentences with a traceable citation


def build_explainability_report(
    grounding_report: SentenceGroundingReport,
    evidence_chunks: list[dict],
) -> ExplainabilityReport:
    """Build a structured explanation trace from a sentence-grounding result.

    Args:
        grounding_report: output of SentenceLevelGrounder.check().
        evidence_chunks: the same ranked chunks passed to the prompt builder,
            each expected to carry chunk_id/text/metadata (source_file,
            page_number) as produced by the existing retrieval/rerank stages.

    Returns:
        An ExplainabilityReport, ready to serialize alongside the answer
        (e.g. as an extra field in the API response next to `evidence`).
    """
    by_index: dict[int, EvidenceExplanation] = {}

    for sent in grounding_report.sentences:
        for idx in sent.cited_passage_indices:
            if idx < 1 or idx > len(evidence_chunks):
                continue
            chunk = evidence_chunks[idx - 1]
            meta = chunk.get("metadata", {}) or {}

            if idx not in by_index:
                by_index[idx] = EvidenceExplanation(
                    passage_tag=f"P{idx}",
                    chunk_id=chunk.get("chunk_id", ""),
                    source_file=meta.get("source_file", "unknown"),
                    page_number=meta.get("page_number", "?"),
                    rerank_position=idx,
                )

            by_index[idx].used_by_sentences.append(sent.sentence)
            by_index[idx].confidence = max(by_index[idx].confidence, sent.combined_score)

    unsupported = [s.sentence for s in grounding_report.sentences if s.status == "unsupported"]
    low_confidence = [s.sentence for s in grounding_report.sentences if s.status == "low_confidence"]

    traceable = sum(1 for s in grounding_report.sentences if s.cited_passage_indices)
    explainability_score = (
        round(traceable / len(grounding_report.sentences), 4) if grounding_report.sentences else 0.0
    )

    return ExplainabilityReport(
        answer=grounding_report.cleaned_answer,
        evidence_explanations=sorted(by_index.values(), key=lambda e: e.rerank_position),
        unsupported_sentences=unsupported,
        low_confidence_sentences=low_confidence,
        explainability_score=explainability_score,
    )