"""Stage 16: Evaluation (Hallucination Analysis).

Aggregates the fast, model-free ``EvidenceGroundingChecker`` signal (Stage
15) across an evaluation set to produce a hallucination-rate report,
independent of and complementary to the LLM-judge-based RAGAS/DeepEval
faithfulness scores. Reporting a non-LLM-judge hallucination proxy
alongside LLM-judge metrics is a methodological safeguard requested by the
research design (judge models can themselves hallucinate their
judgments — a lexical-overlap floor check is a useful sanity cross-check).
"""

from __future__ import annotations

from dataclasses import dataclass

from medgraphrag.generation.evidence_grounding import EvidenceGroundingChecker, GroundingReport
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HallucinationReport:
    """Summary hallucination statistics across an evaluation batch."""

    n_samples: int
    n_well_grounded: int
    hallucination_rate: float          # fraction of answers failing the grounding check
    mean_sentence_overlap: float
    mean_uncited_sentence_ratio: float
    mean_invalid_citation_count: float


class HallucinationAnalyzer:
    """Computes batch-level hallucination statistics from per-answer grounding reports."""

    def __init__(self, checker: EvidenceGroundingChecker | None = None) -> None:
        """Initialize with an (optionally custom-configured) grounding checker.

        Args:
            checker: An ``EvidenceGroundingChecker`` instance; a default one
                is created if not provided.
        """
        self.checker = checker or EvidenceGroundingChecker()

    def analyze_batch(
        self, answers: list[str], evidence_chunks_per_answer: list[list[dict]]
    ) -> HallucinationReport:
        """Run grounding checks across a batch and aggregate hallucination statistics.

        Args:
            answers: Generated answer texts.
            evidence_chunks_per_answer: For each answer, the list of
                evidence chunk dicts it was generated from.

        Returns:
            A ``HallucinationReport`` summarizing the batch.
        """
        reports: list[GroundingReport] = [
            self.checker.check(answer, chunks)
            for answer, chunks in zip(answers, evidence_chunks_per_answer, strict=True)
        ]

        n = len(reports)
        if n == 0:
            return HallucinationReport(0, 0, 0.0, 0.0, 0.0, 0.0)

        n_well_grounded = sum(1 for r in reports if r.is_well_grounded)
        hallucination_rate = 1.0 - (n_well_grounded / n)
        mean_overlap = sum(r.avg_sentence_overlap for r in reports) / n
        mean_uncited = sum(r.uncited_sentence_ratio for r in reports) / n
        mean_invalid = sum(len(r.invalid_citation_numbers) for r in reports) / n

        report = HallucinationReport(
            n_samples=n,
            n_well_grounded=n_well_grounded,
            hallucination_rate=round(hallucination_rate, 4),
            mean_sentence_overlap=round(mean_overlap, 4),
            mean_uncited_sentence_ratio=round(mean_uncited, 4),
            mean_invalid_citation_count=round(mean_invalid, 4),
        )
        logger.info(f"Hallucination analysis: {report}")
        return report
