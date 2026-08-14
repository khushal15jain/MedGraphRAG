"""grounding_checker.py
---------------------
Sentence-level and claim-level grounding verification module.

Provides fine-grained claim validation across generated clinical answers:
- Evaluates individual claims against retrieved gold evidence chunks using BGE embeddings.
- Computes token-level lexical overlap to verify exact medical terminology and entity matches.
- Integrates NLI entailment scoring to verify factual support and flag ungrounded claims.
"""
   rather than silently dropped, EXCEPT when similarity is near-zero (no
   plausible support at all), in which case the claim is dropped as
   hallucinated content, not merely flagged.

FILES TO MODIFY
----------------
- grounding/grounding_checker.py  (NEW - this file)
- generation/generator.py         (uses check_claims() both for the
                                    retry gate and for final filtering)

FUNCTIONS TO ADD
-----------------
- claim_evidence_similarity(claim_text, evidence_texts, embedder) -> float
- lexical_overlap(claim_text, evidence_texts) -> float
- check_claims(claims, evidence_list, embedder, nli_model=None,
                high_thresh=0.62, low_thresh=0.40) -> List[Claim]  (mutates
                .grounded / .grounding_score in place and returns them)
- sentence_level_report(grounded_answer) -> dict  (for explainability /
                                                     evaluation pipeline)

EXPECTED METRIC IMPROVEMENT
-----------------------------
Groundedness  : 67.5% -> 88-92% (now measured AND enforced per-claim,
                                  matching what the evaluator measures)
Faithfulness  : 47.15% -> 80-85% (grounding gate feeds generator's retry
                                   loop, so ungrounded claims are caught
                                   before the user ever sees them)
Hallucination : 52.85% -> 12-18% (near-zero-similarity claims are dropped
                                   outright, not just down-weighted)
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)
_embedding_cache = {}

_STOPWORDS = set(
    "the a an of to in on for and or is are was were be been being with "
    "as by at from this that these those it its into over under between".split()
)


def _tokenize(text: str) -> set:
    tokens = re.findall(r"[a-zA-Z0-9\-]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def lexical_overlap(claim_text: str, evidence_texts: Sequence[str]) -> float:
    """Evidence-recall-style overlap: fraction of claim's content tokens
    that appear in the union of cited evidence text. Cheap, model-free,
    catches terminology mismatches embeddings can miss."""
    claim_tokens = _tokenize(claim_text)
    if not claim_tokens:
        return 0.0
    evidence_tokens = set()
    for t in evidence_texts:
        evidence_tokens |= _tokenize(t)
    if not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    return (2 * len(overlap)) / (len(claim_tokens) + len(evidence_tokens))


def claim_evidence_similarity(claim_text: str, evidence_texts: Sequence[str],
                               embedder) -> float:
    """
    embedder: your existing BGE-base wrapper, assumed to expose either
        embedder.encode(List[str]) -> np.ndarray  (sentence-transformers-style)
    Adapt this call if your wrapper's method name differs -- the
    similarity math below does not need to change.
    """
    if not evidence_texts or not claim_text.strip():
        return 0.0
    try:
        import numpy as np
        vecs = embedder.encode([claim_text] + list(evidence_texts), normalize_embeddings=True)
        claim_vec = vecs[0]
        evidence_vecs = vecs[1:]
        sims = evidence_vecs @ claim_vec
        return float(np.max(sims))
    except Exception as e:
        logger.warning("Embedding similarity failed, falling back to lexical only: %s", e)
        return 0.0


def _nli_entailment_score(claim_text: str, evidence_texts: Sequence[str],
                           nli_model) -> Optional[float]:
    """Optional stronger check for boundary cases. nli_model is expected
    to expose: nli_model.predict([(premise, hypothesis), ...]) -> scores
    over [contradiction, neutral, entailment] (standard HF NLI cross-
    encoder convention), or an equivalent .entailment_score(p, h) -> float.
    Returns entailment probability against the BEST-matching evidence
    text, or None if unavailable/failed."""
    if nli_model is None or not evidence_texts:
        return None
    try:
        best = 0.0
        for ev in evidence_texts:
            if hasattr(nli_model, "entailment_score"):
                score = nli_model.entailment_score(premise=ev, hypothesis=claim_text)
            else:
                # generic cross-encoder predict returning [contra, neutral, entail]
                preds = nli_model.predict([(ev, claim_text)])
                score = float(preds[0][-1])
            best = max(best, score)
        return best
    except Exception as e:
        logger.warning("NLI entailment check failed: %s", e)
        return None


def check_claims(
    claims,  # List[Claim] from generator.py (duck-typed to avoid circular import)
    evidence_list,  # List[EvidenceItem] from prompt_builder.py
    embedder,
    nli_model=None,
    high_thresh: float = 0.68,
    low_thresh: float = 0.45,
    boundary_band: float = 0.06,
):
    evidence_by_id = {ev.evidence_id: ev.text for ev in evidence_list}

    for claim in claims:
        cited_texts = [evidence_by_id[eid] for eid in claim.evidence_ids if eid in evidence_by_id]

        if claim.confidence == "unsupported" or not claim.evidence_ids:
            # Model already declared this unsupported -- respect it,
            # score stays 0, will be surfaced as low-confidence in the
            # final answer per generator.assemble_answer_text().
            claim.grounded = False if not cited_texts else None
            claim.grounding_score = 0.0
            continue

        sem_sim = claim_evidence_similarity(claim.text, cited_texts, embedder)
        lex_overlap = lexical_overlap(claim.text, cited_texts)
        combined = 0.55 * sem_sim + 0.45 * lex_overlap

        if high_thresh - boundary_band <= combined <= high_thresh + boundary_band and nli_model is not None:
            entail = _nli_entailment_score(claim.text, cited_texts, nli_model)
            if entail is not None:
                combined = 0.7 * combined + 0.3 * entail

        claim.grounding_score = round(combined, 4)
        if combined >= high_thresh:
            claim.grounded = True
        elif combined >= low_thresh:
            # Ambiguous band: keep but caller (assemble_answer_text) marks
            # it low-confidence rather than dropping -- matches the
            # requirement "mark as low confidence" instead of removing.
            claim.confidence = "low"
            claim.grounded = True
        else:
            claim.grounded = False  # dropped as unsupported/hallucinated

    return claims

def get_embedding(text, embedder):
    if text not in _embedding_cache:
        _embedding_cache[text] = embedder.encode(
            [text],
            normalize_embeddings=True
        )[0]
    return _embedding_cache[text]

def sentence_level_report(grounded_answer) -> dict:
    """Produces the per-sentence grounding audit trail consumed by
    explainability.py and by the evaluation pipeline's Groundedness /
    Faithfulness metric computation."""
    report = {
        "question": grounded_answer.question,
        "total_claims": len(grounded_answer.claims),
        "grounded": sum(1 for c in grounded_answer.claims if c.grounded is True),
        "dropped_ungrounded": sum(1 for c in grounded_answer.claims if c.grounded is False),
        "low_confidence": sum(1 for c in grounded_answer.claims if c.confidence == "low"),
        "claims": [
            {
                "text": c.text,
                "evidence_ids": c.evidence_ids,
                "confidence": c.confidence,
                "grounded": c.grounded,
                "grounding_score": c.grounding_score,
            }
            for c in grounded_answer.claims
        ],
        "unanswerable_aspects": grounded_answer.unanswerable_aspects,
    }
    report["groundedness_rate"] = (
        report["grounded"] / report["total_claims"] if report["total_claims"] else 0.0
    )
    return report
