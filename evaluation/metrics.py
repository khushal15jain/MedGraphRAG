"""evaluation/metrics.py
----------------------
Extended Evaluation Metrics Suite for MedGraphRAG.

Computes comprehensive lexical, retrieval ranking, generation, citation, and clinical metrics:
- Lexical Overlap: BLEU-1/2/4, ROUGE-1/2/L, METEOR, SQuAD-style Answer F1.
- Retrieval Ranking: Precision@5, Recall@5, MRR, nDCG@5, HitRate@5.
- Context Interface: Context Precision, Context Recall, Context Relevancy / Groundedness.
- Citation & Evidence: Citation Precision, Citation Recall, Citation Coverage, Explainability.
- Semantic Similarity: Embedding cosine similarity between generated and reference answers.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Lazy imports: keep this module importable even if optional deps aren't
# installed yet, but fail loudly & specifically at call time.
# --------------------------------------------------------------------------


def _lazy_nltk():
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
            try:
                nltk.data.find(f"tokenizers/{pkg}")
            except LookupError:
                try:
                    nltk.data.find(f"corpora/{pkg}")
                except LookupError:
                    try:
                        nltk.download(pkg, quiet=True)
                    except Exception:
                        pass
        return sentence_bleu, SmoothingFunction, meteor_score
    except ImportError as e:
        raise ImportError("pip install nltk  (required for BLEU/METEOR)") from e


def _lazy_rouge():
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer
    except ImportError as e:
        raise ImportError("pip install rouge-score  (required for ROUGE)") from e


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\-]+", text.lower())


# --------------------------------------------------------------------------
# 1. BLEU-1 / BLEU-2 / BLEU-4
# --------------------------------------------------------------------------


def compute_bleu(candidate: str, reference: str) -> Dict[str, float]:
    sentence_bleu, SmoothingFunction, _ = _lazy_nltk()
    ref_tokens = [_tokenize(reference)]
    cand_tokens = _tokenize(candidate)
    smoothie = SmoothingFunction().method1
    if not cand_tokens:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu4": 0.0}
    return {
        "bleu1": sentence_bleu(ref_tokens, cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie),
        "bleu2": sentence_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        "bleu4": sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }


# --------------------------------------------------------------------------
# 2. ROUGE-1 / ROUGE-2 / ROUGE-L
# --------------------------------------------------------------------------


def compute_rouge(candidate: str, reference: str) -> Dict[str, float]:
    rouge_scorer = _lazy_rouge()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


# --------------------------------------------------------------------------
# 3. METEOR
# --------------------------------------------------------------------------


def compute_meteor(candidate: str, reference: str) -> float:
    _, _, meteor_score = _lazy_nltk()
    try:
        return meteor_score([_tokenize(reference)], _tokenize(candidate))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# 4. Answer F1 (SQuAD-style token F1)
# --------------------------------------------------------------------------


def _extract_clean_text(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    s = raw.strip()
    if (s.startswith("{") and "answer" in s) or s.startswith("```"):
        try:
            fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
            payload_str = fenced.group(1) if fenced else s
            data = json.loads(payload_str)
            if isinstance(data, dict) and "answer" in data and isinstance(data["answer"], str):
                s = data["answer"]
        except Exception:
            match = re.search(r'"answer"\s*:\s*"(.*?)"', s, re.DOTALL)
            if match:
                s = match.group(1).replace("\\n", "\n").replace('\\"', '"')
    match = re.search(r"\n?\s*Confidence:\s*(High|Medium|Low)\s*\.?\s*$", s, re.IGNORECASE)
    if match:
        s = s[: match.start()].strip()
    return s.strip()


def compute_answer_f1(candidate: str, reference: str) -> float:
    cand_tokens = _tokenize(_extract_clean_text(candidate))
    ref_tokens = _tokenize(_extract_clean_text(reference))
    if not cand_tokens or not ref_tokens:
        return 0.0
    common: Dict[str, int] = {}
    for t in cand_tokens:
        common[t] = common.get(t, 0) + 1
    ref_counts: Dict[str, int] = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = sum(min(common.get(t, 0), c) for t, c in ref_counts.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# 5. MRR / nDCG (retrieval ranking quality)
# --------------------------------------------------------------------------


def compute_mrr(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: Optional[int] = None) -> float:
    relevant_set = set(relevant_ids)
    ids = retrieved_ids[:k] if k else retrieved_ids
    dcg = sum(
        (1.0 / math.log2(rank + 1)) for rank, doc_id in enumerate(ids, start=1) if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), len(ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------
# 6. Context Precision / Recall / Relevancy
# --------------------------------------------------------------------------


def compute_context_precision(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    if not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits = sum(1 for d in retrieved_ids if d in relevant_set)
    return hits / len(retrieved_ids)


def compute_context_recall(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    if not relevant_ids:
        return 0.0
    retrieved_set = set(retrieved_ids)
    hits = sum(1 for d in relevant_ids if d in retrieved_set)
    return hits / len(relevant_ids)


def compute_context_relevancy(retrieved_texts: Sequence[str], question: str, embedder) -> float:
    """Average embedding similarity of each retrieved chunk to the
    question -- measures whether retrieved context stays on-topic even
    when gold relevance labels aren't available for every chunk (useful
    for chunks retrieved via expansion that aren't in the hand-labeled
    gold set)."""
    if not retrieved_texts:
        return 0.0
    import numpy as np
    vecs = embedder.encode([question] + list(retrieved_texts), normalize_embeddings=True)
    q_vec, ctx_vecs = vecs[0], vecs[1:]
    sims = ctx_vecs @ q_vec
    return float(np.mean(sims))


# --------------------------------------------------------------------------
# 7. Citation Precision / Recall / Coverage, Evidence Coverage
# (novel metrics enabled by the structured citation contract)
# --------------------------------------------------------------------------


def compute_citation_precision(claims, evidence_by_id: Dict[str, str]) -> float:
    """Of all evidence_ids cited across claims, what fraction point to
    evidence that is actually grounded (per grounding_checker)?"""
    total_citations = 0
    correct_citations = 0
    for c in claims:
        for _ in c.evidence_ids:
            total_citations += 1
            if c.grounded is True:
                correct_citations += 1
    return correct_citations / total_citations if total_citations else 0.0


def compute_citation_recall(claims, gold_evidence_ids: Sequence[str]) -> float:
    """Of the gold evidence chunks known to support the reference answer,
    what fraction were actually cited by SOME claim in the generated
    answer?"""
    if not gold_evidence_ids:
        return 0.0
    cited = set()
    for c in claims:
        cited.update(c.evidence_ids)
    hits = sum(1 for g in gold_evidence_ids if g in cited)
    return hits / len(gold_evidence_ids)


def compute_citation_coverage(claims) -> float:
    """Fraction of claims that carry at least one citation (vs. bare,
    uncited assertions) -- a direct, cheap proxy for explainability."""
    if not claims:
        return 0.0
    cited = sum(1 for c in claims if c.evidence_ids)
    return cited / len(claims)


def compute_evidence_coverage(claims, evidence_list) -> float:
    """Fraction of RETRIEVED evidence that was actually used (cited by
    at least one claim) -- low values indicate retrieval is bringing back
    excess/irrelevant context relative to what generation needed."""
    if not evidence_list:
        return 0.0
    used_ids = set()
    for c in claims:
        used_ids.update(c.evidence_ids)
    evidence_ids = {e.evidence_id for e in evidence_list}
    return len(used_ids & evidence_ids) / len(evidence_ids)


# --------------------------------------------------------------------------
# 8. Semantic similarity
# --------------------------------------------------------------------------


def compute_semantic_similarity(candidate: str, reference: str, embedder) -> float:
    import numpy as np
    vecs = embedder.encode([candidate, reference], normalize_embeddings=True)
    return float(np.dot(vecs[0], vecs[1]))


# --------------------------------------------------------------------------
# 9. Single entry point
# --------------------------------------------------------------------------


def compute_all_metrics(
    candidate_answer: str,
    reference_answer: str,
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    retrieved_texts: Sequence[str],
    question: str,
    claims,
    evidence_list,
    gold_evidence_ids: Sequence[str],
    embedder,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update({f"bleu_{k[4:]}": v for k, v in compute_bleu(candidate_answer, reference_answer).items()})
    metrics.update(compute_rouge(candidate_answer, reference_answer))
    metrics["meteor"] = compute_meteor(candidate_answer, reference_answer)
    metrics["answer_f1"] = compute_answer_f1(candidate_answer, reference_answer)
    metrics["mrr"] = compute_mrr(retrieved_ids, relevant_ids)
    metrics["ndcg"] = compute_ndcg(retrieved_ids, relevant_ids, k=5)
    metrics["context_precision"] = compute_context_precision(retrieved_ids, relevant_ids)
    metrics["context_recall"] = compute_context_recall(retrieved_ids, relevant_ids)
    metrics["context_relevancy"] = compute_context_relevancy(retrieved_texts, question, embedder)
    evidence_by_id = {e.evidence_id: e.text for e in evidence_list}
    metrics["citation_precision"] = compute_citation_precision(claims, evidence_by_id)
    metrics["citation_recall"] = compute_citation_recall(claims, gold_evidence_ids)
    metrics["citation_coverage"] = compute_citation_coverage(claims)
    metrics["evidence_coverage"] = compute_evidence_coverage(claims, evidence_list)
    metrics["semantic_similarity"] = compute_semantic_similarity(candidate_answer, reference_answer, embedder)
    return metrics
