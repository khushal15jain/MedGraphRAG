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

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

# --------------------------------------------------------------------------
# Lazy imports: keep this module importable even if optional deps aren't
# installed yet, but fail loudly & specifically at call time.
# --------------------------------------------------------------------------


def compute_clinical_reliability(
    faithfulness: float,
    groundedness: float,
    hallucination: float | None = None,
    safety: float = 1.0,
    completeness: float = 1.0,
) -> float:
    """Compute Clinical Reliability deterministically via the formal publication equation:

    Clinical Reliability = 0.30 * Faithfulness + 0.30 * Groundedness + 0.20 * (1 - Hallucination)
                           + 0.10 * Safety + 0.10 * Completeness
    where Safety and Completeness are normalized in [0, 1].
    """
    if hallucination is None:
        hallucination = max(0.0, 1.0 - faithfulness)
    s = min(1.0, max(0.0, safety))
    c = min(1.0, max(0.0, completeness))
    f = min(1.0, max(0.0, faithfulness))
    g = min(1.0, max(0.0, groundedness))
    h = min(1.0, max(0.0, hallucination))
    score = 0.30 * f + 0.30 * g + 0.20 * (1.0 - h) + 0.10 * s + 0.10 * c
    return float(round(score, 4))


def _lazy_nltk():
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
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


def _lcs_length(x: List[str], y: List[str]) -> int:
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _ngram_f1(cand_tokens: List[str], ref_tokens: List[str], n: int) -> float:
    if len(cand_tokens) < n or len(ref_tokens) < n:
        return 0.0
    cand_ngrams: Dict[Tuple[str, ...], int] = {}
    for i in range(len(cand_tokens) - n + 1):
        g = tuple(cand_tokens[i : i + n])
        cand_ngrams[g] = cand_ngrams.get(g, 0) + 1
    ref_ngrams: Dict[Tuple[str, ...], int] = {}
    for i in range(len(ref_tokens) - n + 1):
        g = tuple(ref_tokens[i : i + n])
        ref_ngrams[g] = ref_ngrams.get(g, 0) + 1
    overlap = sum(min(cand_ngrams.get(g, 0), c) for g, c in ref_ngrams.items())
    if overlap == 0:
        return 0.0
    precision = overlap / len(cand_tokens) if n == 1 else overlap / (len(cand_tokens) - n + 1)
    recall = overlap / len(ref_tokens) if n == 1 else overlap / (len(ref_tokens) - n + 1)
    return 2 * precision * recall / (precision + recall)


def compute_rouge(candidate: str, reference: str) -> Dict[str, float]:
    cand_tokens = _tokenize(_extract_clean_text(candidate))
    ref_tokens = _tokenize(_extract_clean_text(reference))
    if not cand_tokens or not ref_tokens:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    r1 = _ngram_f1(cand_tokens, ref_tokens, 1)
    r2 = _ngram_f1(cand_tokens, ref_tokens, 2)
    lcs = _lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        rl = 0.0
    else:
        prec = lcs / len(cand_tokens)
        rec = lcs / len(ref_tokens)
        rl = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
    return {
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rl,
    }


def rouge_1(candidate: str, reference: str) -> float:
    return compute_rouge(candidate, reference)["rouge1"]


def rouge_2(candidate: str, reference: str) -> float:
    return compute_rouge(candidate, reference)["rouge2"]


def rouge_l(candidate: str, reference: str) -> float:
    return compute_rouge(candidate, reference)["rougeL"]


compute_rouge_1 = rouge_1
compute_rouge_2 = rouge_2
compute_rouge_l = rouge_l


def bleu_n(candidate: str, reference: str, n: int = 4) -> float:
    res = compute_bleu(candidate, reference)
    return res.get(f"bleu{n}", 0.0)


def meteor(candidate: str, reference: str) -> float:
    return compute_meteor(candidate, reference)


def answer_f1(candidate: str, reference: str) -> float:
    return compute_answer_f1(candidate, reference)


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    if not retrieved_ids or k <= 0:
        return 0.0
    sub = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    return sum(1 for d in sub if d in rel_set) / len(sub)


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    if not relevant_ids:
        return 0.0
    sub = set(retrieved_ids[:k])
    rel_set = set(relevant_ids)
    return sum(1 for d in rel_set if d in sub) / len(rel_set)


def hit_rate_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    sub = set(retrieved_ids[:k])
    rel_set = set(relevant_ids)
    return 1.0 if (sub & rel_set) else 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    return compute_ndcg(retrieved_ids, relevant_ids, k=k)


# --------------------------------------------------------------------------
# 3. METEOR
# --------------------------------------------------------------------------


def compute_meteor(candidate: str, reference: str) -> float:
    cand_tokens = _tokenize(_extract_clean_text(candidate))
    ref_tokens = _tokenize(_extract_clean_text(reference))
    if not cand_tokens or not ref_tokens:
        return 0.0
    try:
        _, _, meteor_score = _lazy_nltk()
        return meteor_score([ref_tokens], cand_tokens)
    except Exception:
        cand_counts: Dict[str, int] = {}
        for t in cand_tokens:
            cand_counts[t] = cand_counts.get(t, 0) + 1
        ref_counts: Dict[str, int] = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        matches = sum(min(cand_counts.get(t, 0), c) for t, c in ref_counts.items())
        if matches == 0:
            return 0.0
        p = matches / len(cand_tokens)
        r = matches / len(ref_tokens)
        f_mean = (10 * p * r) / (r + 9 * p) if (r + 9 * p) > 0 else 0.0
        penalty = 0.5 * ((1.0 / matches) ** 3)
        return float(max(0.0, f_mean * (1.0 - penalty)))


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


def mrr_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    return compute_mrr(retrieved_ids[:k], relevant_ids)


def compute_ndcg(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: Optional[int] = None) -> float:
    relevant_set = set(relevant_ids)
    ids = retrieved_ids[:k] if k else retrieved_ids
    dcg = sum(
        (1.0 / math.log2(rank + 1)) for rank, doc_id in enumerate(ids, start=1) if doc_id in relevant_set
    )
    ideal_hits = min(len(relevant_set), len(ids))
    idcg = sum((1.0 / math.log2(rank + 1)) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int = 5) -> float:
    return compute_ndcg(retrieved_ids, relevant_ids, k=k)


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


def compute_publication_metrics_summary(data_or_path: Any = None) -> Dict[str, Any]:
    """Return the authoritative publication results summary matching results/publication_results.json."""
    pub_file = Path(__file__).resolve().parent.parent / "results" / "publication_results.json"
    if pub_file.exists():
        with open(pub_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "dataset": "MedGraphRAG Medical Oncology QA Benchmark",
        "n_main_dataset": 200,
        "n_ablation_dataset": 100,
        "ablation_seed": 42,
        "random_seed": 42,
        "human_expert_validation": "Human expert validation was not conducted. No synthetic human scores generated.",
        "configuration": {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "reranker_model": "BAAI/bge-reranker-base",
            "llm_generator": "Local Qwen2.5-3B-Instruct / Llama-3.2-3B",
            "graph_traversal": "SciSpaCy NER + NetworkX BFS (Hop-Distance Decay S_graph = 1/(1+d))",
            "grounding_threshold_tau_g": 0.65,
            "refusal_threshold_tau_l": 0.45,
        },
    }
