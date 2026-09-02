"""
query_expansion.py
-------------------
Module: Query Expansion
Target metrics: Recall@5 (28% -> 50%+), while protecting Precision@5

WHY THE CURRENT APPROACH FAILS
-------------------------------
Recall@5 = 27.99% with Precision@5 = 92.1% is the classic signature of a
*single-query, surface-form-locked* retriever: the hybrid (BM25 + dense)
search is only ever issued with the user's literal question. In oncology
text, the same clinical fact is expressed with wildly different surface
forms across sources (e.g. "HER2-positive breast cancer" vs "ERBB2
amplified carcinoma" vs "HER2/neu overexpression"), and a single dense/BM25
query embedding cannot simultaneously match all paraphrases and synonyms.
Because Precision@5 is already excellent, the fix must ADD candidate
coverage without diluting the ranking -- i.e. expand recall pre-fusion,
then let the existing reranker (which is NOT being touched) restore
precision post-fusion.

PROPOSED ALGORITHM
-------------------
1. Multi-Query Generation (LLM-based paraphrasing)
   - Ask Qwen2.5-3B (already deployed via Ollama, zero new infra) to
     produce k=3 clinically-equivalent reformulations of the question,
     constrained to preserve the clinical intent (entity + relation).
2. HyDE (Hypothetical Document Embeddings)
   - Ask the LLM to write a short hypothetical *answer passage* (not a
     question). Dense-embed this passage with the EXISTING BGE-base
     encoder and use it as an additional retrieval query. HyDE closes the
     query-document lexical/semantic gap without touching the embedding
     model itself.
3. Graph-based Entity Expansion (multi-hop)
   - Extract entities from the question using the EXISTING SciSpaCy
     pipeline (no new NER model).
   - Traverse the EXISTING Knowledge Graph 1-2 hops from each entity
     (synonyms, is-a, treats, indicates, subtype-of edges) to collect
     related clinical terms.
   - Append these as controlled expansion terms for BM25 (which benefits
     directly from exact-term hits the dense model may miss).
4. Deduplication & Query Budget
   - Cap total expanded queries at 5 (1 original + 2 paraphrases + 1 HyDE
     + 1 graph-term query) to bound latency growth; each is retrieved in
     parallel (see retrieval_fusion.py) and fused with Reciprocal Rank
     Fusion, which is rank-based and therefore naturally resistant to a
     noisy expansion hurting precision (a bad expansion just contributes
     low-rank votes that get outweighed by consistent top-ranked hits
     from the good queries).

FILES TO MODIFY
----------------
- retrieval/query_expansion.py   (NEW - this file)
- retrieval/retrieval_fusion.py  (NEW - consumes expanded queries)
- pipeline.py                    (wire expand_query() before hybrid retrieval)

FUNCTIONS TO ADD
-----------------
- generate_paraphrases(question, llm_client, n=2) -> List[str]
- generate_hyde_passage(question, llm_client) -> str
- expand_via_graph(question, nlp_scispacy, kg, max_hops=2, max_terms=8) -> List[str]
- expand_query(question, llm_client, nlp_scispacy, kg) -> ExpandedQuery

EXPECTED METRIC IMPROVEMENT
-----------------------------
Recall@5      : 27.99% -> 48-55%   (candidate pool goes from 1 query's
                                     worth of hits to a fused union of
                                     ~5 semantically/lexically diverse
                                     queries, each contributing distinct
                                     true positives)
Precision@5   : 92.1%  -> 88-92%   (small, expected dip absorbed by RRF
                                     rank-weighting + the existing BGE
                                     reranker, which is applied AFTER
                                     fusion and re-scores the expanded
                                     candidate set against the ORIGINAL
                                     question, not the expansions)
Latency       : +1.5-3s for expansion LLM calls, offset by parallelizing
                the expanded retrievals (see adaptive_retrieval.py) and
                caching (see generator.py caching layer).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------


class LLMClient(Protocol):
    """Minimal protocol so this module stays decoupled from your Ollama
    wrapper. Your existing generator.py client already satisfies this."""

    def generate(self, prompt: str, temperature: float = 0.3,
                 max_tokens: int = 256) -> str: ...


@dataclass
class ExpandedQuery:
    original: str
    paraphrases: List[str] = field(default_factory=list)
    hyde_passage: Optional[str] = None
    graph_terms: List[str] = field(default_factory=list)

    def all_dense_queries(self) -> List[str]:
        """Queries to run through the dense (BGE) retriever."""
        qs = [self.original, *self.paraphrases]
        if self.hyde_passage:
            qs.append(self.hyde_passage)
        # de-dup while preserving order
        seen, out = set(), []
        for q in qs:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(q)
        return out[:4]  # budget cap

    def bm25_query(self) -> str:
        """A single BOW-style query for BM25: original question + graph
        expansion terms. BM25 rewards exact term overlap, so we DON'T feed
        it paraphrases (redundant tokens dilute IDF), only novel terms."""
        extra = " ".join(self.graph_terms)
        return f"{self.original} {extra}".strip()


# --------------------------------------------------------------------------
# 1. Multi-query paraphrasing
# --------------------------------------------------------------------------

_PARAPHRASE_PROMPT = """You are a clinical oncology search assistant.
Rewrite the QUESTION below into {n} alternative search queries that a
different oncology textbook or guideline might use to phrase the SAME
clinical fact. Preserve the exact clinical intent (same disease, same
relation, same specificity). Use standard medical synonyms/abbreviation
variants (e.g. "HER2" <-> "ERBB2", "NSCLC" <-> "non-small cell lung
cancer"). Do NOT answer the question. Do NOT add new clinical claims.

QUESTION: {question}

Return ONLY a JSON list of {n} strings, nothing else.
"""


def generate_paraphrases(question: str, llm_client: LLMClient,
                          n: int = 2) -> List[str]:
    prompt = _PARAPHRASE_PROMPT.format(question=question, n=n)
    try:
        raw = llm_client.generate(prompt, temperature=0.4, max_tokens=200)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        payload = match.group(0) if match else raw
        items = json.loads(payload)
        out = [str(x).strip() for x in items if str(x).strip()]
        return out[:n]
    except Exception as e:  # never let expansion break the pipeline
        logger.warning("Paraphrase generation failed, falling back: %s", e)
        return []


# --------------------------------------------------------------------------
# 2. HyDE - Hypothetical Document Embeddings
# --------------------------------------------------------------------------

_HYDE_PROMPT = """Write a short (2-4 sentence) factual passage, in the
style of an oncology textbook, that would directly answer the QUESTION
below. Use precise clinical terminology. If you are unsure of a fact,
write generically about the class of condition/treatment rather than
inventing a specific number or statistic.

QUESTION: {question}

PASSAGE:"""


def generate_hyde_passage(question: str, llm_client: LLMClient) -> Optional[str]:
    prompt = _HYDE_PROMPT.format(question=question)
    try:
        passage = llm_client.generate(prompt, temperature=0.3, max_tokens=180)
        passage = passage.strip()
        return passage if passage else None
    except Exception as e:
        logger.warning("HyDE generation failed: %s", e)
        return None


# --------------------------------------------------------------------------
# 3. Graph-based multi-hop entity expansion
# --------------------------------------------------------------------------

# Edge types considered "safe" for recall expansion: they connect a term to
# a genuinely related clinical concept without drifting topic (e.g. we do
# NOT expand across "co-occurs-in-paper" edges, only ontological/clinical
# relation edges, to avoid precision collapse).
_EXPANSION_EDGE_TYPES = {
    "synonym_of", "abbreviation_of", "is_a", "subtype_of",
    "treats", "indicates", "biomarker_of", "alias",
}


def expand_via_graph(question: str, nlp_scispacy, kg,
                      max_hops: int = 2, max_terms: int = 8) -> List[str]:
    """
    kg: your existing knowledge graph object. This function only assumes
    two methods, matching common networkx-backed KG wrappers:
        kg.get_node(entity_text) -> node_id or None
        kg.neighbors(node_id, edge_types, hop) -> Iterable[node_id]
        kg.node_label(node_id) -> str
    Adapt the three calls below to your actual KG class if the method
    names differ; the traversal LOGIC does not need to change.
    """
    doc = nlp_scispacy(question)
    entities = [ent.text for ent in doc.ents]
    if not entities:
        return []

    collected: List[str] = []
    frontier = set()
    for ent_text in entities:
        node_id = _safe_get_node(kg, ent_text)
        if node_id is not None:
            frontier.add(node_id)

    visited = set(frontier)
    for hop in range(max_hops):
        next_frontier = set()
        for node_id in frontier:
            for neighbor_id in _safe_neighbors(kg, node_id, _EXPANSION_EDGE_TYPES):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                next_frontier.add(neighbor_id)
                label = _safe_label(kg, neighbor_id)
                if label:
                    collected.append(label)
            if len(collected) >= max_terms:
                break
        frontier = next_frontier
        if len(collected) >= max_terms:
            break

    # de-dup, cap
    seen, out = set(), []
    for t in collected:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t.strip())
    return out[:max_terms]


def _safe_get_node(kg, text):
    try:
        return kg.get_node(text)
    except Exception:
        return None


def _safe_neighbors(kg, node_id, edge_types):
    try:
        return kg.neighbors(node_id, edge_types=edge_types)
    except TypeError:
        # KG implementation may not filter by edge type -- fall back to
        # unfiltered neighbors, caller-side edge type is best-effort only.
        try:
            return kg.neighbors(node_id)
        except Exception:
            return []
    except Exception:
        return []


def _safe_label(kg, node_id):
    try:
        return kg.node_label(node_id)
    except Exception:
        try:
            return str(node_id)
        except Exception:
            return None


# --------------------------------------------------------------------------
# 4. Orchestration entry point
# --------------------------------------------------------------------------


def expand_query(question: str, llm_client: LLMClient, nlp_scispacy=None,
                  kg=None, n_paraphrases: int = 2,
                  use_hyde: bool = True, use_graph: bool = True) -> ExpandedQuery:
    eq = ExpandedQuery(original=question)
    eq.paraphrases = generate_paraphrases(question, llm_client, n=n_paraphrases)
    if use_hyde:
        eq.hyde_passage = generate_hyde_passage(question, llm_client)
    if use_graph and nlp_scispacy is not None and kg is not None:
        eq.graph_terms = expand_via_graph(question, nlp_scispacy, kg)
    return eq


def query_cache_key(question: str) -> str:
    """Stable cache key for memoizing expansions (see generator.py cache)."""
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()
