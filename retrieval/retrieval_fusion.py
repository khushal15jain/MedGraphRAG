"""
retrieval_fusion.py
--------------------
Module: Retrieval Fusion (multi-query + multi-hop graph, rank fused)
Target metrics: Recall@5, Latency (via parallelism)

Does NOT replace: ChromaDB, BM25 implementation, BGE embeddings, BGE
reranker, or the Knowledge Graph. This module only orchestrates MULTIPLE
calls into your EXISTING retrievers and MERGES the result lists before
handing the fused candidate set to your EXISTING reranker.

WHY THE CURRENT APPROACH FAILS
-------------------------------
Today the pipeline runs exactly one hybrid retrieval call per question and
one graph retrieval call, then reranks. Any relevant chunk that doesn't
lexically/semantically match the ORIGINAL question phrasing is simply
never in the candidate set the reranker sees -- reranking cannot recover
recall that retrieval never produced. This is why Precision@5 is high
(the reranker is good at sorting what it's given) but Recall@5 is low
(it's not given enough).

PROPOSED ALGORITHM
-------------------
1. Fan-out: issue the dense retriever against every query in
   ExpandedQuery.all_dense_queries(), and BM25 against
   ExpandedQuery.bm25_query(), IN PARALLEL (ThreadPoolExecutor -- I/O
   bound calls to ChromaDB/BM25 index, so threads are sufficient and
   avoid multiprocessing overhead/pickling of your models).
2. Graph retrieval: run your existing graph retrieval starting from the
   question entities AND the 1-2 hop expansion entities, merging the
   resulting chunk/passage sets.
3. Reciprocal Rank Fusion (RRF) across all result lists:
       score(doc) = sum_over_lists( 1 / (k + rank_in_list(doc)) )
   RRF is chosen (over e.g. weighted score summation) because:
     - It needs no score calibration across BM25/dense/graph, which use
       incomparable score scales.
     - It is empirically robust to a low-quality expansion query: a query
       that returns junk simply contributes a few small 1/(k+rank) terms
       that a well-matching document from a good query easily outranks.
     - It is O(n log n) and adds negligible latency.
4. Deduplicate by chunk_id, cap the fused candidate list (default 30) and
   hand off to the EXISTING BGE reranker, which then re-scores against the
   ORIGINAL (unexpanded) question -- this is what protects Precision@5,
   since the final ordering is still governed by relevance to the real
   question, not to the expansions.

FILES TO MODIFY
----------------
- retrieval/retrieval_fusion.py  (NEW - this file)
- pipeline.py                    (replace single hybrid_retrieve() call
                                   with fused_retrieve())

FUNCTIONS TO ADD
-----------------
- reciprocal_rank_fusion(ranked_lists, k=60) -> List[FusedResult]
- parallel_dense_retrieve(queries, dense_retriever, top_k) -> List[List[RetrievalResult]]
- fused_retrieve(question, expanded_query, dense_retriever, bm25_retriever,
                  graph_retriever, reranker, final_k) -> List[RetrievalResult]

EXPECTED METRIC IMPROVEMENT
-----------------------------
Recall@5     : 27.99% -> 48-55% (see query_expansion.py for the
               decomposition of where the gain comes from)
Precision@5  : 92.1%  -> 88-92%
Latency      : fan-out is parallelized so wall-clock cost is
               ~max(individual call latency) + fusion overhead (~10-30ms),
               not sum(call latency). Combined with adaptive_retrieval.py's
               early-exit, net latency impact is close to neutral.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    source: str = ""
    page: Optional[int] = None
    score: float = 0.0
    retriever: str = ""  # "dense" | "bm25" | "graph"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult:
    chunk_id: str
    rrf_score: float
    contributing_retrievers: List[str]
    result: RetrievalResult


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]], k: int = 60
) -> List[FusedResult]:
    """
    ranked_lists: each inner list is already sorted best-first (rank 0 =
    best) coming from one (retriever, query) pair.
    k: RRF damping constant. k=60 is the standard value from the original
    RRF paper (Cormack et al., 2009) and is not a sensitive
    hyperparameter -- values 20-100 give near-identical fused orderings.
    """
    scores: Dict[str, float] = {}
    best_result: Dict[str, RetrievalResult] = {}
    contributors: Dict[str, List[str]] = {}

    for result_list in ranked_lists:
        for rank, res in enumerate(result_list):
            scores[res.chunk_id] = scores.get(res.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            contributors.setdefault(res.chunk_id, []).append(res.retriever)
            # Keep the highest-original-score instance of this chunk for
            # its text/metadata (retriever scores aren't comparable across
            # sources, so we just need ANY faithful copy of the chunk).
            if res.chunk_id not in best_result or res.score > best_result[res.chunk_id].score:
                best_result[res.chunk_id] = res

    fused = [
        FusedResult(
            chunk_id=cid,
            rrf_score=score,
            contributing_retrievers=contributors[cid],
            result=best_result[cid],
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda f: f.rrf_score, reverse=True)
    return fused


# --------------------------------------------------------------------------
# Parallel fan-out
# --------------------------------------------------------------------------


def parallel_retrieve(
    jobs: Sequence[Callable[[], List[RetrievalResult]]], max_workers: int = 6
) -> List[List[RetrievalResult]]:
    """Runs each zero-arg retrieval callable concurrently. Any single job
    failure is logged and treated as an empty result list, so one bad
    query never kills the whole retrieval."""
    results: List[Optional[List[RetrievalResult]]] = [None] * len(jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(job): i for i, job in enumerate(jobs)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("Retrieval job %d failed: %s", idx, e)
                results[idx] = []
    return [r or [] for r in results]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def fused_retrieve(
    question: str,
    expanded_query,  # ExpandedQuery from query_expansion.py
    dense_retriever: Callable[[str, int], List[RetrievalResult]],
    bm25_retriever: Callable[[str, int], List[RetrievalResult]],
    graph_retriever: Callable[[str, List[str], int], List[RetrievalResult]],
    reranker: Callable[[str, List[RetrievalResult], int], List[RetrievalResult]],
    dense_top_k: int = 15,
    bm25_top_k: int = 15,
    graph_top_k: int = 15,
    fused_candidate_cap: int = 30,
    final_k: int = 5,
) -> List[RetrievalResult]:
    """
    dense_retriever(query, top_k)        -> your existing ChromaDB call
    bm25_retriever(query, top_k)         -> your existing BM25 call
    graph_retriever(question, terms, k)  -> your existing graph traversal
                                             retrieval; `terms` includes
                                             expanded_query.graph_terms so
                                             multi-hop context is reachable
    reranker(question, candidates, k)    -> your EXISTING BGE reranker,
                                             UNCHANGED. Always reranks
                                             against the ORIGINAL question.
    """
    jobs: List[Callable[[], List[RetrievalResult]]] = []

    for q in expanded_query.all_dense_queries():
        jobs.append(lambda q=q: dense_retriever(q, dense_top_k))

    bm25_q = expanded_query.bm25_query()
    jobs.append(lambda: bm25_retriever(bm25_q, bm25_top_k))

    jobs.append(
        lambda: graph_retriever(question, expanded_query.graph_terms, graph_top_k)
    )

    ranked_lists = parallel_retrieve(jobs)

    fused = reciprocal_rank_fusion(ranked_lists, k=60)
    candidates = [f.result for f in fused[:fused_candidate_cap]]

    if not candidates:
        logger.warning("Fused retrieval returned zero candidates for question: %s", question)
        return []

    # Existing reranker has final say, protecting precision.
    reranked = reranker(question, candidates, final_k)
    return reranked
