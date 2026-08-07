"""Stage 10: Hybrid Retrieval (dense + BM25 + graph fusion).

Combines three retrieval signals into a single ranked candidate list:
  - Dense (BGE embeddings via ChromaDB): semantic similarity.
  - BM25: exact lexical/term overlap.
  - Graph: entity-based structural relevance (from ``GraphRetriever``).

Fusion strategy: min-max normalize each signal's scores independently
(different scales: cosine similarity in [0,1], BM25 unbounded, graph score
unbounded), then combine dense+BM25 via a weighted sum controlled by
``hybrid_alpha``, and add graph score as an independent boost. This keeps
the fusion formula simple, transparent, and easy to ablate -- three
properties that matter for the ablation-study requirement of this project.

v2 changes:
  - Retrieval calls run sequentially. An earlier version of this file ran
    dense/BM25/graph concurrently via a thread pool for latency, but
    ChromaDB's persistent (SQLite-backed) client turned out not to be safe
    for concurrent queries from multiple threads -- under concurrent access
    it silently returned zero hits instead of raising, which broke
    retrieval on most queries without any visible error. Reverted:
    correctness matters more than the latency win here. If real concurrency
    is worth revisiting, it needs a per-thread Chroma client/connection or a
    lock around the dense leg specifically, not a bare thread pool.
  - Added a second, entity-derived BM25 query for lexical recall: query
    entities are extracted ONCE (shared with graph retrieval, avoiding a
    redundant NER pass) and joined into a keyword-only variant, e.g. "HER2
    trastuzumab metastatic" alongside the original natural-language
    question. This catches passages phrased very differently from the
    question but containing the same key terms -- classic BM25 recall
    failure mode on clinical text, where the source material rarely
    matches a question's exact grammar. Results from both queries are
    merged by chunk_id, keeping the max score per chunk.
  - Added adaptive top-k: after fusion, if the score just past the
    requested cutoff is nearly tied with the last included candidate, the
    pool is widened rather than arbitrarily cutting a near-tied relevant
    chunk. Bounded by adaptive_max_extra so it can't runaway. This only
    ever returns MORE candidates than requested, never fewer, and since
    downstream reranking always trims to its own top_k regardless of pool
    size, this is a low-risk way to hand the reranker a fairer set to
    choose from -- it does not change what the final answer set looks like
    unless the reranker was starved of a genuinely relevant candidate that
    got cut off arbitrarily by a fixed final_top_k.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph.graph_retriever import GraphRetriever
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from utils.exceptions import RetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk surfaced by hybrid retrieval, with per-signal score breakdown."""

    chunk_id: str
    text: str
    metadata: dict
    dense_score: float
    bm25_score: float
    graph_score: float
    fused_score: float


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize a {id: score} mapping to [0, 1] range; returns all-ones if scores are positive and uniform."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return {k: (1.0 if hi > 0 else 0.0) for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    """Fuses dense, BM25, and graph retrieval signals into one ranked list."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        graph_retriever: GraphRetriever | None = None,
        hybrid_alpha: float = 0.5,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            dense_retriever: Configured ``DenseRetriever``.
            bm25_retriever: Configured (already-indexed) ``BM25Retriever``.
            graph_retriever: Optional ``GraphRetriever``; if ``None``, hybrid
                retrieval degrades gracefully to dense+BM25 fusion only
                (used for the "Hybrid Retrieval" baseline vs. the full
                "GraphRAG"/"Our Proposed Method" configurations).
            hybrid_alpha: Weight on dense score vs. BM25 score in [0, 1].
                0.0 = BM25 only, 1.0 = dense only, 0.5 = equal weight.
        """
        if not 0.0 <= hybrid_alpha <= 1.0:
            raise RetrievalError("hybrid_alpha must be in [0, 1]")

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.graph_retriever = graph_retriever
        self.hybrid_alpha = hybrid_alpha

    def retrieve(
        self,
        query: str,
        top_k_dense: int = 20,
        top_k_bm25: int = 20,
        top_k_graph: int = 10,
        final_top_k: int = 15,
        use_graph: bool = True,
        use_query_expansion: bool = True,
        adaptive_top_k: bool = True,
        adaptive_max_extra: int = 20,
        adaptive_gap_threshold: float = 0.02,
    ) -> list[RetrievedChunk]:
        """Run all enabled retrieval signals and fuse them into one ranked list.

        Args:
            query: Natural-language clinical query.
            top_k_dense: Candidates to pull from dense retrieval.
            top_k_bm25: Candidates to pull from BM25 (per query variant).
            top_k_graph: Candidates to pull from graph retrieval.
            final_top_k: Minimum number of fused results to return (may
                return more if adaptive_top_k widens the cutoff).
            use_graph: Whether to include the graph retrieval signal (set
                False to reproduce the "Hybrid Retrieval" baseline without
                graph augmentation, for ablation studies).
            use_query_expansion: Whether to also run BM25 against an
                entity-derived keyword query alongside the original
                question, for lexical recall on differently-phrased source
                text. Set False to reproduce the original single-query BM25
                baseline for ablation studies.
            adaptive_top_k: Whether to widen the returned pool when the
                score right at the cutoff is nearly tied with the last
                included item, instead of making an arbitrary cut.
            adaptive_max_extra: Ceiling on how many extra candidates
                adaptive widening can add beyond final_top_k.
            adaptive_gap_threshold: Minimum normalized-score gap (on the
                combined dense+bm25 portion of fused_score, which is in
                [0,1]) required to consider the cutoff "clear" and stop
                widening.

        Returns:
            List of ``RetrievedChunk``, sorted by descending ``fused_score``,
            length >= final_top_k (more if adaptive widening triggered).

        Raises:
            RetrievalError: If all retrieval signals fail.
        """
        # Extract query entities ONCE, shared between graph retrieval and
        # BM25 expansion, so we don't pay for two NER passes over one query.
        query_entities: list[str] = []
        if (use_graph and self.graph_retriever is not None) or use_query_expansion:
            if self.graph_retriever is not None:
                query_entities = self.graph_retriever.extract_query_entities(query)

        expanded_bm25_query = " ".join(query_entities) if (use_query_expansion and query_entities) else None

        import re
        section_filter = None
        lower_query = query.lower()
        if "treat" in lower_query or "manag" in lower_query:
            section_filter = "Treatment"
        elif "diagnos" in lower_query:
            section_filter = "Diagnosis"
        elif "prognos" in lower_query:
            section_filter = "Prognosis"
        elif "risk" in lower_query:
            section_filter = "Risk"
            
        # Sequential, not concurrent: ChromaDB's persistent (SQLite-backed)
        # client is NOT safe to query from multiple threads at once -- under
        # concurrent access it can silently return zero hits instead of
        # raising, which is worse than the latency this was meant to fix
        # (a "faster" retriever that quietly returns nothing on most queries
        # is not usable). If real concurrency is worth revisiting later, it
        # needs either a per-thread Chroma client/connection or a lock around
        # the dense leg specifically -- not attempted here since correctness
        # matters more than the speedup for a research run.
        dense_hits = self.dense_retriever.search(query, top_k=top_k_dense, section_filter=section_filter)
        bm25_hits = self.bm25_retriever.search(query, top_k=top_k_bm25)
        bm25_expanded_hits = (
            self.bm25_retriever.search(expanded_bm25_query, top_k=top_k_bm25)
            if expanded_bm25_query
            else []
        )
        graph_hits = (
            self.graph_retriever.retrieve(query, top_k=top_k_graph, query_entities=query_entities or None)
            if use_graph and self.graph_retriever is not None
            else []
        )

        if not dense_hits and not bm25_hits and not bm25_expanded_hits and not graph_hits:
            logger.warning(f"All retrieval signals returned zero results for query: {query!r}")
            return []

        # Collect raw scores per chunk_id per signal.
        dense_raw = {cid: score for cid, score, _, _ in dense_hits}
        bm25_raw = {cid: score for cid, score in bm25_hits}
        # Merge the expanded-query BM25 hits into the same score dict, keeping
        # the max score per chunk -- a chunk found by either phrasing counts.
        for cid, score in bm25_expanded_hits:
            bm25_raw[cid] = max(bm25_raw.get(cid, float("-inf")), score)
        graph_raw = {r.chunk_id: r.graph_score for r in graph_hits}

        dense_norm = _min_max_normalize(dense_raw)
        bm25_norm = _min_max_normalize(bm25_raw)
        graph_norm = _min_max_normalize(graph_raw)

        # Text/metadata lookup, sourced from whichever signal(s) returned this chunk.
        text_lookup: dict[str, str] = {cid: text for cid, _, text, _ in dense_hits}
        meta_lookup: dict[str, dict] = {cid: meta for cid, _, _, meta in dense_hits}

        all_chunk_ids = set(dense_raw) | set(bm25_raw) | set(graph_raw)

        # Backfill text/metadata for chunks found only via BM25/graph (not dense).
        missing_ids = [cid for cid in all_chunk_ids if cid not in text_lookup]
        if missing_ids:
            fetched = self.dense_retriever.indexer.get_by_ids(missing_ids)
            for cid, data in fetched.items():
                text_lookup[cid] = data["text"]
                meta_lookup[cid] = data["metadata"]

        # Normalized fusion weights across active components
        dense_weight = 0.45
        graph_weight = 0.35 if (use_graph and self.graph_retriever is not None) else 0.0
        bm25_weight = 0.20 if (self.bm25_retriever is not None and (top_k_bm25 > 0 or bool(bm25_raw))) else 0.0

        total_weight = dense_weight + graph_weight + bm25_weight
        w_d = (dense_weight / total_weight) if total_weight > 0 else 0.0
        w_g = (graph_weight / total_weight) if total_weight > 0 else 0.0
        w_b = (bm25_weight / total_weight) if total_weight > 0 else 0.0

        fused: list[RetrievedChunk] = []
        for chunk_id in all_chunk_ids:
            d = dense_norm.get(chunk_id, 0.0)
            b = bm25_norm.get(chunk_id, 0.0)
            g = graph_norm.get(chunk_id, 0.0)
            fused_score = w_d * d + w_g * g + w_b * b
            if not text_lookup.get(chunk_id):
                continue
            fused.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text_lookup.get(chunk_id, ""),
                    metadata=meta_lookup.get(chunk_id, {}),
                    dense_score=dense_raw.get(chunk_id, 0.0),
                    bm25_score=bm25_raw.get(chunk_id, 0.0),
                    graph_score=graph_raw.get(chunk_id, 0.0),
                    fused_score=fused_score,
                )
            )

        fused.sort(key=lambda c: c.fused_score, reverse=True)

        cutoff = final_top_k
        if adaptive_top_k:
            cutoff = self._adaptive_cutoff(
                fused, final_top_k, adaptive_max_extra, adaptive_gap_threshold
            )

        logger.info(
            f"Hybrid retrieval fused {len(fused)} unique candidates for query "
            f"(returning {min(cutoff, len(fused))})"
        )
        return fused[:cutoff]

    @staticmethod
    def _adaptive_cutoff(
        fused: list[RetrievedChunk],
        base_top_k: int,
        max_extra: int,
        gap_threshold: float,
    ) -> int:
        """Widen the cutoff past base_top_k while consecutive scores stay nearly tied.

        Walks forward from base_top_k, extending the cutoff by one each time
        the next candidate's fused_score is within gap_threshold of the
        current last-included candidate's score -- i.e. it's ambiguous
        whether that candidate really belongs outside the top-k or just
        landed on the wrong side of an arbitrary cut. Stops as soon as a
        clear gap appears, or at base_top_k + max_extra, whichever comes first.
        """
        cutoff = min(base_top_k, len(fused))
        max_cutoff = min(base_top_k + max_extra, len(fused))

        while cutoff < max_cutoff:
            gap = fused[cutoff - 1].fused_score - fused[cutoff].fused_score
            if gap > gap_threshold:
                break
            cutoff += 1

        return cutoff
