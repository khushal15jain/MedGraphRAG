"""Stage 17: Benchmarking — Baseline Method Definitions.

Defines the comparison methods required by the research design:
  - Vanilla RAG:      single-pass dense retrieval, top-k, no reranking, no graph.
  - Dense Retrieval:  same as above (kept as an explicit alias for clarity
                      in result tables, since "Vanilla RAG" and "Dense
                      Retrieval" are conceptually the same pipeline here).
  - BM25:             sparse lexical retrieval only.
  - Hybrid Retrieval: dense + BM25 fusion, no graph, no rerank.
  - GraphRAG:         graph retrieval only (entity-seeded traversal).
  - Our Proposed:     full pipeline — hybrid (dense+BM25) + graph fusion + cross-encoder rerank.

Each method exposes a uniform ``run(query) -> list[dict]`` interface
(dicts with at least ``chunk_id`` and ``text`` keys) so ``run_benchmark.py``
can evaluate every method identically with the same RAGAS/DeepEval/
hallucination pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from graph.graph_retriever import GraphRetriever
from reranker.reranker import BGEReranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalMethod(Protocol):
    """Common interface implemented by every benchmarked retrieval method."""

    name: str

    def run(self, query: str, top_k: int) -> list[dict]:
        ...


@dataclass
class VanillaRAGMethod:
    """Dense-only retrieval, no reranking, no graph — the "vanilla RAG" baseline."""

    dense_retriever: DenseRetriever
    name: str = "Vanilla RAG (Dense)"

    def run(self, query: str, top_k: int = 5) -> list[dict]:
        hits = self.dense_retriever.search(query, top_k=top_k)
        return [{"chunk_id": cid, "text": text, "metadata": meta} for cid, _, text, meta in hits]


@dataclass
class BM25Method:
    """Sparse lexical-only retrieval baseline."""

    bm25_retriever: BM25Retriever
    chunk_text_lookup: dict[str, str]
    chunk_meta_lookup: dict[str, dict]
    name: str = "BM25"

    def run(self, query: str, top_k: int = 5) -> list[dict]:
        hits = self.bm25_retriever.search(query, top_k=top_k)
        return [
            {
                "chunk_id": cid,
                "text": self.chunk_text_lookup.get(cid, ""),
                "metadata": self.chunk_meta_lookup.get(cid, {}),
            }
            for cid, _ in hits
        ]


@dataclass
class HybridRetrievalMethod:
    """Dense + BM25 fusion, no graph, no rerank."""

    hybrid_retriever: HybridRetriever
    name: str = "Hybrid Retrieval (Dense+BM25)"

    def run(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.hybrid_retriever.retrieve(query, final_top_k=top_k, use_graph=False)
        return [{"chunk_id": r.chunk_id, "text": r.text, "metadata": r.metadata} for r in results]


@dataclass
class GraphRAGMethod:
    """Graph-only retrieval baseline (entity-seeded traversal, no dense/BM25)."""

    graph_retriever: GraphRetriever
    chunk_text_lookup: dict[str, str]
    chunk_meta_lookup: dict[str, dict]
    name: str = "GraphRAG (Graph-only)"

    def run(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.graph_retriever.retrieve(query, top_k=top_k)
        return [
            {
                "chunk_id": r.chunk_id,
                "text": self.chunk_text_lookup.get(r.chunk_id, ""),
                "metadata": self.chunk_meta_lookup.get(r.chunk_id, {}),
            }
            for r in results
        ]


@dataclass
class ProposedMethod:
    """Full MedGraphRAG pipeline: hybrid (dense+BM25) + graph fusion + cross-encoder rerank."""

    hybrid_retriever: HybridRetriever
    reranker: BGEReranker
    name: str = "MedGraphRAG (Proposed)"

    def run(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = self.hybrid_retriever.retrieve(query, final_top_k=top_k * 2, use_graph=True)
        candidate_dicts = [
            {"chunk_id": r.chunk_id, "text": r.text, "metadata": r.metadata} for r in candidates
        ]
        if not candidate_dicts:
            return []
        return self.reranker.rerank(query, candidate_dicts, top_k=top_k)
