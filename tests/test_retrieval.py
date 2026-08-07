"""Unit tests for retrieval/bm25_retriever.py and retrieval/hybrid_retriever.py.

BM25 tests use the real ``rank_bm25`` library directly (lightweight, pure
Python, no model download). Hybrid retriever tests use lightweight fakes
for the dense/graph retrievers so fusion logic can be verified in
isolation without ChromaDB or spaCy dependencies.
"""

from __future__ import annotations

import pytest

from graph.graph_retriever import GraphRetrievalResult
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever, _min_max_normalize
from utils.exceptions import RetrievalError


class TestMinMaxNormalize:
    def test_normalizes_to_zero_one_range(self) -> None:
        result = _min_max_normalize({"a": 1.0, "b": 3.0, "c": 5.0})
        assert result["a"] == 0.0
        assert result["c"] == 1.0
        assert result["b"] == 0.5

    def test_uniform_scores_return_zero(self) -> None:
        result = _min_max_normalize({"a": 2.0, "b": 2.0})
        assert result == {"a": 0.0, "b": 0.0}

    def test_empty_dict_returns_empty(self) -> None:
        assert _min_max_normalize({}) == {}


class TestBM25Retriever:
    def test_search_before_index_raises(self) -> None:
        retriever = BM25Retriever()
        with pytest.raises(RetrievalError):
            retriever.search("trastuzumab")

    def test_index_mismatched_lengths_raises(self) -> None:
        retriever = BM25Retriever()
        with pytest.raises(RetrievalError):
            retriever.index(["c1", "c2"], ["only one text"])

    def test_index_empty_corpus_raises(self) -> None:
        retriever = BM25Retriever()
        with pytest.raises(RetrievalError):
            retriever.index([], [])

    def test_search_returns_relevant_chunk_first(self, sample_chunks: list[dict]) -> None:
        retriever = BM25Retriever()
        retriever.index([c["chunk_id"] for c in sample_chunks], [c["text"] for c in sample_chunks])

        results = retriever.search("trastuzumab breast cancer HER2", top_k=3)

        assert len(results) > 0
        assert results[0][0] == "c1"  # the trastuzumab chunk should rank first

    def test_search_no_match_returns_empty(self, sample_chunks: list[dict]) -> None:
        retriever = BM25Retriever()
        retriever.index([c["chunk_id"] for c in sample_chunks], [c["text"] for c in sample_chunks])

        results = retriever.search("quantum computing blockchain", top_k=3)
        assert results == []


class FakeDenseRetriever:
    """Fake dense retriever returning fixed (chunk_id, score, text, metadata) tuples."""

    def __init__(self, hits, indexer=None) -> None:
        self._hits = hits
        self.indexer = indexer or FakeIndexer()

    def search(self, query: str, top_k: int = 10, level_filter: str | None = "child"):
        return self._hits[:top_k]


class FakeIndexer:
    """Fake ChromaIndexer providing get_by_ids fallback for hybrid retriever tests."""

    def get_by_ids(self, chunk_ids):
        return {}


class FakeBM25Retriever:
    """Fake BM25 retriever returning fixed (chunk_id, score) tuples."""

    def __init__(self, hits) -> None:
        self._hits = hits

    def search(self, query: str, top_k: int = 10):
        return self._hits[:top_k]


class FakeGraphRetriever:
    """Fake graph retriever returning fixed GraphRetrievalResult objects."""

    def __init__(self, results) -> None:
        self._results = results

    def retrieve(self, query: str, top_k: int = 5):
        return self._results[:top_k]


class TestHybridRetriever:
    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(RetrievalError):
            HybridRetriever(
                dense_retriever=FakeDenseRetriever([]),
                bm25_retriever=FakeBM25Retriever([]),
                hybrid_alpha=1.5,
            )

    def test_fuses_dense_and_bm25_scores(self) -> None:
        dense_hits = [("c1", 0.9, "trastuzumab text", {"source_file": "book.pdf"})]
        bm25_hits = [("c1", 5.0), ("c2", 2.0)]

        retriever = HybridRetriever(
            dense_retriever=FakeDenseRetriever(dense_hits),
            bm25_retriever=FakeBM25Retriever(bm25_hits),
            graph_retriever=None,
            hybrid_alpha=0.5,
        )
        results = retriever.retrieve("trastuzumab", final_top_k=5, use_graph=False)

        assert len(results) == 2
        chunk_ids = {r.chunk_id for r in results}
        assert chunk_ids == {"c1", "c2"}
        # c1 appears in both signals so should outrank c2 (bm25-only).
        assert results[0].chunk_id == "c1"

    def test_graph_signal_included_when_enabled(self) -> None:
        dense_hits = [("c1", 0.9, "text1", {})]
        bm25_hits = [("c2", 3.0)]
        graph_hits = [GraphRetrievalResult(chunk_id="c3", matched_entity="trastuzumab", hop_distance=0, graph_score=2.0)]

        retriever = HybridRetriever(
            dense_retriever=FakeDenseRetriever(dense_hits),
            bm25_retriever=FakeBM25Retriever(bm25_hits),
            graph_retriever=FakeGraphRetriever(graph_hits),
            hybrid_alpha=0.5,
        )
        results = retriever.retrieve("query", final_top_k=5, use_graph=True)

        chunk_ids = {r.chunk_id for r in results}
        assert "c3" in chunk_ids

    def test_use_graph_false_excludes_graph_signal(self) -> None:
        dense_hits = [("c1", 0.9, "text1", {})]
        bm25_hits = []
        graph_hits = [GraphRetrievalResult(chunk_id="c3", matched_entity="x", hop_distance=0, graph_score=2.0)]

        retriever = HybridRetriever(
            dense_retriever=FakeDenseRetriever(dense_hits),
            bm25_retriever=FakeBM25Retriever(bm25_hits),
            graph_retriever=FakeGraphRetriever(graph_hits),
            hybrid_alpha=0.5,
        )
        results = retriever.retrieve("query", final_top_k=5, use_graph=False)

        chunk_ids = {r.chunk_id for r in results}
        assert "c3" not in chunk_ids

    def test_empty_signals_return_empty_list(self) -> None:
        retriever = HybridRetriever(
            dense_retriever=FakeDenseRetriever([]),
            bm25_retriever=FakeBM25Retriever([]),
            graph_retriever=None,
        )
        assert retriever.retrieve("nothing matches", use_graph=False) == []
