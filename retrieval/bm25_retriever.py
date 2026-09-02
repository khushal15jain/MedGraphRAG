"""BM25 sparse retrieval (one leg of Stage 10: Hybrid Retrieval).

Uses ``rank-bm25``'s Okapi BM25 implementation, a pure-Python, dependency-
light library chosen specifically to avoid the memory/setup overhead of
Elasticsearch or Lucene-based alternatives. BM25 complements dense
embeddings by capturing exact lexical matches (e.g. precise drug names,
dosages, or gene symbols) that dense embeddings can sometimes under-weight.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from utils.exceptions import RetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-]*")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer (keeps hyphens for terms like 'HER2-positive')."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Wraps rank-bm25's Okapi BM25 over a fixed corpus of chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []

    def index(self, chunk_ids: list[str], texts: list[str]) -> None:
        """Build the BM25 index over a corpus of chunk texts.

        Args:
            chunk_ids: Unique chunk identifiers, aligned positionally with ``texts``.
            texts: Chunk text content.

        Raises:
            RetrievalError: If inputs are empty or mismatched in length.
        """
        if len(chunk_ids) != len(texts):
            raise RetrievalError("chunk_ids and texts must have equal length")
        if not texts:
            raise RetrievalError("Cannot build BM25 index over an empty corpus")

        self._chunk_ids = chunk_ids
        self._texts = texts
        tokenized_corpus = [_tokenize(t) for t in texts]

        try:
            self._bm25 = BM25Okapi(tokenized_corpus)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Failed to build BM25 index: {exc}") from exc

        logger.info(f"Built BM25 index over {len(texts)} chunks")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Retrieve the top-k chunks by BM25 score for a query.

        Args:
            query: Natural-language query.
            top_k: Number of results to return.

        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by descending score.

        Raises:
            RetrievalError: If the index has not been built yet.
        """
        if self._bm25 is None:
            raise RetrievalError("BM25 index has not been built — call index() first")

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = [(self._chunk_ids[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]

        return results
