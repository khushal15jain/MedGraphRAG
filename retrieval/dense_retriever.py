"""Dense retrieval (second leg of Stage 10: Hybrid Retrieval).

Thin orchestration layer that embeds a query with ``BGEEmbedder`` and
searches the persisted ``ChromaIndexer`` collection. Kept as its own module
(rather than folded into the hybrid retriever) so it can also be used
standalone as the "Dense Retrieval" baseline in the benchmark comparisons
required by the research design.
"""

from __future__ import annotations

from embeddings.chroma_indexer import ChromaIndexer
from embeddings.embedder import BGEEmbedder
from utils.exceptions import RetrievalError
from utils.logger import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """Dense (embedding-based) retrieval over a ChromaDB collection."""

    def __init__(self, embedder: BGEEmbedder, indexer: ChromaIndexer) -> None:
        """Initialize with shared embedder and indexer instances.

        Args:
            embedder: A loaded ``BGEEmbedder``.
            indexer: A ``ChromaIndexer`` pointing at the populated collection.
        """
        self.embedder = embedder
        self.indexer = indexer

    def search(
        self, query: str, top_k: int = 10, level_filter: str | None = "child", section_filter: str | None = None
    ) -> list[tuple[str, float, str, dict]]:
        """Retrieve the top-k chunks by dense cosine similarity to a query.

        Args:
            query: Natural-language query.
            top_k: Number of results to return.
            level_filter: Restrict retrieval to chunks of that hierarchy level.
            section_filter: Optional keyword to filter metadata heading by.

        Returns:
            List of (chunk_id, similarity_score, text, metadata) tuples,
            sorted by descending similarity. Similarity is computed as
            ``1 - cosine_distance`` so higher is always better.
        """
        try:
            query_embedding = self.embedder.embed_query(query)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Failed to embed query for dense retrieval: {exc}") from exc

        where = None
        if level_filter:
            where = {"level": level_filter}

        # If we need to filter by section_filter, retrieve more chunks so we can filter post-retrieval
        fetch_k = top_k * 10 if section_filter else top_k

        try:
            hits = self.indexer.query(query_embedding, top_k=fetch_k, where=where)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Dense retrieval query failed: {exc}") from exc

        results = []
        for hit in hits:
            # Post-retrieval section filtering in Python
            if section_filter:
                heading = hit["metadata"].get("heading", "")
                if not heading or section_filter.lower() not in heading.lower():
                    continue
            
            results.append((hit["chunk_id"], 1.0 - hit["distance"], hit["text"], hit["metadata"]))
            if len(results) == top_k:
                break

        logger.debug(f"Dense retrieval returned {len(results)} results for query")
        return results
