"""Stage 12: Reranking.

Applies ``BAAI/bge-reranker-base``, a cross-encoder that jointly scores
(query, chunk) pairs, to re-rank the fused hybrid-retrieval candidate list.
Cross-encoders are substantially more accurate than bi-encoder (dense)
similarity alone because they attend across the full query-document pair
rather than comparing independently-computed vectors — but they are too
slow to run over an entire corpus, so they are applied only to the small
top-N candidate set produced by hybrid retrieval. This two-stage
retrieve-then-rerank design is standard in production RAG and is
computationally cheap enough for 8GB RAM since only ~10-20 pairs are
scored per query.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from utils.exceptions import RerankingError
from utils.logger import get_logger

logger = get_logger(__name__)


class BGEReranker:
    """Cross-encoder reranker wrapping BAAI/bge-reranker-base."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 8,
    ) -> None:
        """Load the cross-encoder reranking model.

        Args:
            model_name: HuggingFace model identifier.
            device: "cpu" or "mps".
            batch_size: Pairs scored per forward pass, kept small for 8GB RAM.

        Raises:
            RerankingError: If the model fails to load.
        """
        self.batch_size = batch_size
        try:
            self.model = CrossEncoder(model_name, device=device, max_length=512, local_files_only=True)
            logger.info(f"Loaded reranker model '{model_name}' on device '{device}'")
        except Exception as exc:  # noqa: BLE001
            raise RerankingError(f"Failed to load reranker model '{model_name}': {exc}") from exc

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text",
        top_k: int = 5,
    ) -> list[dict]:
        """Score and re-sort candidate chunks by cross-encoder relevance to the query.

        Args:
            query: Natural-language query.
            candidates: List of candidate dicts (e.g. ``RetrievedChunk``
                converted to dict) each containing at least ``text_key``.
            text_key: Dict key holding the chunk's text content.
            top_k: Number of top candidates to return after reranking.

        Returns:
            The input candidate dicts, each augmented with a
            ``rerank_score`` key, sorted by descending ``rerank_score`` and
            truncated to ``top_k``.

        Raises:
            RerankingError: If scoring fails.
        """
        if not candidates:
            return []

        pairs = [(query, c[text_key]) for c in candidates]

        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RerankingError(f"Cross-encoder scoring failed: {exc}") from exc

        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)

        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        logger.info(f"Reranked {len(candidates)} candidates, keeping top {top_k}")
        return ranked[:top_k]
