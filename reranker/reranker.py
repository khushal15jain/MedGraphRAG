"""reranker/reranker.py
------------------
Stage 12: Cross-Encoder Reranking & Evidence Quality Filtering.

Applies ``BAAI/bge-reranker-base`` cross-encoder to score query-document pairs,
followed by candidate deduplication and evidence quality filtering before selecting top-k.
"""

from __future__ import annotations

import re
from sentence_transformers import CrossEncoder

from utils.exceptions import RerankingError
from utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
_GENERIC_STOPWORDS = {"patient", "treatment", "clinical", "study", "data", "results", "cancer", "disease", "may", "used"}


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _jaccard_similarity(text1: str, text2: str) -> float:
    t1 = _tokenize(text1)
    t2 = _tokenize(text2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


class BGEReranker:
    """Cross-encoder reranker wrapping BAAI/bge-reranker-base with quality filtering."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 8,
        min_word_count: int = 15,
        dedup_threshold: float = 0.75,
    ) -> None:
        """Load the cross-encoder reranking model and set quality thresholds."""
        self.batch_size = batch_size
        self.min_word_count = min_word_count
        self.dedup_threshold = dedup_threshold
        try:
            self.model = CrossEncoder(model_name, device=device, max_length=512, local_files_only=True)
            logger.info(f"Loaded reranker model '{model_name}' on device '{device}'")
        except Exception as exc:  # noqa: BLE001
            raise RerankingError(f"Failed to load reranker model '{model_name}': {exc}") from exc

    def _filter_and_deduplicate(
        self, query: str, candidates: list[dict], text_key: str = "text"
    ) -> list[dict]:
        """Apply evidence quality filtering and candidate deduplication."""
        query_tokens = _tokenize(query) - _GENERIC_STOPWORDS
        selected: list[dict] = []

        for candidate in candidates:
            text = candidate.get(text_key, "").strip()
            tokens = _tokenize(text)

            # 1. Low information filter (< 15 words)
            if len(text.split()) < self.min_word_count:
                candidate["rerank_score"] -= 1.5

            # 2. Query term overlap penalty if chunk lacks query keywords
            if query_tokens and len(query_tokens & tokens) == 0:
                candidate["rerank_score"] -= 0.8

            # 3. Near-duplicate filtering against already selected chunks
            is_duplicate = False
            for sel in selected:
                if _jaccard_similarity(text, sel.get(text_key, "")) > self.dedup_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                selected.append(candidate)

        return selected

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        text_key: str = "text",
        top_k: int = 5,
    ) -> list[dict]:
        """Score, filter, and re-sort candidate chunks by cross-encoder relevance to the query."""
        if not candidates:
            return []

        pairs = [(query, c[text_key]) for c in candidates]

        try:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise RerankingError(f"Cross-encoder scoring failed: {exc}") from exc

        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)

        # Apply deduplication and quality filtering
        filtered = self._filter_and_deduplicate(query, candidates, text_key=text_key)

        ranked = sorted(filtered, key=lambda c: c["rerank_score"], reverse=True)
        logger.info(f"Reranked {len(candidates)} candidates down to top {top_k}")
        return ranked[:top_k]
