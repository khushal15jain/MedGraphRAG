"""Stage 8: Embedding Generation.

Wraps ``BAAI/bge-base-en-v1.5`` via ``sentence-transformers`` for dense
embedding generation. BGE-base (109M params, ~440MB) was chosen over larger
BGE variants (e.g. bge-large) specifically to stay within the 8GB RAM
budget while still ranking competitively on the MTEB retrieval benchmark.

BGE models require a specific query-side instruction prefix for optimal
retrieval performance (asymmetric encoding: queries get an instruction,
documents do not) — this module implements that convention.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.exceptions import EmbeddingError
from utils.logger import get_logger

logger = get_logger(__name__)

# Official BGE instruction prefix for retrieval queries (per BAAI model card).
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    """Generates dense embeddings using BAAI/bge-base-en-v1.5."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
        batch_size: int = 16,
    ) -> None:
        """Load the embedding model.

        Args:
            model_name: HuggingFace model identifier.
            device: "cpu" or "mps" — kept to "cpu" by default for stability
                on Apple Silicon (MPS support in sentence-transformers can
                be inconsistent across PyTorch versions).
            max_seq_length: Maximum token length; longer chunks are truncated.
            normalize_embeddings: L2-normalize output vectors, required for
                correct cosine-similarity search in ChromaDB.
            batch_size: Encoding batch size, kept small for 8GB RAM safety.

        Raises:
            EmbeddingError: If the model fails to load.
        """
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

        try:
            self.model = SentenceTransformer(model_name, device=device, local_files_only=True)
            self.model.max_seq_length = max_seq_length
            logger.info(f"Loaded embedding model '{model_name}' on device '{device}'")
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Failed to load embedding model '{model_name}': {exc}") from exc

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document/chunk texts (no instruction prefix).

        Args:
            texts: List of chunk texts to embed.

        Returns:
            NumPy array of shape (len(texts), embedding_dim).

        Raises:
            EmbeddingError: If encoding fails (e.g. OOM).
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        except RuntimeError as exc:
            raise EmbeddingError(
                f"Embedding generation failed (possible OOM with batch_size="
                f"{self.batch_size}): {exc}"
            ) from exc

        logger.debug(f"Embedded {len(texts)} documents -> shape {embeddings.shape}")
        return embeddings

    # Alias for API compatibility across pipeline modules
    embed_passages = embed_documents

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query, applying BGE's required query instruction prefix.

        Args:
            query: Natural-language query text.

        Returns:
            1-D NumPy array (embedding_dim,).

        Raises:
            EmbeddingError: If encoding fails.
        """
        try:
            embedding = self.model.encode(
                _BGE_QUERY_INSTRUCTION + query,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except RuntimeError as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc

        return embedding

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimensionality of the loaded model."""
        return self.model.get_sentence_embedding_dimension()
