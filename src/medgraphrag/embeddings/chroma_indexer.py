"""Stage 9: ChromaDB Indexing.

Persists chunk embeddings and metadata into a local, embedded ChromaDB
instance (DuckDB+Parquet backend — no server process, no Docker), chosen
explicitly over FAISS/Neo4j per project constraints because it bundles
storage + metadata filtering + a simple query API in one lightweight
dependency well suited to a single-machine research pipeline.
"""

from __future__ import annotations

from typing import Any

import os

# Disable ChromaDB telemetry to prevent posthog crash
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import numpy as np
from chromadb.config import Settings

from medgraphrag.utils.exceptions import VectorStoreError
from medgraphrag.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaIndexer:
    """Manages a persistent local ChromaDB collection for chunk embeddings."""

    def __init__(
        self,
        persist_dir: str = "outputs/chroma_db",
        collection_name: str = "medgraphrag_chunks",
    ) -> None:
        """Initialize (or open) a persistent Chroma collection.

        Args:
            persist_dir: Directory where Chroma persists its DuckDB+Parquet store.
            collection_name: Name of the collection holding chunk embeddings.

        Raises:
            VectorStoreError: If the Chroma client or collection cannot be created.
        """
        try:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Opened Chroma collection '{collection_name}' at '{persist_dir}' "
                f"({self.collection.count()} existing vectors)"
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to initialize ChromaDB: {exc}") from exc

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict[str, Any]],
        batch_size: int = 256,
    ) -> None:
        """Insert or update chunk embeddings in the collection.

        Args:
            chunk_ids: Unique chunk IDs (used as Chroma document IDs).
            embeddings: Array of shape (N, embedding_dim).
            documents: Raw chunk text (stored alongside for retrieval-time display).
            metadatas: Per-chunk metadata dicts (e.g. source_file, page_number, level).
                Chroma only supports flat scalar metadata values (str/int/float/bool);
                callers must pre-flatten any nested fields (e.g. sets -> comma strings).
            batch_size: Number of vectors upserted per Chroma call, to bound memory.

        Raises:
            VectorStoreError: If lengths mismatch or the upsert call fails.
        """
        if not (len(chunk_ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise VectorStoreError(
                "chunk_ids, embeddings, documents, and metadatas must have equal length "
                f"(got {len(chunk_ids)}, {len(embeddings)}, {len(documents)}, {len(metadatas)})"
            )

        try:
            for start in range(0, len(chunk_ids), batch_size):
                end = start + batch_size
                self.collection.upsert(
                    ids=chunk_ids[start:end],
                    embeddings=embeddings[start:end].tolist(),
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc

        logger.info(f"Upserted {len(chunk_ids)} chunks into Chroma collection")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the collection for the nearest chunks to a query embedding.

        Args:
            query_embedding: 1-D embedding vector for the query.
            top_k: Number of nearest neighbors to return.
            where: Optional Chroma metadata filter (e.g. {"level": "child"}).

        Returns:
            List of result dicts with keys ``chunk_id``, ``text``, ``metadata``,
            ``distance`` (cosine distance, lower = more similar), sorted by
            ascending distance.

        Raises:
            VectorStoreError: If the query fails.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids, documents, metadatas, distances, strict=True):
            hits.append({"chunk_id": chunk_id, "text": text, "metadata": meta, "distance": dist})

        return hits

    def get_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch text and metadata for specific chunk IDs (no similarity search).

        Used as a fallback when a chunk was surfaced by BM25 or graph
        retrieval (which only know chunk_ids) but not by dense retrieval,
        so the hybrid retriever can still display its text.

        Args:
            chunk_ids: List of chunk IDs to fetch.

        Returns:
            Mapping of chunk_id -> {"text": ..., "metadata": ...}. IDs not
            found in the collection are simply omitted.

        Raises:
            VectorStoreError: If the underlying Chroma ``get`` call fails.
        """
        if not chunk_ids:
            return {}
        try:
            result = self.collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"ChromaDB get-by-ids failed: {exc}") from exc

        out: dict[str, dict[str, Any]] = {}
        for cid, doc, meta in zip(result.get("ids", []), result.get("documents", []), result.get("metadatas", []), strict=True):
            out[cid] = {"text": doc, "metadata": meta}
        return out

    def count(self) -> int:
        """Return the number of vectors currently stored in the collection."""
        return self.collection.count()
