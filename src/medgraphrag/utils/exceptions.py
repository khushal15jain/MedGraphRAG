"""Custom exception hierarchy for MedGraphRAG.

Defining explicit exception types (rather than raising bare ``Exception`` or
``ValueError`` everywhere) lets calling code catch failures precisely and
lets us log/report failures with stage-level granularity, which matters when
debugging a 20-stage research pipeline.
"""

from __future__ import annotations


class MedGraphRAGError(Exception):
    """Base exception for all MedGraphRAG-specific errors."""


class PDFLoadError(MedGraphRAGError):
    """Raised when a PDF file cannot be opened, parsed, or is empty/corrupt."""


class ChunkingError(MedGraphRAGError):
    """Raised when hierarchical chunking fails or produces invalid output."""


class EntityExtractionError(MedGraphRAGError):
    """Raised when spaCy/SciSpaCy NER pipeline fails to process text."""


class RelationExtractionError(MedGraphRAGError):
    """Raised when relationship extraction between entities fails."""


class GraphConstructionError(MedGraphRAGError):
    """Raised when the NetworkX knowledge graph cannot be built or queried."""


class EmbeddingError(MedGraphRAGError):
    """Raised when embedding generation fails (e.g. model load or OOM)."""


class VectorStoreError(MedGraphRAGError):
    """Raised when ChromaDB indexing or querying fails."""


class RetrievalError(MedGraphRAGError):
    """Raised when dense, sparse, hybrid, or graph retrieval fails."""


class RerankingError(MedGraphRAGError):
    """Raised when the cross-encoder reranker fails to score candidates."""


class GenerationError(MedGraphRAGError):
    """Raised when the LLM (Ollama) fails to generate a response."""


class EvaluationError(MedGraphRAGError):
    """Raised when RAGAS/DeepEval metric computation fails."""


class ConfigurationError(MedGraphRAGError):
    """Raised when Hydra/OmegaConf configuration is missing or invalid."""
