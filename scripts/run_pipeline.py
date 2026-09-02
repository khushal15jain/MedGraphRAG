"""MedGraphRAG — End-to-End Pipeline Orchestrator.

Runs the full offline pipeline: PDF loading -> cleaning -> metadata
extraction -> hierarchical chunking -> entity/relation extraction ->
knowledge graph construction -> embedding generation -> ChromaDB indexing.

After this script completes, the system is ready for querying either via
``app/api.py`` (FastAPI) or directly via the retrieval/generation modules
in a notebook, and for benchmarking via ``benchmark/run_benchmark.py``.

Usage:
    python main.py                                  # run with default config
    python main.py paths.data_raw=data/raw/textbooks # override any config value
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import hydra
from omegaconf import DictConfig

from medgraphrag.embeddings.vector_store import ChromaIndexer
from medgraphrag.embeddings.encoder import BGEEmbedder
from medgraphrag.extraction.ner import MedicalEntityExtractor
from medgraphrag.extraction.relations import RelationExtractor
from medgraphrag.graph.builder import KnowledgeGraphBuilder
from medgraphrag.preprocessing.chunker import HierarchicalChunker
from medgraphrag.preprocessing.cleaner import TextCleaner
from medgraphrag.preprocessing.metadata_extractor import MetadataExtractor
from medgraphrag.preprocessing.pdf_loader import PDFLoader
from medgraphrag.utils.exceptions import MedGraphRAGError, VectorStoreError
from medgraphrag.utils.io import save_pickle, write_jsonl
from medgraphrag.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Stage 8-9: streaming, resumable, memory-bounded embedding + indexing
# ============================================================================

# Batch size for the embed -> upsert loop. Kept modest so that only one
# batch of raw texts / embeddings is ever resident in RAM at a time.
# This is independent of BGEEmbedder's own internal encode batch_size
# (which controls GPU/CPU micro-batching inside sentence-transformers).
_INDEXING_BATCH_SIZE = 128


class _IndexingCheckpoint:
    """Tracks indexing progress on disk so a crash/interrupt can resume.

    The checkpoint stores the offset (into the deterministic chunk order)
    of the next chunk that still needs to be embedded + indexed. It is
    written atomically (write-to-temp + os.replace) after every batch so a
    crash mid-write can never leave a corrupt/half-written checkpoint file.
    """

    def __init__(self, checkpoint_path: str) -> None:
        self.path = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> int:
        """Return the last completed offset, or 0 if no checkpoint exists."""
        if not self.path.exists():
            return 0
        try:
            data = json.loads(self.path.read_text())
            return int(data.get("next_offset", 0))
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(f"Could not read checkpoint '{self.path}' ({exc}); starting from 0")
            return 0

    def save(self, next_offset: int, total: int) -> None:
        """Atomically persist progress."""
        payload = {
            "next_offset": next_offset,
            "total": total,
            "updated_at": time.time(),
        }
        fd, tmp_path = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, self.path)  # atomic on POSIX
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


def _iter_batches(
    chunk_dicts: list[dict[str, Any]], start: int, batch_size: int
) -> Iterator[tuple[int, int, list[dict[str, Any]]]]:
    """Yield (batch_start, batch_end, batch_slice) without copying the full list.

    Slicing a list still copies that slice, but only one batch-sized slice
    exists in memory at a time rather than duplicating the entire chunk set
    (e.g. into separate parallel texts/ids/metadatas lists up front).
    """
    total = len(chunk_dicts)
    for batch_start in range(start, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        yield batch_start, batch_end, chunk_dicts[batch_start:batch_end]


def _format_eta(elapsed_s: float, done: int, total: int) -> str:
    if done <= 0:
        return "unknown"
    rate = elapsed_s / done
    remaining = rate * (total - done)
    mins, secs = divmod(int(remaining), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"


def run_embedding_indexing_stage(
    chunk_dicts: list[dict[str, Any]],
    cfg: DictConfig,
    paths: DictConfig,
) -> None:
    """Stage 8-9: embed chunks and index into ChromaDB.

    Streams data in fixed-size batches end-to-end: for each batch, only
    that batch's texts/embeddings ever exist in RAM. Each batch is embedded
    then immediately upserted into ChromaDB, memory is freed, progress is
    logged, and a checkpoint is written — so an interrupted run resumes
    from the last completed batch instead of restarting from scratch.

    Args:
        chunk_dicts: All chunks produced by Stages 1-4 (dicts, not the full
            embedding matrix — embeddings are never materialized for more
            than one batch at a time).
        cfg: Composed Hydra config (used for embedding model settings).
        paths: ``cfg.paths`` — Chroma persist dir/collection + checkpoint dir.

    Raises:
        MedGraphRAGError: If embedding or indexing fails unrecoverably after
            the current progress has been safely checkpointed.
    """
    total = len(chunk_dicts)
    if total == 0:
        logger.info("No chunks to embed/index. Skipping Stage 8-9.")
        return

    checkpoint_path = getattr(paths, "chroma_checkpoint_file", None) or os.path.join(
        paths.chroma_persist_dir, "_indexing_checkpoint.json"
    )
    checkpoint = _IndexingCheckpoint(checkpoint_path)

    logger.info("Loading embedding model...")
    embedder = BGEEmbedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        max_seq_length=cfg.embedding.max_seq_length,
        normalize_embeddings=cfg.embedding.normalize_embeddings,
        batch_size=cfg.embedding.batch_size,
    )

    logger.info("Opening ChromaDB collection...")
    indexer = ChromaIndexer(
        persist_dir=paths.chroma_persist_dir,
        collection_name=paths.chroma_collection_name,
    )

    # Resume point: prefer the explicit checkpoint (accurate even if chunks
    # were reordered/regenerated between runs); fall back to the collection's
    # existing vector count for a first-time migration from the old scheme.
    checkpoint_offset = checkpoint.load()
    existing_vectors = indexer.count()
    start_index = checkpoint_offset if checkpoint_offset > 0 else min(existing_vectors, total)

    if start_index >= total:
        logger.info(
            f"All {total} chunks already indexed (checkpoint/collection confirms). "
            "Skipping Stage 8-9."
        )
        checkpoint.clear()
        return

    logger.info(
        f"Resuming indexing at chunk {start_index}/{total} "
        f"(checkpoint_offset={checkpoint_offset}, existing_in_chroma={existing_vectors})"
    )

    start_time = time.monotonic()
    indexed_so_far = start_index

    try:
        for batch_start, batch_end, batch in _iter_batches(chunk_dicts, start_index, _INDEXING_BATCH_SIZE):
            batch_num = batch_start // _INDEXING_BATCH_SIZE + 1
            embeddings = None

            try:
                # Build only this batch's parallel arrays — never the full
                # dataset's worth of texts/ids/metadatas at once.
                batch_texts = [c["text"] for c in batch]
                batch_ids = [c["chunk_id"] for c in batch]
                batch_meta = [
                    {
                        "source_file": c["source_file"],
                        "page_number": c["page_number"],
                        "level": c["level"],
                        "heading": c["heading"] or "Unknown",
                    }
                    for c in batch
                ]

                embeddings = embedder.embed_documents(batch_texts)

                # upsert() is idempotent on chunk_id, so re-processing the
                # tail of a batch that partially succeeded before a crash
                # cannot create duplicate vectors.
                indexer.upsert_chunks(
                    chunk_ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_texts,
                    metadatas=batch_meta,
                    batch_size=_INDEXING_BATCH_SIZE,
                )

            except (VectorStoreError, RuntimeError, MemoryError) as exc:
                # Progress up to the *previous* successful batch is already
                # checkpointed, so nothing is lost — just stop and surface it.
                logger.error(
                    f"Stage 8-9 failed on batch {batch_num} "
                    f"(chunks {batch_start}-{batch_end}): {exc}"
                )
                raise MedGraphRAGError(
                    f"Embedding/indexing failed at chunk {batch_start} of {total}; "
                    f"safe to re-run — it will resume from chunk {batch_start}."
                ) from exc

            finally:
                # Drop references to this batch's large objects and reclaim
                # memory immediately, regardless of success/failure.
                embeddings = None
                batch = None
                gc.collect()

            indexed_so_far = batch_end
            checkpoint.save(next_offset=indexed_so_far, total=total)

            elapsed = time.monotonic() - start_time
            pct = 100.0 * indexed_so_far / total
            eta = _format_eta(elapsed, indexed_so_far - start_index, total - start_index)
            logger.info(
                f"[Stage 8-9] batch {batch_num} | "
                f"indexed {indexed_so_far}/{total} ({pct:.2f}%) | "
                f"elapsed {elapsed:.1f}s | ETA {eta}"
            )

    except KeyboardInterrupt:
        logger.warning(
            f"Interrupted by user at chunk {indexed_so_far}/{total}. "
            "Progress checkpointed — re-run to resume."
        )
        raise

    total_elapsed = time.monotonic() - start_time
    logger.info(
        f"=== Stage 8-9 completed: {indexed_so_far}/{total} chunks indexed "
        f"in {total_elapsed:.1f}s ==="
    )
    checkpoint.clear()


def run_pipeline(cfg: DictConfig) -> None:
    """Execute the full offline ingestion-to-indexing pipeline.

    Args:
        cfg: The composed Hydra configuration (config + paths + model + retrieval).
    """
    paths = cfg.paths

    # ---------------------------------------------------------------- Stage 1-4: Preprocessing
    logger.info("=== Stage 1-4: PDF Loading, Cleaning, Metadata, Chunking ===")
    loader = PDFLoader()
    cleaner = TextCleaner()
    metadata_extractor = MetadataExtractor()
    chunker = HierarchicalChunker(
        parent_chunk_size=cfg.chunking.parent_chunk_size,
        child_chunk_size=cfg.chunking.child_chunk_size,
        chunk_overlap=cfg.chunking.chunk_overlap,
    )

    documents = loader.load_directory(paths.data_raw)
    all_chunks = []

    for filename, pages in documents.items():
        page_texts = [p.text for p in pages]
        cleaned_pages = cleaner.clean_document(page_texts)

        pdf_path = f"{paths.data_raw}/{filename}"
        headings = metadata_extractor.detect_headings(pdf_path)
        headings_by_page = {
            p.page_number: metadata_extractor.get_heading_for_page(p.page_number, headings)
            for p in pages
        }

        doc_chunks = chunker.chunk_document(cleaned_pages, filename, headings_by_page)
        all_chunks.extend(doc_chunks)

    chunk_dicts = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "level": c.level,
            "parent_id": c.parent_id,
            "source_file": c.source_file,
            "page_number": c.page_number,
            "heading": c.heading or "",
        }
        for c in all_chunks
    ]
    write_jsonl(chunk_dicts, paths.chunks_file)
    logger.info(f"Total chunks produced: {len(chunk_dicts)}")

    # ---------------------------------------------------------------- Stage 5-6: Entity/Relation Extraction
    logger.info("=== Stage 5-6: Medical Entity & Relation Extraction ===")
    entity_extractor = MedicalEntityExtractor(spacy_model=cfg.ner.spacy_model)
    relation_extractor = RelationExtractor(spacy_model=cfg.ner.spacy_model)

    child_chunks = [c for c in chunk_dicts if c["level"] == "child"]
    entities = entity_extractor.extract_from_chunks(child_chunks)
    relations = relation_extractor.extract_from_chunks(child_chunks)

    write_jsonl(
        [
            {
                "entity_id": e.entity_id,
                "text": e.text,
                "normalized_text": e.normalized_text,
                "label": e.label,
                "chunk_id": e.chunk_id,
                "source_file": e.source_file,
            }
            for e in entities
        ],
        paths.entities_file,
    )
    write_jsonl(
        [
            {
                "relation_id": r.relation_id,
                "subject_text": r.subject_text,
                "predicate": r.predicate,
                "object_text": r.object_text,
                "chunk_id": r.chunk_id,
                "source_file": r.source_file,
                "confidence": r.confidence,
            }
            for r in relations
        ],
        paths.relations_file,
    )

    # ---------------------------------------------------------------- Stage 7: Knowledge Graph
    logger.info("=== Stage 7: Knowledge Graph Construction ===")
    graph_builder = KnowledgeGraphBuilder()
    graph_builder.add_entities(
        [{"normalized_text": e.normalized_text, "text": e.text, "label": e.label, "chunk_id": e.chunk_id} for e in entities]
    )
    graph_builder.add_relations(
        [
            {
                "subject_text": r.subject_text,
                "predicate": r.predicate,
                "object_text": r.object_text,
                "confidence": r.confidence,
                "chunk_id": r.chunk_id,
            }
            for r in relations
        ]
    )
    save_pickle(graph_builder.graph, paths.graph_file)
    logger.info(f"Knowledge graph stats: {graph_builder.stats()}")

    logger.info("Freeing memory before embedding stage...")
    del entities
    del relations
    del graph_builder
    gc.collect()

    # ---------------------------------------------------------------- Stage 8-9: Embeddings + ChromaDB
    logger.info("=== Stage 8-9: Embedding Generation & ChromaDB Indexing ===")
    run_embedding_indexing_stage(chunk_dicts, cfg, paths)

    logger.info("=== Pipeline complete. System ready for querying ===")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: compose config and run the ingestion pipeline.

    Args:
        cfg: Auto-injected composed configuration from ``configs/config.yaml``
            and its defaults (``paths``, ``model``, ``retrieval``).
    """
    try:
        run_pipeline(cfg)
    except MedGraphRAGError as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()