#!/usr/bin/env python3
"""MedGraphRAG: Main End-to-End Clinical QA & Ingestion Pipeline.

Supports PDF document ingestion, hierarchical chunking, SciSpaCy NER,
knowledge graph construction, BGE dense embedding indexing, and
graph-augmented hybrid retrieval-generation.

Usage:
    python scripts/run_pipeline.py                           # run with default config
    python scripts/run_pipeline.py paths.data_raw=data/raw  # override config values
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig

from embeddings.chroma_indexer import ChromaIndexer
from embeddings.embedder import BGEEmbedder
from entity_extraction.ner_extractor import MedicalEntityExtractor
from entity_extraction.relation_extractor import RelationExtractor
from graph.graph_builder import KnowledgeGraphBuilder
from preprocessing.chunker import HierarchicalChunker
from preprocessing.cleaner import TextCleaner
from preprocessing.metadata_extractor import MetadataExtractor
from preprocessing.pdf_loader import PDFLoader
from utils.exceptions import MedGraphRAGError, VectorStoreError
from utils.io_utils import save_pickle, write_jsonl
from utils.logger import get_logger

logger = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the complete MedGraphRAG ingestion and indexing pipeline."""
    start_time = time.time()
    logger.info("==================================================")
    logger.info("Starting MedGraphRAG Pipeline Execution")
    logger.info("==================================================")

    data_raw = Path(cfg.paths.data_raw)
    pdf_files = list(data_raw.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in {data_raw}. Ingestion skipped.")
        return

    logger.info(f"Found {len(pdf_files)} PDF documents to ingest.")

    # 1. Document Loading & Cleaning
    loader = PDFLoader()
    cleaner = TextCleaner()
    metadata_extractor = MetadataExtractor()

    documents = {}
    for pdf in pdf_files:
        logger.info(f"Loading document: {pdf.name}")
        pages = loader.load(pdf)
        clean_pages = cleaner.clean_document([p["text"] for p in pages])
        doc_meta = metadata_extractor.extract_document_metadata(pdf, clean_pages)
        documents[pdf.name] = {
            "pages": clean_pages,
            "metadata": doc_meta,
        }

    # 2. Hierarchical Chunking
    chunker = HierarchicalChunker(
        parent_chunk_size=cfg.chunking.parent_chunk_size if hasattr(cfg.chunking, 'parent_chunk_size') else 3500,
        child_chunk_size=cfg.chunking.child_chunk_size if hasattr(cfg.chunking, 'child_chunk_size') else 300,
        chunk_overlap=cfg.chunking.chunk_overlap if hasattr(cfg.chunking, 'chunk_overlap') else 35,
    )

    all_chunks = []
    for doc_name, doc_data in documents.items():
        chunks = chunker.chunk_document(doc_data["pages"], source_file=doc_name)
        all_chunks.extend(chunks)

    logger.info(f"Extracted {len(all_chunks)} total chunks across documents.")

    # 3. Biomedical NER & Relation Extraction
    ner = MedicalEntityExtractor(spacy_model=cfg.ner.spacy_model if hasattr(cfg.ner, 'spacy_model') else "en_core_sci_sm")
    rel_extractor = RelationExtractor()

    all_entities = []
    all_relations = []
    for chunk in all_chunks:
        if chunk.level == "child":
            ents = ner.extract_from_chunk(chunk.text, chunk.chunk_id, chunk.source_file)
            rels = rel_extractor.extract_from_chunk(chunk.text, ents, chunk.chunk_id)
            all_entities.extend(ents)
            all_relations.extend(rels)

    logger.info(f"Extracted {len(all_entities)} entity mentions and {len(all_relations)} relations.")

    # 4. Knowledge Graph Construction
    kg_builder = KnowledgeGraphBuilder()
    kg_builder.add_entities([
        {
            "normalized_text": e.normalized_text,
            "text": e.text,
            "label": e.label,
            "chunk_id": e.chunk_id,
        }
        for e in all_entities
    ])
    kg_builder.add_relations([
        {
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "chunk_id": r.chunk_id,
        }
        for r in all_relations
    ])

    graph_stats = kg_builder.stats()
    logger.info(f"Built Knowledge Graph: {graph_stats['num_nodes']} nodes, {graph_stats['num_edges']} edges.")

    # Save Graph
    graph_out = Path(cfg.paths.graph_file)
    graph_out.parent.mkdir(parents=True, exist_ok=True)
    kg_builder.save_graph(graph_out)
    logger.info(f"Saved Knowledge Graph to {graph_out}")

    # 5. Dense Embedding & Vector Store Indexing
    embedder = BGEEmbedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
    )
    indexer = ChromaIndexer(
        persist_directory=cfg.paths.chroma_persist_dir,
        collection_name=cfg.paths.chroma_collection_name,
    )

    child_chunks = [c for c in all_chunks if c.level == "child"]
    texts = [c.text for c in child_chunks]
    ids = [c.chunk_id for c in child_chunks]
    metadatas = [
        {
            "source_file": c.source_file,
            "page_number": c.page_number,
            "parent_id": c.parent_id or "",
            "heading": c.heading or "",
        }
        for c in child_chunks
    ]

    embeddings = embedder.embed_passages(texts)
    indexer.add_chunks(chunk_ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info(f"Indexed {len(child_chunks)} child chunks in Chroma vector store.")

    elapsed = time.time() - start_time
    logger.info(f"Pipeline completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()