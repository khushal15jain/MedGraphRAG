"""Integration test for full end-to-end pipeline flow using top-level package imports."""

import pytest
from preprocessing.cleaner import TextCleaner
from preprocessing.chunker import HierarchicalChunker
from entity_extraction.ner_extractor import MedicalEntityExtractor
from graph.graph_builder import KnowledgeGraphBuilder
from retrieval.bm25_retriever import BM25Retriever
from evaluation.metrics import precision_at_k, recall_at_k, hit_rate_at_k


def test_end_to_end_pipeline_flow():
    """Test full flow: clean text -> chunk -> extract entities -> build graph -> retrieve."""
    page_texts = [
        "Pembrolizumab is indicated for first-line treatment of metastatic NSCLC in patients with PD-L1 expression >= 50%.",
        "Tamoxifen is a selective estrogen receptor modulator for breast cancer treatment.",
        "Trastuzumab targets HER2 receptor in HER2-positive metastatic breast carcinoma."
    ]

    # 1. Clean
    cleaner = TextCleaner()
    clean_pages = [cleaner.clean_page(p) for p in page_texts]
    assert all(len(p) > 0 for p in clean_pages)

    # 2. Chunk
    chunker = HierarchicalChunker(child_chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_document(clean_pages, source_file="DOC_001.pdf")
    assert len(chunks) >= 1

    # 3. Extract entities
    extractor = MedicalEntityExtractor()
    entities = extractor.extract_from_chunk(clean_pages[0], chunk_id="C001", source_file="DOC_001.pdf")
    assert isinstance(entities, list)

    # 4. Build Graph
    graph_builder = KnowledgeGraphBuilder()
    entity_dicts = [
        {
            "normalized_text": getattr(e, "normalized_text", str(e).lower()),
            "text": getattr(e, "text", str(e)),
            "label": getattr(e, "label", "ENTITY"),
            "chunk_id": getattr(e, "chunk_id", "C001"),
        }
        for e in entities
    ]
    graph_builder.add_entities(entity_dicts)

    stats = graph_builder.stats()
    assert "num_nodes" in stats

    # 5. BM25 Retrieve
    corpus = [c.text for c in chunks]
    doc_ids = [c.chunk_id for c in chunks]
    bm25 = BM25Retriever()
    bm25.index(chunk_ids=doc_ids, texts=corpus)

    results = bm25.search("pembrolizumab", top_k=2)
    assert len(results) >= 1

    # 6. Evaluation metrics
    retrieved = [r[0] for r in results]
    gold = [doc_ids[0]]

    prec = precision_at_k(retrieved, gold, k=5)
    rec = recall_at_k(retrieved, gold, k=5)
    hit = hit_rate_at_k(retrieved, gold, k=5)

    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert hit in (0.0, 1.0)
