# MedGraphRAG: System Architecture Specification

This document details the component-based system architecture of **MedGraphRAG**, an enterprise Graph-Augmented Retrieval-Augmented Generation framework tailored for Medical Oncology Clinical Decision Support.

---

## 🏛 Component Overview

```
MedGraphRAG Architecture
├── app/                  # FastAPI Web Server & Clinical QA Dashboard
├── preprocessing/        # PDF Loading, Text Cleaner & Hierarchical Chunker
├── entity_extraction/    # Biomedical NER & SciSpaCy Relation Extraction
├── graph/                # NetworkX Knowledge Graph & Hop-Decay Subgraph Retriever
├── embeddings/           # BAAI/bge-base-en-v1.5 Dense Embedder & Chroma Vector DB
├── retrieval/            # BM25, Dense, & Graph-Augmented Hybrid Fusion Retriever
├── reranker/             # Cross-Encoder Reranker (bge-reranker-base)
├── generator/            # Ollama/vLLM LLM Interface & DeBERTa NLI Evidence Grounding
├── evaluation/           # Closed-Form Metric Calculation & Holm-Bonferroni Testing
├── benchmark/            # Baseline Model Architectures (Naïve, Dense, Graph-Only)
└── utils/                # Logging & Exception Hierarchy
```

---

## 🔍 Module Responsibilities

### 1. Document Preprocessing (`preprocessing/`)
- **`cleaner.py`**: Removes running headers/footers, page numbers, inline citations, and rejoins hyphenated terms across page boundaries.
- **`chunker.py`**: Splits cleaned documents into parent (3,500 token target) and child (300 token target, 35 token overlap) chunks, preserving page-level provenance.
- **`pdf_loader.py`**: Loads raw medical oncology PDF guidelines using PyMuPDF.

### 2. Biomedical Extraction (`entity_extraction/`)
- **`ner_extractor.py`**: Extracts medical concepts (diseases, drugs, genes, dosages) using SciSpaCy (`en_core_sci_sm`).
- **`relation_extractor.py`**: Extracts directed semantic relations between entity pairs within a sentence window.

### 3. Knowledge Graph Engine (`graph/`)
- **`graph_builder.py`**: Constructs a NetworkX multigraph of normalized entity nodes and relation edges with mention frequency counts.
- **`graph_retriever.py`**: Implements hop-distance decay scoring:
  $$S_{\text{graph}}(c) = \sum_{e \in E_c} \gamma^{h(e, Q)}$$
  where $\gamma = 0.5$ is the decay factor and $h(e, Q)$ is the shortest path hop distance to query entity nodes.

### 4. Vector Storage & Dense Retrieval (`embeddings/` & `retrieval/`)
- **`embedder.py`**: Encodes passage chunks using `BAAI/bge-base-en-v1.5` (512-dim normalized vectors).
- **`chroma_indexer.py`**: Manages persistent Chroma vector database indexing.
- **`bm25_retriever.py`**: Rank-BM25 Okapi sparse search engine.
- **`hybrid_retriever.py`**: Fuses BM25, dense, and graph scores with min-max score normalization:
  $$S_{\text{hybrid}}(c) = \alpha S_{\text{dense}}(c) + \beta S_{\text{bm25}}(c) + \gamma S_{\text{graph}}(c)$$

### 5. Evidence Grounding & Generation (`generator/`)
- **`llm_generator.py`**: Interfaces with local LLMs (Qwen2.5 / Llama-3.2) via Ollama.
- **`sentence_grounder.py`**: Computes sentence-level grounding against retrieved evidence using a dual Jaccard + Cosine Similarity threshold ($\tau_g = 0.65$).
- **`evidence_grounding.py`**: Performs NLI entailment/contradiction verification using `DeBERTa-v3`.
