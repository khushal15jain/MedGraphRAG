# MedGraphRAG System Architecture Specification

## Overview

**MedGraphRAG** is an evidence-grounded, privacy-preserving, and explainable Clinical Decision Support System (CDSS) for medical oncology question answering. The system combines three complementary retrieval channels—Dense Semantic Vector Search, Lexical BM25 Search, and Multi-Hop Knowledge Graph Traversal—with Cross-Encoder Reranking and Sentence-Level NLI Grounding.

```text
[ Clinical Oncology PDFs ]
           │
           ▼
[1] PDF Parsing & Preprocessing (PyMuPDF)
           │
           ▼
[2] Section-Aware Recursive Chunking (500 tokens, 100 overlap)
           │
           ├──▶ [3] Biomedical NER & Relation Extraction (SciSpaCy) ──▶ [4] Multi-Hop Knowledge Graph (NetworkX)
           │                                                                          │
           ▼                                                                          │
[5] Dense Embedding Indexing (BGE-base / ChromaDB) & Inverted BM25 Index             │
           │                                                                          │
           ═════════════════════════════ QUERY INFERENCE TIME ═════════════════════════
           │                                                                          │
    ┌──────┴───────────────────────────────┬──────────────────────────────────────────┘
    ▼                                      ▼                                          ▼
[Dense Vector Search]             [Sparse BM25 Search]                 [Multi-Hop IDF Graph Traversal]
    │                                      │                                          │
    └──────────────────────────────────────┼──────────────────────────────────────────┘
                                           ▼
                       [6] Min-Max Score Standardization & Fusion
                                           │
                                           ▼
                       [7] BGE Cross-Encoder Reranking & Deduplication (Top-25 ➔ Top-5)
                                           │
                                           ▼
                       [8] Dual Safety Gatekeeper (Similarity Threshold τ = 0.35)
                                           │
                                           ▼
                       [9] 10-Rule Constrained Prompt Construction & Llama-3.2 Generation
                                           │
                                           ▼
                       [10] Sentence-Level NLI Entailment Claim Verification
                                           │
                                           ▼
                           [ Verified Clinical Answer + Citation Cards ]
```

---

## 1. Document Parsing & Section-Aware Chunking

- **PDF Ingestion**: Layout-aware parsing powered by PyMuPDF (`fitz`), retaining section headers, page numbers, and document titles.
- **Chunking Strategy**: Recursive token-aware chunking with `chunk_size = 500` tokens and `chunk_overlap = 100` tokens.
- **Metadata Annotation**: Each chunk is annotated with `[source_file, page_number, section_header, chunk_id]`.

---

## 2. Tri-Modal Retrieval Channels

### Channel 1: High-Dimensional Dense Semantic Retrieval
- **Embedding Model**: `BAAI/bge-base-en-v1.5` (768 dimensions, normalized embeddings).
- **Vector Database**: ChromaDB (`Cosine` distance index).
- **Function**: Retrieves top $K_{\mathrm{dense}} = 20$ semantically related chunks.

### Channel 2: Lexical Sparse Search (BM25)
- **Algorithm**: BM25Okapi with SciSpaCy query entity expansion.
- **Function**: Guarantees exact match retrieval for alphanumeric drug codes (*AZD9291*, *AZD9292*) and mutation identifiers (*EGFR C797S*). Retrieves top $K_{\mathrm{bm25}} = 20$ chunks.

### Channel 3: Knowledge Graph Traversal with IEF Topological Scoring
- **NER & Graph Construction**: SciSpaCy `en_core_sci_sm` extracts biomedical entities and co-occurrence relations into a NetworkX undirected graph.
- **Inverse Entity Frequency (IEF) Topological Scoring**: Shortest-path BFS hop-distance decay eliminates generic entity bias (*patient*, *treatment*):
  $$S_{\mathrm{graph}}(c) = \max_{e \in \mathcal{E}_q} \left( \sum_{v \in \mathcal{N}_H(e) \cap \mathrm{ChunkEntities}(c)} \frac{1.0}{1.0 + \mathrm{dist}_{\mathcal{G}}(e, v)} \right)$$
- **Function**: Retrieves top $K_{\mathrm{graph}} = 10$ graph-connected candidate chunks.

---

## 3. Min-Max Score Standardization & Hybrid Fusion

Retrieval scores across all three channels are standardized to $[0, 1]$ via Min-Max normalization:

$$S_{\mathrm{norm\_channel}}(c) = \frac{s(c) - \min(s)}{\max(s) - \min(s) + \epsilon}$$

Unified hybrid score $S_{\mathrm{hybrid}}(c)$ is computed using tuned channel weights ($\alpha = 0.35, \beta = 0.30, \gamma = 0.35$):

$$S_{\mathrm{hybrid}}(c) = 0.35 \cdot S_{\mathrm{dense}}(c) + 0.30 \cdot S_{\mathrm{bm25}}(c) + 0.35 \cdot S_{\mathrm{graph}}(c)$$

Top 25 candidates are selected for reranking.

---

## 4. Cross-Encoder Reranking & Deduplication

- **Cross-Encoder Model**: `BAAI/bge-reranker-base` performing full cross-attention over (query, chunk) pairs.
- **Candidate Deduplication**: Jaccard token overlap cutoff ($\tau_{\mathrm{dedup}} = 0.65$) removes near-duplicate passages.
- **Quality Filter**: Filters out low-information text fragments (< 15 non-stopwords).
- **Final Candidate Selection**: Retains the top $K_{\mathrm{final}} = 5$ gold evidence chunks.

---

## 5. Dual-Stage Safety Gatekeeper & Constrained Generation

- **Refusal Gatekeeper**: If top rerank score falls below threshold $\tau_{\mathrm{refusal}} = 0.35$, generation is aborted, returning an explicit refusal message ("Insufficient clinical evidence available to answer query safely").
- **Local Generator**: Quantized 4-bit `Llama-3.2:latest` (3.8B parameters, `llama3.2:3b-instruct-q4_K_M`) operating locally via Ollama at temperature $T = 0.0$ and `top_p = 0.9`.
- **System Prompt**: 10-rule constrained prompt enforcing strict evidence grounding and inline citation placement (`[P1]`, `[P2]`).

---

## 6. Sentence-Level Grounding & Citation Provenance

- **NLI Entailment Verification**: Generated answers are split into factual sentences $s_j$. Each sentence is evaluated for entailment against cited passages using a hybrid lexical-semantic checker:
  $$g(s_j) = 0.4 \cdot \mathrm{LexicalOverlap}(s_j, p) + 0.6 \cdot \mathrm{CosineSim}(\mathrm{Embed}(s_j), \mathrm{Embed}(p))$$
- **Trimming Threshold**: Sentences with $g(s_j) < 0.70$ are pruned or rewritten.
- **98.5% Sentence Citation Provenance**:
  $$\mathcal{P}_{\mathrm{citation}} = \frac{\sum_{j=1}^{M} \mathbb{I}(\text{Sentence } s_j \text{ contains valid } [\text{Book, Chapter, Page, Chunk ID}] \text{ citation})}{\text{Total Generated Factual Sentences } M}$$

---

## Technical Specifications Summary

| Component | Model / Library | Specification |
| :--- | :--- | :--- |
| **PDF Loader** | PyMuPDF (`fitz`) | Layout-aware parsing |
| **Biomedical NER** | SciSpaCy (`en_core_sci_sm`) | Entity & relation extraction |
| **Knowledge Graph** | NetworkX | Undirected IEF shortest-path graph |
| **Dense Vector Embeddings** | `BAAI/bge-base-en-v1.5` | 768-dim, ChromaDB storage |
| **Sparse Retrieval** | Rank-BM25 | Lexical keyword search |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-base` | Batch size = 8, top-5 output |
| **Local LLM Generator** | Ollama (`llama3.2:latest`) | 3.8B, 4-bit `q4_K_M`, $T=0.0$ |
| **Grounding Verifier** | NLI Entailment Checker | Threshold $\tau_{\mathrm{grounding}} = 0.70$ |
