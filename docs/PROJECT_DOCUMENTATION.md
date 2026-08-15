# MedGraphRAG Project Documentation

## Project Overview

**MedGraphRAG** is an evidence-grounded, privacy-preserving, and explainable Clinical Decision Support System (CDSS) designed specifically for medical oncology question answering. MedGraphRAG integrates three complementary retrieval channels—Dense Semantic Vector Search, Lexical BM25 Keyword Search, and Multi-Hop Knowledge Graph Traversal—with Cross-Encoder Reranking and Sentence-Level NLI Grounding.

---

## Technical Architecture & Workflow

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

## Canonical Performance Metrics

Evaluated across the 200 Gold Clinical Question Benchmark ($N=200$ main benchmark, $N=500$ ablation evaluations across a 100-question subset):

- **Retrieval Accuracy**: **0.9300 ± 0.2551**
- **Precision@5**: **0.8950 ± 0.1857**
- **Recall@5**: **0.9776 ± 0.0534**
- **Faithfulness**: **0.9080 ± 0.0649**
- **Answer Relevance**: **0.9150 ± 0.0515**
- **Groundedness**: **0.9120 ± 0.3962**
- **Hallucination Rate**: **0.0920 ± 0.0649**
- **Explainability (Citation Coverage)**: **0.9850 ± 0.0594**
- **Clinical Reliability**: **0.9240 ± 0.1181**
- **Latency**: **25.57s ± 9.79s**

---

## Technology Stack & Provenance

| Component | Technology | Exact Version / Tag |
| :--- | :--- | :--- |
| **Language** | Python | 3.11.8 |
| **PDF Loader** | PyMuPDF | `fitz` v1.23 |
| **Biomedical NER** | SciSpaCy | `en_core_sci_sm` v0.5.4 |
| **Knowledge Graph** | NetworkX | Undirected Graph |
| **Dense Embeddings** | ChromaDB | `BAAI/bge-base-en-v1.5` (768-dim) |
| **Sparse Search** | Rank-BM25 | BM25Okapi |
| **Reranker** | Cross-Encoder | `BAAI/bge-reranker-base` |
| **Local LLM Generator** | Ollama | `llama3.2:latest` (3.8B, 4-bit `q4_K_M`, $T=0.0$) |
| **Grounding Engine** | NLI Checker | Sentence-level claim verifier ($\tau = 0.70$) |

---

## Quickstart Execution Commands

```bash
# 1. Environment Setup
git clone git@github.com:khushal15jain/RAGupdated.git
cd RAGupdated
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Local LLM Pull
ollama pull llama3.2:latest

# 3. Test Suite & Main Pipeline Execution
pytest tests/test_reproducibility.py tests/test_publication_reproducibility.py
python main.py
python run_ablations.py --num-questions 100
python evaluation/p_test_evaluator.py
python generate_paper_tables.py
```
