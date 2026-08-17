# MedGraphRAG: Medical Oncology Graph-Augmented RAG System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/khushal15jain/MegGraphRAG/blob/main/LICENSE) [![Framework](https://img.shields.io/badge/Architecture-GraphRAG%20%2B%20Hybrid%20Retrieval-green.svg)](https://github.com/khushal15jain/MegGraphRAG/blob/main) [![LLM](https://img.shields.io/badge/LLM-Llama--3.2%20%2F%20Ollama-orange.svg)](https://ollama.com/)

An end-to-end, deterministic, explainable, and hallucination-resistant clinical question-answering architecture designed specifically for medical oncology guidelines and textbooks.

MedGraphRAG integrates three complementary retrieval channels:

1. **High-Dimensional Dense Semantic Search** (`BAAI/bge-base-en-v1.5` in ChromaDB),
2. **Lexical Sparse Keyword Search** (BM25Okapi with query entity expansion), and
3. **Multi-Hop Knowledge Graph Traversal** (SciSpaCy biomedical NER with **Hub-Suppressed Hop-Distance Decay Scoring** $S_{\mathrm{graph}}(e, q) = \frac{1}{1 + d(e, q)}$ in NetworkX).

Candidates undergo Min-Max score fusion and are dynamically reranked using a Cross-Encoder (`BAAI/bge-reranker-base`) before context injection into a quantized `Llama-3.2` local clinical generator.

---

## 🌟 Key Features & Breakthroughs

- **Privacy-Preserving & Cloud-Free Execution**: Designed for local, cloud-free execution to support data-privacy-sensitive clinical environments using Ollama (`Llama-3.2`), ChromaDB, and NetworkX. No patient data leaves local hardware.
- **Hub-Suppressed Hop-Distance Decay Graph Scoring**: Suppresses hub entity frequency bias in Knowledge Graphs by applying topological shortest-path distance decay ($S_{\mathrm{graph}}(e, q) = \frac{1}{1 + d(e, q)}$), promoting high-specificity biomedical entities over generic clinical stop-words.
- **Cross-Encoder Reranking**: Utilizes `BAAI/bge-reranker-base` full cross-attention over fused candidate pools, driving **Precision@5 up to 0.8950** (vs. 0.3280 without reranker).
- **Deterministic Hallucination Resistance & Sentence Grounding**: Features dual-stage refusal gating and sentence-level grounding confidence thresholds ($\tau_g = 0.65$, $\tau_{\mathrm{low}} = 0.45$).
- **98.5% Sentence Citation Provenance**: Every generated claim includes verifiable document, section header, page number, and chunk attribution citations (`[Source: Document, Section, Page X, Chunk ID]`).

---

## 📊 Benchmark & Ablation Study Results ($N=500$ Evaluations)

Evaluated across a benchmark of 100 clinical oncology guideline questions across 5 distinct ablation modes ($N=500$ total inferences). Statistical significance tested against Baseline via paired two-tailed Wilcoxon signed-rank tests (\* $p < 0.05$, \*\* $p < 0.01$, \*\*\* $p < 0.001$); Latency via paired $t$-test.

### Key Scientific Takeaways:

1. **Full GraphRAG Superiority:** Fusing the Knowledge Graph outperforms the No Graph ablation across Retrieval Accuracy (**93.00% vs 92.00%**), Evidence Recall (**97.76% vs 97.00%**), Faithfulness (**0.9080 vs 0.6964**), and Hallucination reduction (**0.0920 vs 0.3036**).
2. **Explainability & Provenance (98.5%):** Measured as sentence-level citation coverage ($\frac{\text{Traceable Sentences}}{\text{Total Sentences}}$). Fused pipelines achieve **98.5% citation coverage**, while Dense-Only RAG collapses to **87.00%** ($p < 0.001$) due to ungrounded assertions.
3. **Indispensability of BM25:** Disabling BM25 keyword matching triggers a statistically significant drop in Groundedness (**0.6383 vs 0.9120**, $p < 0.01$) and Accuracy ($p < 0.05$).
4. **Cross-Encoder Precision:** Removing the reranker collapses Precision@5 (**0.3280 vs 0.8950**, $p < 0.001$) and Retrieval Accuracy ($p < 0.05$).
5. **Naive Vector RAG Collapse:** Dense Only RAG suffers a **+29.9% surge in hallucinations** ($0.3913$ vs $0.0920$) and significant drops in accuracy ($p < 0.01$).
6. **Groundedness Dynamics & Nuance:** The Baseline achieves top groundedness (**0.9120 ± 0.0370**) under NLI sentence grounding thresholding ($\tau_g = 0.65$). On un-thresholded passes, No Graph reaches 0.7550 vs 0.7517, which is not statistically significant ($p > 0.05$). Baseline leads across all primary grounding metrics (Faithfulness, Hallucination Reduction, Explainability).

---

## 🏗 System Architecture

```
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
                       [7] BGE Cross-Encoder Reranking (Top-15 ➔ Top-3 Gold Context)
                                           │
                                           ▼
                       [8] Dual Safety Gatekeeper (Similarity Threshold τ = 0.35)
                                           │
                                           ▼
                       [9] Constrained Prompt Construction & Llama-3.2 Local Generation
                                           │
                                           ▼
                       [10] NLI Sentence-Level Groundedness & Provenance Verification
                                           │
                                           ▼
                           [ Verified Clinical Answer + Citation Cards ]
```

---

## 📂 Project Structure

```
MedGraphRAG/
├── app/                     FastAPI web backend & Streamlit interface
├── benchmark/               Baseline methods & benchmark evaluation runners
├── configs/                 YAML configuration files (model, retrieval, paths)
├── data/                    Raw PDF guidelines, interim files, processed chunks, and gold_standard_dataset.json
├── docs/                    System documentation, IEEE papers, CITATION.cff, CONTRIBUTING.md, and reproducibility guides
├── embeddings/              BGE dense embedding wrapper & ChromaDB indexing pipeline
├── entity_extraction/       SciSpaCy NER and dependency-parsing relation extraction
├── evaluation/              Evaluators (Faithfulness, Groundedness, Accuracy, BLEU, ROUGE)
├── explainability/          Provenance tracking & source attribution models
├── generator/               Prompt templates, Ollama generation, and sentence grounding
├── graph/                   NetworkX Knowledge Graph construction & IDF graph retrieval
├── grounding/               Sentence-level NLI entailment checkers
├── notebooks/               Exploratory analysis and prototyping notebooks
├── preprocessing/           Layout-aware PDF parsing, text cleaning, section detection, & chunking
├── prompts/                 System prompts and QA prompt templates
├── reranker/                BAAI/bge-reranker-base cross-encoder integration
├── results/                 Ablation JSON results, summary tables, qualitative examples, and charts/
│   └── charts/              Generated publication figures (PNG)
├── retrieval/                Dense, BM25, query expansion, and hybrid fusion algorithms
├── tests/                   Unit and integration test suite
├── utils/                   Shared I/O, exception, and helper utilities
├── evaluate_extended_metrics.py  Resumable evaluator with BLEU/ROUGE/METEOR/F1 metrics
├── run_ablations.py         Full 500-evaluation ablation sweep runner
├── generate_publication_figures.py Statistical significance tester & chart generator
├── main.py                  Full pipeline end-to-end execution script
├── requirements.txt         Python environment dependencies
└── pyproject.toml           Project package definitions & metadata
```

---

## ⚡ Quick Start & Setup Guide

### 1. Prerequisites

- **OS**: macOS (Apple Silicon M-series recommended) or Linux.
- **Python**: Version 3.11.
- **Ollama**: Download and install [Ollama](https://ollama.com).

### 2. Environment Installation

```bash
# Clone the repository
git clone https://github.com/khushal15jain/MegGraphRAG.git
cd MedGraphRAG

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pull Ollama LLM Model

```bash
ollama pull llama3.2:latest
```

### 4. Configure Environment

```bash
cp .env.example .env
```

---

## 🚀 Running the Pipeline

### 1. Full Ingestion & Execution Pipeline

To run the full end-to-end ingestion (parsing, chunking, graph construction, vector indexing, and test querying):

```bash
python main.py
```

### 2. Execute Full 500-Evaluation Ablation Benchmark

To run the complete 100-question evaluation benchmark across all 5 configurations (Baseline, No Graph, No BM25, No Reranker, Dense Only):

```bash
python run_ablations.py
```

### 3. Generate Publication Figures & Statistical Significance Tables

To run paired Wilcoxon signed-rank tests, calculate p-values, and generate high-resolution figures:

```bash
python generate_publication_figures.py
```

*Outputs generated (to `results/charts/`):* `retrieval_accuracy_chart.png`, `faithfulness_chart.png`, `groundedness_chart.png`, `hallucination_chart.png`, `clinical_reliability_chart.png`, `latency_chart.png`, `radar_chart.png`.

### 4. Launch FastAPI Server & Frontend Interface

```bash
uvicorn app.api:app --reload --port 8000
```

Open `http://localhost:8000` in your web browser to access the interactive clinical QA interface.

### 5. Extended-Metrics Evaluation (BLEU/ROUGE/METEOR/F1)

To run a resumable evaluation pass over pre-generated answers with additional NLG metrics not covered by the ablation sweep:

```bash
python evaluate_extended_metrics.py
```

Skips already-scored questions in `results/fast_eval_results.csv` on rerun.

---

## 🧪 Unit & Integration Tests

Run the unit test suite:

```bash
pytest
```

The test suite utilizes lightweight mocks for embedding and model weights to verify pipeline components quickly.

---

## 📄 Citation

If you use this software, please see [`CITATION.cff`](CITATION.cff) for citation metadata, or use the "Cite this repository" button on GitHub.

---

## 📄 License & Attribution

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.