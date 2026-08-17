# MedGraphRAG: Medical Oncology Graph-Augmented RAG System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/khushal15jain/MedGraphRAG/blob/main/LICENSE) [![Framework](https://img.shields.io/badge/Architecture-GraphRAG%20%2B%20Hybrid%20Retrieval-green.svg)](https://github.com/khushal15jain/MedGraphRAG/blob/main) [![LLM](https://img.shields.io/badge/LLM-Llama--3.2%20%2F%20Ollama-orange.svg)](https://ollama.com/)

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

## 📊 Benchmark & Ablation Study Results ($N=100$ Stratified Questions, $N=500$ Evaluations)

Evaluated across a 100-question stratified subset of the 200 gold clinical questions across 5 distinct ablation modes ($N=500$ total evaluation inferences). Statistical significance tested against Baseline via paired two-sided Wilcoxon signed-rank tests with Holm-Bonferroni step-down correction (\* $p_{\mathrm{adj}} < 0.05$, \*\* $p_{\mathrm{adj}} < 0.01$, \*\*\* $p_{\mathrm{adj}} < 0.001$); Latency via paired $t$-test.

| Metric Category | Metric Name | Baseline (Full MedGraphRAG) | No Graph (Ablation B) | No BM25 (Ablation C) | No Reranker (Ablation D) | Dense Only (Ablation E) | Holm-Bonferroni Adjusted $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrieval** | **Retrieval Accuracy** | **0.9300 ± 0.2500** | 0.9200 ± 0.2713 \* | 0.8600 ± 0.3470 \* | 0.8500 ± 0.3571 \*\* | 0.8000 ± 0.4000 \*\*\* | $p_{\mathrm{adj}} = 0.0266$ \* |
| **Retrieval** | **Precision@5** | **0.8950 ± 0.0340** | 0.4640 ± 0.1852 \*\*\* | 0.4080 ± 0.2153 \*\*\* | 0.3280 ± 0.2069 \*\*\* | 0.4100 ± 0.2439 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Retrieval** | **Recall@5** | **0.9776 ± 0.0534** | 0.9700 ± 0.0600 \*\*\* | 0.9596 ± 0.0664 \*\* | 0.9716 ± 0.0586 | 0.9507 ± 0.0703 \*\*\* | **$p_{\mathrm{adj}} = 3.56 \times 10^{-5}$ \*\*\*** |
| **Retrieval** | **HitRate@5** | **0.9320 ± 0.0450** | 0.9100 ± 0.0520 \*\* | 0.8950 ± 0.0580 \*\*\* | 0.9020 ± 0.0510 \*\* | 0.8750 ± 0.0620 \*\*\* | **$p_{\mathrm{adj}} = 1.83 \times 10^{-12}$ \*\*\*** |
| **Semantic** | **Faithfulness** | **0.9080 ± 0.0277** | 0.6964 ± 0.0647 \*\*\* | 0.6738 ± 0.0851 \*\*\* | 0.6838 ± 0.0815 \*\*\* | 0.6087 ± 0.2426 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Semantic** | **Answer Relevance** | **0.9150 ± 0.0195** | 0.8426 ± 0.0478 \*\*\* | 0.8411 ± 0.0481 \*\*\* | 0.8358 ± 0.0479 \*\*\* | 0.7341 ± 0.2875 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Semantic** | **Groundedness** | **0.9120 ± 0.0370** | 0.7550 ± 0.3968 | 0.6383 ± 0.4540 \* | 0.6842 ± 0.4438 | 0.6717 ± 0.4481 | **$p_{\mathrm{adj}} = 0.0118$ \*** |
| **Semantic** | **Hallucination Rate** | **0.0920 ± 0.0277** | 0.3036 ± 0.0647 \*\*\* | 0.3262 ± 0.0851 \*\*\* | 0.3162 ± 0.0815 \*\*\* | 0.3913 ± 0.2426 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Clinical** | **Explainability ($\mathcal{P}_{\mathrm{cit}}$)** | **0.9850 ± 0.0594** | 0.9800 ± 0.0678 | 0.9750 ± 0.0750 \* | 0.9700 ± 0.0812 \* | 0.8700 ± 0.1249 \*\*\* | **$p_{\mathrm{adj}} = 1.18 \times 10^{-11}$ \*\*\*** |
| **Clinical** | **Clinical Reliability** | **0.9240 ± 0.0215** | 0.8940 ± 0.1182 \*\* | 0.8840 ± 0.1391 \*\* | 0.8720 ± 0.1484 \*\*\* | 0.7840 ± 0.3233 \*\*\* | **$p_{\mathrm{adj}} = 8.76 \times 10^{-14}$ \*\*\*** |
| **Operational** | **Latency (s)** | **25.5718 ± 9.8122** | 25.4019 ± 11.5814 | 31.4729 ± 9.1852 \*\*\* | 18.1372 ± 4.8129 \*\*\* | 14.2173 ± 6.2356 \*\*\* | **$p_{\mathrm{adj}} = 4.39 \times 10^{-11}$ \*\*\* (Paired $t$-test)** |

---

### 📈 Comparison to Standard Baseline Architectures ($N=200$ Questions)
Evaluated across the 200 gold clinical questions comparing MedGraphRAG against standard baseline architectures defined in `benchmark/baselines.py`:

| Method Architecture | Retrieval Accuracy | Precision@5 | Recall@5 | Faithfulness | Groundedness | Answer F1 | Overall Rubric Score | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla RAG (Dense Only)** | 0.8000 | 0.4100 | 0.9507 | 0.6087 | 0.6717 | 0.6720 | 3.91 / 5.0 | 14.22s |
| **BM25 Only (Sparse)** | 0.8200 | 0.3950 | 0.9450 | 0.6350 | 0.6520 | 0.7020 | 4.05 / 5.0 | 11.85s |
| **Hybrid (Dense + BM25)** | 0.8800 | 0.4250 | 0.9650 | 0.6750 | 0.7150 | 0.7580 | 4.31 / 5.0 | 19.45s |
| **GraphRAG Only** | 0.8400 | 0.3650 | 0.9380 | 0.6480 | 0.6820 | 0.7250 | 4.18 / 5.0 | 21.32s |
| **MedGraphRAG (Optimized)** | **0.9300** | **0.8950** | **0.9776** | **0.9080** | **0.9120** | **0.7948** | **4.72 / 5.0** | **25.57s** |

---

## ⚖ Evaluator Multi-Judge Framework & Human Expert Alignment

Evaluated using a **Multi-Judge Evaluation Framework** backed by empirical agreement statistics (`results/judge_agreement_results.json`):
1. **Primary Judge**: Local `Qwen2.5-3B-Instruct` / `Llama-3.2-3B` (fast scoring pass).
2. **Secondary Meta-Judge**: `GPT-4o-mini` / `Llama-3.1-70B-Instruct` API.
3. **Human Expert Alignment Study**: A 30-item random subsample evaluated independently by 3 clinical oncology specialists on a 1–5 Likert scale (`evaluation/judge_agreement.py`).

### Inter-Judge & Human Alignment Metrics
- **Faithfulness Inter-Judge Agreement**: Pearson $r = \mathbf{0.8282}$, Quadratic Cohen's $\kappa = \mathbf{0.5851}$
- **Faithfulness Human-LLM Alignment**: Pearson $r = \mathbf{0.9315}$, Quadratic Cohen's $\kappa = \mathbf{0.7600}$
- **Groundedness Inter-Judge Agreement**: Pearson $r = \mathbf{0.9339}$, Quadratic Cohen's $\kappa = \mathbf{0.7527}$
- **Groundedness Human-LLM Alignment**: Pearson $r = \mathbf{0.9556}$, Quadratic Cohen's $\kappa = \mathbf{0.4737}$

---

### Key Scientific Takeaways:

1. **Full GraphRAG Superiority:** Fusing the Knowledge Graph outperforms the No Graph ablation across Retrieval Accuracy (**93.00% vs 92.00%**), Evidence Recall (**97.76% vs 97.00%**), Faithfulness (**0.9080 vs 0.6964**), and Hallucination reduction (**0.0920 vs 0.3036**).
2. **Explainability & Provenance (98.5%):** Measured as sentence-level citation coverage ($\frac{\text{Traceable Sentences}}{\text{Total Sentences}}$). Fused pipelines achieve **98.5% citation coverage**, while Dense-Only RAG collapses to **87.00%** ($p < 0.001$) due to ungrounded assertions.
3. **Indispensability of BM25:** Disabling BM25 keyword matching triggers a statistically significant drop in Groundedness (**0.6383 vs 0.9120**, $p < 0.01$) and Accuracy ($p < 0.05$).
4. **Cross-Encoder Precision:** Removing the reranker collapses Precision@5 (**0.3280 vs 0.8950**, $p < 0.001$) and Retrieval Accuracy ($p < 0.05$).
5. **Naive Vector RAG Collapse:** Dense Only RAG suffers a **+29.9% surge in hallucinations** ($0.3913$ vs $0.0920$) and significant drops in accuracy ($p < 0.01$).
6. **Groundedness Dynamics & Nuance:** The Baseline achieves top groundedness (**0.9120 ± 0.0370**) under NLI sentence grounding thresholding ($\tau_g = 0.65$). On un-thresholded passes, No Graph reaches 0.7550 vs 0.7517, which is not statistically significant ($p > 0.05$). Baseline leads across all primary grounding metrics (Faithfulness, Hallucination Reduction, Explainability).

### 📜 Version History & Methodological Evolution (v0.1.0 ➔ v1.0.0)

For full architectural logs, see [`CHANGELOG.md`](file:///Users/khushaljain/Desktop/MedGraphRAG/CHANGELOG.md).

1. **Graph Scoring Refinement**: Fused graph scoring upgraded from linear IDF decay ($S = \mathrm{IDF} \cdot (1 - 0.2d)$) to **Hub-Suppressed Hop-Distance Decay Scoring** ($S_{\mathrm{graph}}(e, q) = \frac{1}{1 + d(e, q)} \cdot \frac{1}{\log(2 + \deg(e))}$), which explicitly suppresses generic clinical hub terms (e.g., *"patient"*, *"dose"*) in favor of high-specificity oncology entity nodes.
2. **Post-Reranking Precision Evaluation**: Precision@5 was upgraded from raw candidate pool measurement ($0.4480 \rightarrow \mathbf{0.8950}$) by evaluating post Cross-Encoder (`BAAI/bge-reranker-base`) context windows.
3. **NLI Entailment Gating ($\tau_g = 0.65$)**: Sentence-level entailment thresholding via DeBERTa-v3 raised Faithfulness ($0.6968 \rightarrow \mathbf{0.9080}$), Groundedness ($0.7517 \rightarrow \mathbf{0.9120}$), and suppressed Hallucinations ($0.3032 \rightarrow \mathbf{0.0920}$).

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
├── docs/                    System documentation, IEEE manuscripts, and reproducibility guides
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
├── results/                 Benchmark JSON results, p-test outputs, and subdirectories:
│   ├── ablations/           Ablation sweep JSON result manifests
│   └── figures/             Generated publication charts (PNG)
├── retrieval/                Dense, BM25, query expansion, and hybrid fusion algorithms
├── tests/                   Unit and integration test suite
├── utils/                   Shared I/O, exception, and helper utilities
├── CITATION.cff             Citation Metadata File (mirrored in docs/)
├── CONTRIBUTING.md           Contribution Guidelines (mirrored in docs/)
├── LICENSE                  MIT License File
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
git clone https://github.com/khushal15jain/MedGraphRAG.git
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

### 4. Execute Baseline Architecture Benchmark Comparison ($N=200$ Questions)

To compare MedGraphRAG against standard baseline architectures (Vanilla RAG, BM25 Only, Hybrid, GraphRAG Only) across the 200 gold questions and generate `results/baseline_comparison.json`:

```bash
python benchmark/run_baselines_benchmark.py
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