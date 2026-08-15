# MedGraphRAG Reproducibility Guide

This document details the exact environment, dataset splits, model tags, and commands required to reproduce all benchmark results reported for MedGraphRAG.

---

## 1. Reproducibility Status Matrix

| Pipeline Stage | Reproducibility Status | Notes / Requirements |
| :--- | :--- | :--- |
| **Main Evaluation ($N=200$)** | **ARTIFACT-REPRODUCIBLE** | Full 200-question dataset included in `data/gold_standard_dataset.json`. Requires local Ollama `llama3.2:latest` model server. |
| **Ablation Benchmark ($N=1000$)** | **FULLY REPRODUCIBLE** | 100-question subset across 5 ablation conditions ($100 \times 5 = 500$ inferences per benchmark run). Execution logged in `ablation_*.json`. |
| **Statistical Analysis** | **FULLY REPRODUCIBLE** | SciPy paired Wilcoxon signed-rank and paired $t$-tests executed via `evaluation/p_test_evaluator.py`. |
| **Figure Generation** | **FULLY REPRODUCIBLE** | Re-rendered using `generate_publication_figures.py`. Saved to `docs/images/`. |
| **Source PDF Ingestion** | **PARTIALLY REPRODUCIBLE** | Ingestion pipeline is fully reproducible; copyrighted PDF files must be independently acquired per `docs/source_corpus.md`. |

---

## 2. Canonical Experimental Environment

- **Operating System**: macOS (Apple Silicon M3 Max) / x86_64 Ubuntu Linux 22.04 LTS.
- **Python Runtime**: Python 3.11.8 (`.venv` virtual environment).
- **LLM Engine**: Ollama `v0.1.28` hosting `llama3.2:latest` (3B parameters, 4-bit quantized, `T = 0.0`).
- **Dense Embedding Model**: `BAAI/bge-base-en-v1.5` (768-dim, CPU execution).
- **Cross-Encoder Reranker**: `BAAI/bge-reranker-base` (batch size = 8, top-5 output).
- **Biomedical NER Engine**: SciSpaCy `en_core_sci_sm` (v0.5.4).
- **Random Seed**: `seed = 42` (Python, NumPy, PyTorch, Hydra).

---

## 3. Reproduction Commands

```bash
# 1. Clone repository and setup environment
git clone git@github.com:khushal15jain/RAGupdated.git
cd RAGupdated
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull local LLM generator
ollama pull llama3.2:latest

# 3. Execute main end-to-end RAG pipeline
python main.py

# 4. Run target metric optimization
python evaluation/run_full_optimization.py

# 5. Run full 5-way ablation benchmark (100 questions x 5 conditions = 500 evaluations)
python run_ablations.py

# 6. Compute paired Wilcoxon statistical p-values & effect sizes
python evaluation/p_test_evaluator.py

# 7. Generate publication-grade figures
python generate_publication_figures.py
```
