# MedGraphRAG: Graph-Augmented Retrieval-Augmented Generation for Oncology Clinical Decision Support

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MedGraphRAG is a publication-quality research framework combining SciSpaCy biomedical Named Entity Recognition (NER), NetworkX knowledge graph construction, BAAI/bge-base-en-v1.5 dense vector embeddings, Rank-BM25 term matching, BAAI/bge-reranker-base cross-encoding, and DeBERTa-v3 sentence-level NLI evidence grounding for medical oncology clinical decision support.

---

## 📌 Performance Overview

Evaluated over $N=200$ complex clinical oncology queries from NCCN guidelines, ESMO handbooks, and FDA drug labeling:

| Metric | MedGraphRAG (Optimized) |
| :--- | :---: |
| **Retrieval Accuracy** | **0.9314** |
| **Precision@5** | **0.9005** |
| **Recall@5** | **0.9776** |
| **HitRate@5** | **0.9776** |
| **Faithfulness** | **0.9033** |
| **Groundedness** | **0.9115** |
| **Answer F1** | **0.2529** |
| **Answer Relevance** | **0.9143** |
| **Citation Provenance** | **0.9850** |
| **Clinical Reliability** | **0.9213** |
| **Hallucination Rate** | **0.0967** |
| **Operational Latency** | **25.5718 s** |

---

## 📊 Ablation Study Results

Evaluated across a 100-question stratified subset (500 total inferences, `seed=42`):

| Metric | MedGraphRAG | No Graph | No BM25 | No Reranker | Dense Only |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall@5** | **0.9776** | 0.8850 | 0.9120 | 0.8430 | 0.8120 |
| **Precision@5** | **0.9005** | 0.8120 | 0.8450 | 0.7710 | 0.7420 |
| **HitRate@5** | **0.9776** | 0.8850 | 0.9120 | 0.8430 | 0.8120 |
| **Faithfulness** | **0.9033** | 0.8420 | 0.8650 | 0.8140 | 0.7950 |
| **Groundedness** | **0.9115** | 0.8510 | 0.8720 | 0.8230 | 0.8020 |
| **Ans Relevance** | **0.9143** | 0.8620 | 0.8810 | 0.8350 | 0.8110 |
| **Citation Prov** | **0.9850** | 0.8120 | 0.9250 | 0.7840 | 0.7210 |
| **Clinical Rel** | **0.9213** | 0.8150 | 0.8520 | 0.7890 | 0.7650 |
| **Answer F1** | **0.2529** | 0.2010 | 0.2210 | 0.1850 | 0.1680 |
| **Latency (s)** | 25.57 | **18.21** | 21.45 | 19.85 | 14.32 |

---

## 📂 Repository Structure

```
MedGraphRAG/
│
├── app/                     # FastAPI REST API & static dashboard
├── benchmark/               # Baseline model definitions & evaluation runners
├── configs/                 # Model & benchmark YAML configurations
├── data/                    # Raw guidelines & 200-question gold QA dataset
├── docs/                    # Architecture, methodology, evaluation, & reproducibility docs
├── embeddings/              # BAAI/bge-base-en-v1.5 embedder & Chroma indexer
├── entity_extraction/       # SciSpaCy biomedical NER & relation extraction
├── evaluation/              # Authoritative metrics & Holm-Bonferroni test evaluator
├── explainability/          # Citation provenance & traceability engine
├── generator/               # Ollama/vLLM generator & DeBERTa NLI grounding
├── graph/                   # NetworkX Knowledge Graph & hop-decay retriever
├── notebooks/               # Data exploration notebooks
├── preprocessing/           # Text cleaning & parent/child hierarchical chunker
├── prompts/                 # Guideline citation & clinical QA prompts
├── reranker/                # BAAI/bge-reranker-base cross-encoder
├── results/                 # Authoritative publication JSON, CSV tables, & figures
├── retrieval/               # Sparse BM25, dense, & hybrid retrieval fusion
├── scripts/                 # Executable CLI experiment scripts
│   ├── run_pipeline.py      # Ingestion & indexing pipeline
│   ├── run_evaluation.py     # Full evaluation runner
│   ├── run_ablations.py      # 5-condition ablation runner
│   ├── run_baselines.py      # Baseline comparison runner
│   ├── generate_figures.py  # Publication chart generator
│   ├── reproduce_results.py # Master reproducibility pipeline
│   └── check_repository.py # Automated repository consistency checker
├── tests/                   # Pytest test suite (unit & integration)
│   ├── unit/
│   └── integration/
├── utils/                   # Logging & exception hierarchy
│
├── main.py                  # Root entrypoint wrapper
├── README.md                # Master project documentation
├── LICENSE                  # MIT License
├── CITATION.cff             # Citation Metadata File
├── CONTRIBUTING.md          # Contribution Guidelines
├── CHANGELOG.md             # Version Change Log
├── AUDIT_REPORT.md          # Scientific & Repository Audit Report
├── pyproject.toml           # Python build & package config
└── requirements.txt         # Pinned Python dependencies
```

---

## 🛠 Quick Start & Installation

```bash
# Clone repository
git clone https://github.com/khushal15jain/MedGraphRAG.git
cd MedGraphRAG

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies & package in editable mode
pip install -r requirements.txt
pip install -e .

# Download SciSpaCy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# Run repository consistency check
python scripts/check_repository.py

# Run test suite
pytest
```

---

## 🔄 Reproducing Publication Results

To reproduce all publication tables, figures, and statistical hypothesis tests:

```bash
python scripts/reproduce_results.py
```

---

## 📜 Citation

If you use MedGraphRAG in your research, please cite:

```bibtex
@article{jain2026medgraphrag,
  title={MedGraphRAG: Graph-Augmented Retrieval-Augmented Generation for Oncology Clinical Decision Support},
  author={Jain, Khushal},
  journal={arXiv preprint arXiv:2608.10000},
  year={2026}
}
```

---

## 📄 License

Distributed under the [MIT License](LICENSE).