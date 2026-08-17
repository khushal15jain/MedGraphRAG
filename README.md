# MedGraphRAG: Medical Oncology Graph-Augmented Retrieval-Augmented Generation

<p align="center">
  <b>Evidence-grounded GraphRAG for Medical Oncology Question Answering</b><br>
  Hybrid dense, sparse, and multi-hop graph retrieval with reranking and sentence-level grounding.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/khushal15jain/MegGraphRAG/actions"><img src="https://img.shields.io/badge/CI-Passing-brightgreen.svg" alt="CI"></a>
  <a href="https://github.com/khushal15jain/MegGraphRAG">Repository</a> ·
  <a href="docs/REVISED_MANUSCRIPT.md">Revised Manuscript</a> ·
  <a href="docs/reproducibility.md">Reproducibility Guide</a>
</p>

---

## Overview

**MedGraphRAG** is a research-oriented Retrieval-Augmented Generation (RAG) system for medical oncology question answering. It combines complementary retrieval strategies rather than relying on a single vector search:

- **Dense semantic retrieval** using `BAAI/bge-base-en-v1.5` in ChromaDB
- **Sparse lexical retrieval** using BM25 with entity expansion
- **Hub-suppressed multi-hop graph retrieval with hop-distance decay** ($S_{\mathrm{graph}}(e, q) = \frac{1}{1 + d(e, q)}$) using SciSpaCy and NetworkX
- **Cross-encoder reranking** using `BAAI/bge-reranker-base`
- **Sentence-level grounding**: $\tau_g = 0.65$ (grounded threshold), $\tau_{\mathrm{low}} = 0.45$ (low confidence threshold)
- **Source-level provenance tracking** (Sentence citation coverage: $98.5\%$)
- **Local LLM generation through Ollama** (`llama3.2:latest`, 3.8B, $T=0.0$)

The repository contains a 200-question gold-standard benchmark, a 100-question 5-condition ablation sweep ($N=500$ evaluations), statistical evaluation utilities, and publication-oriented result artifacts.

> **Research disclaimer:** MedGraphRAG is a research prototype designed for evidence-grounded medical information retrieval and question answering. It is **not a medical device and must not be used as a substitute for qualified clinical judgment, diagnosis, or treatment decisions.**

---

## 🛠 Project Structure & Documentation Index

```text
MedGraphRAG/
├── .github/workflows/ci.yml       # GitHub Actions automated test workflow
├── benchmark/                      # Baseline comparison benchmark suite
│   ├── baselines.py                # Vanilla Dense, BM25, Hybrid, GraphRAG definitions
│   └── run_baselines_benchmark.py  # Baseline benchmark execution script
├── configs/                        # Hydra YAML configurations (model, retrieval, paths)
├── data/                           # Clinical QA gold-standard dataset & raw guidelines
│   ├── qa_dataset.json             # 200 gold-standard oncology QA pairs
│   └── raw/                        # ESMO / NCCN guideline metadata and source text
├── docs/                           # Publication manuscripts & technical documentation
│   ├── REVISED_MANUSCRIPT.md       # IEEE revised manuscript (scoped ablation framing)
│   ├── ABLATION_STUDY_REPORT.md    # Detailed 5-condition ablation evaluation report
│   ├── PROJECT_DOCUMENTATION.md    # Comprehensive system architecture specification
│   ├── ablation_summary_table.md   # Canonical metric ablation summary table
│   ├── qualitative_examples.md     # Qualitative clinical QA outputs & citation cards
│   ├── reproducibility.md          # Full evaluation setup & hardware replication guide
│   └── source_corpus.md            # Guidelines corpus provenance & entity graph spec
├── embeddings/                     # BGE dense embedder & ChromaDB vector store wrapper
├── entity_extraction/              # SciSpaCy NER & dependency-parsed relation extractor
├── evaluation/                     # Statistical evaluation, p-tests & Holm-Bonferroni
│   ├── p_test_evaluator.py         # Paired Wilcoxon signed-rank test & effect size calculator
│   └── judge_agreement.py          # Inter-judge & human clinical alignment metrics
├── generator/                      # Citation-constrained prompt builder & local LLM
├── graph/                          # NetworkX knowledge graph builder & BFS retriever
├── reranker/                       # BGE cross-encoder reranker wrapper
├── retrieval/                      # Hybrid retrieval fusion & adaptive graph expansion
├── tests/                          # Automated Pytest reproducibility test suite
│   ├── test_reproducibility.py     # Schema, ID alignment, and scoring logic tests
│   └── test_publication_reproducibility.py # Manifest schema and statistical tests
├── CITATION.cff                    # Citation metadata file
├── CONTRIBUTING.md                 # Contribution & PR guidelines
├── LICENSE                         # MIT Open-Source License
├── main.py                         # End-to-end pipeline execution entrypoint
├── README.md                       # Main repository overview & quickstart
└── requirements.txt                # Pinned dependencies (==) for 100% reproducibility
```

---

## 🔬 Dataset Provenance & Hardware Specifications

- **Dataset Origin**: 200 clinical oncology question-answer pairs curated from peer-reviewed clinical guidelines (ESMO Handbook of Immuno-Oncology, Oxford Handbook of Oncology 4th Ed, MD Anderson Manual 3rd Ed).
- **Category Breakdown**: Treatment (50), Diagnosis (30), Epidemiology (26), Prognosis (21), Biomarkers (19), Mechanism (12), Staging (7), Pathology (7), Other (22).
- **Target Hardware**: Single Apple Silicon machine (M-series, 8 GB RAM, CPU-only execution, `device: cpu`).
- **Runtime Performance**: Average latency of **25.57 seconds** per query inference. The full 100-question 5-condition ablation sweep ($N=500$ inferences) executes in approximately 3.5 hours on CPU.

---

## 📊 Benchmark & Ablation Study Results ($N=100$ Stratified Questions, $N=500$ Evaluations)

Evaluated across a 100-question stratified subset of the 200 gold clinical questions across 5 distinct ablation modes ($N=500$ total evaluation inferences). Statistical significance tested against Baseline via paired two-sided Wilcoxon signed-rank tests with **Holm-Bonferroni step-down correction** for 50 hypothesis tests ($5 \text{ conditions} \times 10 \text{ metrics}$) (\* $p_{\mathrm{adj}} < 0.05$, \*\* $p_{\mathrm{adj}} < 0.01$, \*\*\* $p_{\mathrm{adj}} < 0.001$); Latency via paired $t$-test.

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

> **Note:** These figures correspond to the ablation results reported in Tables 3–6 of the accompanying manuscript (`docs/REVISED_MANUSCRIPT.md`).

---

## 💬 Sample Clinical QA Output & Citation Card

```text
Query: What is the first-line treatment for EGFR exon 19 deletion non-small cell lung cancer?

Generated Answer:
First-line treatment for metastatic non-small cell lung cancer harboring EGFR exon 19 deletion is 
Osimertinib 80mg orally once daily [P1]. Osimertinib demonstrates superior progression-free survival 
compared to first-generation EGFR TKIs such as gefitinib or erlotinib [P2].

Citation Cards:
---------------------------------------------------------------------------------------------------
[P1] Document: ESMO Handbook of Immuno-Oncology | Section: 4.2 Non-Small Cell Lung Cancer
     Page: 114 | Chunk ID: chunk_nsclc_egfr_042 | Grounding Confidence: 0.9420 (Grounded)
[P2] Document: Oxford Handbook of Oncology 4th Ed | Section: Targeted Therapies in NSCLC
     Page: 289 | Chunk ID: chunk_nsclc_tki_289 | Grounding Confidence: 0.9150 (Grounded)
---------------------------------------------------------------------------------------------------
```

---

## ⚖ Evaluator Judge Framework & Human Expert Alignment

Evaluation metrics were assessed using a **Multi-Judge Evaluation Framework**:
1. **Primary Judge**: Local `Qwen2.5-3B-Instruct` (fast 1–5 scoring pass).
2. **Secondary Meta-Judge**: `GPT-4o-mini` / `Llama-3.1-70B-Instruct`.
3. **Human Expert Alignment Study**: A 30-item random subsample evaluated independently by 3 clinical oncology specialists on a 1–5 Likert scale across Factual Accuracy, Patient Safety, and Completeness (`evaluation/judge_agreement.py`).

### Inter-Judge & Human Alignment Metrics
- **Faithfulness Inter-Judge Agreement**: Pearson $r = \mathbf{0.9644}$, Cohen's $\kappa = \mathbf{0.6815}$.
- **Faithfulness Human-LLM Alignment**: Pearson $r = \mathbf{0.9683}$, Cohen's $\kappa = \mathbf{0.5408}$.
- **Groundedness Inter-Judge Agreement**: Pearson $r = \mathbf{0.9998}$, Cohen's $\kappa = \mathbf{1.0000}$.
- **Groundedness Human-LLM Alignment**: Pearson $r = \mathbf{0.9996}$, Cohen's $\kappa = \mathbf{1.0000}$.

---

## 💻 Quickstart & Reproduction Commands

```bash
# 1. Clone repository and setup virtual environment
git clone git@github.com:khushal15jain/MegGraphRAG.git
cd MegGraphRAG
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Pull local LLM model via Ollama
ollama pull llama3.2:latest

# 3. Run automated reproducibility test suite
pytest tests/test_reproducibility.py tests/test_publication_reproducibility.py

# 4. Run main pipeline and ablation benchmark
python main.py
python run_ablations.py --num-questions 100
python evaluation/p_test_evaluator.py
python generate_paper_tables.py
```

---

## 📜 License & Citation

Distributed under the **MIT Open-Source License**.

If you use MedGraphRAG in your research, please cite:

```bibtex
@misc{jain2026medgraphrag,
  author       = {Khushal Jain},
  title        = {MedGraphRAG: An Ablation Study of Hybrid Dense–Sparse–Graph Retrieval for Evidence-Grounded Clinical Question Answering},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/khushal15jain/MegGraphRAG}}
}
```