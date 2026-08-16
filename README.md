# MedGraphRAG: Medical Oncology Graph-Augmented Retrieval-Augmented Generation

<p align="center">
  <b>Evidence-grounded GraphRAG for Medical Oncology Question Answering</b><br>
  Hybrid dense, sparse, and multi-hop graph retrieval with reranking and sentence-level grounding.
</p>

<p align="center">
  <a href="https://github.com/khushal15jain/MegGraphRAG">Repository</a> ·
  <a href="https://github.com/khushal15jain/MegGraphRAG/tree/main/docs">Documentation</a>
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
- **Local LLM generation through Ollama** (configuration specifies `llama3.2:latest`, 3.8B, $T=0.0$)

The repository contains a 200-question gold-standard benchmark, a 100-question 5-condition ablation sweep ($N=500$ evaluations), statistical evaluation utilities, and publication-oriented result artifacts.

> **Research disclaimer:** MedGraphRAG is a research prototype designed for evidence-grounded medical information retrieval and question answering. It is **not a medical device and must not be used as a substitute for qualified clinical judgment, diagnosis, or treatment decisions.**

---

## Research Motivation

Standard RAG systems can retrieve semantically similar passages but may struggle with:

1. Multi-hop relationships between biomedical entities
2. Rare but clinically important biomarkers and disease concepts
3. Lexical terminology variation
4. Retrieval of evidence distributed across multiple document sections
5. Unsupported generated statements
6. Traceability from generated claims back to source evidence

MedGraphRAG addresses these limitations by combining:

```text
Dense Semantic Retrieval
          +
Sparse BM25 Retrieval
          +
Multi-Hop Biomedical Graph Retrieval
          ↓
      Score Fusion
          ↓
   Cross-Encoder Reranking
          ↓
    Local LLM Generation
          ↓
 Sentence-Level NLI Grounding
          ↓
 Answer + Source Provenance
```

---

# Key Contributions

## 1. Hybrid Retrieval

Three complementary retrieval channels are combined:

| Retrieval Channel | Purpose | Implementation Framework |
|---|---|---|
| Dense BGE Retrieval | Semantic similarity | `BAAI/bge-base-en-v1.5` + ChromaDB |
| BM25 | Exact lexical and terminology matching | Rank-BM25 + SciSpaCy entity expansion |
| Knowledge Graph | Entity relationships and multi-hop evidence | SciSpaCy (`en_core_sci_sm`) + NetworkX |

This combination reduces dependence on any single retrieval mechanism.

## 2. Hub-Suppressed Hop-Distance-Decayed Graph Scoring

The graph retriever uses BFS shortest-path distance decay with hub suppression (skipping nodes with degree $>100$):

$$S_{\mathrm{graph}}(e, q) = \frac{1.0}{1.0 + d(e, q)}$$

where $d(e, q)$ is the exact shortest-path hop distance in NetworkX ($0$ for seed entity, $1$ for 1-hop neighbor, $2$ for 2-hop neighbor). This prevents common generic terms (*patient*, *treatment*) from dominating graph retrieval while granting specific biomarkers (*Osimertinib*, *EGFR*) greater weight.

## 3. Cross-Encoder Reranking

Candidate passages from the retrieval layer are reranked with `BAAI/bge-reranker-base`. The pipeline uses Jaccard candidate deduplication (threshold = 0.65) followed by reranking and selection of top-k retrieved chunks handed to the LLM context window.

## 4. Sentence Citation Coverage

Every generated factual claim undergoes sentence-level verification, measuring structured citation presence ($98.5\%$ sentence citation coverage):

$$\mathcal{P}_{\mathrm{citation}} = \frac{\sum_{j=1}^{M} \mathbb{I}(\text{Sentence } s_j \text{ contains valid } [\text{Book, Chapter, Page, Chunk ID}] \text{ citation})}{\text{Total Generated Factual Sentences } M}$$

*Note:* This metric measures valid citation presence according to the structural definition, rather than clinical correctness.

---

# Model Provenance & Model Roles

To ensure experimental clarity, system models used during inference are explicitly distinguished from evaluator models used in the benchmark suite:

### System Models (Core MedGraphRAG Pipeline)
- **Dense Embedding Encoder**: `BAAI/bge-base-en-v1.5` (768 dimensions)
- **Cross-Encoder Reranker**: `BAAI/bge-reranker-base`
- **Biomedical NER Engine**: SciSpaCy `en_core_sci_sm` (v0.5.4)
- **Local LLM Generator**: Ollama local LLM; the released configuration specifies `llama3.2:latest`, while an alternate constructor default specifies `qwen2.5:3b-instruct`. The exact generator used for the reported benchmark should be confirmed from the experiment log.

### Evaluator Models (Offline Benchmark Suite)
- **Primary Judge**: `Qwen2.5-3B-Instruct`
- **Secondary Strong Judge**: `GPT-4o-mini` / `Llama-3.1-70B-Instruct` API
- **Human Clinical Alignment**: 30-item random subsample evaluated across 3 independent clinical oncology scoring dimensions

---

# Benchmark & Ablation Study Results ($N=100$ Stratified Questions, $N=500$ Evaluations)

Evaluated across a 100-question stratified subset of the 200 gold clinical questions across 5 distinct ablation modes ($N=500$ total evaluation inferences). Statistical significance tested against Baseline via paired two-sided Wilcoxon signed-rank tests with **Holm-Bonferroni step-down correction** (\* $p_{\mathrm{adj}} < 0.05$, \*\* $p_{\mathrm{adj}} < 0.01$, \*\*\* $p_{\mathrm{adj}} < 0.001$); Latency via paired $t$-test.

| Metric Category | Metric Name | Baseline (Full MedGraphRAG) | No Graph (Ablation B) | No BM25 (Ablation C) | No Reranker (Ablation D) | Dense Only (Ablation E) | Holm-Bonferroni Adjusted $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrieval** | **Retrieval Accuracy** | **0.9300 ± 0.2500** | 0.9200 ± 0.2713 \* | 0.8600 ± 0.3470 \* | 0.8500 ± 0.3571 \*\* | 0.8000 ± 0.4000 \*\*\* | $p_{\mathrm{adj}} = 0.0266$ \* |
| **Retrieval** | **Precision@5** | **0.8950 ± 0.0340** | 0.4640 ± 0.1852 \*\*\* | 0.4080 ± 0.2153 \*\*\* | 0.3280 ± 0.2069 \*\*\* | 0.4100 ± 0.2439 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Retrieval** | **Recall@5** | **0.9776 ± 0.0534** | 0.9700 ± 0.0600 \*\*\* | 0.9596 ± 0.0664 \*\* | 0.9716 ± 0.0586 | 0.9507 ± 0.0703 \*\*\* | **$p_{\mathrm{adj}} = 3.56 \times 10^{-5}$ \*\*\*** |
| **Semantic** | **Faithfulness** | **0.9080 ± 0.0277** | 0.6964 ± 0.0647 \*\*\* | 0.6738 ± 0.0851 \*\*\* | 0.6838 ± 0.0815 \*\*\* | 0.6087 ± 0.2426 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Semantic** | **Answer Relevance** | **0.9150 ± 0.0195** | 0.8426 ± 0.0478 \*\*\* | 0.8411 ± 0.0481 \*\*\* | 0.8358 ± 0.0479 \*\*\* | 0.7341 ± 0.2875 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Semantic** | **Groundedness** | **0.9120 ± 0.0370** | 0.7550 ± 0.3968 | 0.6383 ± 0.4540 \* | 0.6842 ± 0.4438 | 0.6717 ± 0.4481 | **$p_{\mathrm{adj}} = 0.0118$ \*** |
| **Semantic** | **Hallucination Rate** | **0.0920 ± 0.0277** | 0.3036 ± 0.0647 \*\*\* | 0.3262 ± 0.0851 \*\*\* | 0.3162 ± 0.0815 \*\*\* | 0.3913 ± 0.2426 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Clinical** | **Explainability ($\mathcal{P}_{\mathrm{cit}}$)** | **0.9850 ± 0.0594** | 0.9800 ± 0.0678 | 0.9750 ± 0.0750 \* | 0.9700 ± 0.0812 \* | 0.8700 ± 0.1249 \*\*\* | **$p_{\mathrm{adj}} = 1.18 \times 10^{-11}$ \*\*\*** |
| **Clinical** | **Clinical Reliability** | **0.9240 ± 0.0215** | 0.8940 ± 0.1182 \*\* | 0.8840 ± 0.1391 \*\* | 0.8720 ± 0.1484 \*\*\* | 0.7840 ± 0.3233 \*\*\* | **$p_{\mathrm{adj}} = 8.76 \times 10^{-14}$ \*\*\*** |
| **Operational** | **Latency (s)** | **25.5718 ± 9.8122** | 25.4019 ± 11.5814 | 31.4729 ± 9.1852 \*\*\* | 18.1372 ± 4.8129 \*\*\* | 14.2173 ± 6.2356 \*\*\* | **$p_{\mathrm{adj}} = 4.39 \times 10^{-11}$ \*\*\* (Paired $t$-test)** |

> **Note:** These figures correspond to the ablation results reported in Tables 3–6 of the accompanying manuscript.

---

## ⚖ Evaluator Judge Framework & Human Expert Alignment

Evaluation metrics were verified using a **Dual-Judge Framework**:
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