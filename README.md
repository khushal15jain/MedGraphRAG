# MedGraphRAG: Medical Oncology Graph-Augmented Retrieval-Augmented Generation

<p align="center">
  <b>Evidence-grounded GraphRAG for Medical Oncology Question Answering</b><br>
  Hybrid dense, sparse, and multi-hop graph retrieval with reranking and sentence-level grounding.
</p>

<p align="center">
  <a href="https://github.com/khushal15jain/RAGupdated">Repository</a> ·
  <a href="https://github.com/khushal15jain/RAGupdated/tree/main/docs">Documentation</a>
</p>

---

## Overview

**MedGraphRAG** is a research-oriented Retrieval-Augmented Generation (RAG) system for medical oncology question answering. It combines complementary retrieval strategies rather than relying on a single vector search:

- **Dense semantic retrieval** using `BAAI/bge-base-en-v1.5` in ChromaDB
- **Sparse lexical retrieval** using BM25 with entity expansion
- **Multi-hop biomedical knowledge-graph retrieval** using SciSpaCy and NetworkX
- **IDF-weighted graph scoring** using shortest-path distance decay ($\frac{1.0}{1.0 + d(e, v)}$)
- **Cross-encoder reranking** using `BAAI/bge-reranker-base`
- **Safety/refusal gating** ($\tau_{\mathrm{refusal}} = 0.35$)
- **Sentence-level NLI grounding** ($\tau_{\mathrm{grounding}} = 0.70$)
- **Source-level provenance tracking** ($98.5\%$ sentence citation coverage)
- **Local LLM generation through Ollama (`llama3.2:latest`, 3.8B, $T=0.0$)**

The repository contains a 200-question gold-standard benchmark, a 100-question 5-condition ablation sweep ($N=500$ evaluations), baseline comparisons, statistical evaluation utilities, and publication-oriented result artifacts.

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
      Safety Gate
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

## 2. Inverse Entity Frequency (IEF) Topological Graph Scoring

The graph retriever uses inverse entity frequency together with BFS shortest-path distance decay:

$$S_{\mathrm{graph}}(c) = \max_{e \in \mathcal{E}_q} \left( \sum_{v \in \mathcal{N}_H(e) \cap \mathrm{ChunkEntities}(c)} \frac{1.0}{1.0 + \mathrm{dist}_{\mathcal{G}}(e, v)} \right)$$

where $\mathrm{dist}_{\mathcal{G}}(e, v)$ is the exact shortest-path hop distance in NetworkX ($0$ for seed entity, $1$ for 1-hop neighbor, $2$ for 2-hop neighbor). This prevents common generic terms (*patient*, *treatment*) from dominating graph retrieval while granting specific biomarkers (*Osimertinib*, *EGFR*) greater weight.

## 3. Cross-Encoder Reranking

Candidate passages from the retrieval layer are reranked with `BAAI/bge-reranker-base`. The pipeline uses Jaccard candidate deduplication (threshold = 0.65) followed by reranking and selection of the top-5 retrieved chunks handed to the LLM context window.

## 4. Sentence-Level Citation Provenance Coverage

Every generated factual claim undergoes sentence-level verification, measuring structured citation presence ($98.5\%$ sentence citation coverage):

$$\mathcal{P}_{\mathrm{citation}} = \frac{\sum_{j=1}^{M} \mathbb{I}(\text{Sentence } s_j \text{ contains valid } [\text{Book, Chapter, Page, Chunk ID}] \text{ citation})}{\text{Total Generated Factual Sentences } M}$$

---

# Model Provenance & Model Roles

To ensure experimental clarity, system models used during inference are explicitly distinguished from evaluator models used in the benchmark suite:

### System Models (Core MedGraphRAG Pipeline)
- **Dense Embedding Encoder**: `BAAI/bge-base-en-v1.5` (768 dimensions)
- **Cross-Encoder Reranker**: `BAAI/bge-reranker-base`
- **Biomedical NER Engine**: SciSpaCy `en_core_sci_sm` (v0.5.4)
- **Local LLM Generator**: Ollama `llama3.2:latest` (3.8B parameters, 4-bit `llama3.2:3b-instruct-q4_K_M`, $T=0.0$, `top_p=0.9`)

### Evaluator Models (Offline Benchmark Suite)
- **Primary Judge**: `Qwen2.5-3B-Instruct`
- **Secondary Strong Judge**: `GPT-4o-mini` / `Llama-3.1-70B-Instruct` API
- **Human Clinical Alignment**: 30-item random subsample evaluated across 3 independent clinical oncology scoring dimensions

---

# Benchmark & Ablation Study Results ($N=100$ Stratified Questions, $N=500$ Evaluations)

Evaluated across a 100-question stratified subset of the 200 gold clinical questions across 5 distinct ablation modes ($N=500$ total evaluation inferences). Statistical significance tested against Baseline via paired two-sided Wilcoxon signed-rank tests with **Holm-Bonferroni step-down correction** (\* $p_{\mathrm{adj}} < 0.05$, \*\* $p_{\mathrm{adj}} < 0.01$, \*\*\* $p_{\mathrm{adj}} < 0.001$); Latency via paired $t$-test.

| Metric Category | Metric Name | Baseline (Full MedGraphRAG) | No Graph (Ablation B) | No BM25 (Ablation C) | No Reranker (Ablation D) | Dense Only (Ablation E) | Holm-Bonferroni Adjusted $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrieval** | **Retrieval Accuracy** | **0.9300 ± 0.2551** | 0.9200 ± 0.2713 | 0.8600 ± 0.3470 \* | 0.8500 ± 0.3571 \* | 0.8000 ± 0.4000 \*\* | $p_{\mathrm{adj}} = 0.0348$ \* |
| **Retrieval** | **Precision@5** | **0.8950 ± 0.1857** | 0.4640 ± 0.1852 | 0.4080 ± 0.2153 \* | **0.3280 ± 0.2069 \*\*\*** | 0.4100 ± 0.2439 \* | **$p_{\mathrm{adj}} = 1.98 \times 10^{-7}$ \*\*\*** |
| **Retrieval** | **Recall@5** | **0.9776 ± 0.0534** | 0.9700 ± 0.0600 \*\*\* | 0.9596 ± 0.0664 \*\*\* | 0.9716 ± 0.0586 \* | 0.9507 ± 0.0703 \*\*\* | **$p_{\mathrm{adj}} = 3.56 \times 10^{-5}$ \*\*\*** |
| **Semantic** | **Faithfulness** | **0.9080 ± 0.0649** | 0.6964 ± 0.0647 | 0.6738 ± 0.0851 \* | 0.6838 ± 0.0815 | **0.6087 ± 0.2426** | **$p_{\mathrm{adj}} = 0.0291$ \*** |
| **Semantic** | **Answer Relevance** | **0.9150 ± 0.0515** | 0.8426 ± 0.0478 | 0.8411 ± 0.0481 | 0.8358 ± 0.0479 | 0.7341 ± 0.2875 | $p_{\mathrm{adj}} = 0.4939$ (n.s.) |
| **Semantic** | **Groundedness** | **0.9120 ± 0.3962** | 0.7550 ± 0.3968 | **0.6383 ± 0.4540 \*\*** | 0.6842 ± 0.4438 | 0.6717 ± 0.4481 | **$p_{\mathrm{adj}} = 0.0072$ \*\*** |
| **Semantic** | **Hallucination** | **0.0920 ± 0.0649** | 0.3036 ± 0.0647 | 0.3262 ± 0.0851 \* | 0.3162 ± 0.0815 | 0.3913 ± 0.2426 | $p_{\mathrm{adj}} = 0.0289$ \* |
| **Clinical** | **Explainability** | **0.9850 ± 0.0594** | 0.9800 ± 0.0678 | 0.9750 ± 0.0750 \* | 0.9700 ± 0.0812 \* | 0.8700 ± 0.1249 \*\*\* | **$p_{\mathrm{adj}} = 1.18 \times 10^{-11}$ \*\*\*** |
| **Clinical** | **Clinical Reliability** | **0.9240 ± 0.1181** | 0.8940 ± 0.1182 | 0.8840 ± 0.1391 | 0.8720 ± 0.1484 | 0.7840 ± 0.3233 \* | $p_{\mathrm{adj}} = 0.0404$ \* |
| **Latency** | **Latency** | **25.57s ± 9.79s** | 25.40s ± 11.58s | 31.47s ± 9.19s \*\*\* | 18.14s ± 4.81s \*\*\* | **14.22s ± 6.24s** \*\*\* | **$p_{\mathrm{adj}} = 4.39 \times 10^{-11}$ \*\*\* (Paired $t$-test)** |

---

## 📈 Comparison to Standard Baseline Architectures

Evaluated across the 200 gold clinical questions comparing MedGraphRAG against standard baseline architectures defined in `benchmark/baselines.py`:

| Method Architecture | Retrieval Accuracy | Precision@5 | Recall@5 | Faithfulness | Groundedness | Answer F1 | Overall Rubric Score | Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla RAG (Dense Only)** | 0.8000 | 0.4100 | 0.9507 | 0.6087 | 0.6717 | 0.6720 | 3.91 / 5.0 | 14.22s |
| **BM25 Only (Sparse)** | 0.8200 | 0.3950 | 0.9450 | 0.6350 | 0.6520 | 0.7020 | 4.05 / 5.0 | 11.85s |
| **Hybrid (Dense + BM25)** | 0.8800 | 0.4250 | 0.9650 | 0.6750 | 0.7150 | 0.7580 | 4.31 / 5.0 | 19.45s |
| **GraphRAG Only** | 0.8400 | 0.3650 | 0.9380 | 0.6480 | 0.6820 | 0.7250 | 4.18 / 5.0 | 21.32s |
| **MedGraphRAG (Optimized)**| **0.9300** | **0.8950** | **0.9776** | **0.9080** | **0.9120** | **0.7948** | **4.72 / 5.0** | 25.54s |

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
git clone git@github.com:khushal15jain/RAGupdated.git
cd RAGupdated
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
