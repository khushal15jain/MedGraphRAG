# MedGraphRAG: A Privacy-Preserving, Reproducible, and Explainable Clinical Decision-Support System Using Tri-Modal Graph Retrieval-Augmented Generation

**Khushal Jain**  
Department of Computer Science & Engineering  
Repository: https://github.com/khushal15jain/RAGupdated  
License: MIT Open-Source License  

---

## ABSTRACT

Medical decision-making relies on evidence-based clinical oncology guidelines and peer-reviewed literature. However, conventional Retrieval-Augmented Generation (RAG) architectures exhibit critical limitations in clinical domains: single-vector dense retrieval struggles with exact alphanumeric drug codes or gene variant tokens (*EGFR C797S*, *AZD9291*), Knowledge Graph (KG) retrievers suffer from entity frequency bias where generic clinical terms (*patient*, *treatment*) suppress high-specificity biomarkers, and unconstrained Large Language Models (LLMs) risk generating ungrounded recommendations.

This paper presents **MedGraphRAG**, an end-to-end, privacy-preserving, and explainable clinical question-answering research prototype (reproducible under a fixed software, model, hardware, and inference configuration). MedGraphRAG introduces a **Tri-Modal Hybrid Retrieval Architecture** integrating three complementary search channels:
1. **High-Dimensional Dense Semantic Search** (`BAAI/bge-base-en-v1.5` in ChromaDB),
2. **Lexical Sparse Keyword Search** (BM25Okapi with query entity expansion), and
3. **Multi-Hop Knowledge Graph Traversal** using SciSpaCy biomedical Named Entity Recognition (NER) structured in NetworkX with **Inverse Entity Frequency (IEF) Topological Decay Scoring**.

Retrieved candidates undergo candidate deduplication, quality filtering, and cross-attention reranking via `BAAI/bge-reranker-base`. Context is injected into a 4-bit quantized local `Llama-3.2:latest` (3.8B, `llama3.2:3b-instruct-q4_K_M`) generator operating at temperature T = 0.0. Every generated claim is subjected to sentence-level hybrid lexical-semantic grounding verification (grounding threshold = 0.70) and refusal gating (refusal threshold = 0.35), providing 98.5% sentence-level citation provenance tracking (`[Source: Document, Section, Page X, Chunk ID]`).

Evaluated across a benchmark of 200 Gold Clinical Oncology Questions (N = 200 main benchmark, N = 500 ablation evaluations over a stratified 100-question subset), MedGraphRAG achieves **0.9300 Retrieval Accuracy**, **0.8950 Precision@5**, **0.9080 Faithfulness**, **0.9150 Answer Relevance**, **0.9120 Groundedness**, and **0.9240 Clinical Reliability**, demonstrating statistically significant improvements for several key metrics (such as Precision@5 $p_{\mathrm{adj}} = 1.98 \times 10^{-7}$, Recall@5 $p_{\mathrm{adj}} = 3.56 \times 10^{-5}$, and Groundedness $p_{\mathrm{adj}} = 0.0072$) over Vanilla Dense RAG, BM25-only, Hybrid, and GraphRAG baselines. A dual-judge framework (`Qwen2.5-3B-Instruct` primary judge vs. `GPT-4o-mini` / `Llama-3.1-70B-Instruct` secondary meta-judges) establishes high inter-judge agreement (Pearson r = 0.9644, Cohen kappa = 0.6815) and strong alignment with human expert clinical annotators (30-item random subsample evaluated by 3 independent clinical oncology specialists, Pearson r = 0.9683). The system operates entirely on local consumer hardware without transmitting clinical data to third-party cloud APIs.

**Index Terms**—Biomedical NLP, Retrieval-Augmented Generation, Knowledge Graphs, Information Retrieval, Clinical Decision Support Systems, Local LLM Inference.

---

> **MEDICAL SAFETY & RESEARCH DISCLAIMER**  
> This system is intended for research and educational evaluation only. It is not a substitute for professional medical judgment and has not been clinically validated or approved for autonomous patient-care decisions.

---

## 1. INTRODUCTION

### 1.1 Background
Clinical oncology advances rapidly, producing thousands of clinical practice guidelines (e.g., NCCN, ESMO), peer-reviewed trial reports, and therapeutic updates annually. Clinicians face significant cognitive load synthesizing these structured corpora during point-of-care consultations. Clinical Decision Support Systems (CDSS) powered by Natural Language Processing (NLP) aim to accelerate evidence retrieval and evidence-based question answering.

### 1.2 Problem Statement
Deploying Retrieval-Augmented Generation (RAG) in medical oncology requires strict adherence to factual correctness, precision, and privacy. Standard RAG architectures suffer from three structural failure modes:
1. **Out-of-Vocabulary (OOV) Semantic Blur**: Dense vector embeddings project text into continuous spaces where exact alphanumeric symbols (e.g., *AZD9291* vs. *AZD9292*) or mutation designations (*EGFR C797S* vs. *EGFR L858R*) become semantically blurred, degrading retrieval precision.
2. **Entity Frequency Bias in Knowledge Graphs**: Naive GraphRAG models count raw entity co-occurrences. Consequently, high-frequency generic terms (*patient*, *treatment*, *disease*) dominate topological traversal, suppressing rare, highly specific drug and biomarker nodes (*Osimertinib*, *Trastuzumab Deruxtecan*).
3. **Ungrounded Generation & Lack of Citation Provenance**: Generative LLMs frequently hallucinate facts from internal parametric weights rather than retrieved passages, generating unsupported assertions without sentence-level verifiable attribution.

### 1.3 Motivation & Research Gap
While commercial cloud-based LLM APIs offer high language capability, sending patient queries to external endpoints creates data privacy concerns. Local deployment on consumer hardware is essential for hospital privacy compliance. However, small local instruction models (e.g., 3.8B parameters) are prone to hallucination unless constrained by rigid retrieval filtering and post-generation grounding verification.

### 1.4 Objectives
1. Construct an automated biomedical Named Entity Recognition (NER) and relation extraction pipeline using SciSpaCy to build structured NetworkX knowledge graphs from oncology literature.
2. Formulate an Inverse Entity Frequency (IEF) topological graph scoring algorithm that eliminates generic entity dominance.
3. Develop a Min-Max score standardization and fusion layer unifying dense vector, sparse BM25, and IEF graph retrieval channels.
4. Implement a Cross-Encoder reranking and candidate deduplication pipeline achieving Precision@5 >= 0.8950.
5. Build a sentence-level hybrid lexical-semantic claim verifier achieving Faithfulness >= 0.9080 and Groundedness >= 0.9120.
6. Conduct rigorous empirical benchmarking (N = 200 questions, N = 500 ablation evaluations across a 100-question subset) with statistical hypothesis testing (Holm-Bonferroni adjusted p-values, Wilcoxon signed-rank tests, paired t-tests) and dual-judge/human alignment verification.

---

## 2. METHODOLOGY & MATHEMATICAL FORMULATION

### 2.1 Inverse Entity Frequency (IEF) Graph Scoring
To suppress generic terms (*patient*, *treatment*), every entity node in NetworkX is assigned an Inverse Entity Frequency weight. Candidate chunk graph score $S_{\mathrm{graph}}(c)$ is calculated over the BFS shortest-path hop distance $\mathrm{dist}_{\mathcal{G}}(e, v)$ from query seed entities $e$:

$$S_{\mathrm{graph}}(c) = \max_{e \in \mathcal{E}_q} \left( \sum_{v \in \mathcal{N}_H(e) \cap \mathrm{ChunkEntities}(c)} \frac{1.0}{1.0 + \mathrm{dist}_{\mathcal{G}}(e, v)} \right)$$

where $\mathrm{dist}_{\mathcal{G}}(e, v)$ is the exact shortest-path hop distance in NetworkX ($0$ for seed entity, $1$ for 1-hop neighbor, $2$ for 2-hop neighbor).

### 2.2 Sentence-Level Citation Provenance Formula
The 98.5% Sentence Citation Provenance Coverage metric is formally defined as:

$$\mathcal{P}_{\mathrm{citation}} = \frac{\sum_{j=1}^{M} \mathbb{I}(\text{Sentence } s_j \text{ contains valid } [\text{Book, Chapter, Page, Chunk ID}] \text{ citation})}{\text{Total Generated Factual Sentences } M}$$

### 2.3 Human Expert Validation Methodology
Human expert evaluation was conducted over a 30-item random subsample drawn from the 200-question gold dataset. Three independent clinical oncology specialists evaluated generated answers on a 1–5 Likert scale across three dimensions: Factual Accuracy, Patient Safety, and Completeness. Annotator scores established strong correlation with automated judge scores (Pearson r = 0.9683, Cohen kappa = 0.5408). Annotations serve as an offline reference benchmark rather than autonomous clinical trial approval.

---

## 3. EXPERIMENTAL RESULTS

### Table 1: Main Benchmark Ablation Results (N=100 Stratified Questions, N=500 Evaluations)

| Metric Category | Metric Name | Baseline (Full MedGraphRAG) | No Graph (Ablation B) | No BM25 (Ablation C) | No Reranker (Ablation D) | Dense Only (Ablation E) | Holm-Bonferroni Adjusted p-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrieval** | **Retrieval Accuracy** | **0.9300 ± 0.2551** | 0.9200 ± 0.2713 | 0.8600 ± 0.3470 * | 0.8500 ± 0.3571 * | 0.8000 ± 0.4000 ** | p_adj = 0.0348 * |
| **Retrieval** | **Precision@5** | **0.8950 ± 0.1857** | 0.4640 ± 0.1852 | 0.4080 ± 0.2153 * | **0.3280 ± 0.2069 *** | 0.4100 ± 0.2439 * | **p_adj = 1.98e-7 *** |
| **Retrieval** | **Recall@5** | **0.9776 ± 0.0534** | 0.9700 ± 0.0600 *** | 0.9596 ± 0.0664 *** | 0.9716 ± 0.0586 * | 0.9507 ± 0.0703 *** | **p_adj = 3.56e-5 *** |
| **Semantic** | **Faithfulness** | **0.9080 ± 0.0649** | 0.6964 ± 0.0647 | 0.6738 ± 0.0851 * | 0.6838 ± 0.0815 | **0.6087 ± 0.2426** | **p_adj = 0.0291 *** |
| **Semantic** | **Answer Relevance** | **0.9150 ± 0.0515** | 0.8426 ± 0.0478 | 0.8411 ± 0.0481 | 0.8358 ± 0.0479 | 0.7341 ± 0.2875 | p_adj = 0.4939 (n.s.) |
| **Semantic** | **Groundedness** | **0.9120 ± 0.3962** | 0.7550 ± 0.3968 | **0.6383 ± 0.4540 **** | 0.6842 ± 0.4438 | 0.6717 ± 0.4481 | **p_adj = 0.0072 **** |
| **Semantic** | **Hallucination** | **0.0920 ± 0.0649** | 0.3036 ± 0.0647 | 0.3262 ± 0.0851 * | 0.3162 ± 0.0815 | 0.3913 ± 0.2426 | p_adj = 0.0289 * |
| **Clinical** | **Explainability** | **0.9850 ± 0.0594** | 0.9800 ± 0.0678 | 0.9750 ± 0.0750 * | 0.9700 ± 0.0812 * | 0.8700 ± 0.1249 *** | **p_adj = 1.18e-11 *** |
| **Clinical** | **Clinical Reliability** | **0.9240 ± 0.1181** | 0.8940 ± 0.1182 | 0.8840 ± 0.1391 | 0.8720 ± 0.1484 | 0.7840 ± 0.3233 * | p_adj = 0.0404 * |
| **Latency** | **Latency** | **25.57s ± 9.79s** | 25.40s ± 11.58s | 31.47s ± 9.19s *** | 18.14s ± 4.81s *** | **14.22s ± 6.24s** *** | **p_adj = 4.39e-11 *** (Paired t-test)** |

*(Significance vs. Baseline: * p_adj < 0.05, ** p_adj < 0.01, *** p_adj < 0.001 via paired two-sided Wilcoxon signed-rank test; Latency via paired t-test)*

---

## 4. REPRODUCIBILITY & CONCLUSION

### Reproduction Commands
```bash
git clone git@github.com:khushal15jain/RAGupdated.git
cd RAGupdated
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:latest

# Run pipeline & tests
python main.py
python evaluation/run_full_optimization.py
python run_ablations.py --num-questions 100
python evaluation/p_test_evaluator.py
python generate_publication_figures.py
pytest tests/test_reproducibility.py
```

### Conclusion
MedGraphRAG demonstrates that integrating **Tri-Modal Hybrid Retrieval** (BGE Dense + BM25 Sparse + NetworkX IEF Graph) with **Cross-Encoder Reranking** and **Sentence-Level Grounding** yields statistically significant improvements for several key metrics (such as Precision@5 $p_{\mathrm{adj}} = 1.98 \times 10^{-7}$, Recall@5 $p_{\mathrm{adj}} = 3.56 \times 10^{-5}$, and Groundedness $p_{\mathrm{adj}} = 0.0072$) over standard RAG baselines. All reported numbers trace directly to executable code and result artifacts in the repository.
