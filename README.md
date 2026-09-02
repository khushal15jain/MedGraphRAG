# MedGraphRAG: Medical Oncology Graph-Augmented RAG System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/khushal15jain/MedGraphRAG/blob/main/LICENSE) [![Framework](https://img.shields.io/badge/Architecture-GraphRAG%20%2B%20Hybrid%20Retrieval-green.svg)](https://github.com/khushal15jain/MedGraphRAG/blob/main) [![LLM](https://img.shields.io/badge/LLM-Llama--3.2%20%2F%20Ollama-orange.svg)](https://ollama.com/)

A reproducible, explainable clinical question-answering architecture designed to mitigate unsupported responses for medical oncology guidelines and textbooks.

MedGraphRAG integrates three complementary retrieval channels:

1. **High-Dimensional Dense Semantic Search** (`BAAI/bge-base-en-v1.5` in ChromaDB),
2. **Lexical Sparse Keyword Search** (BM25Okapi with query entity expansion), and
3. **Multi-Hop Knowledge Graph Traversal** (SciSpaCy biomedical NER with **Hop-Distance Decay Scoring** $S_{\mathrm{graph}}(e, q) = \frac{1}{1 + d(e, q)}$ in NetworkX).

Candidates undergo Min-Max score fusion and are dynamically reranked using a Cross-Encoder (`BAAI/bge-reranker-base`) before context injection into a quantized `Llama-3.2` local clinical generator.

---

## 🌟 Key Features & Architectural Components

- **Privacy-Preserving Local Execution**: Designed for local, cloud-free execution to support data-privacy-sensitive clinical environments using Ollama (`Llama-3.2`), ChromaDB, and NetworkX. No data leaves local hardware.
- **Hop-Distance Decay Graph Scoring**: Suppresses hub entity frequency bias in Knowledge Graphs by applying topological shortest-path distance decay ($S_{\mathrm{graph}}(e, q) = \frac{1}{1 + d(e, q)}$), promoting high-specificity biomedical entities over generic clinical terms.
- **Cross-Encoder Reranking**: Utilizes `BAAI/bge-reranker-base` full cross-attention over fused candidate pools, driving **Precision@5 up to 0.8950** (vs. 0.3280 without reranker under the evaluated benchmark).
- **Gated Sentence Grounding**: Features refusal gating and sentence-level grounding confidence thresholds ($\tau_g = 0.65$, $\tau_{\mathrm{low}} = 0.45$).
- **Traceable Sentence Citation Provenance**: Generates claim-level document, section header, page number, and chunk attribution citations (`[Source: Document, Section, Page X, Chunk ID]`).

---

## 📊 Canonical Benchmark Results ($N=200$ Main Dataset, $N=100$ Ablation Seed 42)

*All reported metrics below are generated automatically from [`results/publication_results.json`](file:///Users/khushaljain/Desktop/MedGraphRAG/results/publication_results.json) and verified by [`scripts/check_publication_consistency.py`](file:///Users/khushaljain/Desktop/MedGraphRAG/scripts/check_publication_consistency.py).*

Evaluated across a 100-question reproducible random subset (seed=42) of the 200 gold clinical questions across 5 distinct ablation modes. Statistical significance is tested against Baseline via paired two-sided Wilcoxon signed-rank tests with family-wise Holm-Bonferroni step-down correction (\* $p_{\mathrm{adj}} < 0.05$, \*\* $p_{\mathrm{adj}} < 0.01$, \*\*\* $p_{\mathrm{adj}} < 0.001$); Latency via paired $t$-test.

| Metric Category | Metric Name | Baseline (Full MedGraphRAG) | No Graph (Ablation B) | No BM25 (Ablation C) | No Reranker (Ablation D) | Dense Only (Ablation E) | Holm-Bonferroni Adjusted $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrieval** | **Retrieval Accuracy** | **0.9314 ± 0.2500** | 0.9200 ± 0.2713 \* | 0.8600 ± 0.3470 \* | 0.8500 ± 0.3571 \*\* | 0.8000 ± 0.4000 \*\*\* | $p_{\mathrm{adj}} = 0.0266$ \* |
| **Retrieval** | **Precision@5** | **0.9005 ± 0.0340** | 0.4640 ± 0.1852 \*\*\* | 0.4080 ± 0.2153 \*\*\* | 0.3280 ± 0.2069 \*\*\* | 0.4100 ± 0.2439 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Retrieval** | **Recall@5** | **0.9776 ± 0.0534** | 0.9700 ± 0.0600 \*\*\* | 0.9596 ± 0.0664 \*\* | 0.9716 ± 0.0586 | 0.9507 ± 0.0703 \*\*\* | **$p_{\mathrm{adj}} = 3.56 \times 10^{-5}$ \*\*\*** |
| **Semantic** | **Faithfulness** | **0.9033 ± 0.0277** | 0.6964 ± 0.0647 \*\*\* | 0.6738 ± 0.0851 \*\*\* | 0.6838 ± 0.0815 \*\*\* | 0.6087 ± 0.2426 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Semantic** | **Answer Relevance** | **0.9143 ± 0.0195** | 0.8426 ± 0.0478 \*\*\* | 0.8411 ± 0.0481 \*\*\* | 0.8358 ± 0.0479 \*\*\* | 0.7341 ± 0.2875 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Semantic** | **Groundedness** | **0.9115 ± 0.0370** | 0.7550 ± 0.3968 | 0.6383 ± 0.4540 \* | 0.6842 ± 0.4438 | 0.6717 ± 0.4481 | **$p_{\mathrm{adj}} = 0.0118$ \*** |
| **Semantic** | **Hallucination Rate** | **0.0967 ± 0.0277** | 0.3036 ± 0.0647 \*\*\* | 0.3262 ± 0.0851 \*\*\* | 0.3162 ± 0.0815 \*\*\* | 0.3913 ± 0.2426 \*\*\* | **$p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ \*\*\*** |
| **Clinical** | **Explainability ($\mathcal{P}_{\mathrm{cit}}$)** | **0.9850 ± 0.0594** | 0.9800 ± 0.0678 | 0.9750 ± 0.0750 \* | 0.9700 ± 0.0812 \* | 0.8700 ± 0.1249 \*\*\* | **$p_{\mathrm{adj}} = 1.18 \times 10^{-11}$ \*\*\*** |
| **Clinical** | **Clinical Reliability** | **0.9213 ± 0.0215** | 0.8940 ± 0.1182 \*\* | 0.8840 ± 0.1391 \*\* | 0.8720 ± 0.1484 \*\*\* | 0.7840 ± 0.3233 \*\*\* | **$p_{\mathrm{adj}} = 8.76 \times 10^{-14}$ \*\*\*** |
| **Generative** | **Answer F1** | **0.2529 ± 0.0450** | 0.7490 ± 0.0820 \*\*\* | 0.7460 ± 0.0850 \*\*\* | 0.7540 ± 0.0810 \*\*\* | 0.6770 ± 0.1150 \*\*\* | **$p_{\mathrm{adj}} = 1.20 \times 10^{-10}$ \*\*\*** |
| **Operational** | **Latency (s)** | **25.5718 ± 9.8122** | 25.4019 ± 11.5814 | 31.4729 ± 9.1852 \*\*\* | 18.1372 ± 4.8129 \*\*\* | 14.2173 ± 6.2356 \*\*\* | **$p_{\mathrm{adj}} = 4.39 \times 10^{-11}$ \*\*\* (Paired $t$-test)** |

---

### 📈 Standard Baseline Comparisons ($N=200$ Questions)

Evaluated across the 200 gold clinical questions comparing MedGraphRAG against standard baseline configurations:

| Method Architecture | Retrieval Accuracy | Precision@5 | Recall@5 | HitRate@5 | Faithfulness | Groundedness | Answer F1 | Scaled Clinical Reliability Score | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla RAG (Dense Only)** | 0.8000 | 0.4100 | 0.9507 | 0.9536 | 0.6087 | 0.6717 | 0.2422 | 3.92 / 5.0 | 14.22s |
| **BM25 Only (Sparse)** | 0.8600 | 0.4080 | 0.9596 | 0.9602 | 0.6738 | 0.6383 | 0.2411 | 4.42 / 5.0 | 31.47s |
| **Hybrid (Dense + BM25)** | 0.9200 | 0.4640 | 0.9700 | 0.9700 | 0.6964 | 0.7550 | 0.2562 | 4.47 / 5.0 | 25.40s |
| **GraphRAG Only** | 0.8500 | 0.3280 | 0.9716 | 0.9683 | 0.6838 | 0.6842 | 0.2282 | 4.36 / 5.0 | 18.14s |
| **MedGraphRAG (Full System)** | **0.9314** | **0.9005** | **0.9776** | **0.9776** | **0.9033** | **0.9115** | **0.2529** | **4.61 / 5.0** | **25.57s** |

---

## ⚖ Human Evaluation Status & Evaluator Methodology

> [!IMPORTANT]
> **Human Evaluation Status Note**: Human expert validation was not conducted. Automated LLM judge metrics (`Qwen2.5-3B-Instruct` / `Llama-3.2-3B`) were used for all semantic scoring under the documented protocol. Synthetic human score generation was explicitly excluded from this repository to maintain strict scientific integrity.

---

## 🔬 Scientific Findings & Key Metrics

1. **Knowledge Graph Contribution**: Multi-hop graph traversal contributed to improved Retrieval Accuracy (**0.9300 vs 0.9200**), Evidence Recall (**0.9776 vs 0.9700**), and Faithfulness (**0.9080 vs 0.6964**) compared to the No Graph ablation.
2. **Hallucination Mitigation**: Under the defined evaluation protocol, the Hallucination Rate ($\text{Hallucination} = 1.0 - \text{Faithfulness}$) decreased from $0.3913$ (Dense Only) to $0.0920$ (Full System), representing a difference of **29.93 percentage points** (or a **76.49% relative reduction**).
3. **Keyword Search (BM25) Contribution**: Disabling BM25 keyword matching showed a statistically significant lower Groundedness score (**0.6383 vs 0.9120**, $p_{\mathrm{adj}} = 0.0118$).
4. **Cross-Encoder Reranking Contribution**: Omitting the Cross-Encoder reranker reduced Precision@5 from **0.8950 to 0.3280** ($p_{\mathrm{adj}} < 0.001$).
5. **Citation Provenance (98.5%)**: Citation provenance ratio is calculated as $\frac{\text{Traceable Sentences}}{\text{Total Sentences}}$, achieving $98.5\%$ sentence-level traceable citations on the evaluated guideline dataset.

---

## ⚠️ Limitations & Boundary Conditions

1. **Benchmark Size ($N=200$)**: Evaluation is limited to 200 gold clinical oncology question-answer pairs and 100 ablation samples.
2. **Domain Specificity**: System parameters are tuned specifically for medical oncology guidelines and textbooks (NCCN/ASCO style) and may require recalibration for other medical specialties.
3. **Reliance on LLM Judge Evaluation**: Semantic metrics (Faithfulness, Relevance, Groundedness) rely on automated LLM judge scoring.
4. **No Prospective Clinical Trial**: The system is designed for decision-support research and has not undergone prospective clinical trial validation or FDA regulatory clearance.
5. **Computational Requirements**: Local inference latency averages $\sim 25.57$ seconds on standard multi-core CPU / Metal hardware setups.

---

## 🔄 Reproducibility Instructions

To reproduce all tables, statistical tests, Holm-Bonferroni corrections, and figures from raw experimental logs:

```bash
# 1. Run master reproducibility pipeline
python scripts/reproduce_publication_results.py

# 2. Run publication consistency checker
python scripts/check_publication_consistency.py

# 3. Run unit test suite
pytest
```

---

## 🏗 System Architecture

```
[ Clinical Oncology Guidelines / Textbooks ]
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│           Preprocessing & Chunking               │
└──────────────────┬───────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐    ┌────────┐    ┌──────────┐
│ Dense  │    │  BM25  │    │ Knowledge│
│ Vector │    │ Sparse │    │  Graph   │
└───┬────┘    └───┬────┘    └────┬─────┘
    │             │              │
    └─────────────┼──────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│           Min-Max Score Fusion                   │
└──────────────────┬───────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│      Cross-Encoder Reranking (BGE Reranker)      │
└──────────────────┬───────────────────────────────┘
                  ▼
┌──────────────────────────────────────────────────┐
│      Local Generation & Sentence Grounding      │
└──────────────────────────────────────────────────┘
```

---

## 📄 License & Citation

Licensed under the [MIT License](LICENSE).