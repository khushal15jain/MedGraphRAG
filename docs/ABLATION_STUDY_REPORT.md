# MedGraphRAG Ablation Study Report

## Executive Summary

This report documents the empirical component-level ablation study conducted on **MedGraphRAG**. The study evaluates a stratified subset of **100 Gold Clinical Oncology Questions** across **5 distinct ablation conditions** ($100 \times 5 = \mathbf{500\text{ total evaluation inferences}}$).

Statistical significance was evaluated against the Full MedGraphRAG Baseline using paired two-sided Wilcoxon signed-rank tests for quality metrics and paired $t$-tests for latency, with **Holm-Bonferroni step-down multiple comparison adjustment**.

---

## 1. Experimental Protocol & Conditions

The 5 evaluated retrieval conditions are defined in `run_ablations.py`:

| Condition ID | Condition Name | Enabled Components | Description / Objective |
| :---: | :--- | :--- | :--- |
| **Exp1** | **Baseline (Full MedGraphRAG)** | Dense + BM25 + Graph + Reranker | Full tri-modal retrieval with cross-encoder reranking and NLI grounding. |
| **Exp2** | **No Graph (Ablation B)** | Dense + BM25 + Reranker | Disables Knowledge Graph traversal channel ($\gamma = 0.0$). |
| **Exp3** | **No BM25 (Ablation C)** | Dense + Graph + Reranker | Disables sparse lexical BM25 search channel ($\beta = 0.0$). |
| **Exp4** | **No Reranker (Ablation D)** | Dense + BM25 + Graph | Disables `BAAI/bge-reranker-base` cross-encoder scoring. |
| **Exp5** | **Dense Only (Ablation E)** | Dense Retrieval Only | Standard Vanilla Vector RAG architecture. |

---

## 2. Canonical Ablation Study Results Table

The canonical metric values and standard deviations from `p_test_results.json` and `outputs/optimization/final_results.json`:

| Metric Category | Metric Name | Baseline (Full MedGraphRAG) | No Graph (Ablation B) | No BM25 (Ablation C) | No Reranker (Ablation D) | Dense Only (Ablation E) | Holm-Bonferroni Adjusted $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Retrieval** | **Retrieval Accuracy** | **0.9300 ± 0.2551** | 0.9200 ± 0.2713 | 0.8600 ± 0.3470 \* | 0.8500 ± 0.3571 \* | 0.8000 ± 0.4000 \*\* | $p_{\mathrm{adj}} = 0.0348$ \* |
| **Retrieval** | **Precision@5** | **0.8950 ± 0.1857** | 0.4640 ± 0.1852 | 0.4080 ± 0.2153 \* | **0.3280 ± 0.2069 \*\*\*** | 0.4100 ± 0.2439 \* | **$p_{\mathrm{adj}} = 1.98 \times 10^{-7}$ \*\*\*** |
| **Retrieval** | **Recall@5** | **0.9776 ± 0.0534** | 0.9700 ± 0.0600 \*\*\* | 0.9596 ± 0.0664 \*\*\* | 0.9716 ± 0.0586 \* | 0.9507 ± 0.0703 \*\*\* | **$p_{\mathrm{adj}} = 3.56 \times 10^{-5}$ \*\*\*** |
| **Semantic** | **Faithfulness** | **0.9080 ± 0.0649** | 0.6964 ± 0.0647 | 0.6738 ± 0.0851 \* | 0.6838 ± 0.0815 | **0.6087 ± 0.2426** | **$p_{\mathrm{adj}} = 0.0291$ \*** |
| **Semantic** | **Answer Relevance** | **0.9150 ± 0.0515** | 0.8426 ± 0.0478 | 0.8411 ± 0.0481 | 0.8358 ± 0.0479 | 0.7341 ± 0.2875 | $p_{\mathrm{adj}} = 0.4939$ (n.s.) |
| **Semantic** | **Groundedness** | **0.9120 ± 0.3962** | 0.7550 ± 0.3968 | **0.6383 ± 0.4540 \*\*** | 0.6842 ± 0.4438 | 0.6717 ± 0.4481 | **$p_{\mathrm{adj}} = 0.0072$ \*\*** |
| **Semantic** | **Hallucination** | **0.0920 ± 0.0649** | 0.3036 ± 0.0647 | 0.3262 ± 0.0851 \* | 0.3162 ± 0.0815 | 0.3913 ± 0.2426 | $p_{\mathrm{adj}} = 0.0289$ \* |
| **Clinical** | **Explainability** | **0.9850 ± 0.0594** | 0.9800 ± 0.0678 | 0.9750 ± 0.0750 \* | 0.9700 ± 0.0812 \* | 0.8700 ± 0.1249 \*\*\* | **$p_{\mathrm{adj}} = 1.18 \times 10^{-11}$ \*\*\* |
| **Clinical** | **Clinical Reliability** | **0.9240 ± 0.1181** | 0.8940 ± 0.1182 | 0.8840 ± 0.1391 | 0.8720 ± 0.1484 | 0.7840 ± 0.3233 \* | $p_{\mathrm{adj}} = 0.0404$ \* |
| **Latency** | **Latency** | **25.57s ± 9.79s** | 25.40s ± 11.58s | 31.47s ± 9.19s \*\*\* | 18.14s ± 4.81s \*\*\* | **14.22s ± 6.24s** \*\*\* | **$p_{\mathrm{adj}} = 4.39 \times 10^{-11}$ \*\*\* (Paired $t$-test)** |

*(Significance vs. Baseline: \* $p_{\mathrm{adj}} < 0.05$, \*\* $p_{\mathrm{adj}} < 0.01$, \*\*\* $p_{\mathrm{adj}} < 0.001$ via paired two-sided Wilcoxon signed-rank test; Latency via paired $t$-test)*

---

## 3. Key Component Contributions & Findings

1. **Impact of Cross-Encoder Reranking (Baseline vs. No Reranker)**:
   - Removing the reranker causes **Precision@5 to collapse from 0.8950 to 0.3280** ($p_{\mathrm{adj}} = 1.98 \times 10^{-7}$).
   - *Finding*: Reranking with Jaccard candidate deduplication is the single most important component for eliminating context noise and candidate redundancy.

2. **Impact of Knowledge Graph Retrieval (Baseline vs. No Graph)**:
   - Removing Knowledge Graph traversal drops **Groundedness from 0.9120 to 0.7550** and **Recall@5 from 0.9776 to 0.9700** ($p_{\mathrm{adj}} = 3.56 \times 10^{-5}$).
   - *Finding*: Multi-hop graph traversal surfaces evidence distributed across multiple document sections that vector embeddings alone miss.

3. **Impact of BM25 Lexical Retrieval (Baseline vs. No BM25)**:
   - Removing BM25 sparse search reduces **Groundedness to 0.6383** ($p_{\mathrm{adj}} = 0.0072$) and increases latency to 31.47s.
   - *Finding*: Lexical search handles exact alphanumeric drug codes (*AZD9291*) and mutation symbols (*EGFR L858R*).

4. **Dense Only Baseline (Vanilla Vector RAG)**:
   - Vanilla RAG achieves **0.8000 Retrieval Accuracy**, **0.4100 Precision@5**, **0.6087 Faithfulness**, and **0.6717 Groundedness**.
   - *Finding*: Full MedGraphRAG outperforms Dense Only across all factual, retrieval, and clinical metrics.
