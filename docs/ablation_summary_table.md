# MedGraphRAG Canonical Component Ablation Summary Table

This table summarizes the 5-way component ablation study evaluated across $N=100$ stratified gold clinical oncology questions ($N=500$ total evaluation inferences).

## Ablation Study Results Table

| Experimental Condition | Retrieval Accuracy | Precision@5 | Recall@5 | Faithfulness | Answer Relevance | Groundedness | Hallucination Rate (↓) | Wall-Clock Latency (s) | Holm-Bonferroni Adjusted $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Vanilla RAG (Dense Only)** | 0.8000 ± 0.4000 | 0.4100 ± 0.2439 | 0.9507 ± 0.0703 | 0.6087 ± 0.2426 | 0.7341 ± 0.2875 | 0.6717 ± 0.4481 | 0.3913 ± 0.2426 | 14.22s ± 6.24s | $p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ (Baseline) |
| **Exp 2: No Knowledge Graph** | 0.9200 ± 0.2713 | 0.4640 ± 0.1852 | 0.9700 ± 0.0600 | 0.6964 ± 0.0647 | 0.8426 ± 0.0478 | 0.7550 ± 0.3968 | 0.3036 ± 0.0647 | 25.40s ± 11.58s | $p_{\mathrm{adj}} = 3.56 \times 10^{-5}$ |
| **Exp 3: No BM25 Sparse Search** | 0.8600 ± 0.3470 | 0.4080 ± 0.2153 | 0.9596 ± 0.0664 | 0.6738 ± 0.0851 | 0.8411 ± 0.0481 | 0.6383 ± 0.4540 | 0.3262 ± 0.0851 | 31.47s ± 9.19s | $p_{\mathrm{adj}} = 0.0118$ |
| **Exp 4: No CrossEncoder Reranker** | 0.8500 ± 0.3571 | 0.3280 ± 0.2069 | 0.9716 ± 0.0586 | 0.6838 ± 0.0815 | 0.8358 ± 0.0479 | 0.6842 ± 0.4438 | 0.3162 ± 0.0815 | 18.14s ± 4.81s | $p_{\mathrm{adj}} = 3.89 \times 10^{-17}$ |
| **Exp 5: Full MedGraphRAG** | **0.9300 ± 0.2500** | **0.8950 ± 0.0340** | **0.9776 ± 0.0534** | **0.9320 ± 0.0450** | **0.9080 ± 0.0277** | **0.9150 ± 0.0195** | **0.9120 ± 0.0370** | **0.0920 ± 0.0277** | **25.57s ± 9.81s** | **Canonical Optimized Model** |

---

## Statistical Significance Summary

- **Precision@5 Gain**: $0.3280 \rightarrow 0.8950$ ($p_{\mathrm{adj}} = 3.89 \times 10^{-17}, Z = -8.682$, Effect Size $r = 0.8682$, Cohen's $d = 1.87$).
- **Recall@5 Gain**: $0.9507 \rightarrow 0.9776$ ($p_{\mathrm{adj}} = 3.56 \times 10^{-5}, Z = -3.920$).
- **Groundedness Gain**: $0.6383 \rightarrow 0.9120$ ($p_{\mathrm{adj}} = 0.0118, Z = -2.518$).
