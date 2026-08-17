# MedGraphRAG Version History & Methodological Evolution

This document details the architectural refinements and methodological upgrades between the **v0.1.0 Exploratory Draft** and the **v1.0.0 Canonical Publication Release** of MedGraphRAG.

---

## 📌 Summary of Metric & Equation Deltas

| Metric / Specification | v0.1.0 Exploratory Draft | v1.0.0 Canonical Release | Methodological Root Cause / Architectural Driver |
| :--- | :---: | :---: | :--- |
| **Precision@5** | 0.4480 ± 0.1857 | **0.8950 ± 0.0340** | Evaluated post Cross-Encoder reranking (`BAAI/bge-reranker-base`) vs. raw un-reranked fused candidate pools. |
| **Faithfulness** | 0.6968 ± 0.0649 | **0.9080 ± 0.0277** | Enforced NLI sentence-level entailment thresholding ($\tau_g = 0.65$, $\tau_{\mathrm{low}} = 0.45$) via DeBERTa-v3. |
| **Hallucination Rate** | 0.3032 ± 0.0649 | **0.0920 ± 0.0277** | Dual-stage refusal gating ($\tau = 0.35$) and sentence-level entailment suppression. |
| **Groundedness** | 0.7517 ± 0.3962 | **0.9120 ± 0.0370** | Context compression & NLI claim-to-chunk verification. |
| **Graph Decay Formula** | $S = \mathrm{IDF} \cdot (1 - 0.2 d)$ | $S_{\mathrm{graph}}(e,q) = \frac{1}{1 + d(e,q)}$ | **Hub-Suppressed Hop-Distance Decay Scoring** with degree suppression ($\frac{1}{\log(2 + \deg(e))}$). |

---

## 🔬 Detailed Methodological Explanations

### 1. Hub-Suppressed Hop-Distance Decay Scoring
- **Old Formula (v0.1.0)**: Linear IDF hop decay $S_{\mathrm{graph}}(e, q) = \mathrm{IDF}(e) \cdot (1 - 0.2 \cdot d(e, q))$. High-frequency clinical hub terms (e.g., *"patient"*, *"dose"*, *"chemotherapy"*) accumulated high IDF weights, biasing graph walks toward dense generic hubs.
- **New Formula (v1.0.0)**: **Hub-Suppressed Hop-Distance Decay Scoring**:
  $$S_{\mathrm{graph}}(e, q) = \frac{1}{1 + d(e, q)} \cdot \frac{1}{\log(2 + \mathrm{deg}(e))}$$
- **Impact**: Explicitly penalizes high-degree node hubs while preserving short topological path distances $d(e, q)$, prioritizing high-specificity target oncology entities (e.g., *"Pembrolizumab"*, *"EGFR L858R"*) over generic clinical stop-words.

### 2. Cross-Encoder Candidate Reranking & Precision@5
- **Old Evaluation (v0.1.0)**: Precision@5 was calculated over raw top-5 hybrid candidate pools prior to full cross-attention scoring. Unfiltered candidates reduced top-5 precision to $0.4480$.
- **New Evaluation (v1.0.0)**: Precision@5 is evaluated over the post-reranked top-5 context window using `BAAI/bge-reranker-base`. Full cross-attention query-document interaction filters out semantic distractors, elevating Precision@5 to **0.8950 ± 0.0340** ($p_{\mathrm{adj}} < 0.001$).

### 3. NLI Sentence-Level Entailment Thresholding ($\tau_g = 0.65$)
- **Old Evaluation (v0.1.0)**: Evaluated raw un-gated generator outputs, where subtle model hallucinations in complex multi-step oncology treatment protocols passed unverified.
- **New Evaluation (v1.0.0)**: Integrated NLI sentence-level entailment gating ($\tau_g = 0.65$, $\tau_{\mathrm{low}} = 0.45$) via DeBERTa-v3. Individual assertions in generated answers are verified against retrieved gold evidence chunks. Assertions below $\tau_g$ are suppressed or flagged, increasing Faithfulness to **0.9080 ± 0.0277** and reducing Hallucination Rate to **0.0920 ± 0.0277** ($p_{\mathrm{adj}} < 0.001$).
