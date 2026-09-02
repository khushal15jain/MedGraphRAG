# MedGraphRAG: Evaluation & Scientific Validation Protocol

This document specifies the closed-form metric definitions, dataset sampling protocols, judge agreement setup, and statistical significance testing methodology for **MedGraphRAG**.

---

## 📊 Dataset Evaluation Settings & Sample Sizes

To maintain statistical rigor while balancing computational throughput during ablation studies, evaluation is partitioned into two distinct settings:

1. **Main Benchmark Evaluation ($N=200$)**:
   - Primary evaluation across the full gold QA dataset ($N=200$ complex medical oncology queries) derived from NCCN guidelines, ESMO handbooks, and FDA labeling.
   - Evaluates full MedGraphRAG performance on all 12 closed-form and LLM-as-a-Judge metrics.

2. **Stratified Ablation Study Evaluation ($N=100$)**:
   - System ablation study evaluating 5 system configurations (MedGraphRAG, No Graph, No BM25, No Reranker, Dense Only).
   - Uses a 100-question stratified random subset (`seed=42`, documented in `results/ablation_question_ids.json`).
   - Total inference runs: $100 \text{ questions} \times 5 \text{ configurations} = 500 \text{ evaluations}$.

---

## 🎯 Ground-Truth Retrieval Metric Verification

Retrieval metrics assess top-$K$ ($K=5$) passage selection against ground-truth evidence chunk IDs ($C_{\text{gold}}$) manually curated in `data/qa_dataset.json`.

1. **Precision@5**:
   $$\text{Precision}@5 = \frac{|C_{\text{retrieved\_5}} \cap C_{\text{gold}}|}{5}$$

2. **Recall@5**:
   $$\text{Recall}@5 = \frac{|C_{\text{retrieved\_5}} \cap C_{\text{gold}}|}{|C_{\text{gold}}|}$$

3. **HitRate@5**:
   $$\text{HitRate}@5 = \mathbb{I}\left(|C_{\text{retrieved\_5}} \cap C_{\text{gold}}| > 0\right)$$

4. **Retrieval Accuracy**:
   $$\text{Retrieval Accuracy} = \mathbb{I}\left(\frac{|C_{\text{retrieved\_5}} \cap C_{\text{gold}}|}{|C_{\text{gold}}|} \ge 0.50\right)$$

---

## 🛡 NLI Grounding & Hallucination Protocol

1. **Faithfulness**:
   Ratio of generated answer sentences $s_i$ supported by retrieved context $P$ using `DeBERTa-v3` NLI entailment ($\tau_g = 0.65$).

2. **Hallucination Rate**:
   Derived directly as the exact sentence-level inverse of Faithfulness:
   $$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$$
   *Note: Hallucination Rate is not an independently measured metric.*

---

## ⚖️ Inter-Judge Agreement & Human Validation Status

- **Human Annotation Status**: Human expert clinical validation was not conducted for this benchmark release. No synthetic or simulated human scores are generated or claimed.
- **LLM-as-a-Judge Consensus**: Evaluation relies on multi-judge consensus between local evaluators (`Qwen2.5-3B-Instruct`, `Llama-3.2-3B`) and strong evaluators.

---

## 📈 Statistical Significance & Hypothesis Testing

1. **Question ID Pairing**:
   Observations across system pairs (e.g., MedGraphRAG vs No Graph) are matched strictly by unique `question_id` using `align_by_question_id()`. Positional array matching is forbidden.

2. **Hypothesis Tests**:
   - Two-sided paired Wilcoxon signed-rank test for non-parametric metrics (Accuracy, Precision, Recall, Faithfulness, Groundedness).
   - Paired $t$-test for operational latency.

3. **Multiplicity Correction**:
   Step-down **Holm-Bonferroni correction** applied across all raw $p$-values to control Family-Wise Error Rate ($\alpha = 0.05$).

4. **Confidence Intervals**:
   $95\%$ non-parametric percentile bootstrap confidence intervals ($1,000$ iterations, `seed=42`).
