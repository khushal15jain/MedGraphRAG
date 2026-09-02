# MedGraphRAG: Evaluation Methodology & Metric Specifications

This document outlines the evaluation framework, metric definitions, grounding protocols, and statistical significance testing procedures implemented in **MedGraphRAG**.

---

## 📐 Authoritative Evaluation Engine (`medgraphrag.evaluation`)

All metrics in MedGraphRAG are evaluated through a centralized, closed-form metric engine:
`src/medgraphrag/evaluation/metrics.py`.

### 1. Retrieval & Ranking Metrics ($K=5$)

- **Retrieval Accuracy**:
  $$\text{Retrieval Accuracy} = \mathbb{I}\left( \frac{|C_{\text{top\_5}} \cap C_{\text{gold}}|}{|C_{\text{gold}}|} \ge 0.50 \right)$$
  Evaluates whether top-5 retrieved passages cover at least 50% of ground-truth evidence passages.

- **Precision@5**:
  $$\text{Precision}@5 = \frac{|C_{\text{top\_5}} \cap C_{\text{gold}}|}{5}$$

- **Recall@5**:
  $$\text{Recall}@5 = \frac{|C_{\text{top\_5}} \cap C_{\text{gold}}|}{|C_{\text{gold}}|}$$

- **HitRate@5**:
  $$\text{HitRate}@5 = \mathbb{I}\left( |C_{\text{top\_5}} \cap C_{\text{gold}}| > 0 \right)$$

- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{\text{rank}_{\text{first\_gold\_match}}}$$

- **NDCG@5**:
  $$\text{NDCG}@5 = \frac{\text{DCG}@5}{\text{IDCG}@5}$$

---

### 2. Semantic, NLI & Safety Metrics

- **Faithfulness**:
  $$\text{Faithfulness} = \frac{N_{\text{supported\_sentences}}}{N_{\text{total\_sentences}}}$$
  Sentence-level NLI verification using `DeBERTa-v3` / entailment status.

- **Answer Relevance**:
  $$\text{Answer Relevance} = \frac{1}{N_{\text{sentences}}} \sum_{i=1}^{N_{\text{sentences}}} \cos\left( \mathbf{e}(s_i), \mathbf{e}(q) \right)$$
  Cosine similarity between sentence embeddings (`bge-base-en-v1.5`) and query embedding.

- **Groundedness**:
  $$\text{Groundedness} = \frac{|\{s_i \mid (0.4 \cdot \text{Jaccard} + 0.6 \cdot \text{CosSim}) \ge 0.65\}|}{N_{\text{total\_sentences}}}$$
  Sentence-level grounding thresholding ($\tau_g = 0.65$).

- **Hallucination Rate**:
  $$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$$
  Derived inverse metric representing unsupported sentence fraction.

- **Explainability ($\mathcal{P}_{\text{cit}}$)**:
  $$\text{Explainability} = \frac{N_{\text{sentences\_with\_valid\_[Pn]\_citation}}}{N_{\text{total\_sentences}}}$$

- **Clinical Reliability**:
  $$\text{Clinical Reliability} = \frac{0.4 \cdot \text{Accuracy}_{\text{judge}} + 0.4 \cdot \text{Safety}_{\text{judge}} + 0.2 \cdot \text{Completeness}_{\text{judge}} - 1.0}{4.0}$$
  LLM-as-a-Judge 5-point Likert scale evaluation normalized to $[0.0, 1.0]$.

---

### 3. Lexical NLP Metrics

- **SQuAD Answer F1**: Token-level precision and recall harmonic mean post JSON-stripping.
- **BLEU-1 / BLEU-2 / BLEU-4**: NLTK `sentence_bleu` with `SmoothingFunction().method1`.
- **ROUGE-1 / ROUGE-2 / ROUGE-L**: Dynamic Programming LCS F1 score ($\beta = 1.0$).
- **METEOR**: Unigram precision/recall alignment with fragmentation chunk penalty.

---

## 📊 Statistical Significance & ID Alignment Protocol

1. **Paired Observations**: All statistical tests match baseline and ablation records strictly by `question_id`. Positional array index pairing is prohibited.
2. **Wilcoxon Signed-Rank Test**: Applied to all non-parametric metrics.
3. **Paired $t$-Test**: Applied to Operational Latency.
4. **Holm-Bonferroni Step-Down Adjustment**: Multiplicity correction applied across the family of statistical hypotheses:
   $$p_{(k)} = \min\left(1.0, p_{\text{raw}} \cdot (m - k + 1)\right)$$
5. **Effect Size**: $r = \frac{|z|}{\sqrt{N}}$.
6. **Confidence Intervals**: $95\%$ non-parametric percentile bootstrap CIs ($1,000$ resamples, `seed=42`).
