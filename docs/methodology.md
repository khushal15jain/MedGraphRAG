# MedGraphRAG: Scientific Methodology

This document presents the authoritative mathematical formulations, scoring functions, and module mappings of **MedGraphRAG**.

---

## 🧮 1. Multi-Hop Graph Traversal & Scoring

For a query $q$ with extracted entity set $E_q$, graph traversal operates over a NetworkX biomedical entity graph built from SciSpaCy NER extractions.

For each discovered entity node $e$ at BFS shortest-path distance $d(e, q)$ from a query seed entity, the topological decay score is:

$$\text{score}_{\text{graph}}(e, q) = \frac{1}{1 + d(e, q)}$$

- **Seed node ($d = 0$)**: $\text{score} = 1.0$
- **1-hop neighbor ($d = 1$)**: $\text{score} = 0.5$
- **2-hop neighbor ($d = 2$)**: $\text{score} = 0.333$
- **Hops $> 2$**: Attenuated to $0$.

A chunk $c$ referenced by multiple entity nodes receives the supremum (maximum) score across discovered nodes:

$$S_{\text{graph}}(c, q) = \max_{e \in E_c} \left( \frac{1}{1 + d(e, q)} \right)$$

This pure topological decay function eliminates corpus mention frequency bias so common generic terms do not dominate rare, clinically significant entities.

---

## ⚖️ 2. Tri-Modal Hybrid Retrieval Fusion

Candidate chunks retrieved from dense semantic search ($S_{\text{dense}}$), sparse lexical BM25 search ($S_{\text{bm25}}$), and multi-hop graph traversal ($S_{\text{graph}}$) are standardized using min-max normalization into $[0, 1]$:

$$\hat{S}_{\text{channel}}(c) = \frac{S_{\text{channel}}(c) - \min(S_{\text{channel}})}{\max(S_{\text{channel}}) - \min(S_{\text{channel}}) + \epsilon}$$

The standardized scores are combined via a linear weighted sum:

$$S_{\text{hybrid}}(c, q) = w'_d \hat{S}_{\text{dense}}(c, q) + w'_b \hat{S}_{\text{bm25}}(c, q) + w'_g \hat{S}_{\text{graph}}(c, q)$$

### Base Channel Weights & Active Channel Normalization
The base weights configured in `configs/retrieval.yaml` and `configs/optimized_retrieval.yaml` are:
- $w_d = 0.40$ (Dense BGE-base-en-v1.5)
- $w_b = 0.30$ (BM25Okapi with query entity expansion)
- $w_g = 0.30$ (Multi-Hop Knowledge Graph)

In ablation conditions where one or more channels are toggled off (e.g., No Graph, No BM25, or Dense Only), weights are dynamically renormalized across active channels:

$$w'_i = \frac{w_i}{\sum_{j \in \text{active}} w_j}$$

---

## 🎯 3. Cross-Encoder Reranking & Quality Penalties

Top candidate chunks from hybrid retrieval are scored by a cross-encoder (`BAAI/bge-reranker-base`):

$$\text{score}_{\text{CE}}(q, c) = \text{CrossEncoder}_\phi([q \mathbin{\Vert} c])$$

Before final sorting, two quality penalty adjustments are applied to prevent degenerate snippets and off-topic fragments from entering the generator prompt:

$$\text{score}_{\text{final}}(q, c) = \text{score}_{\text{CE}}(q, c) - 1.5 \cdot \mathbb{I}(\text{len}_{\text{words}}(c) < 15) - 0.8 \cdot \mathbb{I}(\text{tokens}(q) \cap \text{tokens}(c) = \emptyset)$$

- **Low-information penalty ($-1.5$)**: Penalizes fragments with fewer than $W_{\min} = 15$ words.
- **Query keyword mismatch penalty ($-0.8$)**: Penalizes chunks that share zero non-stopword tokens with the user query.
- **Deduplication filtering**: Candidates are sequentially accepted into the top-$k$ pool only if their Jaccard token similarity with all previously accepted candidates does not exceed $\theta_{\text{dedup}} = 0.65$:
  $$J(c, c_{\text{selected}}) \le 0.65$$

---

## 🛡 4. Grounding & Confidence Classification

Generated answers undergo sentence-level decomposition and verification against the top-ranked retrieved evidence passages using `generator/sentence_grounder.py`:

$$\text{conf}(s) = 0.40 \cdot \text{overlap}_{\text{lexical}}(s, p^*(s)) + 0.60 \cdot \cos(\mathbf{e}_s, \mathbf{e}_{p^*(s)})$$

where $p^*(s)$ is the best-matching evidence passage for sentence $s$. Grounding status is classified as:

$$\text{status}(s) = \begin{cases}
\text{grounded}, & \text{conf}(s) \ge \tau_g \quad (0.65) \\
\text{low\_confidence}, & \tau_l \le \text{conf}(s) < \tau_g \quad (0.45 \le \text{conf} < 0.65) \\
\text{unsupported}, & \text{conf}(s) < \tau_l \quad (0.45)
\end{cases}$$

- **Faithfulness**: Mean sentence-level confidence score: $\frac{1}{|S|} \sum_{s \in S} \text{conf}(s)$.
- **Groundedness**: Ratio of grounded sentences: $\frac{|\{s \in S \mid \text{status}(s) = \text{grounded}\}|}{|S|}$.
- **Hallucination Rate**: Evaluated directly as: $\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$.

---

## 🏥 5. Clinical Reliability Formulation

Clinical Reliability is computed deterministically in code via the formal weighted linear combination:

$$\text{Clinical Reliability} = 0.30 \cdot \text{Faithfulness} + 0.30 \cdot \text{Groundedness} + 0.20 \cdot (1.0 - \text{Hallucination}) + 0.10 \cdot \text{Safety} + 0.10 \cdot \text{Completeness}$$

where Safety and Completeness are evaluated against medical criteria in $[0, 1]$.

---

## 📚 6. Authoritative Metric Sources of Truth

| Metric Name | Authoritative Python Module | Function / Class | Status |
| :--- | :--- | :--- | :--- |
| **Precision@5** | `evaluation/metrics.py` | `precision_at_k(retrieved, gold, k=5)` | **Authoritative (Ground-Truth Chunks)** |
| **Recall@5** | `evaluation/metrics.py` | `recall_at_k(retrieved, gold, k=5)` | **Authoritative (Ground-Truth Chunks)** |
| **HitRate@5** | `evaluation/metrics.py` | `hit_rate_at_k(retrieved, gold, k=5)` | **Authoritative (Ground-Truth Chunks)** |
| **MRR** | `evaluation/metrics.py` | `compute_mrr(retrieved, gold)` | **Authoritative** |
| **NDCG@5** | `evaluation/metrics.py` | `compute_ndcg(retrieved, gold, k=5)` | **Authoritative** |
| **Retrieval Accuracy** | `evaluation/metrics.py` | `hit_rate_at_k(retrieved, gold, k=5)` | **Authoritative** |
| **Faithfulness** | `generator/sentence_grounder.py` | `SentenceLevelGrounder.check()` | **Authoritative (Sentence NLI/Cosine)** |
| **Groundedness** | `generator/sentence_grounder.py` | `SentenceLevelGrounder.check()` | **Authoritative (Grounded Ratio)** |
| **Hallucination Rate** | `generator/sentence_grounder.py` | `1.0 - Faithfulness` | **Authoritative (Inverse Faithfulness)** |
| **Clinical Reliability**| `evaluation/metrics.py` | `compute_clinical_reliability()` | **Authoritative (Code Enforced)** |
| **Answer Relevance** | `embeddings/embedder.py` | `cosine_similarity(q_vec, a_vec)` | **Authoritative** |
| *RAGAS Evaluator* | `evaluation/ragas_evaluator.py` | `RagasEvaluator` | Experimental Reference Only |
| *DeepEval Evaluator*| `evaluation/deepeval_evaluator.py` | `DeepEvalEvaluator` | Experimental Reference Only |
| *LLM Rubric Judge* | `generator/llm_evaluator.py` | `LLMEvaluator` | Exploratory / Secondary Pass |
