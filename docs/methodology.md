# MedGraphRAG: Scientific Methodology

This document presents the complete mathematical formulation and scientific methodology of **MedGraphRAG**.

---

## 🧮 Graph Scoring & Subgraph Propagation Formula

For a query $Q$ with extracted entity set $E_Q$, the graph retrieval signal score for candidate passage chunk $c$ containing entity set $E_c$ is calculated using a hop-distance decay formulation:

$$S_{\text{graph}}(c) = \sum_{e \in E_c} \sum_{e_q \in E_Q} \gamma^{d(e, e_q)}$$

where:
- $d(e, e_q)$ is the shortest path length (in graph edge hops) between entity node $e$ and query entity node $e_q$ in the NetworkX knowledge graph.
- $\gamma = 0.5$ is the attenuation decay constant.
- If $d(e, e_q) > 2$ hops, the term evaluates to $0$.

---

## ⚖️ Hybrid Retrieval Fusion

The final retrieval score $S_{\text{hybrid}}(c)$ for candidate chunk $c$ combines normalized dense vector similarity, BM25 term matching, and graph propagation signals:

$$\hat{S}_{\text{dense}}(c) = \frac{S_{\text{dense}}(c) - \min(S_{\text{dense}})}{\max(S_{\text{dense}}) - \min(S_{\text{dense}})}$$

$$\hat{S}_{\text{bm25}}(c) = \frac{S_{\text{bm25}}(c) - \min(S_{\text{bm25}})}{\max(S_{\text{bm25}}) - \min(S_{\text{bm25}})}$$

$$\hat{S}_{\text{graph}}(c) = \frac{S_{\text{graph}}(c) - \min(S_{\text{graph}})}{\max(S_{\text{graph}}) - \min(S_{\text{graph}})}$$

$$S_{\text{hybrid}}(c) = \alpha \hat{S}_{\text{dense}}(c) + \beta \hat{S}_{\text{bm25}}(c) + \gamma \hat{S}_{\text{graph}}(c)$$

Hyperparameters: $\alpha = 0.4$, $\beta = 0.3$, $\gamma = 0.3$.

---

## 🛡 Grounding & Hallucination Protocol

1. **Sentence-Level NLI Verification**:
   Each generated sentence $s_i$ is evaluated against retrieved evidence passages $P$ using `DeBERTa-v3` entailment.

2. **Faithfulness**:
   $$\text{Faithfulness} = \frac{N_{\text{entailed\_sentences}}}{N_{\text{total\_sentences}}}$$

3. **Hallucination Rate**:
   $$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$$
