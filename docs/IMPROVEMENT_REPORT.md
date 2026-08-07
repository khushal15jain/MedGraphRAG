# MedGraphRAG Targeted Improvement Report

Scope: **only** Prompt Builder, Generator, Grounding Checker, Explainability,
Retrieval Fusion, Query Expansion, Adaptive Retrieval, and the Evaluation
Pipeline are modified. Chunking, SciSpaCy NER, Knowledge Graph
construction, BGE embeddings, ChromaDB, BM25 indexing, Hybrid Retrieval's
core scoring, and the BGE Reranker model are **untouched**, per your
constraints.

## 1. Root-cause diagnosis (why each metric is where it is)

| Metric | Value | Root cause |
|---|---|---|
| Precision@5 | 92.1% | Reranker is doing its job well on whatever candidates it receives. |
| Recall@5 | 27.99% | Single-query retrieval; the candidate pool the reranker sees never contains most true positives. |
| Faithfulness | 47.15% | Free-text generation with no citation contract; nothing prevents parametric-knowledge fill-in. |
| Hallucination | 52.85% | Direct consequence of the above — no refusal mechanism for uncovered claims. |
| Groundedness | 67.50% | Grounding check (if any) is answer-level, not claim/sentence-level, so partial hallucination is masked by the supported majority. |
| Explainability | 38.11% | Evidence provenance (rank, retriever, matched terms, which claim it supports) is discarded after reranking instead of surfaced. |
| Clinical Reliability | 66.50% | No terminology-fidelity or no-fabrication constraint at generation time. |
| Latency | 37.76s | Single-threaded sequential calls; no caching; fixed large context regardless of question complexity. |

## 2. Module-by-module changes and expected deltas

### Query Expansion (`retrieval/query_expansion.py`)
- **Why current fails**: one query embedding cannot match all clinical paraphrases/synonyms.
- **Algorithm**: LLM paraphrasing (n=2) + HyDE passage + 1-2 hop KG entity expansion, budget-capped at 5 total queries.
- **Recall@5**: 27.99% → **48-55%**
- **Precision@5**: 92.1% → **88-92%** (small expected dip, recovered by fusion + existing reranker)

### Retrieval Fusion (`retrieval/retrieval_fusion.py`)
- **Why current fails**: no mechanism to merge multiple queries' results before reranking.
- **Algorithm**: parallel fan-out (ThreadPoolExecutor) across expanded queries + Reciprocal Rank Fusion (k=60) + dedup, capped at 30 candidates, handed to the **existing, unmodified** reranker scored against the **original** question.
- **Recall@5**: contributes the majority of the above gain.
- **Latency**: fan-out parallelism keeps wall-clock ≈ max(call latency), not sum.

### Adaptive Retrieval (`retrieval/adaptive_retrieval.py`)
- **Why current fails**: fixed top-5 either truncates multi-fact questions or pads single-fact ones.
- **Algorithm**: score-gap ("elbow") adaptive K ∈ [3,10]; adjacent-chunk stitching across chunk boundaries; LRU caching for query→answer, expansion, and embeddings.
- **Recall (effective)**: multi-fact questions get up to 10 chunks → recall rises further on the subset that needed it.
- **Latency**: 37.76s → **12-18s** typical, cache hits near-zero on repeats.

### Prompt Builder (`generation/prompt_builder.py`)
- **Why current fails**: no structural mechanism forcing claim-to-evidence attribution or refusal.
- **Algorithm**: numbered evidence blocks `[E1]…`, mandatory structured JSON output `{claims:[{text, evidence_ids, confidence}], unanswerable_aspects:[]}`, explicit no-fabrication / terminology-fidelity rules, few-shot exemplars (grounded + correctly-declined).
- **Faithfulness**: 47.15% → **78-85%**
- **Clinical Reliability**: 66.5% → **85-90%**

### Generator (`generation/generator.py`)
- **Why current fails**: single-shot generation with no verification-before-serving gate.
- **Algorithm**: low-temperature JSON generation, robust JSON-repair parsing, grounding-gated single retry (re-prompt listing specifically which claims failed the cheap grounding check), LRU caching keyed on (question, evidence-set).
- **Faithfulness**: reinforced to **80-85%**
- **Hallucination**: 52.85% → **12-18%**

### Grounding Checker (`grounding/grounding_checker.py`)
- **Why current fails**: coarse/answer-level check masks partial hallucination.
- **Algorithm**: per-claim combined score = 0.65·(BGE embedding similarity to cited evidence only) + 0.35·(lexical/term overlap), optional NLI entailment for boundary-band claims only (cost-controlled); decision bands: ≥0.62 grounded, 0.40-0.62 grounded-but-low-confidence (marked, not removed), <0.40 dropped as hallucinated.
- **Groundedness**: 67.5% → **88-92%**

### Explainability (`explainability/explainability.py`)
- **Why current fails**: retrieval provenance is discarded post-rerank.
- **Algorithm**: aggregates data ALREADY produced by fusion (contributing retrievers), rerank (score), and grounding (confidence) stages into a per-evidence record: rank, retriever(s), rerank score, matched question-entities ("why selected"), which claim(s) it supports, and that claim's grounding confidence. No new model calls required.
- **Explainability**: 38.11% → **75-82%**

### Evaluation Pipeline (`evaluation/metrics.py`, `evaluation/ragas_deepeval.py`)
- Adds BLEU-1/2/4, ROUGE-1/2/L, METEOR, Answer F1, MRR, nDCG, Context Precision/Recall/Relevancy, Citation Precision/Recall/Coverage, Evidence Coverage, Semantic Similarity — all model-free or reusing the existing BGE embedder, so no new inference infra.
- Wraps **RAGAS** and **DeepEval**, configured to use your **local Ollama** model as judge (no external API dependency), for Faithfulness / Answer Relevance / Groundedness / Hallucination computed by an independent, peer-reviewed-standard framework.
- Adds `correlate_with_internal_scores()` — Pearson/Spearman correlation between your cheap internal grounding score and RAGAS's LLM-judged faithfulness, which is the evidence a reviewer will want to see that the fast internal check (used at inference time for the retry gate) is a valid proxy for the expensive judge-based metric.

## 3. Consolidated expected improvement table

| Metric | Before | After (expected) |
|---|---|---|
| Retrieval Accuracy | 99.5% | ~99.5% (unchanged, already saturated) |
| Precision@5 | 92.1% | 88-92% |
| Recall@5 | 27.99% | 48-55% |
| Faithfulness | 47.15% | 78-85% |
| Answer Relevance | 83.29% | 85-90% (indirect benefit from citation-constrained relevance) |
| Groundedness | 67.50% | 88-92% |
| Hallucination | 52.85% | 12-18% |
| Explainability | 38.11% | 75-82% |
| Clinical Reliability | 66.50% | 85-90% |
| Latency | 37.76s | 12-18s |

These are **engineering estimates based on the mechanism of each fix**, not measured results — run the extended evaluation pipeline (Section 4) on your 200-item benchmark before/after to get real numbers for publication; report exact deltas with confidence intervals (e.g. bootstrap over the 200 items), not point estimates, in the paper.

## 4. Recommended paper structure implications

- **Ablation table**: report metrics with each module toggled independently (expansion-only, fusion-only, grounding-only, etc.) — this is the single strongest thing you can add for an IEEE/Springer/Elsevier submission, since it isolates which architectural change drives which metric, directly answering "why does this work" rather than only "does this work."
- **Retrieval vs. generation decomposition**: use Context Precision/Recall (retrieval-side) vs. Faithfulness/Groundedness (generation-side) to show the paper's contribution cleanly maps to the RAG-triad framework, which reviewers from IR/NLP venues will recognize.
- **Internal-vs-external metric agreement**: report the Pearson/Spearman correlation between your `grounding_checker.py` score and RAGAS faithfulness as evidence the lightweight in-pipeline check is valid, not just convenient.
- **Latency-quality tradeoff plot**: adaptive-K and caching trade a small, quantifiable precision cost for large recall/latency gains — a Pareto-style plot (Recall@5 vs. Latency across gap_threshold / max_k settings) is a natural figure.
- **Limitations section**: disclose that RAGAS/DeepEval judge quality depends on the local judge model size; using a 3B model as both generator and judge should be flagged, and results with a larger local judge (if available) should be reported as a robustness check.

## 5. Files added (mirrors existing project structure — nothing removed)

```
generation/prompt_builder.py
generation/generator.py
grounding/grounding_checker.py
explainability/explainability.py
retrieval/query_expansion.py
retrieval/retrieval_fusion.py
retrieval/adaptive_retrieval.py
evaluation/metrics.py
evaluation/ragas_deepeval.py
pipeline_integration_example.py   (reference wiring only)
```

Add to `requirements.txt`:
```
nltk>=3.8
rouge-score>=0.1.2
ragas>=0.1.9
deepeval>=1.0
datasets>=2.14
scipy>=1.11
langchain-community>=0.2
```
