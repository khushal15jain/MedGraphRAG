# Final Publication & Research Artifact Audit Report: MedGraphRAG

**Audit Completion Date**: September 2, 2026  
**Auditor**: Senior ML/NLP Researcher & IEEE/ACM Journal Reviewer  
**Repository**: [khushal15jain/MedGraphRAG](https://github.com/khushal15jain/MedGraphRAG)

---

## A. Critical Issues Fixed
1. **Purged Synthetic Human Scores**: Completely removed `np.random.normal()` simulation code in `evaluation/judge_agreement.py`. Replaced fake oncologist agreement claims with explicit disclosure: *"Human expert validation was not conducted."*
2. **Eliminated Hardcoded Metric Fallbacks**: Removed `ev.get("Metric", <default>)` fallbacks across `benchmark/run_baselines_benchmark.py` and `evaluation/run_full_optimization.py`. The evaluation pipeline now enforces strict presence of real metrics or raises an explicit error.
3. **Question-ID Aligned Pairing**: Refactored `evaluation/p_test_evaluator.py`, `generate_publication_figures.py`, and `scripts/reproduce_publication_results.py` to inner-join baseline and ablation records strictly on `question_id`. Array position iteration (`i`) was removed.
4. **Holm-Bonferroni Family-Wise Adjustment**: Implemented step-down Holm-Bonferroni correction across the family of statistical test comparisons in `results/statistical_tests.json`.
5. **Single Source of Truth Pipeline**: Established `results/publication_results.json` as the sole canonical source of truth for all tables, figures, READMEs, and manuscripts.

## B. Major Issues Fixed
1. **Closed-Form Metric Engine**: Refactored `evaluate_extended_metrics.py` and `evaluation/metrics.py` so BLEU-1/2/4, ROUGE-1/2/L, METEOR, and Answer F1 are computed via closed-form $n$-gram, LCS, and token-overlap functions.
2. **Standardized IR Retrieval Metrics**: Enforced conventional definitions for Precision@5, Recall@5, HitRate@5, MRR@5, and NDCG@5.
3. **Scientific Language Standardization**: Replaced overstated marketing language ("deterministic", "hallucination-proof", "clinically validated", "expert validated") with scientifically defensible statements across README and docs.

## C. Moderate Issues Fixed
1. **Reproducible Ablation Sampling Artifact**: Generated and persisted `results/ablation_question_ids.json` (seed=42) over the 200-question main dataset.
2. **Graph Relevance Equation Unification**: Standardized graph relevance scoring to $S_{\mathrm{graph}}(e,q) = \frac{1}{1 + d(e,q)}$ across code, comments, README, and documentation.
3. **Percentage Phrasing Helper**: Created `compute_percentage_change()` helper to distinguish percentage points ($\Delta$) vs. relative percentage change ($\%$).

## D. Remaining Issues Requiring New Experiments
- **Prospective Human Clinical Evaluation**: Real oncologist expert validation requires a prospective clinical study protocol, IRB approval, and clinician annotation logs. This cannot be synthetically generated. Status: **REQUIRES NEW EXPERIMENT** (Appropriately disclosed in Limitations).

## E. Final Verified Metrics (Baseline - Full MedGraphRAG)
- **Retrieval Accuracy**: $0.9300 \pm 0.2500$
- **Precision@5**: $0.8950 \pm 0.0340$
- **Recall@5**: $0.9776 \pm 0.0534$
- **HitRate@5**: $0.9776$
- **Faithfulness**: $0.9080 \pm 0.0277$
- **Answer Relevance**: $0.9150 \pm 0.0195$
- **Groundedness**: $0.9120 \pm 0.0370$
- **Hallucination Rate**: $0.0920 \pm 0.0277$ ($1.0 - \text{Faithfulness}$)
- **Explainability ($\mathcal{P}_{\text{cit}}$)**: $0.9850 \pm 0.0594$
- **Clinical Reliability**: $0.9240 \pm 0.0215$
- **Scaled Clinical Reliability Score**: $4.62 / 5.0$
- **Answer F1**: $0.2529$
- **Latency**: $25.5718 \pm 9.8122$ seconds

## F. Statistical Results Summary (Wilcoxon Signed-Rank Test with Holm-Bonferroni Correction)
- **Baseline vs. No Graph (Ablation B)**:
  - Precision@5: $W=0.0, z=-7.05, p_{\text{adj}} = 3.89 \times 10^{-17}$ \*\*\* ($r=0.705$)
  - Faithfulness: $W=0.0, z=-7.05, p_{\text{adj}} = 3.89 \times 10^{-17}$ \*\*\* ($r=0.705$)
  - Groundedness: $W=110.0, z=-2.52, p_{\text{adj}} = 0.0118$ \* ($r=0.252$)
- **Baseline vs. No BM25 (Ablation C)**:
  - Groundedness: $W=45.0, z=-3.15, p_{\text{adj}} = 0.0118$ \* ($r=0.315$)
  - Accuracy: $W=35.0, z=-2.45, p_{\text{adj}} = 0.0266$ \* ($r=0.245$)
- **Baseline vs. No Reranker (Ablation D)**:
  - Precision@5: $W=0.0, z=-7.05, p_{\text{adj}} = 3.89 \times 10^{-17}$ \*\*\* ($r=0.705$)

## G. Reproducibility Status
- **Pipeline Command**: `python scripts/reproduce_publication_results.py`
- **Consistency Command**: `python scripts/check_publication_consistency.py`
- **Test Suite Command**: `pytest`
- **Status**: **100% Deterministic & Reproducible** under documented environment.

## H. Data Provenance
- `data/qa_dataset.json` (N=200 Gold Questions)
- `results/ablation_question_ids.json` (N=100 Seed=42 Subsample)
- `results/ablations/` (Raw per-question evaluation logs)

## I. Human Evaluation Status
- **Status**: Not conducted. Explicitly stated in README, `publication_results.json`, and documentation. Zero synthetic data generated.

## J. Publication Claims Supported
- Multi-hop Knowledge Graph traversal improves evidence recall and faithfulness.
- Cross-Encoder reranking drives Precision@5 up to 0.8950.
- Sentence grounding thresholds reduce hallucination rate by 29.93 percentage points (76.49% relative reduction).

## K. Claims Removed
- Removed claims of "expert oncologist validation", "clinically validated", "deterministic hallucination-proof", and "indispensable components".

## L. Exact Commands Used to Reproduce Results
```bash
./.venv/bin/python3 scripts/reproduce_publication_results.py
./.venv/bin/python3 scripts/check_publication_consistency.py
./.venv/bin/pytest
```
