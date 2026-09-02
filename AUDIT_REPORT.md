# Comprehensive Codebase Audit Report: MedGraphRAG

**Audit Date**: September 2, 2026  
**Auditor**: Senior ML/NLP Researcher & IEEE/ACM Journal Reviewer  
**Repository**: [khushal15jain/MedGraphRAG](https://github.com/khushal15jain/MedGraphRAG)

---

## Executive Summary

A comprehensive, global audit of the MedGraphRAG repository was conducted across all source directories (`configs/`, `benchmark/`, `evaluation/`, `retrieval/`, `graph/`, `generator/`, `results/`, `tests/`, `scripts/`, `docs/`, `README.md`).

The audit identified critical issues in data integrity, statistical methodology, baseline pairing, and scientific claims. Most notably:
1. **Simulated Human Evaluation**: `evaluation/judge_agreement.py` used `np.random.normal()` to generate fake oncologist expert scores and agreement statistics ($r \approx 0.9683, \kappa \approx 0.5408$).
2. **Hard-Coded Fallback Constants**: Key evaluation scripts used `ev.get("Metric", <default>)` fallbacks, substituting artificial values when evaluations were missing.
3. **Question-ID Unaligned Pairing**: Baseline and ablation evaluations were paired by array index (`i`) rather than joining on `question_id`.
4. **Independent Uncorrected Statistical Tests**: Figure generation scripts calculated raw $p$-values independently without applying Holm-Bonferroni correction across test families.
5. **Percentage Language Ambiguity**: Mixing percentage points vs. relative percentage reduction (e.g. 0.3913 to 0.0967 is 29.46 percentage points or a 75.29% relative reduction).

Below is the itemized audit matrix.

---

## Audit Matrix

| File | Line(s) | Problem | Severity | Why it is a Problem | Required Fix | Status |
| :--- | :--- | :--- | :---: | :--- | :--- | :---: |
| `evaluation/judge_agreement.py` | L101–L115 | Simulated Human Scores via `np.random.normal()` | **CRITICAL** | Violates scientific ethics by representing synthetic normal noise as real clinical oncology expert validation ($r=0.9683, \kappa=0.5408$). | Remove synthetic human score generation entirely. Replace human evaluation claims in docs/README with *"Human expert validation was not conducted."* Do NOT generate fake replacement data. | Open |
| `benchmark/run_baselines_benchmark.py` | L74, L82, L87, L90, L91, L93 | Hardcoded fallback constants (`0.952`, `0.90`, `0.85`, `0.95`, `0.88`) | **CRITICAL** | Fabricates experimental results when underlying JSON evaluation records lack specific metric keys. | Remove all fallback constants. Raise an explicit error or handle missing keys deterministically without fabricating numbers. | Open |
| `evaluation/run_full_optimization.py` | L61–L65, L75–L80 | Hardcoded result fallbacks (`0.8950`, `0.9080`, `0.9150`, `0.9120`, `0.9240`, `0.9300`, `0.9776`, `0.7517`, `0.9850`, `25.0354`) | **CRITICAL** | Injects synthetic default metrics during optimization runs when evaluations are unpopulated. | Remove fallback constants. Require valid evaluation inputs. | Open |
| `generate_publication_figures.py` | L88–L93 | Unaligned Array Index Pairing (`baseline_data[i]` vs `test_data[i]`) | **CRITICAL** | If question ordering differs between JSON files, statistical tests compare mismatched questions. | Inner-join observations strictly on `question_id`. Fail loudly if IDs are missing or mismatched. | Open |
| `generate_publication_figures.py` | L99–L107, L195–L198 | Independent Uncorrected $p$-value Calculation | **CRITICAL** | Calculates raw uncorrected $p$-values independently instead of consuming Holm-Bonferroni adjusted $p$-values from the canonical statistical tests file. | Refactor script to consume $p$-values and significance levels directly from `results/statistical_tests.json` / `results/publication_results.json`. | Open |
| `evaluation/p_test_evaluator.py` | L120–L135 | Array Position Pairing in Wilcoxon Tests | **CRITICAL** | Array positional iteration assumes identical ordering across ablation files. | Align evaluation records strictly by `question_id` before executing Wilcoxon signed-rank or paired $t$-tests. | Open |
| `README.md` | L46–L60, L200–L250 | Hardcoded Markdown Results Tables & Overstated Claims | **MAJOR** | Manually typed numbers in README violate the single-source-of-truth principle. Oversimplified language ("clinically validated", "expert validated", "deterministic", "hallucination-proof"). | Generate README results section automatically from `results/publication_results.json`. Tone down claims to scientifically defensible statements. | Open |
| `docs/ABLATION_STUDY_REPORT.md` | L95–L110 | Citation of Simulated Human Validation Statistics | **CRITICAL** | Documents fake human oncology agreement stats derived from `np.random.normal()`. | Remove human validation section. State explicitly: *"Human expert validation was not conducted."* | Open |
| `docs/REVISED_MANUSCRIPT.md` | §4.5, §7 | Human Evaluation Section Based on Synthetic Data | **CRITICAL** | Manuscript claims 3 oncology specialists evaluated answers when no human trial occurred. | Remove Section 4.5 human evaluation claims. Add explicit limitation regarding absence of human evaluation. | Open |
| `evaluate_extended_metrics.py` | L148–L158 | Historical LLM JSON Flattening for Closed-Form Metrics | **MAJOR** | Historical pipeline flattened LLM evaluator outputs into BLEU/ROUGE/METEOR/F1 fields. | Ensure single pipeline (`results/publication_results.json`) populates closed-form metrics strictly via `evaluation/metrics.py`. | Open |
| `graph/graph_retriever.py` vs Docs | L17–L30 | Inconsistent Graph Relevance Scoring Equation | **MODERATE** | Code implements $S = \frac{1}{1 + d(e,q)}$ topological decay with degree cap ($>100$), while docs mention frequency-weighted scoring. | Standardize equation in README, paper, code comments, and docs to match the executed implementation: $S_{\text{graph}}(e,q) = \frac{1}{1 + d(e,q)}$. | Open |
| `results/` | Multiple JSONs | Multiple Disparate Result Files without Canonical Pipeline | **MAJOR** | Baseline comparison, ablation results, and p-tests are spread across unlinked JSON files. | Create single source of truth `results/publication_results.json` generated via `scripts/reproduce_publication_results.py`. | Open |
| `data/qa_dataset.json` | Global | Lack of Documented Sampling Seed Artifact for Ablation Subset | **MODERATE** | 100-question ablation subset was sampled without a persisted question ID manifest. | Create `results/ablation_question_ids.json` using reproducible seed `42` over the 200-question main dataset. | Open |
| Overall Repo | Global | Percentage Point vs. Relative Percentage Language | **MINOR** | Inconsistent phrasing of score changes (e.g. 0.3913 to 0.0967 described as "29.5% reduction" instead of "29.46 percentage points" or "75.29% relative reduction"). | Create helper function `compute_percentage_change()` and standardize all prose in README and docs. | Open |

---

## Action Plan for Remediation

1. **Phase 2**: Implement single canonical source of truth `results/publication_results.json`.
2. **Phase 3**: Enforce standard retrieval metric definitions (Precision@5, Recall@5, HitRate@5, MRR@5, NDCG@5).
3. **Phase 4**: Implement strict inner-join on `question_id` for paired statistical testing.
4. **Phase 5 & 6**: Persist `results/ablation_question_ids.json` (seed=42) and enforce configuration determinism.
5. **Phase 7**: Completely purge synthetic human evaluation code (`np.random.normal`). Document that human validation was not conducted.
6. **Phase 8**: Refactor `evaluation/p_test_evaluator.py` to produce `results/statistical_tests.json` with Holm-Bonferroni correction and Wilcoxon signed-rank tests.
7. **Phase 9–13**: Remove all hardcoded fallbacks, standardize graph score formulas, and clarify metric definitions.
8. **Phase 14–22**: Build `results/publication_table.csv`, generate high-resolution figures directly from `publication_results.json`, and update README using cautious scientific language.
9. **Phase 23–25**: Create `scripts/reproduce_publication_results.py`, `scripts/check_publication_consistency.py`, and comprehensive unit tests (`tests/test_publication_metrics.py`).
10. **Phase 26–28**: Complete `PUBLICATION_AUDIT_FINAL.md` and `PUBLICATION_READY_CHECKLIST.md`.
