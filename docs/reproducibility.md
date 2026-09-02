# MedGraphRAG: Master Reproducibility Guide

This document provides step-by-step instructions to reproduce all experimental results, statistical hypothesis tests, tables, and figures reported in the **MedGraphRAG** paper and repository.

---

## 📋 Prerequisites & Environment Setup

1. **Python Environment**:
   Requires Python 3.11+.

2. **Installation**:
   ```bash
   git clone https://github.com/khushaljain/MedGraphRAG.git
   cd MedGraphRAG
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **SpaCy Medical Model**:
   ```bash
   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
   ```

---

## 🔄 Executing the Master Reproducibility Pipeline

To execute the master reproducibility pipeline from raw evaluation logs:

```bash
python scripts/reproduce_results.py
```

### What `scripts/reproduce_results.py` Executes:

1. **Dataset Integrity Verification**:
   - Verifies the $N=200$ question gold QA dataset (`data/qa_dataset.json`).
   - Verifies the reproducible 100-question sampling manifest (`results/ablation_question_ids.json`, `seed=42`).

2. **Metric Evaluation Engine**:
   - Executes closed-form mathematical functions in `medgraphrag.evaluation.metrics` for all 18 parameter metrics (Retrieval Accuracy, Precision@5, Recall@5, HitRate@5, MRR, NDCG@5, Faithfulness, Groundedness, Hallucination Rate, Explainability, Clinical Reliability, Answer F1, Latency).

3. **Paired Statistical Testing**:
   - Matches baseline and ablation evaluation records strictly by `question_id`.
   - Computes paired two-sided Wilcoxon signed-rank tests for non-parametric metrics and paired $t$-tests for Operational Latency.
   - Applies step-down **Holm-Bonferroni correction** across the 10 metric families.
   - Computes effect sizes ($r = |z| / \sqrt{N}$) and $95\%$ bootstrap confidence intervals ($1,000$ iterations, `seed=42`).

4. **Artifact Generation**:
   - Exports `results/publication_results.json`.
   - Exports `results/statistical_tests.json`.
   - Exports `results/publication_table.csv` and `results/publication_table.json`.
   - Triggers `scripts/generate_figures.py` to plot high-resolution radar charts and comparison figures in `results/figures/`.

---

## 🔍 Automated Repository Consistency Check

To verify 100% numerical consistency between exported JSON files, CSV tables, figures, and `README.md`:

```bash
python scripts/check_repository.py
```

**Expected Output**:
```
==========================================================================
Running Repository & Publication Consistency Checker (scripts/check_repository.py)
==========================================================================
[PASS] publication_results.json is valid.
[PASS] statistical_tests.json is valid (Holm-adjusted p >= raw p).
[PASS] Codebase integrity verified (no simulated scores or hardcoded fallbacks).
[PASS] README.md matches publication_results.json completely.

All repository consistency checks PASSED cleanly.
```

---

## 🧪 Running Unit & Integration Tests

Execute the full pytest suite:

```bash
pytest
```

All **62 pytest unit and integration tests** will execute and report $100\%$ pass status.
