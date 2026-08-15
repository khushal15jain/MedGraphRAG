# MedGraphRAG Audit Baseline Record

- **Timestamp**: 2026-08-15T15:28:12+05:30
- **Git Branch**: `main`
- **Git Commit Hash**: `5ecd9b5`
- **Repository URI**: `https://github.com/khushal15jain/RAGupdated`

---

## Pre-Audit State Summary

- `data/qa_dataset.json`: 200 items.
- `ablation_*.json`: 100 items each (generated from slice `[:100]`).
- `run_ablations.py`: Refactored to accept `--num-questions` (default 200).
- `evaluation/p_test_evaluator.py`: Refactored to use `align_by_question_id` and Holm-Bonferroni correction.
- `configs/experiment_manifest.yaml`: Central manifest established.
- `tests/test_reproducibility.py`: 5 pytest tests passing.
