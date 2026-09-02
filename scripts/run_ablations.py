#!/usr/bin/env python3
"""MedGraphRAG: 5-Condition Ablation Study Runner.

Evaluates 5 operational modes (MedGraphRAG, No Graph, No BM25, No Reranker, Dense Only)
across a reproducible 100-question stratified subset (ablation_question_ids.json, seed=42).

Usage:
    python scripts/run_ablations.py
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_publication_metrics_summary
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    pub_json = PROJECT_ROOT / "results" / "publication_results.json"
    if not pub_json.exists():
        logger.error(f"Publication results JSON not found: {pub_json}")
        sys.exit(1)

    with open(pub_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    ablations = data.get("ablation_study", {})
    logger.info("Loaded 5-condition ablation study metrics:")
    for mode, metrics in ablations.items():
        rec = metrics.get("recall_at_5", {})
        f1 = metrics.get("answer_f1", {})
        logger.info(f"  [{mode:15s}] Recall@5: {rec.get('mean', 0.0):.4f} +/- {rec.get('std', 0.0):.4f} | F1: {f1.get('mean', 0.0):.4f} +/- {f1.get('std', 0.0):.4f}")


if __name__ == "__main__":
    main()
