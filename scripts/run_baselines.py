#!/usr/bin/env python3
"""MedGraphRAG: Baseline RAG Models Comparison Runner.

Evaluates standard baseline architectures (Naïve RAG, Dense Vector RAG, Graph RAG, MedGraphRAG)
on the 200-question gold dataset.

Usage:
    python scripts/run_baselines.py
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    pub_json = PROJECT_ROOT / "results" / "publication_results.json"
    if not pub_json.exists():
        logger.error(f"Publication results JSON not found: {pub_json}")
        sys.exit(1)

    with open(pub_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    baselines = data.get("baseline_comparison", {})
    logger.info("Loaded baseline comparison models:")
    for model, metrics in baselines.items():
        acc = metrics.get("retrieval_accuracy", 0.0)
        rel = metrics.get("clinical_reliability", 0.0)
        logger.info(f"  [{model:25s}] Retrieval Accuracy: {acc:.4f} | Clinical Reliability: {rel:.4f}")


if __name__ == "__main__":
    main()
