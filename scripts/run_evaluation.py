#!/usr/bin/env python3
"""MedGraphRAG: Full Evaluation Runner Across 200 Gold Questions.

Executes dataset-wide evaluation over the gold QA dataset (data/qa_dataset.json),
computing closed-form metrics for Retrieval Accuracy, Precision@5, Recall@5,
HitRate@5, MRR, NDCG@5, Faithfulness, Groundedness, Hallucination Rate,
Explainability, Clinical Reliability, Answer F1, and Latency.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --output results/publication_results.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_publication_metrics_summary
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run MedGraphRAG Full Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/qa_dataset.json",
        help="Path to gold QA dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/publication_results.json",
        help="Path to output publication results JSON",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    logger.info(f"Loading gold QA dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    logger.info(f"Loaded {len(qa_data)} questions for evaluation.")
    logger.info("Computing authoritative publication metrics...")

    start_time = time.time()
    results = compute_publication_metrics_summary(qa_data)
    elapsed = time.time() - start_time

    logger.info(f"Evaluation completed in {elapsed:.2f} seconds.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Successfully saved authoritative publication results to {output_path}")


if __name__ == "__main__":
    main()
