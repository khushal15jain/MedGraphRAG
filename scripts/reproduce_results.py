#!/usr/bin/env python3
"""MedGraphRAG: Master Reproducibility Pipeline Script.

Executes dataset integrity validation, metric evaluations, paired statistical hypothesis testing,
and artifact generation.

Usage:
    python scripts/reproduce_results.py
"""

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
    logger.info("==================================================")
    logger.info("Running Master Reproducibility Pipeline")
    logger.info("==================================================")

    data_file = PROJECT_ROOT / "data" / "qa_dataset.json"
    pub_file = PROJECT_ROOT / "results" / "publication_results.json"

    if not data_file.exists():
        logger.error(f"QA dataset not found: {data_file}")
        sys.exit(1)

    with open(data_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    logger.info(f"Loaded {len(qa_data)} gold questions.")
    summary = compute_publication_metrics_summary(qa_data)

    pub_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pub_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Successfully updated publication results JSON at {pub_file}")


if __name__ == "__main__":
    main()
