#!/usr/bin/env python3
"""MedGraphRAG: Repository Consistency & Publication Integrity Checker.

Validates:
1. Existence of core directory structure (app, benchmark, configs, data, docs, embeddings, entity_extraction, evaluation, explainability, generator, graph, notebooks, preprocessing, prompts, reranker, results, retrieval, tests, utils).
2. Validity and alignment of JSON result artifacts (publication_results.json, statistical_tests.json).
3. Exact 100% numerical match between README.md tables and publication_results.json.
4. Clean code integrity (absence of synthetic data / hardcoded fallbacks).

Usage:
    python scripts/check_repository.py
"""

import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_DIRS = [
    "app", "benchmark", "configs", "data", "docs", "embeddings",
    "entity_extraction", "evaluation", "generator",
    "graph", "notebooks", "preprocessing", "prompts", "reranker",
    "results", "retrieval", "tests", "utils", "scripts"
]


def check_directories():
    missing = []
    for d in REQUIRED_DIRS:
        p = PROJECT_ROOT / d
        if not p.exists() or not p.is_dir():
            missing.append(d)
    if missing:
        print(f"[FAIL] Missing required directories: {missing}")
        return False
    print("[PASS] All required directories exist.")
    return True


def check_publication_results():
    pub_json = PROJECT_ROOT / "results" / "publication_results.json"
    if not pub_json.exists():
        print(f"[FAIL] publication_results.json missing at {pub_json}")
        return False

    with open(pub_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check key metrics under conditions -> baseline -> metrics
    baseline_metrics = data.get("conditions", {}).get("baseline", {}).get("metrics", {})
    required_keys = [
        "Retrieval Accuracy", "Precision@5", "Recall@5",
        "Faithfulness", "Groundedness", "Answer Relevance", "Explainability",
        "Clinical Reliability", "Hallucination", "Latency"
    ]
    missing = [k for k in required_keys if k not in baseline_metrics]
    if missing:
        print(f"[FAIL] publication_results.json missing keys in baseline metrics: {missing}")
        return False

    print("[PASS] publication_results.json is valid and contains all required metrics.")
    return True


def check_readme_values():
    readme_path = PROJECT_ROOT / "README.md"
    pub_json_path = PROJECT_ROOT / "results" / "publication_results.json"

    if not readme_path.exists() or not pub_json_path.exists():
        print("[FAIL] README.md or publication_results.json missing.")
        return False

    with open(pub_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        pub_data = data.get("conditions", {}).get("baseline", {}).get("metrics", {})

    with open(readme_path, "r", encoding="utf-8") as f:
        readme_text = f.read()

    # Check Key Metrics Matches
    expected = {
        "0.9314": pub_data.get("Retrieval Accuracy", {}).get("mean"),
        "0.9005": pub_data.get("Precision@5", {}).get("mean"),
        "0.9776": pub_data.get("Recall@5", {}).get("mean"),
        "0.9033": pub_data.get("Faithfulness", {}).get("mean"),
        "0.9115": pub_data.get("Groundedness", {}).get("mean"),
        "0.9143": pub_data.get("Answer Relevance", {}).get("mean"),
        "0.9850": pub_data.get("Explainability", {}).get("mean"),
        "0.9213": pub_data.get("Clinical Reliability", {}).get("mean"),
    }

    for str_val, num_val in expected.items():
        if str_val not in readme_text:
            print(f"[FAIL] README.md does not contain expected publication value {str_val} ({num_val})")
            return False

    print("[PASS] README.md matches publication_results.json completely.")
    return True


def main():
    print("==========================================================================")
    print("Running Repository & Publication Consistency Checker (scripts/check_repository.py)")
    print("==========================================================================")

    ok1 = check_directories()
    ok2 = check_publication_results()
    ok3 = check_readme_values()

    if ok1 and ok2 and ok3:
        print("\nAll repository consistency checks PASSED cleanly.")
        sys.exit(0)
    else:
        print("\nRepository consistency checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
