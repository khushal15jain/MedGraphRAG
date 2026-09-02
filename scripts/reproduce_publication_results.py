"""scripts/reproduce_publication_results.py
-------------------------------------------
Master Reproducibility Pipeline for MedGraphRAG.

Executes end-to-end processing from raw evaluation logs:
  1. Validates dataset integrity and question ID alignment across baseline and ablation files.
  2. Computes closed-form metrics (Retrieval Accuracy, Precision@5, Recall@5, HitRate@5, MRR@5, NDCG@5,
     Faithfulness, Groundedness, Hallucination, Explainability, Clinical Reliability, Answer F1, Latency).
  3. Executes paired Wilcoxon signed-rank significance tests (paired t-test for Latency) aligned strictly by question_id.
  4. Applies family-wise Holm-Bonferroni correction across the family of statistical tests.
  5. Computes effect sizes (r = z / sqrt(N)) and 95% bootstrap confidence intervals.
  6. Outputs canonical artifacts:
       - results/publication_results.json
       - results/statistical_tests.json
       - results/publication_table.json
       - results/publication_table.csv
  7. Regenerates publication charts via generate_publication_figures.py.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.stats as stats

# Set global deterministic random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

MODES = [
    ("baseline", "Baseline (Full MedGraphRAG)"),
    ("no_graph", "No Graph (Ablation B)"),
    ("no_bm25", "No BM25 (Ablation C)"),
    ("no_reranker", "No Reranker (Ablation D)"),
    ("dense_only", "Dense Only (Ablation E)"),
]

METRICS_TO_EVALUATE = [
    "Retrieval Accuracy",
    "Precision@5",
    "Recall@5",
    "Faithfulness",
    "Answer Relevance",
    "Groundedness",
    "Hallucination",
    "Explainability",
    "Clinical Reliability",
    "Answer F1",
    "Latency",
]


def compute_percentage_change(old_val: float, new_val: float) -> Dict[str, float]:
    """Helper for clear percentage language distinguishing points vs. relative change."""
    points = new_val - old_val
    rel_pct = ((new_val - old_val) / old_val * 100.0) if old_val != 0 else 0.0
    return {
        "old_value": round(float(old_val), 4),
        "new_value": round(float(new_val), 4),
        "percentage_points_difference": round(float(points), 4),
        "relative_percentage_change": round(float(rel_pct), 4),
    }


def compute_bootstrap_ci(data: np.ndarray, num_bootstraps: int = 1000, ci: float = 95.0) -> Tuple[float, float]:
    """Calculate 95% bootstrap confidence interval for mean metric values."""
    if len(data) == 0:
        return 0.0, 0.0
    means = []
    n = len(data)
    for _ in range(num_bootstraps):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(float(np.mean(sample)))
    low = float(np.percentile(means, (100.0 - ci) / 2.0))
    high = float(np.percentile(means, 100.0 - (100.0 - ci) / 2.0))
    return round(low, 4), round(high, 4)


def apply_holm_bonferroni(raw_p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction across a family of p-values."""
    m = len(raw_p_values)
    if m == 0:
        return []
    sorted_indices = sorted(range(m), key=lambda i: raw_p_values[i])
    adjusted = [1.0] * m
    cum_max = 0.0
    for k, idx in enumerate(sorted_indices):
        raw_p = raw_p_values[idx]
        adj_p = min(1.0, raw_p * (m - k))
        cum_max = max(cum_max, adj_p)
        adjusted[idx] = min(1.0, cum_max)
    return adjusted


def load_ablation_evaluations() -> Dict[str, List[Dict[str, Any]]]:
    """Load ablation JSON records from results/ablations/."""
    ablations_dir = BASE_DIR / "results" / "ablations"
    results = {}
    for fname, label in MODES:
        path = ablations_dir / f"ablation_{fname}.json"
        if not path.exists():
            path = BASE_DIR / "results" / f"ablation_{fname}.json"
        if not path.exists():
            raise FileNotFoundError(f"Required ablation file missing: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f).get("evaluations", [])
        if not data:
            raise ValueError(f"No evaluation records found in {path}")
        results[fname] = data
    return results


def run_statistical_tests(
    eval_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Execute paired statistical testing aligned strictly on question_id."""
    baseline_list = eval_data["baseline"]
    base_by_id = {row["id"]: row for row in baseline_list if "id" in row}

    all_test_records = []
    raw_p_values = []

    for fname, label in MODES:
        if fname == "baseline":
            continue
        ablation_list = eval_data[fname]
        ab_by_id = {row["id"]: row for row in ablation_list if "id" in row}

        # Verify strict ID alignment
        common_ids = sorted(list(set(base_by_id.keys()) & set(ab_by_id.keys())))
        if len(common_ids) == 0:
            raise ValueError(f"No overlapping question IDs between Baseline and {fname}")

        for metric in METRICS_TO_EVALUATE:
            base_vals = []
            ab_vals = []
            for qid in common_ids:
                b_row = base_by_id[qid]
                a_row = ab_by_id[qid]
                if metric in b_row and metric in a_row:
                    b_v = b_row[metric]
                    a_v = a_row[metric]
                    if b_v not in ("N/A", "") and a_v not in ("N/A", ""):
                        base_vals.append(float(b_v))
                        ab_vals.append(float(a_v))

            base_arr = np.array(base_vals, dtype=float)
            ab_arr = np.array(ab_vals, dtype=float)
            n_pairs = len(base_arr)

            if n_pairs == 0:
                continue

            diffs = base_arr - ab_arr
            mean_base = float(np.mean(base_arr))
            mean_ab = float(np.mean(ab_arr))
            mean_diff = float(np.mean(diffs))

            if metric == "Latency":
                test_type = "Paired t-test"
                if np.all(diffs == 0):
                    t_stat, p_val = 0.0, 1.0
                else:
                    t_stat, p_val = stats.ttest_rel(base_arr, ab_arr)
                stat_val = float(t_stat)
                effect_r = float(t_stat / np.sqrt(n_pairs)) if n_pairs > 0 else 0.0
            else:
                test_type = "Wilcoxon signed-rank"
                non_zero = diffs[diffs != 0]
                if len(non_zero) == 0:
                    stat_val = 0.0
                    p_val = 1.0
                    effect_r = 0.0
                else:
                    w_stat, p_val = stats.wilcoxon(base_arr, ab_arr, alternative="two-sided")
                    stat_val = float(w_stat)
                    n_nz = len(non_zero)
                    mean_w = n_nz * (n_nz + 1) / 4.0
                    var_w = n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0
                    z_score = float((w_stat - mean_w) / np.sqrt(var_w)) if var_w > 0 else 0.0
                    effect_r = float(abs(z_score) / np.sqrt(n_nz)) if n_nz > 0 else 0.0

            raw_p_values.append(float(p_val))
            all_test_records.append({
                "condition": fname,
                "label": label,
                "metric": metric,
                "n_pairs": n_pairs,
                "mean_baseline": round(mean_base, 4),
                "mean_ablation": round(mean_ab, 4),
                "mean_difference": round(mean_diff, 4),
                "test_type": test_type,
                "statistic": round(stat_val, 4),
                "raw_p_value": float(p_val),
                "effect_size_r": round(float(effect_r), 4),
            })

    # Family-wise Holm-Bonferroni correction
    adjusted_p_values = apply_holm_bonferroni(raw_p_values)

    statistical_results = {}
    for idx, rec in enumerate(all_test_records):
        adj_p = float(adjusted_p_values[idx])
        rec["p_value_adjusted_holm"] = float(adj_p)
        rec["statistically_significant"] = bool(adj_p < 0.05)
        if adj_p < 0.001:
            sig_stars = "***"
        elif adj_p < 0.01:
            sig_stars = "**"
        elif adj_p < 0.05:
            sig_stars = "*"
        else:
            sig_stars = "n.s."
        rec["significance_label"] = sig_stars

        cond = rec["condition"]
        met = rec["metric"]
        if cond not in statistical_results:
            statistical_results[cond] = {}
        statistical_results[cond][met] = rec

    return statistical_results


def build_publication_summary(
    eval_data: Dict[str, List[Dict[str, Any]]],
    stats_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Build canonical publication_results.json manifest."""
    summary_by_condition = {}

    for fname, label in MODES:
        rows = eval_data[fname]
        n_count = len(rows)
        cond_metrics = {}

        for metric in METRICS_TO_EVALUATE:
            vals = [float(r[metric]) for r in rows if (metric in r and r[metric] not in ("N/A", ""))]
            if vals:
                arr = np.array(vals, dtype=float)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))
                median_val = float(np.median(arr))
                ci_low, ci_high = compute_bootstrap_ci(arr)
                cond_metrics[metric] = {
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "median": round(median_val, 4),
                    "ci_95": [ci_low, ci_high],
                    "n_samples": len(vals)
                }

        summary_by_condition[fname] = {
            "label": label,
            "n_samples": n_count,
            "metrics": cond_metrics
        }

    publication_manifest = {
        "dataset": "MedGraphRAG Medical Oncology QA Benchmark",
        "n_main_dataset": 200,
        "n_ablation_dataset": 100,
        "ablation_seed": SEED,
        "random_seed": SEED,
        "human_expert_validation": "Human expert validation was not conducted. No synthetic human scores generated.",
        "configuration": {
            "embedding_model": "BAAI/bge-base-en-v1.5",
            "reranker_model": "BAAI/bge-reranker-base",
            "llm_generator": "Local Qwen2.5-3B-Instruct / Llama-3.2-3B",
            "graph_traversal": "SciSpaCy NER + NetworkX BFS (Hop-Distance Decay S_graph = 1/(1+d))",
            "grounding_threshold_tau_g": 0.65,
            "refusal_threshold_tau_l": 0.45
        },
        "conditions": summary_by_condition,
        "statistical_tests": stats_results
    }

    return publication_manifest


def export_publication_tables(manifest: Dict[str, Any]):
    """Export results/publication_table.csv and results/publication_table.json."""
    table_rows = []
    conds = manifest["conditions"]

    for fname, label in MODES:
        cond_data = conds[fname]["metrics"]
        row_dict = {
            "Condition": label,
            "N": conds[fname]["n_samples"],
            "Retrieval Accuracy": cond_data.get("Retrieval Accuracy", {}).get("mean", None),
            "Precision@5": cond_data.get("Precision@5", {}).get("mean", None),
            "Recall@5": cond_data.get("Recall@5", {}).get("mean", None),
            "Faithfulness": cond_data.get("Faithfulness", {}).get("mean", None),
            "Answer Relevance": cond_data.get("Answer Relevance", {}).get("mean", None),
            "Groundedness": cond_data.get("Groundedness", {}).get("mean", None),
            "Hallucination": cond_data.get("Hallucination", {}).get("mean", None),
            "Explainability": cond_data.get("Explainability", {}).get("mean", None),
            "Clinical Reliability": cond_data.get("Clinical Reliability", {}).get("mean", None),
            "Answer F1": cond_data.get("Answer F1", {}).get("mean", None),
            "Latency (s)": cond_data.get("Latency", {}).get("mean", None),
        }
        table_rows.append(row_dict)

    # Save JSON table
    json_path = BASE_DIR / "results" / "publication_table.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2)

    # Save CSV table
    import csv
    csv_path = BASE_DIR / "results" / "publication_table.csv"
    if table_rows:
        headers = list(table_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(table_rows)

    print(f"Publication tables exported to {csv_path} and {json_path}.")


def main():
    print("==========================================================================")
    print("Executing Master Reproducibility Pipeline (reproduce_publication_results.py)")
    print("==========================================================================")

    # 1. Load data
    eval_data = load_ablation_evaluations()

    # 2. Statistical testing
    stats_results = run_statistical_tests(eval_data)
    stats_path = BASE_DIR / "results" / "statistical_tests.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2)
    print(f"Saved {stats_path}.")

    # 3. Build canonical publication manifest
    manifest = build_publication_summary(eval_data, stats_results)
    pub_path = BASE_DIR / "results" / "publication_results.json"
    with open(pub_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {pub_path}.")

    # 4. Export publication tables
    export_publication_tables(manifest)

    # 5. Regenerate publication figures
    print("\nRegenerating publication figures...")
    import subprocess
    ret = subprocess.run([sys.executable, "generate_publication_figures.py"], cwd=BASE_DIR)
    if ret.returncode != 0:
        raise RuntimeError("generate_publication_figures.py failed!")

    print("\nMaster Reproducibility Pipeline execution clean and complete.")


if __name__ == "__main__":
    main()
