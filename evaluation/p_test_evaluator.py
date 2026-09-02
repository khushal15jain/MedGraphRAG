"""evaluation/p_test_evaluator.py
------------------------------
Statistical Hypothesis Testing Module (p-test Evaluation Suite).
Performs explicit Question-ID Alignment (align_by_question_id) across baseline and ablation evaluations.
Computes paired two-tailed Wilcoxon signed-rank tests (non-parametric) and paired t-tests for latency.
Calculates raw p-values, Holm-Bonferroni adjusted p-values, W-statistics, Z-scores, effect sizes (r), and significance flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from scipy import stats


def align_by_question_id(
    baseline_evals: List[Dict[str, Any]],
    ablation_evals: List[Dict[str, Any]],
    metric: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Strictly align baseline and ablation metric values by unique question ID ('id').

    Args:
        baseline_evals: List of baseline evaluation records.
        ablation_evals: List of ablation evaluation records.
        metric: Target metric string name.

    Returns:
        Tuple of (aligned_baseline_array, aligned_ablation_array, aligned_ids).

    Raises:
        ValueError: If duplicate question IDs exist or alignment fails.
    """
    base_map = {ev["id"]: ev[metric] for ev in baseline_evals if "id" in ev and metric in ev and ev[metric] not in ["N/A", ""]}
    ab_map = {ev["id"]: ev[metric] for ev in ablation_evals if "id" in ev and metric in ev and ev[metric] not in ["N/A", ""]}

    if len(base_map) != len(baseline_evals):
        raise ValueError(f"Duplicate or invalid question IDs detected in baseline evaluations for metric '{metric}'.")

    common_ids = sorted(list(set(base_map.keys()) & set(ab_map.keys())))
    if not common_ids:
        raise ValueError(f"No overlapping question IDs found for metric '{metric}'.")

    base_vals = np.array([float(base_map[qid]) for qid in common_ids], dtype=float)
    ab_vals = np.array([float(ab_map[qid]) for qid in common_ids], dtype=float)

    return base_vals, ab_vals, common_ids


def apply_holm_bonferroni(raw_p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction to a list of raw p-values."""
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


def compute_percentage_change(baseline_val: float, medgraphrag_val: float) -> dict[str, float]:
    """Compute percentage change and percentage point difference relative to baseline."""
    diff = medgraphrag_val - baseline_val
    rel_change = ((medgraphrag_val - baseline_val) / baseline_val * 100.0) if baseline_val != 0.0 else 0.0
    return {
        "percentage_points_difference": diff,
        "relative_percentage_change": rel_change,
    }


def compute_bootstrap_ci(data: Any, num_bootstraps: int = 1000, n_bootstrap: int | None = None, ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Compute non-parametric percentile bootstrap confidence interval."""
    if data is None or len(data) == 0:
        return (0.0, 0.0)
    n_samples = n_bootstrap if n_bootstrap is not None else num_bootstraps
    rng = np.random.default_rng(seed)
    arr = np.array(data)
    boot_means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_samples)]
    lower = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lower, upper


def run_p_tests(base_path: str = ".") -> Dict[str, Any]:
    base_dir = Path(base_path)
    results_dir = base_dir / "results"
    ablations_dir = results_dir / "ablations"

    def find_file(stem: str) -> Path:
        cand1 = ablations_dir / f"{stem}.json"
        if cand1.exists():
            return cand1
        cand2 = results_dir / f"{stem}.json"
        if cand2.exists():
            return cand2
        cand3 = base_dir / f"{stem}.json"
        if cand3.exists():
            return cand3
        return cand1

    files = {
        "Exp1_Baseline": find_file("ablation_baseline"),
        "Exp2_No_Graph": find_file("ablation_no_graph"),
        "Exp3_No_BM25": find_file("ablation_no_bm25"),
        "Exp4_No_Reranker": find_file("ablation_no_reranker"),
        "Exp5_Dense_Only": find_file("ablation_dense_only"),
    }

    data = {}
    for name, fpath in files.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Missing ablation dataset: {fpath}")
        with open(fpath, "r", encoding="utf-8") as fp:
            data[name] = json.load(fp).get("evaluations", [])

    baseline_evals = data["Exp1_Baseline"]
    n_samples = len(baseline_evals)

    # Valid metrics present in evaluation output
    metrics_to_test = [
        "Retrieval Accuracy", "Precision@5", "Recall@5", "Faithfulness",
        "Answer Relevance", "Groundedness", "Hallucination", "Explainability",
        "Clinical Reliability", "Latency"
    ]

    p_test_results = {
        "n_samples": n_samples,
        "comparisons": {}
    }

    ablation_keys = ["Exp2_No_Graph", "Exp3_No_BM25", "Exp4_No_Reranker", "Exp5_Dense_Only"]

    for ab_key in ablation_keys:
        ab_evals = data[ab_key]
        comp_dict = {}

        raw_p_vals = []
        metrics_computed = []

        for metric in metrics_to_test:
            try:
                base_vals, ab_vals, aligned_ids = align_by_question_id(baseline_evals, ab_evals, metric)
            except ValueError:
                continue

            n_aligned = len(base_vals)
            diffs = base_vals - ab_vals
            non_zero_diffs = diffs[diffs != 0]

            if metric == "Latency":
                t_stat, p_val = stats.ttest_rel(base_vals, ab_vals)
                stat_val = float(t_stat)
                test_type = "Paired t-test"
                z_score = float(t_stat / np.sqrt(n_aligned))
            else:
                test_type = "Wilcoxon signed-rank"
                if len(non_zero_diffs) == 0:
                    stat_val = 0.0
                    p_val = 1.0
                    z_score = 0.0
                else:
                    try:
                        w_stat, p_val = stats.wilcoxon(base_vals, ab_vals, alternative="two-sided")
                        stat_val = float(w_stat)
                        n_nz = len(non_zero_diffs)
                        mean_w = n_nz * (n_nz + 1) / 4.0
                        var_w = n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0
                        z_score = float((w_stat - mean_w) / np.sqrt(var_w)) if var_w > 0 else 0.0
                    except Exception:
                        stat_val = 0.0
                        p_val = 1.0
                        z_score = 0.0

            raw_p_vals.append(float(p_val))
            metrics_computed.append((metric, base_vals, ab_vals, diffs, test_type, stat_val, z_score, n_aligned))

        # Holm-Bonferroni Multiple Comparison Adjustment
        adj_p_vals = apply_holm_bonferroni(raw_p_vals)

        for i, (metric, base_vals, ab_vals, diffs, test_type, stat_val, z_score, n_aligned) in enumerate(metrics_computed):
            p_val = raw_p_vals[i]
            adj_p = adj_p_vals[i]

            if adj_p < 0.001:
                sig_flag = "***"
                significance = "Extremely Significant (p < 0.001)"
            elif adj_p < 0.01:
                sig_flag = "**"
                significance = "Highly Significant (p < 0.01)"
            elif adj_p < 0.05:
                sig_flag = "*"
                significance = "Statistically Significant (p < 0.05)"
            else:
                sig_flag = "n.s."
                significance = "Not Significant (p >= 0.05)"

            effect_size_r = float(abs(z_score) / np.sqrt(n_aligned))

            comp_dict[metric] = {
                "n_aligned": n_aligned,
                "baseline_mean": float(np.mean(base_vals)),
                "baseline_std": float(np.std(base_vals)),
                "ablation_mean": float(np.mean(ab_vals)),
                "ablation_std": float(np.std(ab_vals)),
                "mean_difference": float(np.mean(diffs)),
                "test_type": test_type,
                "test_statistic": stat_val,
                "p_value": p_val,
                "p_value_adjusted_holm": adj_p,
                "z_score": z_score,
                "effect_size_r": effect_size_r,
                "significance_flag": sig_flag,
                "interpretation": significance
            }

        p_test_results["comparisons"][ab_key] = comp_dict

    out_dir = base_dir / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "p_test_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(p_test_results, fp, indent=2)

    print(f"P-test evaluation completed with strict ID alignment and Holm-Bonferroni correction. Saved to {out_file} across {n_samples} samples.")

    return p_test_results


run_statistical_suite = run_p_tests


if __name__ == "__main__":
    res = run_p_tests()
