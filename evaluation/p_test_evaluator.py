"""evaluation/p_test_evaluator.py
------------------------------
Statistical Hypothesis Testing Module (p-test Evaluation Suite).
Computes paired two-tailed Wilcoxon signed-rank tests (non-parametric)
and paired t-tests for latency across all ablation configurations vs Baseline.
Calculates p-values, W-statistics, Z-scores, effect sizes (r), and significance flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from scipy import stats


def run_p_tests(base_path: str = ".") -> Dict[str, Any]:
    base_dir = Path(base_path)
    files = {
        "Exp1_Baseline": base_dir / "ablation_baseline.json",
        "Exp2_No_Graph": base_dir / "ablation_no_graph.json",
        "Exp3_No_BM25": base_dir / "ablation_no_bm25.json",
        "Exp4_No_Reranker": base_dir / "ablation_no_reranker.json",
        "Exp5_Dense_Only": base_dir / "ablation_dense_only.json",
    }

    data = {}
    for name, fpath in files.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Missing ablation dataset: {fpath}")
        with open(fpath, "r", encoding="utf-8") as fp:
            data[name] = json.load(fp).get("evaluations", [])

    baseline_evals = data["Exp1_Baseline"]
    n_samples = len(baseline_evals)

    # Metrics to test
    metrics_to_test = [
        "Retrieval Accuracy", "Precision@5", "Recall@5", "Faithfulness",
        "Answer Relevance", "Groundedness", "Hallucination", "Explainability",
        "Clinical Reliability", "MRR", "NDCG@5", "HitRate@5", "BLEU-1",
        "BLEU-2", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR",
        "Answer F1", "Safety", "Completeness", "Originality", "Precision",
        "Efficiency", "Overall", "Latency"
    ]

    p_test_results = {
        "n_samples": n_samples,
        "comparisons": {}
    }

    ablation_keys = ["Exp2_No_Graph", "Exp3_No_BM25", "Exp4_No_Reranker", "Exp5_Dense_Only"]

    for ab_key in ablation_keys:
        ab_evals = data[ab_key]
        comp_dict = {}

        for metric in metrics_to_test:
            base_vals = np.array([ev.get(metric, 0.0) for ev in baseline_evals if metric in ev])
            ab_vals = np.array([ev.get(metric, 0.0) for ev in ab_evals if metric in ev])

            if len(base_vals) == 0 or len(ab_vals) == 0 or len(base_vals) != len(ab_vals):
                continue

            diffs = base_vals - ab_vals
            non_zero_diffs = diffs[diffs != 0]

            if metric == "Latency":
                # Continuous metric: Paired Student's t-test
                t_stat, p_val = stats.ttest_rel(base_vals, ab_vals)
                stat_val = float(t_stat)
                test_type = "Paired t-test"
                z_score = float(t_stat / np.sqrt(n_samples))
            else:
                # Non-parametric metrics: Wilcoxon signed-rank test
                if len(non_zero_diffs) == 0:
                    stat_val = 0.0
                    p_val = 1.0
                    z_score = 0.0
                else:
                    try:
                        w_stat, p_val = stats.wilcoxon(base_vals, ab_vals, alternative="two-sided")
                        stat_val = float(w_stat)
                        # Normal approximation for effect size Z
                        n_nz = len(non_zero_diffs)
                        mean_w = n_nz * (n_nz + 1) / 4.0
                        var_w = n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0
                        z_score = float((w_stat - mean_w) / np.sqrt(var_w)) if var_w > 0 else 0.0
                    except Exception:
                        stat_val = 0.0
                        p_val = 1.0
                        z_score = 0.0

            # Significance notation
            if p_val < 0.001:
                sig_flag = "***"
                significance = "Extremely Significant (p < 0.001)"
            elif p_val < 0.01:
                sig_flag = "**"
                significance = "Highly Significant (p < 0.01)"
            elif p_val < 0.05:
                sig_flag = "*"
                significance = "Statistically Significant (p < 0.05)"
            else:
                sig_flag = "n.s."
                significance = "Not Significant (p >= 0.05)"

            effect_size_r = float(abs(z_score) / np.sqrt(n_samples))

            comp_dict[metric] = {
                "baseline_mean": float(np.mean(base_vals)),
                "baseline_std": float(np.std(base_vals)),
                "ablation_mean": float(np.mean(ab_vals)),
                "ablation_std": float(np.std(ab_vals)),
                "mean_difference": float(np.mean(diffs)),
                "test_type": test_type if metric == "Latency" else "Wilcoxon signed-rank",
                "test_statistic": stat_val,
                "p_value": float(p_val),
                "z_score": z_score,
                "effect_size_r": effect_size_r,
                "significance_flag": sig_flag,
                "interpretation": significance
            }

        p_test_results["comparisons"][ab_key] = comp_dict

    out_file = base_dir / "p_test_results.json"
    with open(out_file, "w", encoding="utf-8") as fp:
        json.dump(p_test_results, fp, indent=2)

    return p_test_results


if __name__ == "__main__":
    res = run_p_tests()
    print(f"P-test evaluation completed. Saved results to p_test_results.json across {res['n_samples']} samples.")
