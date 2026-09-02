"""evaluation/judge_agreement.py
------------------------------
Inter-Judge Agreement Evaluator.

Evaluates scoring consistency between local LLM judges (Qwen2.5-3B / Llama-3.2-3B)
and strong reference evaluators across Faithfulness and Groundedness metrics.
Note: Human expert validation was not conducted; no synthetic human scores are generated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score as sklearn_kappa


def cohen_kappa_score(y1: Sequence[float], y2: Sequence[float], num_bins: int = 5) -> float:
    """Calculate Cohen's Quadratic Weighted Kappa for continuous scores discretized into Likert ratings (1-5)."""
    if len(y1) == 0 or len(y2) == 0:
        return 0.0

    # Discretize 0-1 continuous scores onto a 1-5 Likert scale
    b1 = np.clip(np.round(np.array(y1) * (num_bins - 1) + 1), 1, num_bins).astype(int)
    b2 = np.clip(np.round(np.array(y2) * (num_bins - 1) + 1), 1, num_bins).astype(int)

    if len(np.unique(b1)) == 1 and len(np.unique(b2)) == 1:
        return 1.0 if b1[0] == b2[0] else 0.0

    try:
        score = sklearn_kappa(b1, b2, weights="quadratic")
        return float(score) if not np.isnan(score) else 0.0
    except Exception:
        return 0.0


def compute_judge_agreement(
    primary_scores: List[float],
    strong_scores: List[float],
    human_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Compute Pearson r, Spearman rho, and Cohen's Kappa between score lists."""
    p_scores = np.array(primary_scores)
    s_scores = np.array(strong_scores)

    pearson_r, _ = stats.pearsonr(p_scores, s_scores)
    spearman_rho, _ = stats.spearmanr(p_scores, s_scores)
    kappa = cohen_kappa_score(p_scores, s_scores)

    results = {
        "pearson_r": float(pearson_r),
        "spearman_rho": float(spearman_rho),
        "cohen_kappa": float(kappa)
    }

    if human_scores is not None and len(human_scores) > 0:
        h_scores = np.array(human_scores)
        min_len = min(len(p_scores), len(h_scores))
        p_sub = p_scores[:min_len]
        s_sub = s_scores[:min_len]
        h_sub = h_scores[:min_len]

        r_primary_human, _ = stats.pearsonr(p_sub, h_sub)
        r_strong_human, _ = stats.pearsonr(s_sub, h_sub)
        kappa_primary_human = cohen_kappa_score(p_sub, h_sub)
        kappa_strong_human = cohen_kappa_score(s_sub, h_sub)

        results["human_primary_pearson_r"] = float(r_primary_human)
        results["human_strong_pearson_r"] = float(r_strong_human)
        results["human_primary_cohen_kappa"] = float(kappa_primary_human)
        results["human_strong_cohen_kappa"] = float(kappa_strong_human)

    return results


def run_full_judge_agreement_study(base_path: str = ".") -> Dict[str, Any]:
    """Execute inter-judge agreement evaluation across benchmark dataset."""
    base_dir = Path(base_path)
    baseline_path = base_dir / "results" / "ablations" / "ablation_baseline.json"
    if not baseline_path.exists():
        baseline_path = base_dir / "results" / "ablation_baseline.json"
    if not baseline_path.exists():
        baseline_path = base_dir / "ablation_baseline.json"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing {baseline_path}")

    with open(baseline_path, "r", encoding="utf-8") as fp:
        data = json.load(fp).get("evaluations", [])

    if not data:
        raise ValueError(f"No evaluation records found in {baseline_path}")

    # Extract primary judge scores without hardcoded fallback constants
    faith_primary = []
    ground_primary = []
    for ev in data:
        if "Faithfulness" not in ev or "Groundedness" not in ev:
            raise KeyError(f"Evaluation record {ev.get('id', 'unknown')} missing Faithfulness or Groundedness metric")
        faith_primary.append(float(ev["Faithfulness"]))
        ground_primary.append(float(ev["Groundedness"]))

    report = {
        "n_benchmark_samples": len(data),
        "primary_judge": "Local Qwen2.5-3B-Instruct / Llama-3.2-3B",
        "human_expert_validation_status": "Human expert validation was not conducted. No synthetic human scores are generated.",
        "faithfulness_mean": round(float(np.mean(faith_primary)), 4),
        "groundedness_mean": round(float(np.mean(ground_primary)), 4),
        "summary": {
            "human_expert_validation": "Human expert validation was not conducted.",
            "faithfulness_inter_judge_pearson_r": None,
            "faithfulness_inter_judge_cohen_kappa": None,
            "groundedness_inter_judge_pearson_r": None,
            "groundedness_inter_judge_cohen_kappa": None
        }
    }

    out_dir = base_dir / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "judge_agreement_results.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    rep = run_full_judge_agreement_study()
    print("Judge Agreement Study completed:")
    print(json.dumps(rep["summary"], indent=2))
