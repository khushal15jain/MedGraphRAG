"""generate_publication_figures.py
-------------------------------
Publication Figure Generator for MedGraphRAG.

Consumes canonical data directly from:
  - results/publication_results.json
  - results/statistical_tests.json

Generates publication-quality figures in figures/ and results/figures/:
  1. main_comparison.png (radar chart)
  2. ablation.png (ablation performance comparisons with Holm-adjusted significance stars)
  3. retrieval_comparison.png
  4. faithfulness_comparison.png
  5. latency_comparison.png
  6. statistical_significance.png
"""

from __future__ import annotations

import json
import os
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

MODES = [
    ("baseline", "Baseline"),
    ("no_graph", "No Graph"),
    ("no_bm25", "No BM25"),
    ("no_reranker", "No Reranker"),
    ("dense_only", "Dense Only"),
]

METRICS_TO_TEST = [
    "Retrieval Accuracy",
    "Precision@5",
    "Recall@5",
    "Faithfulness",
    "Answer Relevance",
    "Groundedness",
    "Hallucination",
    "Explainability",
    "Clinical Reliability",
    "Latency",
]


def load_publication_results():
    pub_path = BASE_DIR / "results" / "publication_results.json"
    if not pub_path.exists():
        raise FileNotFoundError(f"Missing {pub_path}. Run reproduce_publication_results.py first.")

    with open(pub_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    stats_path = BASE_DIR / "results" / "statistical_tests.json"
    stats_data = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats_data = json.load(f)

    return manifest, stats_data


def plot_radar_chart(manifest, filename):
    conds = manifest["conditions"]
    categories = [
        "Retrieval Acc.",
        "Precision@5",
        "Recall@5",
        "Faithfulness",
        "Answer Rel.",
        "Groundedness",
        "Explainability",
        "Clinical Rel.",
    ]
    raw_keys = [
        "Retrieval Accuracy",
        "Precision@5",
        "Recall@5",
        "Faithfulness",
        "Answer Relevance",
        "Groundedness",
        "Explainability",
        "Clinical Reliability",
    ]

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, color="grey", size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)

    colors = {
        "baseline": "#1f77b4",
        "no_graph": "#ff7f0e",
        "no_bm25": "#2ca02c",
        "no_reranker": "#d62728",
        "dense_only": "#9467bd",
    }

    for fname, label in MODES:
        if fname not in conds:
            continue
        vals = [conds[fname]["metrics"].get(k, {}).get("mean", 0.0) for k in raw_keys]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, linestyle="solid", label=label, color=colors.get(fname))
        ax.fill(angles, vals, color=colors.get(fname), alpha=0.1)

    plt.title("MedGraphRAG Tri-Modal Architecture vs. Ablation Modes", size=13, color="black", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()

    out_dirs = [BASE_DIR / "figures", BASE_DIR / "results" / "figures", BASE_DIR / "results" / "charts"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        plt.savefig(d / filename, dpi=300)
    plt.close()


def plot_bar_chart(metric, manifest, stats_data, filename):
    conds = manifest["conditions"]
    labels = [label for fname, label in MODES if fname in conds]
    means = [conds[fname]["metrics"].get(metric, {}).get("mean", 0.0) for fname, label in MODES if fname in conds]
    stds = [conds[fname]["metrics"].get(metric, {}).get("std", 0.0) for fname, label in MODES if fname in conds]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(labels))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    bars = ax.bar(x_pos, means, yerr=stds, align="center", alpha=0.85, ecolor="black", capsize=5, color=colors)

    ax.set_ylabel(metric)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_title(f"{metric} Comparison across Ablation Conditions (N=100)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Annotate bar heights and Holm-adjusted significance stars from stats_data
    for i, bar in enumerate(bars):
        height = bar.get_height()
        fname = MODES[i][0]
        sig_label = ""
        if fname != "baseline" and fname in stats_data and metric in stats_data[fname]:
            sig_label = " " + stats_data[fname][metric].get("significance_label", "")

        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            1.02 * height,
            f"{height:.4f}{sig_label}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_dirs = [BASE_DIR / "figures", BASE_DIR / "results" / "figures", BASE_DIR / "results" / "charts"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        plt.savefig(d / filename, dpi=300)
    plt.close()


def main():
    manifest, stats_data = load_publication_results()
    print("Generating publication figures directly from canonical publication_results.json...")

    plot_radar_chart(manifest, "radar_chart.png")
    plot_radar_chart(manifest, "main_comparison.png")

    plot_bar_chart("Retrieval Accuracy", manifest, stats_data, "retrieval_accuracy_chart.png")
    plot_bar_chart("Faithfulness", manifest, stats_data, "faithfulness_chart.png")
    plot_bar_chart("Groundedness", manifest, stats_data, "groundedness_chart.png")
    plot_bar_chart("Hallucination", manifest, stats_data, "hallucination_chart.png")
    plot_bar_chart("Clinical Reliability", manifest, stats_data, "clinical_reliability_chart.png")
    plot_bar_chart("Latency", manifest, stats_data, "latency_chart.png")

    print("All publication figures successfully generated.")


if __name__ == "__main__":
    main()
