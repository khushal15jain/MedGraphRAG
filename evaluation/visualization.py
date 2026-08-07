"""Stage 19: Visualization.

Produces the figures needed for the research paper and interactive
exploration during development:
  - An interactive PyVis HTML visualization of the (sub-sampled) knowledge graph.
  - Matplotlib bar charts comparing benchmark methods across metrics (paper figures).
  - Plotly grouped bar charts for interactive exploration of ablation results.

Kept in a single module (rather than split further) since all three are
thin, stateless rendering functions over already-computed results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from pyvis.network import Network

from utils.io_utils import ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)


def plot_knowledge_graph_html(
    graph: nx.MultiDiGraph, output_path: str = "outputs/figures/knowledge_graph.html", max_nodes: int = 150
) -> Path:
    """Render an interactive HTML visualization of the knowledge graph via PyVis.

    Args:
        graph: The full ``KnowledgeGraphBuilder.graph`` NetworkX object.
        output_path: Destination HTML file path.
        max_nodes: The full graph from six textbooks can have thousands of
            nodes, which renders unusably in a browser; this subsamples the
            highest-degree (most-connected) nodes for a legible overview figure.

    Returns:
        Path to the written HTML file.
    """
    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    if graph.number_of_nodes() > max_nodes:
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        subgraph = graph.subgraph(top_nodes).copy()
        logger.info(f"Subsampled graph to top {max_nodes} highest-degree nodes for visualization")
    else:
        subgraph = graph

    net = Network(height="800px", width="100%", directed=True, notebook=False, bgcolor="#ffffff")
    net.barnes_hut()

    for node, data in subgraph.nodes(data=True):
        size = 10 + min(data.get("mention_count", 1), 50)
        net.add_node(node, label=node, title=f"mentions: {data.get('mention_count', 0)}", size=size)

    for u, v, data in subgraph.edges(data=True):
        net.add_edge(u, v, title=data.get("predicate", ""), value=data.get("weight", 1))

    net.write_html(str(out_path))
    logger.info(f"Wrote interactive knowledge graph visualization to {out_path}")
    return out_path


def plot_benchmark_comparison_matplotlib(
    summary_df: pd.DataFrame,
    metrics: list[str],
    output_path: str = "outputs/figures/benchmark_comparison.png",
) -> Path:
    """Render a grouped bar chart comparing methods across metrics (static, for the paper).

    Args:
        summary_df: DataFrame with a "method" column and one column per metric
            (as produced by ``BenchmarkRunner.run_all``).
        metrics: Which metric columns to plot (e.g. ["faithfulness",
            "answer_relevancy", "context_precision", "context_recall"]).
        output_path: Destination PNG file path.

    Returns:
        Path to the written PNG file.
    """
    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    methods = summary_df["method"].tolist()
    x = range(len(methods))
    n_metrics = len(metrics)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.5), 6))
    for i, metric in enumerate(metrics):
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, summary_df[metric], width=width, label=metric)

    ax.set_xticks([xi + width * (n_metrics - 1) / 2 for xi in x])
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("MedGraphRAG: Method Comparison Across Evaluation Metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    logger.info(f"Wrote benchmark comparison figure to {out_path}")
    return out_path


def plot_ablation_interactive(
    ablation_df: pd.DataFrame, metric: str, output_path: str = "outputs/figures/ablation_study.html"
) -> Path:
    """Render an interactive Plotly bar chart of an ablation study for a single metric.

    Args:
        ablation_df: DataFrame with a "method" column and the target metric column,
            where rows are ablation variants (e.g. "Proposed", "Proposed - no rerank",
            "Proposed - no graph").
        metric: Metric column name to visualize.
        output_path: Destination HTML file path.

    Returns:
        Path to the written HTML file.
    """
    out_path = Path(output_path)
    ensure_dir(out_path.parent)

    fig = go.Figure(
        data=[go.Bar(x=ablation_df["method"], y=ablation_df[metric], marker_color="#2E86AB")]
    )
    fig.update_layout(
        title=f"Ablation Study: Effect on {metric}",
        xaxis_title="Configuration",
        yaxis_title=metric,
        template="plotly_white",
    )
    fig.write_html(str(out_path))

    logger.info(f"Wrote ablation study figure to {out_path}")
    return out_path
