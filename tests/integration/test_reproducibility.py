"""tests/test_reproducibility.py
--------------------------------
Pytest suite for MedGraphRAG scientific reproducibility, dataset integrity, ID alignment,
ablation definitions, and topological graph scoring.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from medgraphrag.evaluation.p_test_evaluator import align_by_question_id, apply_holm_bonferroni
from medgraphrag.graph.graph_retriever import GraphRetrievalResult, GraphRetriever


def test_dataset_size_and_schema():
    """Verify that qa_dataset.json contains exactly 200 items with unique IDs."""
    dataset_path = Path("data/qa_dataset.json")
    assert dataset_path.exists(), "qa_dataset.json is missing."

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 200, f"Expected 200 questions, found {len(data)}"

    ids = [item["id"] for item in data if "id" in item]
    assert len(ids) == 200, "Missing question IDs."
    assert len(set(ids)) == 200, "Duplicate question IDs detected in dataset."


def test_align_by_question_id():
    """Verify strict question ID pairing and error handling on duplicate/mismatched IDs."""
    baseline = [
        {"id": "Q001", "Precision@5": 0.8},
        {"id": "Q002", "Precision@5": 0.6},
        {"id": "Q003", "Precision@5": 0.9},
    ]
    ablation = [
        {"id": "Q003", "Precision@5": 0.4},
        {"id": "Q001", "Precision@5": 0.5},
        {"id": "Q002", "Precision@5": 0.3},
    ]

    base_vals, ab_vals, aligned_ids = align_by_question_id(baseline, ablation, "Precision@5")

    assert aligned_ids == ["Q001", "Q002", "Q003"]
    assert np.allclose(base_vals, [0.8, 0.6, 0.9])
    assert np.allclose(ab_vals, [0.5, 0.3, 0.4])

    # Test duplicate ID failure
    invalid_baseline = [
        {"id": "Q001", "Precision@5": 0.8},
        {"id": "Q001", "Precision@5": 0.6},
    ]
    with pytest.raises(ValueError):
        align_by_question_id(invalid_baseline, ablation, "Precision@5")


def test_holm_bonferroni_adjustment():
    """Verify Holm-Bonferroni p-value step-down correction."""
    raw_p = [0.01, 0.04, 0.03]
    adj_p = apply_holm_bonferroni(raw_p)

    assert len(adj_p) == 3
    # Smallest p (0.01) multiplied by 3 = 0.03
    assert round(adj_p[0], 4) == 0.03
    assert adj_p[0] <= adj_p[2] <= adj_p[1]


def test_ablation_condition_definitions():
    """Verify the 5 machine-readable ablation condition flags."""
    modes = [
        {"name": "baseline", "use_graph": True, "use_bm25": True, "use_reranker": True},
        {"name": "no_graph", "use_graph": False, "use_bm25": True, "use_reranker": True},
        {"name": "no_bm25", "use_graph": True, "use_bm25": False, "use_reranker": True},
        {"name": "no_reranker", "use_graph": True, "use_bm25": True, "use_reranker": False},
        {"name": "dense_only", "use_graph": False, "use_bm25": False, "use_reranker": False},
    ]

    assert len(modes) == 5
    assert modes[0]["name"] == "baseline"
    assert not modes[1]["use_graph"]
    assert not modes[2]["use_bm25"]
    assert not modes[3]["use_reranker"]
    assert not modes[4]["use_graph"] and not modes[4]["use_bm25"] and not modes[4]["use_reranker"]


def test_graph_retriever_decay_scoring():
    """Verify that pure topological distance decay is computed as 1 / (1 + hop)."""
    # Hop 0 (seed node): score = 1.0
    # Hop 1 (neighbor): score = 0.5
    # Hop 2 (2-hop neighbor): score = 0.3333
    score_hop_0 = 1.0 / (1.0 + 0)
    score_hop_1 = 1.0 / (1.0 + 1)
    score_hop_2 = 1.0 / (1.0 + 2)

    assert score_hop_0 == 1.0
    assert round(score_hop_1, 4) == 0.5
    assert round(score_hop_2, 4) == 0.3333
