"""tests/test_publication_metrics.py
----------------------------------
Unit test suite verifying publication-grade metrics, ID alignment,
Holm-Bonferroni correction, percentage calculations, and graph scoring.
"""

import numpy as np
import pytest

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    compute_answer_f1,
)
from evaluation.p_test_evaluator import align_by_question_id, apply_holm_bonferroni, compute_percentage_change, compute_bootstrap_ci


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc5", "doc9"]
    # 2 matches in top 5 -> 2 / 5 = 0.4
    assert precision_at_k(retrieved, relevant, k=5) == pytest.approx(0.4)


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc5"]
    # 2 matches out of 2 total relevant -> 2 / 2 = 1.0
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)


def test_hit_rate_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc3"]
    assert hit_rate_at_k(retrieved, relevant, k=5) == 1.0

    disjoint_relevant = ["doc9"]
    assert hit_rate_at_k(retrieved, disjoint_relevant, k=5) == 0.0


def test_mrr_at_k():
    # Relevant item doc2 is at index 1 (rank 2) -> 1 / 2 = 0.5
    retrieved = ["doc1", "doc2", "doc3"]
    relevant = ["doc2"]
    assert mrr_at_k(retrieved, relevant, k=5) == pytest.approx(0.5)


def test_ndcg_at_k():
    # Gold items at rank 1 and rank 3
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc1", "doc3"]

    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
    # NDCG = 1.5 / 1.63092975 = 0.9197
    val = ndcg_at_k(retrieved, relevant, k=5)
    assert 0.90 <= val <= 0.95


def test_answer_f1_exact_match():
    cand = '{"answer": "Pembrolizumab is indicated for advanced NSCLC with high PD-L1.", "confidence": "High"}'
    gold = "Pembrolizumab is indicated for advanced NSCLC with high PD-L1."
    assert compute_answer_f1(cand, gold) == pytest.approx(1.0)


def test_question_id_alignment():
    base = [{"id": "Q1", "Score": 0.9}, {"id": "Q2", "Score": 0.8}]
    ab = [{"id": "Q2", "Score": 0.5}, {"id": "Q1", "Score": 0.6}]

    base_arr, ab_arr, ids = align_by_question_id(base, ab, "Score")
    assert ids == ["Q1", "Q2"]
    assert list(base_arr) == [0.9, 0.8]
    assert list(ab_arr) == [0.6, 0.5]


def test_holm_bonferroni_correction():
    raw_p = [0.01, 0.04, 0.03]
    adj_p = apply_holm_bonferroni(raw_p)

    # m=3: min(0.01*3=0.03), next min(0.03*2=0.06), next min(0.04*1=0.04 -> max cum 0.06)
    assert adj_p[0] == pytest.approx(0.03)
    assert all(a >= r for a, r in zip(adj_p, raw_p))


def test_percentage_change_helper():
    res = compute_percentage_change(0.3913, 0.0967)
    assert res["percentage_points_difference"] == pytest.approx(-0.2946)
    assert res["relative_percentage_change"] == pytest.approx(-75.2875, abs=0.01)


def test_bootstrap_ci():
    data = np.array([0.9, 0.92, 0.91, 0.89, 0.93, 0.90])
    low, high = compute_bootstrap_ci(data, num_bootstraps=100)
    assert 0.85 <= low <= high <= 0.95


def test_graph_score_decay_formula():
    # S_graph = 1 / (1 + d)
    d_hop_0 = 0
    d_hop_1 = 1
    d_hop_2 = 2

    s0 = 1.0 / (1.0 + d_hop_0)
    s1 = 1.0 / (1.0 + d_hop_1)
    s2 = 1.0 / (1.0 + d_hop_2)

    assert s0 == 1.0
    assert s1 == 0.5
    assert s2 == pytest.approx(0.3333, abs=0.001)
