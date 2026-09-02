import pytest
from medgraphrag.evaluation.metrics import (
    compute_answer_f1,
    bleu_n,
    rouge_1,
    rouge_2,
    rouge_l,
    meteor,
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    compute_mrr,
    compute_ndcg,
    _extract_clean_text,
)


def test_identical_strings_metrics():
    text = "The supraglottic larynx, glottis, and subglottis are the three main anatomical divisions."
    assert compute_answer_f1(text, text) == 1.0
    assert rouge_1(text, text) == 1.0
    assert rouge_2(text, text) == 1.0
    assert rouge_l(text, text) == 1.0
    assert bleu_n(text, text, 1) > 0.95
    assert meteor(text, text) > 0.95


def test_disjoint_strings_metrics():
    cand = "Alpha beta gamma delta."
    ref = "One two three four."
    assert compute_answer_f1(cand, ref) == 0.0
    assert rouge_1(cand, ref) == 0.0
    assert rouge_2(cand, ref) == 0.0
    assert rouge_l(cand, ref) == 0.0


def test_clean_text_extraction():
    raw_json = '{"id": "Q1", "answer": "The glottis is the middle section.", "sources": ["P1"]}\nConfidence: High'
    cleaned = _extract_clean_text(raw_json)
    assert cleaned == "The glottis is the middle section."


def test_retrieval_ranking_metrics():
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc4"]
    assert precision_at_k(retrieved, relevant, 5) == 2 / 5
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 5) == 1.0
    assert compute_mrr(retrieved, relevant) == 0.5
    assert compute_ndcg(retrieved, relevant, 5) > 0.0
