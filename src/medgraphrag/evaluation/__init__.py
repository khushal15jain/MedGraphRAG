"""evaluation
----------
Evaluation and metric calculation module for MedGraphRAG.
"""

from medgraphrag.evaluation.metrics import (
    answer_f1,
    bleu_n,
    compute_answer_f1,
    compute_bleu,
    compute_meteor,
    compute_mrr,
    compute_ndcg,
    compute_rouge_1,
    compute_rouge_2,
    compute_rouge_l,
    hit_rate_at_k,
    meteor,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rouge_1,
    rouge_2,
    rouge_l,
)
from medgraphrag.evaluation.p_test_evaluator import align_by_question_id, apply_holm_bonferroni

__all__ = [
    "answer_f1",
    "bleu_n",
    "compute_answer_f1",
    "compute_bleu",
    "compute_meteor",
    "compute_mrr",
    "compute_ndcg",
    "compute_rouge_1",
    "compute_rouge_2",
    "compute_rouge_l",
    "hit_rate_at_k",
    "meteor",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "align_by_question_id",
    "apply_holm_bonferroni",
]
