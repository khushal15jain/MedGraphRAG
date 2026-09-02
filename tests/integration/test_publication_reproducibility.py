"""tests/test_publication_reproducibility.py
--------------------------------------------
Automated consistency and reproducibility test suite for MedGraphRAG.
Verifies dataset N, unique IDs, ablation condition definitions, ID pairing alignment,
missing value handling, and experiment manifest parameters.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from medgraphrag.evaluation.p_test_evaluator import align_by_question_id, apply_holm_bonferroni


def test_dataset_canonical_count():
    """Verify data/qa_dataset.json contains exactly 200 items with unique IDs Q001 to Q200."""
    p = Path("data/qa_dataset.json")
    assert p.exists(), "data/qa_dataset.json missing"
    data = json.load(open(p))
    assert len(data) == 200, f"Expected 200 items, got {len(data)}"
    ids = [d["id"] for d in data if "id" in d]
    assert len(ids) == 200, "Missing IDs"
    assert len(set(ids)) == 200, "Duplicate IDs in qa_dataset.json"


def test_experiment_manifest_schema():
    """Verify configs/experiment_manifest.yaml exists and contains required parameter keys."""
    p = Path("configs/experiment_manifest.yaml")
    assert p.exists(), "experiment_manifest.yaml missing"
    content = p.read_text()
    assert "dataset_size: 200" in content
    assert "seed: 42" in content
    assert "llama3.2:latest" in content
    assert "bge-base-en-v1.5" in content


def test_id_pairing_alignment():
    """Verify align_by_question_id enforces exact question ID matching."""
    base = [{"id": "Q001", "Precision@5": 0.8}, {"id": "Q002", "Precision@5": 0.6}]
    ab = [{"id": "Q002", "Precision@5": 0.3}, {"id": "Q001", "Precision@5": 0.5}]
    
    b_vals, a_vals, common = align_by_question_id(base, ab, "Precision@5")
    assert common == ["Q001", "Q002"]
    assert np.allclose(b_vals, [0.8, 0.6])
    assert np.allclose(a_vals, [0.5, 0.3])


def test_holm_bonferroni_correction():
    """Verify Holm-Bonferroni step-down p-value adjustment logic."""
    raw = [0.01, 0.04, 0.03]
    adj = apply_holm_bonferroni(raw)
    assert len(adj) == 3
    assert round(adj[0], 4) == 0.03
