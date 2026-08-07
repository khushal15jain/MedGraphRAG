"""Shared pytest fixtures for MedGraphRAG unit tests.

Fixtures here avoid loading heavy models (spaCy/SentenceTransformer/Ollama)
wherever a test doesn't strictly need them, keeping the default test suite
fast enough to run in CI or on the 8GB dev laptop without downloading
gigabytes of model weights. Tests that do need a real model are marked and
can be skipped via `pytest -m "not heavy"`.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_page_texts() -> list[str]:
    """A small synthetic 'textbook page' corpus for preprocessing tests."""
    return [
        "Chapter 1: Introduction to Oncology\n\n"
        "Cancer is a group of diseases involving abnormal cell growth. "
        "Trastuzumab is used to treat HER2-positive breast cancer. "
        "Patients receiving trastuzumab should be monitored for cardiotox-\nicity.",
        "12",  # page-number-only junk line, should be stripped by cleaner
        "Chemotherapy regimens often combine multiple cytotoxic agents. "
        "Doxorubicin is commonly paired with cyclophosphamide in breast cancer treatment.",
    ]


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Pre-built chunk dicts, avoiding a dependency on the chunker in retrieval tests."""
    return [
        {
            "chunk_id": "c1",
            "text": "Trastuzumab is used to treat HER2-positive breast cancer.",
            "source_file": "onc_textbook.pdf",
            "level": "child",
        },
        {
            "chunk_id": "c2",
            "text": "Doxorubicin is commonly paired with cyclophosphamide in breast cancer treatment.",
            "source_file": "onc_textbook.pdf",
            "level": "child",
        },
        {
            "chunk_id": "c3",
            "text": "Radiation therapy uses high-energy particles to destroy cancer cells.",
            "source_file": "onc_textbook.pdf",
            "level": "child",
        },
    ]
