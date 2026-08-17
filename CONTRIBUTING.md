# Contributing to MedGraphRAG

We welcome contributions to MedGraphRAG! Please review the guidelines below before submitting a pull request or issue.

## Development Setup

```bash
git clone https://github.com/khushal15jain/MegGraphRAG.git
cd MegGraphRAG
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_reproducibility.py tests/test_publication_reproducibility.py
```

## Pull Request Guidelines

1. Ensure all 9 unit and publication reproducibility tests pass:
   `pytest`
2. Maintain metric consistency across JSON output files and documentation.
3. Keep code modular across `retrieval/`, `graph/`, `generator/`, `embeddings/`, and `evaluation/`.
4. Follow standard PEP 8 formatting.
