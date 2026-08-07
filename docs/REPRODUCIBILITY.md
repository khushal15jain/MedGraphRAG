# MedGraphRAG — Reproducibility Guide

This guide gives exact, ordered steps to reproduce every result reported in
`docs/MedGraphRAG_IEEE_Paper.md`, from a clean checkout to final benchmark tables.

## 1. Environment

```bash
# Requires Python 3.11 exactly (see pyproject.toml)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify installation:
```bash
python -c "import torch, chromadb, spacy, scispacy, sentence_transformers; print('OK')"
python -m spacy validate
```

## 2. Local LLM setup

```bash
# Install Ollama: https://ollama.com/download
ollama serve &                      # start the daemon (if not already running)
ollama pull qwen2.5:3b-instruct     # ~2GB download
ollama list                          # confirm the model is present
```

## 3. Data

Place the six source oncology textbook PDFs in `data/raw/`. This repository
does not redistribute textbook content — you must supply your own licensed
copies. File naming does not matter; `PDFLoader.load_directory()` picks up
every `*.pdf` in the directory.

## 4. Configuration

```bash
cp .env.example .env
```

Review `configs/model.yaml` and `configs/retrieval.yaml` — all
hyperparameters reported in the paper's experimental setup section
(chunk sizes, `hybrid_alpha`, `top_k` values, temperature) are defined here
and were not changed between the described experiments unless explicitly
called out as an ablation.

## 5. Run the ingestion pipeline

```bash
python main.py
```

This is deterministic given the same input PDFs and config (chunking and
extraction are rule-/model-based, not sampled), except for negligible
floating-point non-determinism in the embedding model's batched inference.
Expected outputs:
- `data/processed/chunks.jsonl`, `entities.jsonl`, `relations.jsonl`
- `outputs/knowledge_graph.gpickle`
- `outputs/chroma_db/` (persistent vector store)

Approximate runtime on a MacBook Air M2 (8GB): ~15–40 minutes for six
textbooks, dominated by SciSpaCy NER and embedding generation. Peak memory
should stay under 6GB if `batch_size` values in `configs/model.yaml` are
left at their defaults.

## 6. Build the evaluation question set

The paper's benchmark uses a held-out set of oncology clinical questions
with reference answers. Construct `EvalQuestion` instances (see
`benchmark/run_benchmark.py`) — for full reproducibility, use the same
question set described in the paper's Appendix (a JSONL file of
`{"question": ..., "ground_truth": ...}` records, loaded via
`utils.io_utils.read_jsonl` and mapped to `EvalQuestion`).

## 7. Run the full benchmark

```python
from benchmark.run_benchmark import BenchmarkRunner, EvalQuestion
from benchmark.baselines import (
    VanillaRAGMethod, BM25Method, HybridRetrievalMethod, GraphRAGMethod, ProposedMethod,
)
# ... construct retrievers/generator/evaluators per docs/ARCHITECTURE.md,
# then instantiate each baseline method and call runner.run_all(methods, questions)
```

Or via the convenience script:
```bash
python -m benchmark.run_benchmark
```

Outputs:
- `outputs/benchmark_results/<method_name>.csv` — per-question results
- `outputs/benchmark_results/summary.csv` — aggregated metrics per method (the paper's main results table)
- `outputs/mlruns/` — MLflow experiment tracking data; view with `mlflow ui --backend-store-uri outputs/mlruns`

## 8. Ablation studies

Ablations are produced by constructing `ProposedMethod`/`HybridRetrievalMethod`
variants with individual components toggled:

| Ablation | How to construct |
|---|---|
| No reranking | Use `HybridRetrievalMethod` output directly (skips Stage 12) |
| No graph expansion | `hybrid_retriever.retrieve(..., use_graph=False)` |
| No hybrid (dense only) | `VanillaRAGMethod` |
| No hybrid (BM25 only) | `BM25Method` |
| Full proposed method | `ProposedMethod` |

Run each variant through `BenchmarkRunner.run_method()` with the same
question set and compare rows in the resulting summary table.

## 9. Figures

```python
from evaluation.visualization import (
    plot_knowledge_graph_html, plot_benchmark_comparison_matplotlib, plot_ablation_interactive,
)
# graph = load_pickle("outputs/knowledge_graph.gpickle")
# plot_knowledge_graph_html(graph)
# plot_benchmark_comparison_matplotlib(summary_df, metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"])
```

## 10. Unit tests

```bash
pytest
```

All tests run without downloading model weights (heavy dependencies are
replaced with lightweight fakes — see `tests/test_entity_extraction.py`,
`tests/test_retrieval.py`) and complete in well under a minute.

## Known sources of non-determinism

1. **LLM generation**: `temperature=0.2` (not 0) means generated answers
   vary slightly run-to-run. `RagasEvaluator`/`DeepEvalEvaluator` scores
   should be interpreted as means over the evaluation set, not exact
   per-question reproductions.
2. **LLM-as-judge metrics** (RAGAS, DeepEval): using a 3B local judge
   model rather than a larger model introduces judge noise; we report this
   explicitly as a limitation in the paper rather than presenting these
   scores as ground truth.
3. **Heading detection** (`MetadataExtractor.detect_headings`): a
   font-size heuristic; textbooks with unusual typography may produce
   different heading counts than reported.

## Hardware validation

All steps above were designed and load-tested against an 8GB memory
budget by running each pipeline stage as a separate process invocation
(rather than one long-lived process holding every model in memory at
once) — see `main.py`, which loads and releases the SciSpaCy pipeline
before loading the embedding model, etc. The only place all models coexist
in one process is `app/api.py` at serving time; if resource-constrained,
reduce `configs/model.yaml` batch sizes further or run ingestion and
serving as separate process invocations (never simultaneously) on the same
8GB machine.
