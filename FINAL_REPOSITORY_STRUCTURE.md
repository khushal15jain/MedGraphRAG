# MedGraphRAG: Final Publication-Quality Repository Structure

This document details the standardized Python package layout, module organization, execution scripts, test suites, and data artifacts for the **MedGraphRAG** research repository.

---

## 📂 Repository Directory Tree

```
MedGraphRAG/
│
├── src/
│   └── medgraphrag/                 # Main Python Package
│       ├── __init__.py              # Package version & metadata
│       ├── app/                     # Web Application & REST API
│       │   ├── __init__.py
│       │   ├── api.py               # FastAPI clinical QA endpoints
│       │   └── static/              # Web dashboard frontend
│       ├── preprocessing/           # Document Cleaning & Chunking
│       │   ├── __init__.py
│       │   ├── cleaner.py           # Text cleaning & header/footer removal
│       │   ├── chunker.py           # Hierarchical parent/child chunker
│       │   ├── metadata_extractor.py
│       │   └── pdf_loader.py
│       ├── extraction/              # Biomedical NER & Relation Extraction
│       │   ├── __init__.py
│       │   ├── ner_extractor.py     # SciSpaCy entity extraction
│       │   └── relation_extractor.py# Biomedical relation extraction
│       ├── embeddings/              # Dense Embedding Indexer
│       │   ├── __init__.py
│       │   ├── embedder.py          # BAAI/bge-base-en-v1.5 embedding engine
│       │   └── chroma_indexer.py    # Vector store indexer
│       ├── graph/                   # Knowledge Graph Construction & Retrieval
│       │   ├── __init__.py
│       │   ├── graph_builder.py     # NetworkX Knowledge Graph builder
│       │   └── graph_retriever.py   # Hop-distance decay subgraph retriever
│       ├── retrieval/               # Hybrid Retrieval & Fusion
│       │   ├── __init__.py
│       │   ├── bm25_retriever.py    # Sparse BM25 retrieval engine
│       │   ├── dense_retriever.py   # Dense vector retrieval engine
│       │   ├── hybrid_retriever.py  # Graph-augmented hybrid retriever
│       │   ├── adaptive_retrieval.py
│       │   ├── query_expansion.py
│       │   ├── context_compressor.py
│       │   └── retrieval_fusion.py
│       ├── reranker/                # Cross-Encoder Reranking
│       │   ├── __init__.py
│       │   └── reranker.py          # BAAI/bge-reranker-base cross-encoder
│       ├── generation/              # LLM Generation & Evidence Grounding
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   ├── llm_generator.py     # Local LLM generator (Qwen2.5/Llama-3.2)
│       │   ├── prompt_builder.py
│       │   ├── citation_prompt_builder.py # Guideline citation injector
│       │   ├── sentence_grounder.py # Sentence-level NLI grounding engine
│       │   ├── evidence_grounding.py# NLI contradiction & support verifier
│       │   ├── explainability.py    # Traceability & citation provenance
│       │   └── llm_evaluator.py     # Clinical reliability judge
│       ├── evaluation/              # Authoritative Evaluation Engine
│       │   ├── __init__.py          # Module exports
│       │   ├── metrics.py           # Closed-form implementations of all 18 metrics
│       │   ├── p_test_evaluator.py  # Paired ID alignment & Holm-Bonferroni test
│       │   ├── judge_agreement.py   # Multi-judge validation statistics
│       │   ├── hallucination_analysis.py
│       │   ├── deepeval_evaluator.py
│       │   ├── ragas_evaluator.py
│       │   └── run_full_optimization.py
│       └── utils/                   # Shared Utilities & Exception Handling
│           ├── __init__.py
│           ├── exceptions.py        # MedGraphRAG exception hierarchy
│           ├── io_utils.py
│           └── logger.py
│
├── scripts/                         # Executable CLI Scripts
│   ├── run_pipeline.py              # Single question QA inference pipeline
│   ├── run_evaluation.py            # Dataset-wide evaluation runner
│   ├── run_ablations.py             # 5-condition ablation experiment runner
│   ├── run_baselines.py             # Standard baseline comparison runner
│   ├── generate_figures.py          # Publication chart generator
│   ├── reproduce_results.py         # Master reproducibility pipeline
│   └── check_repository.py          # Automated repository consistency checker
│
├── configs/                         # Model & Benchmark Configurations
│   ├── config.yaml
│   ├── experiment_manifest.yaml
│   ├── model.yaml
│   ├── optimized_retrieval.yaml
│   ├── optimized_target_metrics.yaml
│   ├── paths.yaml
│   └── retrieval.yaml
│
├── data/                            # Datasets & Guideline Corpora
│   ├── raw/                         # Raw PDF textbooks & guidelines
│   ├── processed/                   # Processed JSONL chunks & entity graphs
│   └── qa_dataset.json              # Canonical 200-question gold oncology QA dataset
│
├── results/                         # Authoritative Publication Results
│   ├── publication_results.json     # Single source of truth for experimental metrics
│   ├── statistical_tests.json       # Paired Wilcoxon & Holm-Bonferroni test outputs
│   ├── publication_table.csv        # Formatted publication results table (CSV)
│   ├── publication_table.json       # Formatted publication results table (JSON)
│   ├── ablation_question_ids.json   # Stratified 100-question sampling manifest
│   └── figures/                     # Publication figures & radar charts
│
├── tests/                           # Unit & Integration Test Suites
│   ├── conftest.py                  # Pytest fixtures & setup
│   ├── unit/                        # Isolated module unit tests
│   │   ├── test_preprocessing.py
│   │   ├── test_extraction.py
│   │   ├── test_graph.py
│   │   ├── test_retrieval.py
│   │   ├── test_metrics.py
│   │   └── test_publication_metrics.py
│   └── integration/                 # End-to-end integration & reproducibility tests
│       ├── test_pipeline_integration.py
│       ├── test_reproducibility.py
│       └── test_publication_reproducibility.py
│
├── docs/                            # Documentation & Method Specifications
│   ├── evaluation.md                # Evaluation methodology & metric specifications
│   ├── reproducibility.md           # Master reproducibility guide
│   └── METRIC_FORMULAS.md           # Complete closed-form formulas (untracked)
│
├── README.md                        # Master project documentation & results tables
├── LICENSE                          # MIT License
├── CITATION.cff                     # Citation Metadata File
├── CONTRIBUTING.md                  # Contribution Guidelines
├── CHANGELOG.md                     # Version Change Log
├── AUDIT_REPORT.md                  # Comprehensive Scientific & Structural Audit
├── pyproject.toml                   # Python build & package config (PEP 518/621)
├── requirements.txt                 # Exact pinned dependencies
├── .gitignore                       # Git exclusion rules
└── .env.example                     # Environment template
```

---

## 🛠 Architectural Highlights

1. **Clean Separation of Concerns**: Reusable library code is encapsulated under `src/medgraphrag/`, while executable entry points live in `scripts/`.
2. **Single Source of Truth**: All experimental outputs are computed deterministically via `scripts/reproduce_results.py` and exported to `results/publication_results.json`.
3. **Reproducibility Verification**: `scripts/check_repository.py` validates 100% numeric consistency between JSON artifacts, figures, and `README.md`.
4. **Structured Testing**: `tests/unit/` covers isolated logic while `tests/integration/` verifies multi-component workflows and data pipeline integrity.
