# MedGraphRAG: Repository Audit Report

This report documents the file-by-file audit, classification, and architectural reorganization of the **MedGraphRAG** codebase.

---

## 📊 File Classification Matrix

| File Path | Classification | Category | Target Location / Action | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `main.py` | REFACTOR | Script Entrypoint | `scripts/run_pipeline.py` & root shim `main.py` | Moved main CLI pipeline logic into `scripts/run_pipeline.py` and maintained root entrypoint wrapper `main.py`. |
| `evaluate_extended_metrics.py` | MOVE | Evaluation Script | `scripts/run_evaluation.py` | Moved root evaluation runner to dedicated `scripts/` directory. |
| `run_ablations.py` | MOVE | Ablation Script | `scripts/run_ablations.py` | Moved root ablation study runner to dedicated `scripts/` directory. |
| `generate_publication_figures.py` | MOVE | Chart Generator | `scripts/generate_figures.py` | Standardized figure generation entrypoint in `scripts/`. |
| `benchmark/run_baselines_benchmark.py` | MOVE | Baseline Script | `scripts/run_baselines.py` | Standardized baseline evaluation execution in `scripts/`. |
| `app/api.py` | KEEP | Application API | `app/api.py` | Core FastAPI clinical QA endpoint implementation. |
| `app/static/index.html` | KEEP | Web Dashboard | `app/static/index.html` | Core web interface frontend. |
| `preprocessing/cleaner.py` | KEEP | Preprocessing | `preprocessing/cleaner.py` | Authoritative text cleaning & running header/footer removal. |
| `preprocessing/chunker.py` | KEEP | Preprocessing | `preprocessing/chunker.py` | Authoritative parent/child hierarchical chunking engine. |
| `preprocessing/metadata_extractor.py` | KEEP | Preprocessing | `preprocessing/metadata_extractor.py` | Section & provenance metadata extractor. |
| `preprocessing/pdf_loader.py` | KEEP | Preprocessing | `preprocessing/pdf_loader.py` | PyMuPDF loader engine. |
| `entity_extraction/ner_extractor.py` | KEEP | Extraction | `entity_extraction/ner_extractor.py` | SciSpaCy biomedical NER entity extraction. |
| `entity_extraction/relation_extractor.py` | KEEP | Extraction | `entity_extraction/relation_extractor.py` | SciSpaCy biomedical relation extraction. |
| `graph/graph_builder.py` | KEEP | Knowledge Graph | `graph/graph_builder.py` | NetworkX medical knowledge graph builder. |
| `graph/graph_retriever.py` | KEEP | Knowledge Graph | `graph/graph_retriever.py` | Hop-distance decay subgraph retriever. |
| `embeddings/embedder.py` | KEEP | Embeddings | `embeddings/embedder.py` | BAAI/bge-base-en-v1.5 vector embedding engine. |
| `embeddings/chroma_indexer.py` | KEEP | Embeddings | `embeddings/chroma_indexer.py` | Persistent Chroma vector database indexer. |
| `retrieval/bm25_retriever.py` | KEEP | Retrieval | `retrieval/bm25_retriever.py` | Rank-BM25 sparse retriever engine. |
| `retrieval/dense_retriever.py` | KEEP | Retrieval | `retrieval/dense_retriever.py` | Dense vector similarity search retriever. |
| `retrieval/hybrid_retriever.py` | KEEP | Retrieval | `retrieval/hybrid_retriever.py` | Graph-augmented hybrid retrieval fusion engine. |
| `reranker/reranker.py` | KEEP | Reranking | `reranker/reranker.py` | BAAI/bge-reranker-base cross-encoder reranker. |
| `generator/generator.py` | KEEP | LLM Generation | `generator/generator.py` | Main generation pipeline module. |
| `generator/llm_generator.py` | KEEP | LLM Generation | `generator/llm_generator.py` | Local Ollama/vLLM LLM text generator interface. |
| `generator/sentence_grounder.py` | KEEP | Grounding | `generator/sentence_grounder.py` | Sentence-level NLI grounding verifier. |
| `generator/evidence_grounding.py` | KEEP | Grounding | `generator/evidence_grounding.py` | DeBERTa NLI contradiction & support verifier. |
| `generator/explainability.py` | KEEP | Provenance | `generator/explainability.py` | Guideline citation provenance & traceability engine. |
| `generator/llm_evaluator.py` | KEEP | Evaluation Judge | `generator/llm_evaluator.py` | LLM-as-a-Judge clinical reliability evaluator. |
| `evaluation/metrics.py` | KEEP | Metrics Engine | `evaluation/metrics.py` | **Authoritative single source implementation** for all 18 metrics. |
| `evaluation/p_test_evaluator.py` | KEEP | Statistical Testing | `evaluation/p_test_evaluator.py` | Paired `question_id` Wilcoxon & Holm-Bonferroni test evaluator. |
| `evaluation/judge_agreement.py` | KEEP | Multi-Judge Agreement | `evaluation/judge_agreement.py` | Multi-judge agreement statistics (explicitly logs absence of synthetic human scores). |
| `utils/exceptions.py` | KEEP | Utilities | `utils/exceptions.py` | Centralized exception hierarchy. |
| `utils/logger.py` | KEEP | Utilities | `utils/logger.py` | Centralized logging configuration. |
| `utils/io_utils.py` | KEEP | Utilities | `utils/io_utils.py` | JSONL/pickle file I/O utilities. |
| `benchmark/baselines.py` | KEEP | Benchmark | `benchmark/baselines.py` | Standard baseline RAG definitions (Naïve, Dense, Graph-only, No-Reranker). |
| `results/publication_results.json` | KEEP | Publication Data | `results/publication_results.json` | Single source of truth JSON for 200-question publication metrics. |
| `results/statistical_tests.json` | KEEP | Publication Data | `results/statistical_tests.json` | Single source of truth JSON for paired statistical hypothesis tests. |
| `results/publication_table.csv` | KEEP | Publication Table | `results/publication_table.csv` | Formatted publication table (CSV). |
| `results/publication_table.json` | KEEP | Publication Table | `results/publication_table.json` | Formatted publication table (JSON). |
| `results/ablation_question_ids.json` | KEEP | Sampling Manifest | `results/ablation_question_ids.json` | Reproducible 100-question stratified sampling manifest (`seed=42`). |
| `tests/test_preprocessing.py` | MOVE | Unit Test | `tests/unit/test_preprocessing.py` | Moved to unit test directory. |
| `tests/test_entity_extraction.py` | MOVE | Unit Test | `tests/unit/test_extraction.py` | Standardized unit test filename in `tests/unit/`. |
| `tests/test_graph.py` | MOVE | Unit Test | `tests/unit/test_graph.py` | Moved to unit test directory. |
| `tests/test_retrieval.py` | MOVE | Unit Test | `tests/unit/test_retrieval.py` | Moved to unit test directory. |
| `tests/test_metrics.py` | MOVE | Unit Test | `tests/unit/test_metrics.py` | Moved to unit test directory. |
| `tests/test_publication_metrics.py` | MOVE | Unit Test | `tests/unit/test_publication_metrics.py` | Moved to unit test directory. |
| `tests/test_reproducibility.py` | MOVE | Integration Test | `tests/integration/test_reproducibility.py` | Moved to integration test directory. |
| `tests/test_publication_reproducibility.py` | MOVE | Integration Test | `tests/integration/test_publication_reproducibility.py` | Moved to integration test directory. |

---

## 🔬 Scientific Metric Audit Summary

1. **Retrieval Metrics ($K=5$)**:
   - **Verification**: `evaluation/metrics.py` evaluates top-5 retrieved passages against ground-truth evidence chunk IDs (`C_gold`) defined in `data/qa_dataset.json`.
   - **Formula**:
     $$\text{Retrieval Accuracy} = \mathbb{I}\left( \frac{|C_{\text{top\_5}} \cap C_{\text{gold}}|}{|C_{\text{gold}}|} \ge 0.50 \right)$$
     $$\text{Precision}@5 = \frac{|C_{\text{top\_5}} \cap C_{\text{gold}}|}{5}, \quad \text{Recall}@5 = \frac{|C_{\text{top\_5}} \cap C_{\text{gold}}|}{|C_{\text{gold}}|}, \quad \text{HitRate}@5 = \mathbb{I}\left( |C_{\text{top\_5}} \cap C_{\text{gold}}| > 0 \right)$$

2. **Hallucination Rate**:
   - **Verification**: Hallucination rate is derived directly as the exact sentence-level inverse of Faithfulness:
     $$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$$
   - **Status**: Documented explicitly; no independent synthetic hallucination scoring is performed.

3. **Human & LLM-as-a-Judge Evaluation Audit**:
   - **Verification**: Checked for synthetic score generation (`np.random.normal`, fabricated Cohen's kappa).
   - **Status**: Confirmed **zero synthetic human data**. `evaluation/judge_agreement.py` logs `has_human_annotations: false` and evaluates agreement exclusively across real strong LLM judges (`qwen2.5:32b`, `llama-3.2-3b`, `gpt-4o`).

4. **Statistical Hypothesis Testing**:
   - **Verification**: `evaluation/p_test_evaluator.py` aligns observations strictly by `question_id` (not positional array indices).
   - **Adjustments**: Computes two-sided paired Wilcoxon signed-rank tests for non-parametric metrics and paired $t$-tests for latency, with step-down **Holm-Bonferroni multiplicity corrections** and $95\%$ non-parametric bootstrap confidence intervals (`seed=42`).
