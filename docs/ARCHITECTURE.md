# MedGraphRAG — Architecture

This document details every module's responsibility, interfaces, and design
rationale, organized by pipeline stage.

## Offline pipeline (run once via `main.py`)

### Stage 1 — PDF Loading (`preprocessing/pdf_loader.py`)
`PDFLoader` uses PyMuPDF (`fitz`) to extract per-page text from each
textbook PDF. Pages below a minimum character threshold (default 20) are
skipped as likely blank/divider pages. Output: `PageContent` objects
(source file, page number, text).

### Stage 2 — Cleaning (`preprocessing/cleaner.py`)
`TextCleaner` fixes hyphenated line-break artifacts, strips page-number-only
lines, and detects+removes repeated running headers/footers via a
frequency threshold across the document's pages (a line appearing as the
first/last line on ≥40% of pages is treated as a header/footer).

### Stage 3 — Metadata Extraction (`preprocessing/metadata_extractor.py`)
`MetadataExtractor` reads embedded PDF metadata (title/author) with a
filename-derived fallback, and heuristically detects chapter/section
headings using font-size-relative-to-median and regex pattern matching
(`Chapter N`, numbered section headings). Headings are mapped to pages so
every chunk can carry its nearest section heading as provenance.

### Stage 4 — Hierarchical Chunking (`preprocessing/chunker.py`)
`HierarchicalChunker` implements "small-to-big" chunking: **parent**
chunks (~1024 tokens) preserve section-level context; **child** chunks
(~256 tokens, with configurable overlap) nest within each parent and are
what's actually indexed for retrieval. This lets retrieval be precise
(child-level) while generation can still receive broader context by
expanding to the parent when needed. Sentence boundaries are respected via
a lightweight regex splitter — no additional heavy tokenizer model.

### Stage 5 — Medical Entity Extraction (`entity_extraction/ner_extractor.py`)
`MedicalEntityExtractor` wraps SciSpaCy's `en_core_sci_sm` pipeline.
Entities are normalized (lowercased, whitespace-collapsed) for consistent
graph node identity. Batched via `nlp.pipe()` for throughput.

### Stage 6 — Relationship Extraction (`entity_extraction/relation_extractor.py`)
`RelationExtractor` uses dependency-parse heuristics rather than a
supervised RE model (too heavy for 8GB alongside everything else): for
each sentence, every pair of co-occurring entities is linked by the
connecting verb found via the dependency tree (confidence 1.0), or by a
generic `co_occurs_with` relation if no verb path is found (confidence
0.4). This is transparent, explainable, and cheap.

### Stage 7 — Knowledge Graph Construction (`graph/graph_builder.py`)
`KnowledgeGraphBuilder` builds a `networkx.MultiDiGraph`. Nodes are
normalized entity text with `mention_count`, `source_chunks`, and
`surface_forms` attributes. Edges carry `predicate`, `confidence`
(max seen), `weight` (co-occurrence count), and `source_chunks`. Duplicate
(subject, predicate, object) triples are merged rather than duplicated.

### Stage 8 — Embedding Generation (`embeddings/embedder.py`)
`BGEEmbedder` wraps `BAAI/bge-base-en-v1.5` via `sentence-transformers`.
Implements BGE's required asymmetric encoding convention: queries get an
instruction prefix (`"Represent this sentence for searching relevant
passages: "`), documents do not. Batch size kept small (default 16) for
memory safety.

### Stage 9 — ChromaDB Indexing (`embeddings/chroma_indexer.py`)
`ChromaIndexer` manages a persistent, embedded (no server) Chroma
collection using cosine similarity (`hnsw:space: cosine`). Supports
batched upsert and both similarity queries and direct ID lookups (the
latter used to backfill text for chunks surfaced by BM25/graph retrieval).

## Online / query-time pipeline

### Stage 10 — Hybrid Retrieval (`retrieval/`)
- `bm25_retriever.py`: `BM25Retriever` wraps `rank_bm25.BM25Okapi` with a
  simple alphanumeric tokenizer that preserves hyphens (important for terms
  like "HER2-positive").
- `dense_retriever.py`: `DenseRetriever` embeds the query with `BGEEmbedder`
  and searches `ChromaIndexer`, restricted to child-level chunks by default.
- `hybrid_retriever.py`: `HybridRetriever` fuses dense + BM25 (weighted by
  `hybrid_alpha`) and optionally graph scores (fixed 0.2 weight), after
  independently min-max normalizing each signal to [0,1] to make the
  fusion weights meaningful despite very different raw score scales.

### Stage 11 — Graph Retrieval (`graph/graph_retriever.py`)
`GraphRetriever` extracts entities from the query (same extractor used at
ingestion, for consistent normalization), looks them up as seed nodes,
expands via `KnowledgeGraphBuilder.get_neighbors()` by a configurable
number of hops, and maps seed + expanded nodes back to their source
chunks. Score = `mention_count / (1 + hop_distance)`, so well-attested,
directly-matched entities outrank distant or rare expansions.

### Stage 12 — Reranking (`reranker/reranker.py`)
`BGEReranker` wraps `BAAI/bge-reranker-base` (`sentence_transformers.CrossEncoder`)
to jointly score (query, chunk) pairs for the small top-N hybrid-retrieval
candidate set — cross-encoders are far more accurate than bi-encoder cosine
similarity but too slow to run over an entire corpus, so retrieve-then-rerank
is used.

### Stage 13 — Prompt Construction (`generator/prompt_builder.py`)
`PromptBuilder` loads `prompts/system_prompt.txt` (persona + grounding
rules) and `prompts/qa_prompt_template.txt`, and formats retrieved,
reranked chunks into a numbered `[P1] (source, p.X): text` evidence block
so the LLM can produce matching inline citations.

### Stage 14 — LLM Generation (`generator/llm_generator.py`)
`OllamaGenerator` calls a local Ollama daemon (default `qwen2.5:3b-instruct`)
via the official `ollama` Python client, with retry-with-backoff
(`tenacity`) for transient daemon errors. Temperature kept low (0.2) for
factual consistency.

### Stage 15 — Evidence Grounding (`generator/evidence_grounding.py`)
`EvidenceGroundingChecker` verifies (a) every `[Pn]` citation refers to a
real evidence passage, and (b) each cited sentence has sufficient lexical
(token Jaccard) overlap with its cited passage. This is a fast, model-free
first-pass faithfulness signal, independent of the LLM-judge-based RAGAS/
DeepEval metrics computed later — providing a methodological cross-check
against judge-model hallucination.

## Evaluation & benchmarking

### Stage 16 — Evaluation (`evaluation/`)
- `ragas_evaluator.py`: computes faithfulness, answer_relevancy,
  context_precision, context_recall via RAGAS, using the same local Ollama
  model as judge (a documented trade-off vs. GPT-4-as-judge, discussed in
  the paper's limitations).
- `deepeval_evaluator.py`: a second, independently-implemented metric suite
  (faithfulness, answer relevancy) via DeepEval, guarding against
  single-library metric idiosyncrasies.
- `hallucination_analysis.py`: aggregates `EvidenceGroundingChecker` results
  across a batch into a hallucination-rate report.
- `visualization.py`: PyVis interactive knowledge-graph HTML, Matplotlib
  static comparison bar charts (for the paper), Plotly interactive ablation charts.

### Stage 17–18 — Benchmarking & Ablation Studies (`benchmark/`)
- `baselines.py`: defines `VanillaRAGMethod` (dense-only), `BM25Method`,
  `HybridRetrievalMethod` (dense+BM25, no graph/rerank), `GraphRAGMethod`
  (graph-only), and `ProposedMethod` (full hybrid+graph+rerank pipeline) —
  all behind a uniform `run(query, top_k) -> list[dict]` interface.
- `run_benchmark.py`: `BenchmarkRunner` retrieves → generates → evaluates
  (RAGAS + DeepEval + hallucination) for every method over a shared
  question set, logs to MLflow, and writes per-method and summary CSVs to
  `outputs/benchmark_results/`. Ablations are just additional method
  instances with one component toggled (e.g. `ProposedMethod` built with
  `use_graph=False` reproduces a "no graph" ablation).

## Serving

### `app/api.py`
Optional FastAPI service. All models are loaded once via a `lifespan`
context manager at process startup (not per-request) to respect the 8GB
memory budget under load. Exposes `GET /health` and `POST /query`.

## Cross-cutting concerns

- **Configuration** (`configs/`): Hydra + OmegaConf compose `config.yaml`,
  `paths.yaml`, `model.yaml`, `retrieval.yaml` into one config object,
  overridable from the CLI (e.g. `python main.py llm.temperature=0.3`).
- **Logging** (`utils/logger.py`): a single Loguru sink configuration
  (colored console + rotating file) shared by every module via
  `get_logger(__name__)`.
- **Exceptions** (`utils/exceptions.py`): a stage-specific exception
  hierarchy rooted at `MedGraphRAGError`, so failures can be caught and
  logged with precise provenance.
- **I/O** (`utils/io_utils.py`): JSONL read/write/stream helpers (used for
  all intermediate artifacts so no stage needs to hold the full dataset in
  memory at once) and pickle helpers (used for the NetworkX graph).
