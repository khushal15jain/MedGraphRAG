# MedGraphRAG: A Graph-Augmented Retrieval-Augmented Generation System for Oncology Clinical Decision Support on Consumer Hardware

**Abstract**—Retrieval-Augmented Generation (RAG) systems for clinical
decision support typically assume access to substantial compute
infrastructure — large embedding models, GPU-backed vector databases, and
cloud-hosted large language models (LLMs). This dependency limits
deployability in resource-constrained settings such as individual clinician
workstations, low-resource healthcare facilities, or offline research
environments. We present MedGraphRAG, a Graph-Augmented RAG system for
oncology clinical decision support that operates entirely on consumer
hardware (an 8GB-RAM Apple Silicon laptop), combining hierarchical document
chunking, a lightweight medical knowledge graph built with NetworkX, hybrid
dense-sparse retrieval, cross-encoder reranking, and a locally-hosted
3-billion-parameter instruction-tuned LLM (Qwen2.5-3B-Instruct via Ollama).
We evaluate MedGraphRAG against five baseline configurations — dense-only
retrieval, sparse (BM25) retrieval, hybrid dense+sparse retrieval,
graph-only retrieval, and our full proposed method — across faithfulness,
answer relevancy, context precision, context recall, and a model-free
lexical hallucination-rate metric, over a corpus derived from six oncology
textbooks. [Results pending execution on the full six-textbook corpus —
see Section VI for reporting template and Section VII for limitations.] We
release the complete, modular, open-source implementation to support
reproducible research on efficient clinical RAG.

**Index Terms**—Retrieval-Augmented Generation, Knowledge Graphs, Clinical
Decision Support, Oncology, Medical NLP, Efficient NLP, Hybrid Retrieval,
Hallucination Detection.

## I. Introduction

Large language models have shown strong performance on medical
question-answering benchmarks, but their use in clinical decision support
is limited by hallucination risk: an LLM may generate plausible-sounding
but factually incorrect or unsupported statements, a failure mode with
direct patient-safety implications in oncology, where treatment decisions
depend on precise, evidence-specific details (drug dosages, staging
criteria, contraindications). Retrieval-Augmented Generation addresses this
by grounding generation in retrieved source documents, but standard RAG
pipelines retrieve documents based purely on semantic similarity, which can
miss clinically important relationships that are not lexically or
semantically adjacent in embedding space — for instance, a drug and its
delayed adverse effect described in a different section of a textbook.

Graph-Augmented RAG (GraphRAG) approaches address this by additionally
representing entities and their relationships in a structured knowledge
graph, enabling retrieval via graph traversal (e.g., expanding from a named
drug to its documented interactions) in addition to embedding similarity.
However, published GraphRAG systems typically rely on graph databases such
as Neo4j and large embedding/LLM models, incurring memory and
infrastructure requirements incompatible with commodity laptop hardware.

This paper makes the following contributions:

1. We present the design and implementation of MedGraphRAG, a
   GraphRAG system for oncology clinical decision support engineered
   end-to-end for an 8GB-RAM constraint, using only in-memory graph
   structures (NetworkX), an embedded vector store (ChromaDB), and a
   locally-hosted small instruction LLM (Qwen2.5-3B via Ollama).
2. We describe a hierarchical ("small-to-big") chunking strategy and a
   transparent, dependency-parse-based relation extraction method suited
   to compute-constrained medical NLP pipelines.
3. We propose a lightweight, model-free evidence-grounding check
   (citation validity + lexical overlap) as a fast complement to
   LLM-judge-based faithfulness metrics, addressing the risk of judge-model
   hallucination when the judge itself is a small local model.
4. We provide a full benchmarking and ablation framework comparing five
   retrieval configurations against RAGAS and DeepEval metrics, and release
   the complete implementation for reproducibility.

## II. Related Work

**Retrieval-Augmented Generation.** RAG augments LLM generation with
retrieved passages from an external corpus, reducing hallucination
relative to closed-book generation by grounding outputs in retrievable
evidence. Dense retrieval methods embed queries and documents into a shared
vector space and retrieve by similarity; sparse methods such as BM25
retrieve by lexical term overlap. Hybrid approaches combine both signals to
capture complementary strengths — dense retrieval generalizes across
paraphrase and synonymy, while sparse retrieval preserves precision on
exact terminology, which is particularly relevant in biomedical text where
drug names, gene symbols, and dosages must match exactly.

**Graph-Augmented RAG.** GraphRAG approaches construct a knowledge graph
over the corpus and use graph traversal to identify related evidence that
may not be captured by embedding similarity alone. This is especially
relevant in medicine, where clinically important relationships (a drug's
mechanism of action, an indication's staging criteria, a procedure's
contraindications) are frequently distributed across non-adjacent sections
of source text.

**Medical NLP under compute constraints.** SciSpaCy provides biomedical
named-entity-recognition models at multiple size/accuracy tradeoffs,
enabling entity extraction without the memory footprint of larger
transformer-based clinical NER models. Similarly, small instruction-tuned
LLMs (in the 3–4B parameter range) increasingly approach the task
performance of substantially larger models on constrained domains,
particularly when paired with strong retrieval grounding — motivating our
use of Qwen2.5-3B-Instruct rather than a larger model.

**RAG Evaluation.** RAGAS and DeepEval provide automated, LLM-judge-based
metrics for faithfulness, answer relevancy, and context precision/recall,
avoiding the cost of large-scale human evaluation while still capturing
semantic (not just lexical) alignment between generated answers and source
evidence. A known limitation of LLM-judge metrics is that judge-model
errors can themselves introduce noise or bias into reported scores — a
concern we address partially via a model-free grounding check (Section IV-G).

## III. System Architecture

MedGraphRAG implements a twenty-stage pipeline spanning offline ingestion
and online query serving. Figure 1 (see `docs/ARCHITECTURE.md` and the
repository's README for the full diagram) summarizes the data flow: PDF
textbooks are loaded, cleaned, and hierarchically chunked; medical entities
and relations are extracted and assembled into a knowledge graph in
parallel with dense embedding and vector indexing; at query time, hybrid
dense-sparse retrieval is fused with graph-based retrieval, reranked by a
cross-encoder, and used to construct a grounded prompt for local LLM
generation, followed by evidence-grounding verification.

### A. Hierarchical Document Processing

Text is extracted per-page via PyMuPDF and cleaned to remove OCR/extraction
artifacts (hyphenated line breaks, repeated running headers/footers,
isolated page-number lines). Cleaned text is chunked using a two-level
hierarchy: parent chunks (~1024 tokens) preserve section-scale context,
while child chunks (~256 tokens, with token-level overlap) are the unit
actually indexed for retrieval, following the "small-to-big" retrieval
pattern established in prior long-document RAG work — retrieval precision
benefits from small chunks, while generation quality benefits from broader
surrounding context, and the parent-child link allows both simultaneously.

### B. Medical Entity and Relation Extraction

Entities are extracted with SciSpaCy's `en_core_sci_sm` pipeline and
normalized (case-folded, whitespace-collapsed) for consistent identity
across mentions, forming the basis of medical entity linking at the graph
level. Relations between co-occurring entities within a sentence are
extracted via a dependency-parse heuristic: the connecting verb between two
entities (if found in the parse tree) becomes the relation predicate with
high confidence; absent a clear verb path, a generic co-occurrence relation
is recorded with lower confidence. This heuristic avoids the memory and
compute cost of a supervised relation-extraction model while remaining
fully interpretable — a property we consider valuable for a clinical
decision-support tool.

### C. Knowledge Graph Construction

Extracted entities and relations populate a `networkx.MultiDiGraph`.
Duplicate (subject, predicate, object) triples across chunks are merged,
accumulating a co-occurrence weight and tracking every source chunk that
contributed evidence for each node and edge — this chunk-provenance
tracking is what allows the graph to be queried for *retrieval* purposes
(mapping a graph node back to retrievable text) rather than purely for
visualization.

### D. Hybrid Retrieval

At query time, three retrieval signals are computed independently: (1)
dense retrieval via cosine similarity search over BGE-base embeddings in
ChromaDB; (2) sparse retrieval via BM25 over the same child-chunk corpus;
(3) graph retrieval, which extracts entities from the query, seeds them
into the knowledge graph, expands by a configurable number of hops, and
maps resulting nodes back to source chunks, scored by
mention-count-weighted proximity. Because these three signals have
incompatible raw score scales (bounded cosine similarity, unbounded BM25
scores, unbounded graph scores), each is independently min-max normalized
before fusion. Dense and BM25 scores are combined via a weighted sum
(controlled by a tunable α parameter); the graph score is added as an
independent, fixed-weight boost.

### E. Cross-Encoder Reranking

The fused candidate set (typically 10–20 chunks) is reranked using
BAAI/bge-reranker-base, a cross-encoder that jointly scores each
(query, chunk) pair. Cross-encoders are substantially more accurate than
bi-encoder similarity because they can attend across the full query-passage
pair rather than comparing independently-computed vectors, but are too
computationally expensive to apply corpus-wide — the retrieve-then-rerank
pattern applies this expensive scoring only to the small candidate set
produced by Stage D.

### F. Grounded Generation

The final top-k reranked chunks are formatted into a numbered evidence
block and combined with a fixed system prompt instructing the model to
answer strictly from the provided evidence, cite every claim with a
matching `[Pn]` marker, and explicitly flag insufficient evidence rather
than extrapolate. Generation uses Qwen2.5-3B-Instruct hosted locally via
Ollama, at low sampling temperature (0.2) to favor factual consistency.

### G. Evidence Grounding Verification

Following generation, we apply a fast, model-free verification pass: every
`[Pn]` citation is checked against the actual set of provided passages
(citation validity), and every cited sentence's token overlap with its
cited passage is computed as a lexical grounding proxy. This check runs
without any additional model inference, making it always available even
when a judge-model-based metric might be unavailable or untrusted, and
serves as an independent cross-check against the LLM-judge-based
faithfulness metrics computed during formal evaluation.

## IV. Experimental Setup

**Corpus.** Six oncology textbooks (source, editions, and total page
count to be specified by the researcher deploying this pipeline, per
licensing constraints — this repository does not redistribute textbook
content).

**Baselines.** We compare five retrieval configurations, all sharing the
same generation and evaluation pipeline: (1) Vanilla RAG / Dense
Retrieval — BGE-base cosine similarity only; (2) BM25 — sparse lexical
retrieval only; (3) Hybrid Retrieval — dense+BM25 fusion, no graph, no
reranking; (4) GraphRAG — graph traversal retrieval only; (5) MedGraphRAG
(Proposed) — full hybrid dense+BM25+graph fusion with cross-encoder
reranking.

**Metrics.** We report RAGAS faithfulness, answer relevancy, context
precision, and context recall (judge model: Qwen2.5-3B-Instruct, the same
model used for generation, for full local reproducibility); a second,
independently-implemented faithfulness/relevancy pair via DeepEval; and a
model-free hallucination rate derived from the evidence-grounding checker
described in Section III-G.

**Hyperparameters.** Parent/child chunk sizes of 1024/256 tokens
(48-token overlap); hybrid retrieval α = 0.5 (equal dense/BM25 weighting);
graph expansion of 1 hop; final top-k = 5 passages after reranking;
generation temperature = 0.2. Full configuration is version-controlled in
`configs/model.yaml` and `configs/retrieval.yaml`.

**Hardware.** All experiments are designed to run on a MacBook Air M2
(8GB unified memory), with pipeline stages executed sequentially rather
than holding all models in memory concurrently, to remain within the
target memory budget throughout ingestion; the API serving configuration
(Section III, `app/api.py`) is the only point where all models are
resident simultaneously.

## V. Ablation Studies

To isolate the contribution of each architectural component, we evaluate:
(a) the proposed method without cross-encoder reranking (candidates taken
directly from hybrid fusion); (b) the proposed method without graph
expansion (`use_graph=False`, degrading to hybrid dense+BM25 fusion only);
(c) the proposed method without the BM25 signal (dense + graph only,
α = 1.0); (d) the proposed method without the dense signal (BM25 + graph
only, α = 0.0). Each ablation is executed as an independent method
configuration through the shared `BenchmarkRunner`, ensuring identical
generation and evaluation treatment across all variants (see
`docs/REPRODUCIBILITY.md`, Section 8).

## VI. Results

*This section reports the template for results tables; populate with
actual measured values after running `benchmark/run_benchmark.py` over
your corpus and question set, per `docs/REPRODUCIBILITY.md`.*

**Table I. Method Comparison**

| Method | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Hallucination Rate |
|---|---|---|---|---|---|
| BM25 | — | — | — | — | — |
| Vanilla RAG (Dense) | — | — | — | — | — |
| Hybrid Retrieval (Dense+BM25) | — | — | — | — | — |
| GraphRAG (Graph-only) | — | — | — | — | — |
| **MedGraphRAG (Proposed)** | — | — | — | — | — |

**Table II. Ablation Study**

| Configuration | Faithfulness | Context Precision | Hallucination Rate |
|---|---|---|---|
| Proposed, full | — | — | — |
| − Reranking | — | — | — |
| − Graph expansion | — | — | — |
| − BM25 (dense+graph only) | — | — | — |
| − Dense (BM25+graph only) | — | — | — |

## VII. Discussion and Limitations

**Judge-model reliability.** Using a 3B-parameter local model as both
generator and RAGAS/DeepEval judge is a deliberate trade-off for full local
reproducibility and hardware-constraint compliance; it is documented prior
art that larger judge models (e.g., GPT-4-class) correlate more strongly
with human judgment. We partially mitigate this via the model-free
evidence-grounding check (Section III-G), but acknowledge that
LLM-judge-based scores in this paper should be interpreted with this
caveat, and recommend a human-evaluation follow-up study for any
deployment-oriented conclusions.

**Relation extraction recall.** The dependency-parse heuristic used for
relation extraction (Section III-B) favors precision and interpretability
over recall relative to a supervised relation-extraction model; some
clinically relevant relationships spanning multiple sentences or requiring
world knowledge beyond the immediate dependency parse will not be captured.

**Entity linking.** Entities are merged by normalized surface form rather
than linked to a canonical ontology (e.g., UMLS); this is a known
simplification appropriate at the six-textbook scale evaluated here but
would need augmentation (e.g., SciSpaCy's `EntityLinker` with UMLS) for
larger or more heterogeneous corpora.

**Scope of clinical claims.** MedGraphRAG is designed and evaluated as an
evidence-retrieval and grounded-summarization aid; it does not and should
not be used to provide diagnostic or treatment directives independent of
qualified clinician judgment, consistent with the system prompt's explicit
framing (Section III-F).

## VIII. Conclusion

We presented MedGraphRAG, a Graph-Augmented RAG system for oncology
clinical decision support engineered to run entirely within an 8GB-RAM
consumer hardware budget, combining hierarchical chunking, a lightweight
NetworkX-based medical knowledge graph, hybrid dense-sparse-graph
retrieval, cross-encoder reranking, and local LLM generation with
model-free evidence-grounding verification. We provide a complete,
modular, and reproducible open-source implementation along with a
benchmarking framework spanning five retrieval configurations and both
LLM-judge and model-free evaluation metrics, to support further research
into efficient, deployable clinical RAG systems.

## References

*[Populate with the specific literature you cite when finalizing this
paper for submission — e.g. foundational RAG papers, GraphRAG papers,
BGE/SciSpaCy/RAGAS/DeepEval technical reports, and relevant oncology
clinical-NLP prior work. Citations were intentionally left as a template
here rather than fabricated, to ensure every reference in your submitted
paper is real and independently verifiable.]*

[1] Author(s), "Title," *Venue*, Year.
[2] Author(s), "Title," *Venue*, Year.
[3] ...
