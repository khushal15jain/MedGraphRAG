# MedGraphRAG: Medical Oncology Graph-Augmented Retrieval-Augmented Generation

<p align="center">
  <b>Evidence-grounded GraphRAG for Medical Oncology Question Answering</b><br>
  Hybrid dense, sparse, and multi-hop graph retrieval with reranking and sentence-level grounding.
</p>

<p align="center">
  <a href="https://github.com/khushal15jain/RAGupdated">Repository</a> ·
  <a href="https://github.com/khushal15jain/RAGupdated/tree/main/docs">Documentation</a>
</p>

---

## Overview

**MedGraphRAG** is a research-oriented Retrieval-Augmented Generation (RAG) system for medical oncology question answering. It combines complementary retrieval strategies rather than relying on a single vector search:

- **Dense semantic retrieval** using `BAAI/bge-base-en-v1.5` and ChromaDB
- **Sparse lexical retrieval** using BM25
- **Multi-hop biomedical knowledge-graph retrieval** using SciSpaCy and NetworkX
- **IDF-weighted graph scoring** to reduce the influence of generic high-frequency entities
- **Cross-encoder reranking** using `BAAI/bge-reranker-base`
- **Safety/refusal gating**
- **Sentence-level NLI grounding**
- **Source-level provenance and citation tracking**
- **Local LLM generation through Ollama**

The current repository contains a 200-question gold-standard benchmark, five retrieval ablation conditions, baseline comparisons, statistical evaluation utilities, and publication-oriented result artifacts.

> **Research disclaimer:** MedGraphRAG is a research prototype for evidence-grounded medical information retrieval and question answering. It is **not a medical device and must not be used as a substitute for qualified clinical judgment, diagnosis, or treatment decisions.**

---

## Research Motivation

Standard RAG systems can retrieve semantically similar passages but may struggle with:

1. Multi-hop relationships between biomedical entities
2. Rare but clinically important biomarkers and disease concepts
3. Lexical terminology variation
4. Retrieval of evidence distributed across multiple document sections
5. Unsupported generated statements
6. Traceability from generated claims back to source evidence

MedGraphRAG addresses these limitations by combining:

```text
Dense Semantic Retrieval
          +
Sparse BM25 Retrieval
          +
Multi-Hop Biomedical Graph Retrieval
          ↓
      Score Fusion
          ↓
   Cross-Encoder Reranking
          ↓
      Safety Gate
          ↓
    Local LLM Generation
          ↓
 Sentence-Level NLI Grounding
          ↓
 Answer + Source Provenance
```

---

# Key Contributions

## 1. Hybrid Retrieval

Three complementary retrieval channels are combined:

| Retrieval Channel | Purpose |
|---|---|
| Dense BGE Retrieval | Semantic similarity |
| BM25 | Exact lexical and terminology matching |
| Knowledge Graph | Entity relationships and multi-hop evidence |

This combination is designed to reduce dependence on any single retrieval mechanism.

## 2. IDF-Weighted Multi-Hop Graph Retrieval

The graph retriever uses inverse entity frequency together with graph-distance decay.

Conceptually:

\[
Score(e,q) =
\frac{\log(1 + N/count(e))}
{1+d(e,q)}
\]

where:

- `N` = corpus/entity population size used by the implementation
- `count(e)` = entity frequency
- `d(e,q)` = shortest-path distance from the query-related entity

The purpose is to reduce the tendency of very common entities such as *patient* or *treatment* to dominate graph retrieval while giving more specific biomedical entities greater influence.

## 3. Cross-Encoder Reranking

Candidate passages from the retrieval layer are reranked with:

```text
BAAI/bge-reranker-base
```

The repository's documented inference pipeline uses candidate deduplication followed by reranking and selection of a smaller final context.

## 4. Grounded Generation

The generation stage is followed by sentence-level NLI verification.

The system can:

- identify unsupported claims,
- evaluate entailment against retrieved evidence,
- apply grounding/refusal logic,
- retain source attribution for generated answers.

## 5. Local Inference

The intended architecture supports local inference through Ollama, reducing the need to transmit clinical source material to external inference APIs.

---

# System Architecture

```text
                         ┌──────────────────────────┐
                         │ Clinical Oncology PDFs   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ PDF Parsing / Cleaning   │
                         │ PyMuPDF + Preprocessing  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Section-Aware Chunking   │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
             ┌────────────┐    ┌────────────┐   ┌──────────────┐
             │ Dense BGE  │    │    BM25    │   │ Biomedical KG│
             │ + ChromaDB │    │  Retrieval │   │ SciSpaCy +   │
             └──────┬─────┘    └─────┬──────┘   │ NetworkX     │
                    │                │           └──────┬───────┘
                    └────────────────┼──────────────────┘
                                     ▼
                         ┌──────────────────────────┐
                         │ Score Normalization &    │
                         │ Hybrid Score Fusion     │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Cross-Encoder Reranker  │
                         │ BGE Reranker + Dedup    │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Safety / Refusal Gate   │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Local LLM Generation    │
                         │ via Ollama              │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Sentence-Level NLI      │
                         │ Grounding Verification  │
                         └────────────┬─────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Grounded Answer +       │
                         │ Source Provenance       │
                         └──────────────────────────┘
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| PDF processing | PyMuPDF |
| Biomedical NER | SciSpaCy |
| Knowledge graph | NetworkX |
| Dense embeddings | `BAAI/bge-base-en-v1.5` |
| Vector database | ChromaDB |
| Sparse retrieval | BM25 |
| Reranker | `BAAI/bge-reranker-base` |
| LLM runtime | Ollama |
| LLM | Llama-family model configured by the repository |
| Grounding | Sentence-level NLI |
| Web/API layer | FastAPI |
| UI | Streamlit |
| Configuration | YAML |
| Testing | Pytest |

> **Model provenance note:** For publication, the exact generator model tag/version used for each reported experiment should be taken from the experiment manifest/result metadata rather than inferred from a floating tag such as `llama3.2:latest`.

---

# Repository Structure

```text
RAGupdated/
│
├── app/                         # FastAPI backend and Streamlit interface
├── benchmark/                   # Baselines and benchmark runners
├── configs/                     # Model and retrieval configuration
├── data/                        # Raw/sample data and processed artifacts
├── docs/                        # Documentation and publication material
├── embeddings/                  # BGE embedding and ChromaDB indexing
├── entity_extraction/           # SciSpaCy NER and relation extraction
├── evaluation/                  # Metrics, statistics, judge agreement
├── explainability/              # Provenance and source attribution
├── generator/                   # Prompting, Ollama generation, grounding
├── graph/                       # Knowledge graph construction/retrieval
├── grounding/                   # NLI-based sentence grounding
├── notebooks/                   # Experimental notebooks
├── preprocessing/               # PDF parsing, cleaning, chunking
├── prompts/                     # Prompt templates
├── reranker/                    # Cross-encoder reranking
├── retrieval/                   # Dense, BM25 and hybrid retrieval
├── tests/                       # Unit/integration tests
├── utils/                       # Shared utilities
│
├── gold_standard_dataset.json   # Gold clinical QA benchmark
├── full_evaluation_results.json # Full evaluation artifact
│
├── ablation_baseline.json
├── ablation_no_graph.json
├── ablation_no_bm25.json
├── ablation_no_reranker.json
├── ablation_dense_only.json
│
├── baseline_comparison.json
├── p_test_results.json
├── judge_agreement_results.json
│
├── main.py
├── run_ablations.py
├── evaluate_200.py
├── evaluate_generated.py
├── evaluate_to_json.py
├── generate_real_answers.py
├── generate_publication_figures.py
│
├── requirements.txt
├── pyproject.toml
├── .env.example
├── LICENSE
└── README.md
```

---

# Installation

## Requirements

Recommended:

- Python 3.11
- macOS Apple Silicon or Linux
- Ollama
- Sufficient RAM for the selected local LLM
- Internet access during initial model/package installation

Create an environment:

```bash
git clone https://github.com/khushal15jain/RAGupdated.git
cd RAGupdated

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Create the environment file:

```bash
cp .env.example .env
```

Review the values in `.env` and the YAML configuration files before running experiments.

---

# Ollama Setup

Install Ollama using the official installation instructions:

https://ollama.com/

Start the Ollama service:

```bash
ollama serve
```

Pull the exact model specified by your experiment configuration.

For example:

```bash
ollama pull llama3.2:latest
```

> **Important:** `latest` is a moving tag. For publication-grade reproducibility, record the exact model version/digest used for the reported results.

---

# Quick Start

The repository contains a sample clinical guideline workflow.

Run:

```bash
python main.py
```

This initializes the processing/indexing pipeline using the available sample data/configuration.

For a clean reproduction of published benchmark results, use the exact source corpus, configuration, model versions, and experiment metadata associated with the reported artifacts.

---

# Pipeline

The complete processing pipeline is:

### Stage 1 — Document ingestion

Clinical oncology documents are loaded and parsed.

### Stage 2 — Preprocessing

Text is cleaned and document structure such as sections is retained where available.

### Stage 3 — Chunking

Documents are transformed into retrieval units while retaining provenance metadata.

### Stage 4 — Entity extraction

SciSpaCy identifies biomedical entities and relationships.

### Stage 5 — Knowledge graph construction

Entities and relations are represented using NetworkX.

### Stage 6 — Dense indexing

Chunks are embedded with BGE and stored in ChromaDB.

### Stage 7 — Sparse indexing

BM25 supports lexical retrieval and terminology-sensitive matching.

### Stage 8 — Query-time retrieval

A query is processed through:

```text
Dense Retrieval
BM25 Retrieval
Graph Traversal
```

### Stage 9 — Fusion and reranking

Candidate scores are normalized/fused and reranked using the BGE cross-encoder.

### Stage 10 — Generation and grounding

The selected evidence is passed to the configured local LLM, followed by sentence-level grounding verification and provenance generation.

---

# Benchmark Evaluation

The repository contains a gold-standard clinical oncology QA benchmark.

The current README/repository documentation describes:

```text
Gold questions: 200
Ablation conditions: 5
Total ablation evaluations: 1000
```

The five conditions are:

| Condition | Components |
|---|---|
| **Baseline** | Graph + BM25 + Dense + Reranker |
| **No Graph** | BM25 + Dense + Reranker |
| **No BM25** | Graph + Dense + Reranker |
| **No Reranker** | Graph + BM25 + Dense |
| **Dense Only** | Dense retrieval |

Before using stored results as publication evidence, verify that all five artifacts contain the same 200 question IDs and that statistical comparisons are paired by `question_id`.

---

# Reported Benchmark Results

The repository currently reports the following benchmark values:

| Metric | MedGraphRAG |
|---|---:|
| Retrieval Accuracy | 0.9300 |
| Precision@5 | 0.8950 |
| Recall@5 | 0.9776 |
| Faithfulness | 0.9080 |
| Answer Relevance | 0.9150 |
| Groundedness | 0.9120 |
| Hallucination | 0.0920 |
| Explainability | 0.9850 |
| Clinical Reliability | 0.9240 |
| Latency | 25.54 s |

These values are reproduced from the repository's current published result table.

> **Important:** These are reported experimental artifacts, not a claim of universal clinical performance. Exact metric definitions, dataset provenance, model versions, and statistical procedures should be checked against the corresponding evaluation scripts and result metadata before publication.

---

# Ablation Results

The current repository reports the following component-level ablation results:

| Metric | Baseline | No Graph | No BM25 | No Reranker | Dense Only |
|---|---:|---:|---:|---:|---:|
| Retrieval Accuracy | 0.9300 | 0.9200 | 0.8600 | 0.8500 | 0.8000 |
| Precision@5 | 0.8950 | 0.4640 | 0.4080 | 0.3280 | 0.4100 |
| Recall@5 | 0.9776 | 0.9700 | 0.9596 | 0.9716 | 0.9507 |
| Faithfulness | 0.9080 | 0.6964 | 0.6738 | 0.6838 | 0.6087 |
| Answer Relevance | 0.9150 | 0.8426 | 0.8411 | 0.8358 | 0.7341 |
| Groundedness | 0.9120 | 0.7550 | 0.6383 | 0.6842 | 0.6717 |
| Hallucination | 0.0920 | 0.3036 | 0.3262 | 0.3162 | 0.3913 |
| Explainability | 0.9850 | 0.9800 | 0.9750 | 0.9700 | 0.8700 |
| Clinical Reliability | 0.9240 | 0.8940 | 0.8840 | 0.8720 | 0.7840 |
| Latency (s) | 25.54 | 25.40 | 31.47 | 18.14 | 14.22 |

The purpose of this ablation is to quantify the contribution of individual retrieval components rather than to claim that every component improves every metric.

---

# Baseline Comparison

The repository also compares the proposed architecture with conventional retrieval approaches.

| Method | Retrieval Accuracy | Precision@5 | Recall@5 | Faithfulness | Groundedness | Answer F1 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla RAG | 0.8000 | 0.4100 | 0.9507 | 0.6087 | 0.6717 | 0.6720 | 3.91/5 |
| BM25 Only | 0.8200 | 0.3950 | 0.9450 | 0.6350 | 0.6520 | 0.7020 | 4.05/5 |
| Hybrid | 0.8800 | 0.4250 | 0.9650 | 0.6750 | 0.7150 | 0.7580 | 4.31/5 |
| GraphRAG Only | 0.8400 | 0.3650 | 0.9380 | 0.6480 | 0.6820 | 0.7250 | 4.18/5 |
| **MedGraphRAG** | **0.9500** | **0.8950** | **0.9776** | **0.9080** | **0.9120** | **0.7948** | **4.72/5** |

---

# Statistical Evaluation

The repository includes a statistical evaluation pipeline.

The intended ablation comparisons are:

```text
Baseline vs No Graph
Baseline vs No BM25
Baseline vs No Reranker
Baseline vs Dense Only
```

The repository documents paired statistical testing.

Run:

```bash
python evaluation/p_test_evaluator.py
```

For publication-quality analysis, comparisons should:

1. use the same question IDs across conditions,
2. pair observations by `question_id`,
3. report effective sample size,
4. handle missing values explicitly,
5. report effect sizes,
6. apply an appropriate multiple-comparison correction when multiple hypotheses are tested.

The generated statistical artifact is:

```text
p_test_results.json
```

---

# Human / Judge Agreement

The repository contains an agreement-analysis component:

```bash
python evaluation/judge_agreement.py
```

The current project documentation reports a 30-item expert-evaluation sample and dual-judge comparisons.

Reported agreement statistics include:

| Evaluation | Pearson r | Cohen's κ |
|---|---:|---:|
| Faithfulness — Inter Judge | 0.9644 | 0.6815 |
| Faithfulness — Human/LLM | 0.9683 | 0.5408 |
| Groundedness — Inter Judge | 0.9998 | 1.0000 |
| Groundedness — Human/LLM | 0.9996 | 1.0000 |

The underlying artifact is:

```text
judge_agreement_results.json
```

Human-evaluation methodology should be interpreted according to the information actually documented in the repository. No clinical validation claim is made from this limited sample.

---

# Reproducing the Experiments

## Full evaluation

```bash
python evaluate_200.py
```

## Ablation benchmark

```bash
python run_ablations.py
```

## Baseline comparison

```bash
python benchmark/run_baselines_benchmark.py
```

## Statistical analysis

```bash
python evaluation/p_test_evaluator.py
```

## Publication figures

```bash
python generate_publication_figures.py
```

## Judge agreement

```bash
python evaluation/judge_agreement.py
```

## Tests

```bash
pytest
```

---

# Reproducibility Checklist

For a publication-grade reproduction, record:

- [ ] Git commit hash
- [ ] Python version
- [ ] OS
- [ ] CPU/GPU/MPS configuration
- [ ] Python package versions
- [ ] Generator model and exact tag/version
- [ ] Embedding model
- [ ] Reranker model
- [ ] NLI model
- [ ] Dataset version/hash
- [ ] Source corpus version
- [ ] Chunking parameters
- [ ] Retrieval top-k values
- [ ] Fusion parameters
- [ ] Reranking parameters
- [ ] Safety thresholds
- [ ] Generation temperature
- [ ] Random seed
- [ ] Experiment condition
- [ ] Statistical test configuration

For exact publication reproduction, the source corpus and model versions must be identical to those used to generate the reported artifacts.

---

# Data and Source Corpus

The repository can contain sample/derived data for demonstration and testing.

If the benchmark was created from third-party oncology guidelines or textbooks, the original documents may not be redistributed because of licensing restrictions.

In that case, reproducibility should identify:

- document title,
- publisher,
- edition/year,
- source location,
- licensing/access conditions,
- preprocessing procedure.

Do not redistribute copyrighted source documents without permission.

---

# Limitations

MedGraphRAG has several important limitations:

1. **Domain limitation**  
   Evaluation focuses on medical oncology information and should not be generalized to all medical specialties.

2. **Retrieval dependence**  
   Incorrect or incomplete retrieval can lead to incomplete answers.

3. **LLM limitations**  
   Local language models can still generate incorrect statements.

4. **Grounding limitations**  
   NLI-based grounding reduces unsupported claims but does not guarantee clinical correctness.

5. **Benchmark limitations**  
   Benchmark performance depends on dataset composition, source corpus, evaluation definitions, and configuration.

6. **Human evaluation scale**  
   A small expert-evaluation sample cannot establish clinical validity.

7. **Source freshness**  
   Medical guidelines can change. Results should not be interpreted as current clinical guidance unless the underlying source corpus is current.

8. **Hardware/runtime effects**  
   Local model inference and latency can vary substantially across hardware and model/runtime versions.

---

# Safety and Responsible Use

MedGraphRAG is intended for **research and educational evaluation of evidence-grounded RAG systems**.

It should not be used to:

- diagnose patients,
- prescribe treatment,
- replace an oncologist,
- make autonomous clinical decisions,
- determine emergency care,
- provide personalized medical advice without qualified professional oversight.

Always verify medical information against authoritative, current clinical guidelines and qualified healthcare professionals.

---

# Publication Artifacts

The repository includes research and publication-oriented artifacts such as:

```text
PUBLICATION_RESEARCH_REPORT.md
PUBLICATION_RESEARCH_REPORT.pdf
PROJECT_DOCUMENTATION_REPORT.pdf

full_evaluation_results.json
baseline_comparison.json

ablation_baseline.json
ablation_no_graph.json
ablation_no_bm25.json
ablation_no_reranker.json
ablation_dense_only.json

p_test_results.json
judge_agreement_results.json
```

These artifacts should be treated as experimental records. The executable evaluation code is the source of truth for reproducing results.

---

# Citation

If you use this project in academic work, cite the project according to the final publication metadata.

```bibtex
@software{medgraphrag,
  title  = {MedGraphRAG: Medical Oncology Graph-Augmented Retrieval-Augmented Generation},
  author = {Khushal Jain},
  year   = {2026},
  url    = {https://github.com/khushal15jain/RAGupdated}
}
```

---

# License

This project is distributed under the **MIT License**.

See [`LICENSE`](LICENSE) for the complete license text.

---

# Acknowledgements

This project builds on open-source work and pretrained models including:

- BAAI BGE embedding models
- BAAI BGE reranker
- SciSpaCy
- NetworkX
- ChromaDB
- BM25
- PyMuPDF
- Ollama
- Llama-family language models

Please consult the respective project/model licenses and citation requirements before redistribution or publication.

---

# Project Status

**Research prototype — publication-oriented evaluation**

The repository is intended to provide an inspectable implementation of a medical oncology GraphRAG architecture, its benchmark evaluation, ablations, and supporting research artifacts.

The numerical results reported above should always be interpreted together with the exact dataset, model versions, configuration, and experiment artifacts used to generate them.
