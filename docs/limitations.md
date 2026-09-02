# MedGraphRAG: Known Limitations & Scope

This document details the boundaries, clinical scope, and computational constraints of **MedGraphRAG**.

---

## ⚠️ Scope & Operational Boundaries

1. **Domain Focus**: MedGraphRAG is optimized specifically for Medical Oncology clinical decision support (NCCN guidelines, ESMO handbooks, FDA drug labeling). Performance on non-oncological specialties (e.g. cardiology, neurology) has not been systematically evaluated.

2. **Entity Extraction Model**: Biomedical NER relies on SciSpaCy (`en_core_sci_sm`). Rare genetic variants or novel biomarker abbreviations not present in the SciSpaCy vocabulary may require custom dictionary expansion.

3. **Local LLM Dependence**: Response quality and inference speed depend on local hardware capabilities when serving open-weights LLMs (e.g. Qwen2.5-32B or Llama-3.2-3B via Ollama/vLLM).

4. **Static Guideline Snapshots**: Knowledge graph entities are constructed from static PDF text snapshots. Real-time medical guideline updates require re-indexing raw documents through `scripts/run_pipeline.py`.
