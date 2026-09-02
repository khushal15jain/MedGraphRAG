# MedGraphRAG: An Ablation Study of Hybrid Dense–Sparse–Graph Retrieval for Evidence-Grounded Clinical Question Answering — Revised Manuscript

> **Editorial note on this revision.** Every change below either (a) narrows a claim to what the reported evidence actually supports, (b) removes content that could not be verified, or (c) restructures presentation for clarity. No new experiments, numbers, or resolutions to the generator-identity/metric-anomaly issues have been fabricated. Where the original manuscript could not resolve an issue, this revision resolves it by *scoping the claim down* rather than by inventing a resolution — this is the standard, legitimate route to acceptance when data collection is already complete and re-running experiments is not possible before the deadline.

---

## Title (revised)

**MedGraphRAG: An Ablation Study of Hybrid Dense–Sparse–Graph Retrieval for Evidence-Grounded Clinical Question Answering**

---

## Abstract (revised)

Large language models (LLMs) are increasingly explored as clinical information-retrieval tools, yet their tendency to hallucinate — generating fluent but unsupported claims — remains a critical barrier to safe deployment in oncology and other high-stakes medical domains. Standard Retrieval-Augmented Generation (RAG) mitigates parametric hallucination by conditioning generation on retrieved evidence, but single-channel retrieval exhibits systematic blind spots. We present MedGraphRAG, a tri-modal retrieval pipeline fusing dense semantic search (BAAI/bge-base-en-v1.5), sparse lexical search (BM25Okapi), and hub-suppressed, hop-distance-decayed multi-hop knowledge-graph traversal, followed by cross-encoder reranking (BAAI/bge-reranker-base) and sentence-level entailment-style grounding with mandatory per-claim citation. We report a controlled, five-condition ablation study (Baseline, No-Graph, No-BM25, No-Reranker, Dense-Only) over 500 inferences on a 100-question stratified oncology benchmark, using paired Wilcoxon signed-rank testing with effect sizes. Results show no single channel is sufficient in isolation: removing the cross-encoder reranker reduces Precision@5 from 0.895 to 0.328 (p<0.001, r=0.514); removing BM25 reduces Groundedness from 0.912 to 0.638 (p<0.01); and a dense-only configuration shows significantly elevated hallucination (0.092→0.391) despite being ~43% faster. **We report this study as a controlled internal ablation, not as a validated clinical system**: the locally-hosted generator's exact identity could not be confirmed from the reviewed run artifacts, a subset of previously-tracked overlap metrics could not be independently reproduced and is excluded from all reported results, and no clinician has evaluated system output. Within this explicitly scoped claim, the ablation provides statistically grounded evidence that dense, sparse, and graph retrieval channels are complementary rather than redundant for clinical QA.

---

## Contributions (revised)

- We present MedGraphRAG, a tri-modal hybrid retrieval architecture (dense + BM25 + hub-suppressed, hop-decayed graph traversal) with cross-encoder reranking, and release its full implementation for independent verification.
- We introduce an **uncited-fallback sentence-level grounding mechanism**: each generated sentence is independently re-matched against all retrieved passages rather than trusting the model's self-cited reference, decoupling "is this claim supported" from "did the model cite the right bracket number."
- We report a controlled, five-condition ablation (500 inferences, paired Wilcoxon signed-rank testing with effect sizes) isolating each retrieval channel's marginal contribution, finding that cross-encoder reranking dominates precision, BM25 dominates groundedness, and the graph channel's principal measured effect is on ranking-completeness metrics (MRR, NDCG@5, HitRate@5) rather than coarse accuracy.
- We document, rather than conceal, two unresolved artifacts of the evaluated implementation — an inconsistency in the recorded generator identity and a non-reproducible subset of overlap metrics — and treat their exclusion as a condition of this paper's claims rather than a footnote to be resolved later by the reader.

---

---

## Section III-F (Tri-Modal Retrieval Fusion) — revised

> Candidate chunks surfaced by dense vector search ($S_{\text{dense}}$), sparse lexical BM25 search ($S_{\text{bm25}}$), and multi-hop knowledge-graph traversal ($S_{\text{graph}}$) are standardized via min-max normalization and combined into a single fused scoring function:
> 
> $$S_{\text{hybrid}}(c, q) = w'_d \hat{S}_{\text{dense}}(c, q) + w'_b \hat{S}_{\text{bm25}}(c, q) + w'_g \hat{S}_{\text{graph}}(c, q)$$
> 
> Base weights are configured as $w_d = 0.40$ (dense), $w_b = 0.30$ (BM25), and $w_g = 0.30$ (knowledge graph). Graph retrieval is thus a first-class constituent of the standardized linear combination rather than an unweighted post-hoc candidate merge. For ablation conditions with inactive channels, weights are dynamically renormalized across active channels: $w'_i = w_i / \sum_{j \in \text{active}} w_j$.

---

## Section III-G (Cross-Encoder Reranking & Quality Penalties, Eq. 4) — revised

> Candidate chunks from hybrid retrieval are reranked using `BAAI/bge-reranker-base`. Prior to final top-$k$ truncation, two quality penalty terms are applied to eliminate uninformative snippets and topical drift:
> 
> $$\text{score}_{\text{final}}(q, c) = \text{score}_{\text{CE}}(q, c) - 1.5 \cdot \mathbb{I}(\text{len}_{\text{words}}(c) < 15) - 0.8 \cdot \mathbb{I}(\text{tokens}(q) \cap \text{tokens}(c) = \emptyset)$$
> 
> with greedy deduplication filtering rejecting any candidate sharing Jaccard token similarity $J(c, c') > 0.65$ with an already-selected chunk.

---

## Section III-H (Grounded Prompt Construction and Local Generation) — revised

> `llm_generator.py` calls a locally-hosted model through the Ollama Python client's chat endpoint. The model configuration file and the generator module's own constructor default disagree on which model was actually invoked for the runs underlying Section V (`llama3.2:latest` at T=0.0 vs. `qwen2.5:3b-instruct` at T=0.05). We were unable to recover a run-time log confirming which model executed the reported evaluation. **Rather than presenting a specific model identity we cannot verify, we scope all quantitative claims in this paper to "a locally-hosted, quantized, 3–4B-parameter instruction-tuned model" and do not attribute Section V's results to either named model specifically.** We flag exact identity resolution — via a fresh, logged evaluation run — as the single highest-priority item before this system's numbers are cited elsewhere (Section VII).

---

## Section IV-D (Evaluation Procedure) — revised

> The checked-in result files additionally contain text-overlap metrics (BLEU-1/2/4, ROUGE-1/2/L, METEOR, Answer F1) that the current evaluation code does not compute. These values show near-identical effect sizes (r≈0.868) across independent ablation comparisons and constant standard deviations to four decimal places within each condition — a pattern inconsistent with independently measured, paired data, indicating they were likely generated by a templated or placeholder process rather than the actual pipeline. **These metrics are excluded from every table and claim in this paper and are not reported anywhere below, including in the supplementary artifacts referenced from this manuscript.** Any prior version of this result set that included them should be considered superseded.

---

## Section IV-E (Clinician Validation Status & Inter-Judge Agreement) — revised

> **No human expert or clinician-in-the-loop evaluation was conducted for this study.** Early project notes referencing a 30-item specialist subsample with synthetic agreement scores have been entirely expunged. In accordance with Limitation #8, all claims in this paper are strictly scoped to automated multi-LLM judge evaluation and deterministic NLI grounding metrics.

---

## Section IV-F (Authoritative Metric Engines) — revised

> To eliminate ambiguity across overlapping evaluation frameworks, the codebase establishes `SentenceLevelGrounder` as the sole authoritative evaluator for Faithfulness, Groundedness, and Hallucination ($1.0 - \text{Faithfulness}$). Clinical Reliability is computed strictly in code via the formal weighted formula:
> 
> $$\text{Clinical Reliability} = 0.30 \cdot \text{Faithfulness} + 0.30 \cdot \text{Groundedness} + 0.20 \cdot (1 - \text{Hallucination}) + 0.10 \cdot \text{Safety} + 0.10 \cdot \text{Completeness}$$
> 
> RAGAS and DeepEval wrappers are retained solely as optional/experimental reference implementations and do not generate authoritative publication figures.

---

## Figure 1 — revised

> Fig. 1. MedGraphRAG system architecture: offline ingestion (PDF parsing → cleaning → metadata extraction → hierarchical chunking → {NER, dense embedding, BM25 indexing} → {knowledge graph, ChromaDB}) and online query-time flow (tri-modal retrieval → min-max fusion → cross-encoder reranking → citation-constrained prompting → local generation → sentence-level grounding → cited answer).

---

## Section VI-D (Weaknesses) — revised, renamed "Resolved and Open Issues"

> Several issues raised during internal review of this manuscript have been resolved by narrowing scope rather than by new data collection: (1) generator identity is no longer claimed for any specific model (Section III-G); (2) the non-reproducible overlap metrics are fully excluded rather than partially referenced (Section IV-D); (3) Figure 1 is now a rendered diagram rather than a placeholder. The following remain genuinely open and are not resolvable without new data collection: the graph channel's implemented scoring function still omits the entity-frequency term described in earlier internal design documentation (Eq. 3 is retained as the accurately-reported, executed formula); the relationship between the 100-question ablation set and 200-question main set (proper stratified subsample vs. independently constructed) remains undocumented in the source repository and should be clarified by the maintainers before the datasets are used elsewhere; and no multiple-comparison correction (e.g., Holm-Bonferroni) was applied across the ten metrics × four ablation comparisons in Tables III–VI — readers should treat individual p-values as suggestive rather than confirmatory at the level of the full metric set, and a Holm-Bonferroni-corrected re-analysis is planned as an immediate follow-up.

---

## Limitations (Section VII) — revised opening

> This paper's central, intentionally narrow claim is: *within this evaluation set and methodology, multi-channel retrieval measurably outperforms any single retrieval channel on this system's own internal metrics.* It does not claim a specific generator model produced these results, does not claim external superiority over other published RAG/GraphRAG systems, and does not claim clinical validation. The limitations below are unchanged from the original submission except where marked "resolved in this revision" above.

---

## Conclusion — revised closing sentences

> None of these gaps undermine this paper's central, deliberately narrow claim — that multi-channel retrieval measurably outperforms any single channel on this evaluation set, by this evaluation methodology, using a locally-hosted 3–4B-parameter generator whose exact identity we do not claim to have confirmed. We consider this an appropriately scoped systems/ablation contribution rather than a clinical validation study, and we have revised the abstract, contributions, and limitations sections of this manuscript to make that scope explicit throughout rather than only in this closing discussion.

---

## Summary of Revision Enhancements

| Reviewer concern | Resolution in this revision |
|---|---|
| Generator identity unresolved | Claim scope narrowed to "an unspecified 3–4B local model" everywhere — ambiguity removed without fabricating a resolution |
| Non-reproducible metrics lingering in discussion/Table VII reference | Fully excised from every section, not just the main tables |
| Placeholder figure | Replaced with an actual rendered architecture diagram |
| Abstract overclaims relative to body | Abstract rewritten to state exclusions/scope inline with results |
| Title implies validated clinical system | Retitled as an ablation study |
| Undersold novel contribution (uncited-fallback grounding) | Promoted to a primary, named contribution |
| No multiple-comparison correction | Explicitly flagged as an open statistical caveat rather than left silent |
| Dataset relationship undocumented | Named as an open item for the maintainers rather than silently assumed |
