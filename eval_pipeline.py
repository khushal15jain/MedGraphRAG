"""Stage 16: Pipeline Evaluation.

Runs a question/gold-answer dataset through the full MedGraphRAG pipeline
(retrieve -> rerank -> generate -> ground) and produces a per-question CSV
with: Retrieval Accuracy, Precision@5, Recall@5, Faithfulness, Answer
Relevance, Groundedness, Hallucination, Latency, Explainability, Clinical
Reliability -- plus the dataset's own `category` and `difficulty` fields,
so you can slice results by either afterward (e.g. "how does Faithfulness
change on 'complex' questions", "which category has the worst Groundedness").

IMPORTANT -- read this before trusting the retrieval metrics:
Your QA dataset has only {id, q, a, category, difficulty} -- there is no
gold chunk_id / gold source_file per question. That means true Retrieval
Accuracy / Precision@5 / Recall@5 (which require knowing exactly which
chunks are relevant) cannot be computed exactly.

This script instead uses a documented proxy: it embeds the gold answer and
each retrieved chunk with BGEEmbedder and treats cosine similarity above
RELEVANCE_SIM_THRESHOLD as "relevant." This is a common practical stand-in
when you only have QA pairs, but it is NOT the same as human/gold-labeled
relevance judgments -- treat these three metrics as directional, not exact.
If you can get even a handful of manually gold-labeled questions, validate
the threshold against those before trusting the proxy at scale.

Faithfulness / Groundedness / Hallucination / Explainability need no proxy:
they come straight out of SentenceLevelGrounder.check() (a hybrid lexical +
semantic per-sentence score) and build_explainability_report(), both computed
against the real evidence your generator was given.

Clinical Reliability has no automatable ground truth at all. This script
provides an LLM-judge proxy (using your own OllamaGenerator with a rubric
prompt) so the column isn't empty, but a genuinely reliable clinical score
requires a clinician to review a sample of answers. Don't report the judge
score as a clinical validation result on its own.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from embeddings.chroma_indexer import ChromaIndexer
from embeddings.embedder import BGEEmbedder
from entity_extraction.ner_extractor import MedicalEntityExtractor
from generator.citation_prompt_builder import extract_confidence
from generator.evidence_grounding import EvidenceGroundingChecker
from generator.explainability import build_explainability_report
from generator.llm_generator import OllamaGenerator
from generator.citation_prompt_builder import CitationAwarePromptBuilder as PromptBuilder
from generator.citation_prompt_builder import extract_metadata
from generator.sentence_grounder import SentenceLevelGrounder
from graph.graph_builder import KnowledgeGraphBuilder
from graph.graph_retriever import GraphRetriever
from reranker.reranker import BGEReranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from utils.exceptions import MedGraphRAGError
from utils.io_utils import load_pickle, read_jsonl
from utils.logger import get_logger

logger = get_logger(__name__)

# Cosine-similarity threshold for the gold-answer-vs-chunk relevance proxy.
# Tune this against a small hand-labeled sample before trusting it at scale.
RELEVANCE_SIM_THRESHOLD = 0.60

# Size of the pre-rerank candidate pool used as the denominator for the
# Recall@5 proxy (i.e. "of everything retrievable, how much did top-5 catch").
RECALL_POOL_SIZE = 40

_SCORE_PATTERN = re.compile(r"\b([1-5])\b")

_JUDGE_SYSTEM_PROMPT = (
    "You are a clinical oncology reviewer. Score the ANSWER to the QUESTION "
    "on a 1-5 scale for clinical reliability: does it state something a "
    "clinician could safely act on, with no unsafe, contradictory, or "
    "fabricated-sounding claims? 1 = unsafe or fabricated, 5 = clinically "
    "sound and well-supported. Respond with ONLY the single digit."
)

# Every field written per row, in order. id/question/category/difficulty
# come from the QA dataset itself; the rest are computed metrics.
FIELDNAMES = [
    "id",
    "question",
    "gold_answer",
    "generated_answer",
    "category",
    "difficulty",
    "Retrieval Accuracy",
    "Precision@5",
    "Recall@5",
    "Faithfulness",
    "Answer Relevance",
    "Groundedness",
    "Hallucination",
    "Latency",
    "Explainability",
    "Clinical Reliability",
    "Model Confidence",
    "error",
]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (embeddings are pre-normalized,
    but this doesn't assume that, so it's safe even if that changes)."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)


def load_components():
    """Instantiate every pipeline component, mirroring app/api.py's lifespan setup."""
    cfg = OmegaConf.load("configs/config.yaml")
    paths = OmegaConf.load("configs/paths.yaml").paths
    model_cfg = OmegaConf.load("configs/model.yaml")
    retrieval_cfg = OmegaConf.load("configs/retrieval.yaml").retrieval

    embedder = BGEEmbedder(
        model_name=model_cfg.embedding.model_name,
        device=model_cfg.embedding.device,
        batch_size=model_cfg.embedding.batch_size,
    )
    indexer = ChromaIndexer(
        persist_dir=paths.chroma_persist_dir, collection_name=paths.chroma_collection_name
    )
    dense_retriever = DenseRetriever(embedder, indexer)

    chunks = read_jsonl(paths.chunks_file)
    child_chunks = [c for c in chunks if c.get("level") == "child"]
    bm25_retriever = BM25Retriever()
    bm25_retriever.index(
        [c["chunk_id"] for c in child_chunks], [c["text"] for c in child_chunks]
    )

    entity_extractor = MedicalEntityExtractor(spacy_model=model_cfg.ner.spacy_model)
    graph_builder = KnowledgeGraphBuilder()
    graph_builder.graph = load_pickle(paths.graph_file)
    graph_retriever = GraphRetriever(graph_builder, entity_extractor)

    hybrid_retriever = HybridRetriever(
        dense_retriever, bm25_retriever, graph_retriever, hybrid_alpha=retrieval_cfg.hybrid_alpha
    )
    reranker = BGEReranker(
        model_name=model_cfg.reranker.model_name, batch_size=model_cfg.reranker.batch_size
    )
    prompt_builder = PromptBuilder()
    generator = OllamaGenerator(
        model_name=model_cfg.llm.model_name,
        host=model_cfg.llm.host,
        temperature=model_cfg.llm.temperature,
        max_tokens=model_cfg.llm.max_tokens,
    )
    grounding_checker = EvidenceGroundingChecker()
    sentence_grounder = SentenceLevelGrounder(embedder)  # reuses the same embedder instance

    return {
        "embedder": embedder,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "prompt_builder": prompt_builder,
        "generator": generator,
        "grounding_checker": grounding_checker,
        "sentence_grounder": sentence_grounder,
        "retrieval_cfg": retrieval_cfg,
    }


def judge_clinical_reliability(question: str, answer: str, generator) -> float | None:
    """LLM-judge proxy for Clinical Reliability, normalized to 0-1.

    Returns None if the judge's output couldn't be parsed, so callers can
    distinguish "scored low" from "judge failed."
    """
    user_prompt = f"QUESTION: {question}\n\nANSWER: {answer}"
    try:
        raw = generator.generate(_JUDGE_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Clinical-reliability judge call failed: {exc}")
        return None

    match = _SCORE_PATTERN.search(raw)
    if not match:
        logger.warning(f"Could not parse a 1-5 score from judge output: {raw!r}")
        return None
    return int(match.group(1)) / 5.0


def _blank_row(item: dict, error: str) -> dict:
    row = {k: "" for k in FIELDNAMES}
    row.update(
        id=item.get("id", ""),
        question=item.get("q", ""),
        gold_answer=item.get("a", ""),
        category=item.get("category", ""),
        difficulty=item.get("difficulty", ""),
        error=error,
    )
    return row


def evaluate_question(item: dict, components: dict, skip_judge: bool) -> dict:
    """Run one QA item through the pipeline and compute all metrics for it."""
    question = item["q"]
    gold_answer = item["a"]
    cfg = components["retrieval_cfg"]
    embedder = components["embedder"]

    start = time.perf_counter()

    candidate_pool = components["hybrid_retriever"].retrieve(
        question,
        top_k_dense=cfg.top_k_dense,
        top_k_bm25=cfg.top_k_bm25,
        top_k_graph=cfg.top_k_graph,
        final_top_k=RECALL_POOL_SIZE,
    )
    candidate_dicts = [
        {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in candidate_pool
    ]

    if not candidate_dicts:
        latency = time.perf_counter() - start
        row = _blank_row(item, "no_candidates_retrieved")
        row.update(
            **{
                "Retrieval Accuracy": 0,
                "Precision@5": 0.0,
                "Recall@5": 0.0,
                "Faithfulness": 0.0,
                "Answer Relevance": 0.0,
                "Groundedness": 0,
                "Hallucination": 1.0,
                "Latency": round(latency, 3),
                "Explainability": 0.0,
                "Clinical Reliability": "",
            }
        )
        return row

    ranked = components["reranker"].rerank(
        question, candidate_dicts, top_k=cfg.final_top_k
    )

    # --- Generation (moved before the retrieval-metric embeddings below so
    # gold_answer, the candidate pool, and the generated answer can all be
    # embedded in ONE batched call instead of three separate ones) ---
    system_prompt, user_prompt = components["prompt_builder"].build(question, ranked)
    raw_answer = components["generator"].generate(system_prompt, user_prompt)
    answer, metadata = extract_metadata(raw_answer)
    model_confidence = metadata.get("confidence")
    latency = time.perf_counter() - start

    # --- Retrieval metrics (proxy relevance via gold-answer similarity) ---
    # One batched embed_documents call covers gold_answer + the full candidate
    # pool + the generated answer. top5_vecs is NOT re-embedded -- the ranked
    # chunks are always a subset of candidate_dicts (the reranker draws from
    # it), so their vectors are looked up from the pool embedding by chunk_id
    # instead of paying for a redundant model call.
    pool_texts = [c["text"] for c in candidate_dicts]
    combined_texts = [gold_answer] + pool_texts + [answer]
    combined_vecs = embedder.embed_documents(combined_texts)
    gold_vec = combined_vecs[0]
    pool_vecs = combined_vecs[1 : 1 + len(pool_texts)]
    answer_vec = combined_vecs[-1]

    vec_by_chunk_id = {c["chunk_id"]: v for c, v in zip(candidate_dicts, pool_vecs)}
    top5_vecs = [vec_by_chunk_id[c["chunk_id"]] for c in ranked]

    pool_relevant = [cosine_sim(gold_vec, v) >= RELEVANCE_SIM_THRESHOLD for v in pool_vecs]
    top5_relevant = [cosine_sim(gold_vec, v) >= RELEVANCE_SIM_THRESHOLD for v in top5_vecs]

    retrieval_accuracy = 1 if any(top5_relevant) else 0
    precision_at_5 = sum(top5_relevant) / 5.0
    total_relevant_in_pool = sum(pool_relevant)
    recall_at_5 = (sum(top5_relevant) / total_relevant_in_pool) if total_relevant_in_pool else 0.0

    # --- Grounding-derived metrics (sentence-level, lexical+semantic) ---
    sentence_report = components["sentence_grounder"].check(answer, ranked, text_key="text")
    explainability_report = build_explainability_report(sentence_report, ranked)

    faithfulness = sentence_report.faithfulness_score
    groundedness = sentence_report.grounded_ratio  # continuous: fraction of sentences that cleared the grounded threshold
    hallucination = round(1.0 - faithfulness, 4)
    explainability = explainability_report.explainability_score

    # --- Answer relevance (question vs answer, asymmetric BGE convention) ---
    # embed_query uses a different instruction-prefixed encode path than
    # embed_documents, so it can't be folded into the batched call above.
    question_vec = embedder.embed_query(question)
    answer_relevance = round(cosine_sim(question_vec, answer_vec), 4)

    # --- Clinical reliability (LLM-judge proxy, see module docstring) ---
    clinical_reliability = (
        "" if skip_judge else judge_clinical_reliability(question, answer, components["generator"])
    )

    row = _blank_row(item, "")
    row.update(
        **{
            "generated_answer": answer,
            "Retrieval Accuracy": retrieval_accuracy,
            "Precision@5": round(precision_at_5, 4),
            "Recall@5": round(recall_at_5, 4),
            "Faithfulness": faithfulness,
            "Answer Relevance": answer_relevance,
            "Groundedness": groundedness,
            "Hallucination": hallucination,
            "Latency": round(latency, 3),
            "Explainability": explainability,
            "Clinical Reliability": clinical_reliability,
            "Model Confidence": model_confidence or "",
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedGraphRAG against a QA dataset.")
    parser.add_argument("--qa-path", required=True, help="Path to the QA JSON file.")
    parser.add_argument("--output-csv", default="eval_results.csv", help="Where to write results.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N questions.")
    parser.add_argument(
        "--skip-clinical-judge",
        action="store_true",
        help="Skip the LLM-judge call for Clinical Reliability (faster, leaves column blank).",
    )
    args = parser.parse_args()

    qa_items = json.loads(Path(args.qa_path).read_text())
    if args.limit:
        qa_items = qa_items[: args.limit]

    logger.info(f"Loading pipeline components for evaluation of {len(qa_items)} questions...")
    components = load_components()

    rows: list[dict] = []
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, item in enumerate(qa_items, start=1):
            try:
                row = evaluate_question(item, components, args.skip_clinical_judge)
            except MedGraphRAGError as exc:
                logger.error(f"[{item.get('id')}] pipeline error: {exc}")
                row = _blank_row(item, str(exc))
            except Exception as exc:  # noqa: BLE001
                logger.error(f"[{item.get('id')}] unexpected error: {exc}")
                row = _blank_row(item, str(exc))

            writer.writerow(row)
            rows.append(row)
            logger.info(f"[{i}/{len(qa_items)}] {item.get('id')} done")

    _print_summary(rows, args.output_csv)


def _print_summary(rows: list[dict], output_csv: str) -> None:
    numeric_cols = [
        "Retrieval Accuracy", "Precision@5", "Recall@5", "Faithfulness",
        "Answer Relevance", "Groundedness", "Hallucination", "Latency", "Explainability",
    ]
    ok_rows = [r for r in rows if not r.get("error")]
    print(f"\nScored {len(ok_rows)}/{len(rows)} questions successfully.\n")

    print("--- Overall ---")
    for col in numeric_cols:
        vals = [float(r[col]) for r in ok_rows if r.get(col, "") not in ("", "Yes", "No", "Partial")]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            n = len(vals)
            ci = 1.96 * std / np.sqrt(n) if n > 0 else 0
            print(f"{col:20s} mean = {mean:.4f} ± {std:.4f} (95% CI: [{mean - ci:.4f}, {mean + ci:.4f}])")
            
    cr_vals = [
        float(r["Clinical Reliability"]) for r in ok_rows if r.get("Clinical Reliability") not in ("", None)
    ]
    if cr_vals:
        mean = np.mean(cr_vals)
        std = np.std(cr_vals)
        n = len(cr_vals)
        ci = 1.96 * std / np.sqrt(n) if n > 0 else 0
        print(
            f"{'Clinical Reliability':20s} mean = {mean:.4f} ± {std:.4f} (95% CI: [{mean - ci:.4f}, {mean + ci:.4f}])\n"
            "(LLM-judge proxy -- not clinical validation)"
        )

    # Breakdown by difficulty, since it's the field most likely to explain
    # variance in Faithfulness/Groundedness/Retrieval Accuracy.
    print("\n--- By difficulty ---")
    difficulties = sorted({r["difficulty"] for r in ok_rows if r.get("difficulty")})
    for level in difficulties:
        subset = [r for r in ok_rows if r["difficulty"] == level]
        acc_vals = [float(r["Retrieval Accuracy"]) for r in subset if r.get("Retrieval Accuracy") != ""]
        faith_vals = [float(r["Faithfulness"]) for r in subset if r.get("Faithfulness") != ""]
        
        acc_str = f"Retrieval Accuracy={np.mean(acc_vals):.3f}±{np.std(acc_vals):.3f}" if acc_vals else "Retrieval Accuracy=N/A"
        faith_str = f"Faithfulness={np.mean(faith_vals):.3f}±{np.std(faith_vals):.3f}" if faith_vals else "Faithfulness=N/A"
        
        print(f"{level:10s} n={len(subset):3d}  {acc_str}  {faith_str}")

    logger.info(f"Results written to {output_csv}")


if __name__ == "__main__":
    main()