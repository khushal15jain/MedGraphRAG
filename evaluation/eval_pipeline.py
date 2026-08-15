"""evaluation/eval_pipeline.py
--------------------------
End-to-End MedGraphRAG Pipeline Evaluation Harness.

Executes clinical QA datasets through the full MedGraphRAG architecture
(Retrieve -> Rerank -> Generate -> Ground -> Citation Verification) and
produces structured per-question metrics and aggregated benchmark reports.

Evaluates Retrieval Accuracy, Precision@5, Recall@5, Faithfulness, Answer Relevance,
Groundedness, Hallucination, Latency, Explainability, and Clinical Reliability across
question categories and difficulty levels.
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
from generator.evidence_grounding import EvidenceGroundingChecker
from generator.explainability import build_explainability_report
from generator.llm_generator import OllamaGenerator
from generator.prompt_builder import PromptBuilder
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
# Calibrated for BGE-base normalized cosine embedding similarity across clinical oncology text.
RELEVANCE_SIM_THRESHOLD = 0.35

# Size of the pre-rerank candidate pool used as the denominator for the
# Recall@5 proxy (i.e. "of everything retrievable, how much did top-5 catch").
RECALL_POOL_SIZE = 20

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

    ranked = components["reranker"].rerank(question, candidate_dicts, top_k=5)

    # --- Retrieval metrics (proxy relevance via gold-answer similarity) ---
    gold_vec = embedder.embed_documents([gold_answer])[0]
    pool_vecs = embedder.embed_documents([c["text"] for c in candidate_dicts])
    top5_vecs = embedder.embed_documents([c["text"] for c in ranked])

    pool_relevant = [cosine_sim(gold_vec, v) >= RELEVANCE_SIM_THRESHOLD for v in pool_vecs]
    top5_relevant = [cosine_sim(gold_vec, v) >= RELEVANCE_SIM_THRESHOLD for v in top5_vecs]

    retrieval_accuracy = 1 if any(top5_relevant) else 0
    precision_at_5 = sum(top5_relevant) / 5.0
    total_relevant_in_pool = sum(pool_relevant)
    recall_at_5 = (sum(top5_relevant) / total_relevant_in_pool) if total_relevant_in_pool else 0.0

    # --- Generation ---
    system_prompt, user_prompt = components["prompt_builder"].build(question, ranked)
    answer = components["generator"].generate(system_prompt, user_prompt)
    latency = time.perf_counter() - start

    # --- Grounding-derived metrics (sentence-level, lexical+semantic) ---
    sentence_report = components["sentence_grounder"].check(answer, ranked, text_key="text")
    explainability_report = build_explainability_report(sentence_report, ranked)

    faithfulness = sentence_report.faithfulness_score
    groundedness = 1 if sentence_report.grounded_ratio >= 0.7 else 0  # tune this threshold as you like
    hallucination = round(1.0 - faithfulness, 4)
    explainability = explainability_report.explainability_score

    # --- Answer relevance (question vs answer, asymmetric BGE convention) ---
    question_vec = embedder.embed_query(question)
    answer_vec = embedder.embed_documents([answer])[0]
    answer_relevance = round(cosine_sim(question_vec, answer_vec), 4)

    # --- Clinical reliability (LLM-judge proxy, see module docstring) ---
    clinical_reliability = (
        "" if skip_judge else judge_clinical_reliability(question, answer, components["generator"])
    )

    row = _blank_row(item, "")
    row.update(
        **{
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
        vals = [float(r[col]) for r in ok_rows if r[col] != ""]
        if vals:
            print(f"{col:20s} mean = {sum(vals) / len(vals):.4f}")
    cr_vals = [
        float(r["Clinical Reliability"]) for r in ok_rows if r["Clinical Reliability"] not in ("", None)
    ]
    if cr_vals:
        print(
            f"{'Clinical Reliability':20s} mean = {sum(cr_vals) / len(cr_vals):.4f} "
            "(LLM-judge proxy -- not clinical validation)"
        )

    # Breakdown by difficulty, since it's the field most likely to explain
    # variance in Faithfulness/Groundedness/Retrieval Accuracy.
    print("\n--- By difficulty ---")
    difficulties = sorted({r["difficulty"] for r in ok_rows if r["difficulty"]})
    for level in difficulties:
        subset = [r for r in ok_rows if r["difficulty"] == level]
        acc_vals = [float(r["Retrieval Accuracy"]) for r in subset]
        faith_vals = [float(r["Faithfulness"]) for r in subset]
        print(
            f"{level:10s} n={len(subset):3d}  "
            f"Retrieval Accuracy={sum(acc_vals)/len(acc_vals):.3f}  "
            f"Faithfulness={sum(faith_vals)/len(faith_vals):.3f}"
        )

    logger.info(f"Results written to {output_csv}")


if __name__ == "__main__":
    main()