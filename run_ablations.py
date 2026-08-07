import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from embeddings.chroma_indexer import ChromaIndexer
from embeddings.embedder import BGEEmbedder
from entity_extraction.ner_extractor import MedicalEntityExtractor
from generator.citation_prompt_builder import CitationAwarePromptBuilder, extract_confidence
from generator.evidence_grounding import EvidenceGroundingChecker
from generator.explainability import build_explainability_report
from generator.llm_generator import OllamaGenerator
from generator.sentence_grounder import SentenceLevelGrounder
from graph.graph_builder import KnowledgeGraphBuilder
from graph.graph_retriever import GraphRetriever
from reranker.reranker import BGEReranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from utils.io_utils import load_pickle, read_jsonl

RELEVANCE_SIM_THRESHOLD = 0.60
RECALL_POOL_SIZE = 40
_SCORE_PATTERN = re.compile(r"\b([1-5])\b")

_JUDGE_SYSTEM_PROMPT = (
    "You are a clinical oncology reviewer. Score the ANSWER to the QUESTION "
    "on a 1-5 scale for clinical reliability: does it state something a "
    "clinician could safely act on, with no unsafe, contradictory, or "
    "fabricated-sounding claims? 1 = unsafe or fabricated, 5 = clinically "
    "sound and well-supported. Respond with ONLY the single digit."
)

FIELDNAMES = [
    "id", "question", "gold_answer", "generated_answer", "category", "difficulty",
    "Retrieval Accuracy", "Precision@5", "Recall@5", "Faithfulness", "Answer Relevance",
    "Groundedness", "Hallucination", "Latency", "Explainability", "Clinical Reliability",
    "Model Confidence", "error"
]

NUMERIC_COLS = [
    "Retrieval Accuracy", "Precision@5", "Recall@5", "Faithfulness", "Answer Relevance",
    "Groundedness", "Hallucination", "Latency", "Explainability", "Clinical Reliability"
]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

def judge_clinical_reliability(question: str, answer: str, generator) -> float | None:
    user_prompt = f"QUESTION: {question}\n\nANSWER: {answer}"
    try:
        raw = generator.generate(_JUDGE_SYSTEM_PROMPT, user_prompt)
        match = _SCORE_PATTERN.search(raw)
        if match:
            return int(match.group(1)) / 5.0
    except Exception:
        pass
    return None

def load_components():
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
    generator = OllamaGenerator(
        model_name=model_cfg.llm.model_name,
        host=model_cfg.llm.host,
        temperature=model_cfg.llm.temperature,
        max_tokens=model_cfg.llm.max_tokens,
    )
    grounding_checker = EvidenceGroundingChecker()
    sentence_grounder = SentenceLevelGrounder(embedder=embedder)
    prompt_builder = CitationAwarePromptBuilder()

    return {
        "embedder": embedder,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "generator": generator,
        "grounding_checker": grounding_checker,
        "sentence_grounder": sentence_grounder,
        "prompt_builder": prompt_builder,
        "retrieval_cfg": retrieval_cfg,
    }

def main():
    print("Loading dataset...")
    with open("data/qa_dataset.json", "r") as f:
        qa_items = json.load(f)[:100]

    print("Loading pipeline components...")
    components = load_components()
    cfg = components["retrieval_cfg"]
    embedder = components["embedder"]
    generator = components["generator"]

    modes = [
        {"name": "baseline", "use_graph": True, "use_bm25": True, "use_reranker": True},
        {"name": "no_graph", "use_graph": False, "use_bm25": True, "use_reranker": True},
        {"name": "no_bm25", "use_graph": True, "use_bm25": False, "use_reranker": True},
        {"name": "no_reranker", "use_graph": True, "use_bm25": True, "use_reranker": False},
        {"name": "dense_only", "use_graph": False, "use_bm25": False, "use_reranker": False},
    ]

    for mode in modes:
        mode_name = mode["name"]
        print(f"\n=============================================")
        print(f"--- Starting ablation run: {mode_name} ---")
        print(f"=============================================")
        output_json = f"ablation_{mode_name}.json"
        
        rows = []
        evaluated_ids = set()
        if os.path.exists(output_json):
            try:
                with open(output_json, "r") as f:
                    prev_data = json.load(f)
                    for ev in prev_data.get("evaluations", []):
                        evaluated_ids.add(ev["id"])
                        rows.append(ev)
                print(f"Resuming {mode_name}: {len(rows)}/100 already evaluated.")
            except Exception:
                pass

        for item in tqdm(qa_items, desc=f"Ablation [{mode_name}]"):
            item_id = item.get("id")
            if item_id in evaluated_ids:
                continue

            question = item["q"]
            gold_answer = item["a"]

            start = time.perf_counter()
            top_k_bm25 = cfg.top_k_bm25 if mode["use_bm25"] else 0
            
            candidate_pool = components["hybrid_retriever"].retrieve(
                question,
                top_k_dense=cfg.top_k_dense,
                top_k_bm25=top_k_bm25,
                top_k_graph=cfg.top_k_graph,
                final_top_k=RECALL_POOL_SIZE,
                use_graph=mode["use_graph"],
                use_query_expansion=mode["use_bm25"],
            )
            candidate_dicts = [
                {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
                for c in candidate_pool
            ]

            if not candidate_dicts:
                latency = time.perf_counter() - start
                row = {
                    "id": item_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": "",
                    "category": item.get("category", ""),
                    "difficulty": item.get("difficulty", ""),
                    "Retrieval Accuracy": 0.0,
                    "Precision@5": 0.0,
                    "Recall@5": 0.0,
                    "Faithfulness": 0.0,
                    "Answer Relevance": 0.0,
                    "Groundedness": 0.0,
                    "Hallucination": 1.0,
                    "Latency": round(latency, 3),
                    "Explainability": 0.0,
                    "Clinical Reliability": 0.0,
                    "Model Confidence": "",
                    "error": "no_candidates_retrieved"
                }
                rows.append(row)
            else:
                if mode["use_reranker"]:
                    ranked = components["reranker"].rerank(
                        question, candidate_dicts, top_k=cfg.final_top_k
                    )
                else:
                    ranked = candidate_dicts[: cfg.final_top_k]

                system_prompt, user_prompt = components["prompt_builder"].build(question, ranked)
                raw_answer = generator.generate(system_prompt, user_prompt)
                answer, model_confidence = extract_confidence(raw_answer)
                model_confidence = model_confidence or ""
                latency = time.perf_counter() - start

                # --- Retrieval Proxy Metrics ---
                pool_texts = [c["text"] for c in candidate_dicts]
                combined_texts = [gold_answer] + pool_texts + [answer]
                combined_vecs = embedder.embed_documents(combined_texts)
                gold_vec = combined_vecs[0]
                pool_vecs = combined_vecs[1 : 1 + len(pool_texts)]
                answer_vec = combined_vecs[-1]

                vec_by_chunk_id = {c["chunk_id"]: v for c, v in zip(candidate_dicts, pool_vecs)}
                top5_vecs = [vec_by_chunk_id[c["chunk_id"]] for c in ranked if c["chunk_id"] in vec_by_chunk_id]

                pool_relevant = [cosine_sim(gold_vec, v) >= RELEVANCE_SIM_THRESHOLD for v in pool_vecs]
                top5_relevant = [cosine_sim(gold_vec, v) >= RELEVANCE_SIM_THRESHOLD for v in top5_vecs]

                retrieval_accuracy = 1.0 if any(top5_relevant) else 0.0
                precision_at_5 = sum(top5_relevant) / 5.0
                total_relevant_in_pool = sum(pool_relevant)
                recall_at_5 = (sum(top5_relevant) / total_relevant_in_pool) if total_relevant_in_pool else 0.0

                # --- Grounding Metrics ---
                sentence_report = components["sentence_grounder"].check(answer, ranked, text_key="text")
                explainability_report = build_explainability_report(sentence_report, ranked)

                faithfulness = round(sentence_report.faithfulness_score, 4)
                groundedness = round(sentence_report.grounded_ratio, 4)
                hallucination = round(1.0 - faithfulness, 4)
                explainability = round(explainability_report.explainability_score, 4)

                # --- Answer Relevance ---
                question_vec = embedder.embed_query(question)
                answer_relevance = round(cosine_sim(question_vec, answer_vec), 4)

                # --- Clinical Reliability ---
                clinical_rel = judge_clinical_reliability(question, answer, generator)
                if clinical_rel is None:
                    clinical_rel = round((faithfulness + groundedness) / 2.0, 4)

                row = {
                    "id": item_id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": answer,
                    "category": item.get("category", ""),
                    "difficulty": item.get("difficulty", ""),
                    "Retrieval Accuracy": retrieval_accuracy,
                    "Precision@5": round(precision_at_5, 4),
                    "Recall@5": round(recall_at_5, 4),
                    "Faithfulness": faithfulness,
                    "Answer Relevance": answer_relevance,
                    "Groundedness": groundedness,
                    "Hallucination": hallucination,
                    "Latency": round(latency, 3),
                    "Explainability": explainability,
                    "Clinical Reliability": round(clinical_rel, 4),
                    "Model Confidence": model_confidence,
                    "error": ""
                }
                rows.append(row)

            # Continuous Checkpointing
            summary_mean = {}
            summary_median = {}
            summary_std = {}
            for col in NUMERIC_COLS:
                vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))]
                if vals:
                    summary_mean[col] = round(float(np.mean(vals)), 4)
                    summary_median[col] = round(float(np.median(vals)), 4)
                    summary_std[col] = round(float(np.std(vals)), 4)

            final_output = {
                "mode": mode_name,
                "count": len(rows),
                "evaluations": rows,
                "summary": {
                    "mean": summary_mean,
                    "median": summary_median,
                    "std": summary_std
                }
            }
            with open(output_json, "w") as f:
                json.dump(final_output, f, indent=2)

    print("\nAll 5 ablation modes completed successfully!")

if __name__ == "__main__":
    main()
