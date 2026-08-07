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
from generator.llm_generator import OllamaGenerator
from generator.llm_evaluator import LLMEvaluator
from graph.graph_builder import KnowledgeGraphBuilder
from graph.graph_retriever import GraphRetriever
from reranker.reranker import BGEReranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from utils.exceptions import MedGraphRAGError
from utils.io_utils import load_pickle, read_jsonl

RELEVANCE_SIM_THRESHOLD = 0.60
RECALL_POOL_SIZE = 10
FIELDNAMES = [
    "id", "question", "gold_answer", "generated_answer", "category", "difficulty",
    "Faithfulness", "Context Relevancy", "Answer Relevancy",
    "Retrieval Accuracy", "Precision@5", "Recall@5", "MRR", "NDCG@5", "HitRate@5",
    "Groundedness", "Hallucination",
    "BLEU-1", "BLEU-2", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "Answer F1",
    "Explainability", "Clinical Reliability",
    "Safety", "Completeness", "Originality", "Precision", "Efficiency", "Overall",
    "Latency", "Latency (s)"
]

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

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
    llm_generator = OllamaGenerator(
        model_name=model_cfg.llm.model_name,
        host=model_cfg.llm.host,
        temperature=model_cfg.llm.temperature,
        max_tokens=model_cfg.llm.max_tokens,
    )
    llm_evaluator = LLMEvaluator(llm_generator)

    return {
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "llm_evaluator": llm_evaluator,
        "retrieval_cfg": retrieval_cfg,
    }

def _blank_row(item: dict, error: str) -> dict:
    row = {k: "" for k in FIELDNAMES}
    row.update(
        id=item.get("id", ""),
        question=item.get("q", ""),
        gold_answer=item.get("a", ""),
        category=item.get("category", ""),
        difficulty=item.get("difficulty", ""),
    )
    return row

def evaluate_question(item: dict, gen_answer: str, components: dict) -> dict:
    question = item["q"]
    gold_answer = item["a"]
    cfg = components["retrieval_cfg"]

    start = time.perf_counter()
    candidate_pool = components["hybrid_retriever"].retrieve(
        question,
        top_k_dense=cfg.top_k_dense,
        top_k_bm25=cfg.top_k_bm25,
        top_k_graph=cfg.top_k_graph,
        final_top_k=RECALL_POOL_SIZE,
    )
    candidate_dicts = [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in candidate_pool]

    if not candidate_dicts:
        latency = time.perf_counter() - start
        row = _blank_row(item, "no_candidates")
        return row

    ranked = components["reranker"].rerank(question, candidate_dicts, top_k=cfg.final_top_k)
    answer = gen_answer
    latency = time.perf_counter() - start

    eval_inputs = {
        "question": question,
        "gold_answer": gold_answer,
        "generated_answer": answer,
        "retrieved_context": "\n".join([f"P{i+1}: {c['text']}" for i, c in enumerate(ranked)]),
        "retrieved_passage_ids": [f"P{i+1}" for i in range(len(ranked))],
        "ranked_documents": "\n".join([f"Rank {i+1}: {c['chunk_id']}" for i, c in enumerate(ranked)]),
        "retrieval_scores": "N/A",  # Or provide real scores if available
        "response_latency": round(latency, 3)
    }

    print(f"Evaluating Q{item.get('id')} using LLM...")
    llm_results = components["llm_evaluator"].evaluate(eval_inputs)

    row = _blank_row(item, "")
    row.update(
        generated_answer=answer,
        Latency=round(latency, 3)
    )
    
    # Flatten the JSON results into the row dictionary
    for section, metrics in llm_results.items():
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if k in FIELDNAMES:
                    row[k] = v
                elif k == "Answer Relevance": # Handle naming mismatch between prompt and old code if any
                    row["Answer Relevancy"] = v
        else:
            if section in FIELDNAMES:
                row[section] = metrics
                
    return row

def main():
    print("Loading datasets...")
    with open("data/qa_dataset.json", "r") as f:
        qa_items = json.load(f)
    
    with open("result_generated.json", "r") as f:
        generated_items = {item["id"]: item.get("answer", "") for item in json.load(f)}

    print("Loading pipeline components...")
    components = load_components()

    rows = []
    output_json = "full_evaluation_results.json"
    
    # Load existing evaluated IDs to avoid repetition
    evaluated_ids = set()
    import os
    if os.path.exists(output_json):
        try:
            with open(output_json, "r") as f:
                prev_data = json.load(f)
                for ev in prev_data.get("evaluations", []):
                    evaluated_ids.add(ev["id"])
                    rows.append(ev)
        except:
            pass

    for i, item in enumerate(qa_items[:200], start=1):
        if item["id"] in evaluated_ids:
            print(f"[{i}/200] {item['id']} already evaluated, skipping.")
            continue
            
        try:
            gen_answer = generated_items.get(item["id"], "")
            row = evaluate_question(item, gen_answer, components)
        except Exception as exc:
            print(f"[{item.get('id')}] unexpected error: {exc}")
            row = _blank_row(item, str(exc))

        # Keep only requested keys
        filtered_row = {"id": row.get("id")}
        for k in FIELDNAMES:
            if k not in ["question", "gold_answer", "generated_answer", "category", "difficulty", "id"]:
                filtered_row[k] = row.get(k)
        
        rows.append(filtered_row)
        print(f"[{i}/{len(qa_items)}] {item.get('id')} done", flush=True)

        # Save checkpoint periodically
        if i % 10 == 0:
            with open(output_json, "w") as f:
                json.dump({"evaluations": rows}, f, indent=2)

    numeric_cols = [
        "Faithfulness", "Context Relevancy", "Answer Relevancy",
        "Precision@5", "Recall@5", "MRR", "NDCG@5", "HitRate@5",
        "BLEU-1", "BLEU-2", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR", "Answer F1",
        "Safety", "Completeness", "Originality", "Precision", "Efficiency", "Overall",
        "Latency"
    ]
    import numpy as np
    print("\nScored successfully.\n")
    
    summary_mean = {}
    summary_median = {}
    
    for col in numeric_cols:
        raw_vals = [r.get(col) for r in rows if r.get(col)]
        vals = []
        for v in raw_vals:
            try:
                vals.append(float(v))
            except ValueError:
                pass
        
        if vals:
            mean = np.mean(vals)
            median = np.median(vals)
            summary_mean[col] = float(mean)
            summary_median[col] = float(median)
            print(f"{col:20s} mean = {mean:.4f}, median = {median:.4f}")
            
    final_output = {
        "evaluations": rows,
        "summary": {
            "mean": summary_mean,
            "median": summary_median
        }
    }
    
    with open(output_json, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved final results to {output_json}")

if __name__ == "__main__":
    main()
