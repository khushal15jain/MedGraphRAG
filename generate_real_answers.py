import json
import time
from tqdm import tqdm
from omegaconf import OmegaConf
import os
import re

from embeddings.chroma_indexer import ChromaIndexer
from embeddings.embedder import BGEEmbedder
from entity_extraction.ner_extractor import MedicalEntityExtractor
from generator.citation_prompt_builder import CitationAwarePromptBuilder as PromptBuilder
from generator.citation_prompt_builder import extract_confidence
from generator.llm_generator import OllamaGenerator
from generator.sentence_grounder import SentenceLevelGrounder
from graph.graph_builder import KnowledgeGraphBuilder
from graph.graph_retriever import GraphRetriever
from reranker.reranker import BGEReranker
from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from utils.io_utils import load_pickle, read_jsonl


def init_pipeline():
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
    sentence_grounder = SentenceLevelGrounder(embedder)

    return hybrid_retriever, reranker, prompt_builder, generator, sentence_grounder, retrieval_cfg

def main():
    print("Initializing pipeline...")
    hybrid_retriever, reranker, prompt_builder, generator, sentence_grounder, retrieval_cfg = init_pipeline()
    
    print("Loading qa_dataset.json...")
    with open("data/qa_dataset.json", "r") as f:
        dataset = json.load(f)
        
    results = []
    
    existing_results = {}
    if os.path.exists("result_generated.json"):
        try:
            with open("result_generated.json", "r") as f:
                prev_results = json.load(f)
                for res in prev_results:
                    if "Simulated" not in res.get("answer", "") and res.get("answer", "").strip() != "":
                        existing_results[res.get("id")] = res
            print(f"Loaded {len(existing_results)} valid pre-generated answers.")
        except:
            pass
            
    results = []
    
    for item in tqdm(dataset[:200], desc="Processing questions", total=200):
        question = item.get("q", "")
        q_id = item.get("id", "")
        
        if q_id in existing_results:
            results.append(existing_results[q_id])
            continue
            
        try:
            candidates = hybrid_retriever.retrieve(
                question,
                top_k_dense=retrieval_cfg.top_k_dense,
                top_k_bm25=retrieval_cfg.top_k_bm25,
                top_k_graph=retrieval_cfg.top_k_graph,
                final_top_k=15,
            )
            candidate_dicts = [
                {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in candidates
            ]
            
            if len(candidate_dicts) == 0:
                answer = "I could not find sufficient information in the uploaded medical documents to answer this question."
            else:
                ranked = reranker.rerank(question, candidate_dicts, top_k=5)
                
                max_attempts = 2
                for attempt in range(max_attempts):
                    if attempt == 1:
                        print(f"[Processing Q{item.get('id')}] Grounding failed. Attempting query expansion (Attempt 2)...")
                        expansion_system_prompt = "You are a search assistant. Extract 3 to 5 core keywords or concepts from the user's question to use in a vector search. Output ONLY the keywords separated by spaces. Do not output any explanation."
                        expanded_query = generator.generate(expansion_system_prompt, question)
                        print(f"  Expanded query: {expanded_query}")
                        
                        expanded_candidates = hybrid_retriever.retrieve(
                            expanded_query,
                            top_k_dense=retrieval_cfg.top_k_dense,
                            top_k_bm25=retrieval_cfg.top_k_bm25,
                            top_k_graph=retrieval_cfg.top_k_graph,
                            final_top_k=15,
                        )
                        expanded_dicts = [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in expanded_candidates]
                        all_candidates_dict = {c["chunk_id"]: c for c in candidate_dicts + expanded_dicts}
                        merged_candidates = list(all_candidates_dict.values())
                        ranked = reranker.rerank(question, merged_candidates, top_k=5)
                        
                    system_prompt, user_prompt = prompt_builder.build(question, ranked)
                    
                    print(f"\n[Processing Q{item.get('id')}] Generating answer using Llama 3.2 model (Attempt {attempt+1}/{max_attempts})...")
                    raw_answer = generator.generate(system_prompt, user_prompt)
                    
                    # The prompt now instructs the LLM to output ONLY JSON
                    parsed_answer = "I could not find sufficient information in the uploaded medical documents to answer this question."
                    final_sources = []
                    
                    try:
                        # Attempt to extract JSON from the output
                        json_match = re.search(r'\{.*\}', raw_answer, re.DOTALL)
                        if json_match:
                            json_obj = json.loads(json_match.group(0))
                            parsed_answer = json_obj.get("answer", parsed_answer)
                            final_sources = json_obj.get("sources", [])
                        else:
                            parsed_answer = json.loads(raw_answer).get("answer", parsed_answer)
                    except Exception as e:
                        print(f"Failed to parse JSON from LLM: {e}")
                        # Fallback: just take the raw answer if it isn't JSON
                        if "I could not find sufficient information" not in raw_answer:
                            parsed_answer = raw_answer
                            
                    if "I could not find sufficient information" in parsed_answer or "Insufficient evidence" in parsed_answer:
                        answer = "I could not find sufficient information in the uploaded medical documents to answer this question."
                        if attempt == max_attempts - 1:
                            break
                        continue
                        
                    # We still run grounding to check faithfulness for query expansion
                    sentence_report = sentence_grounder.check(parsed_answer, ranked, text_key="text")
                    
                    # User explicitly requested: "The answer field is the single source of truth... Never generate another version."
                    # Therefore, we DO NOT override parsed_answer with sentence_report.cleaned_answer.
                    answer = parsed_answer
                    
                    if not answer.strip() or sentence_report.faithfulness_score < 0.5 or sentence_report.grounded_ratio < 0.5:
                        answer = "I could not find sufficient information in the uploaded medical documents to answer this question."
                        if attempt == max_attempts - 1:
                            break
                        continue
                    else:
                        break
                
        except Exception as e:
            print(f"Error processing question {item.get('id')}: {e}")
            answer = f"Error: {e}"
            final_sources = []
            
        if not final_sources and answer and "I could not find sufficient information" not in answer and not answer.startswith("Error"):
            # Fallback if the LLM JSON didn't contain sources, but the answer passed grounding
            if 'sentence_report' in locals():
                unique_sources = set()
                for s in sentence_report.sentences:
                    for idx in s.cited_passage_indices:
                        unique_sources.add(f"P{idx}")
                final_sources = sorted(list(unique_sources))

        results.append({
            "id": item.get("id", ""),
            "question": question,
            "answer": answer,
            "difficulty": item.get("difficulty", "moderate"),
            "category": item.get("category", "Uncategorized").capitalize(),
            "sources": final_sources,
            "supporting_chunks": len(final_sources)
        })
        
        with open("result_generated.json", "w") as f:
            json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
