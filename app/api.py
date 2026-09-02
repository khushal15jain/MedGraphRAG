"""Optional FastAPI layer exposing MedGraphRAG as an HTTP query service.

Run with: uvicorn app.api:app --reload --port 8000

All heavy models (embedder, reranker, spaCy) are loaded once at startup via
FastAPI's lifespan context, not per-request, since repeated loading would
both be slow and risk exceeding the 8GB RAM budget under concurrent requests.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
import json
import os
import uuid
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from embeddings.chroma_indexer import ChromaIndexer
from embeddings.embedder import BGEEmbedder
from entity_extraction.ner_extractor import MedicalEntityExtractor
from generator.citation_prompt_builder import CitationAwarePromptBuilder as PromptBuilder
from generator.citation_prompt_builder import extract_confidence
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
from utils.exceptions import MedGraphRAGError
from utils.io_utils import load_pickle, read_jsonl
from utils.logger import get_logger

logger = get_logger(__name__)

_pipeline: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load configuration and all pipeline components once at process startup."""
    logger.info("Loading MedGraphRAG pipeline components for API service...")
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

    _pipeline.update(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=generator,
        grounding_checker=grounding_checker,
        sentence_grounder=sentence_grounder,
        retrieval_cfg=retrieval_cfg,
    )
    logger.info("MedGraphRAG pipeline ready.")
    yield
    _pipeline.clear()
    logger.info("MedGraphRAG pipeline shut down.")


app = FastAPI(
    title="MedGraphRAG API",
    description="Graph-Augmented RAG for Oncology Clinical Decision Support",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=3, description="Clinical oncology question")
    top_k: int = Field(default=5, ge=1, le=20)


class EvidenceItem(BaseModel):
    """A single cited evidence passage returned alongside the answer."""

    citation: str
    source_file: str
    page_number: int | str
    text: str


class EvidenceExplanationItem(BaseModel):
    """Explanation trace for one cited passage: why it was used and how confidently."""

    passage_tag: str
    chunk_id: str
    source_file: str
    page_number: int | str
    rerank_position: int
    used_by_sentences: list[str]
    confidence: float


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str
    evidence: list[EvidenceItem]
    is_well_grounded: bool
    avg_sentence_overlap: float
    explainability_score: float
    evidence_explanations: list[EvidenceExplanationItem]
    unsupported_sentences: list[str]
    low_confidence_sentences: list[str]
    model_confidence: str | None  # self-reported by the LLM -- see citation_prompt_builder docstring caveat
    category: str | None
    treatment_type: str | None


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic liveness check."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer an oncology clinical question using the full MedGraphRAG pipeline.

    Args:
        request: The question and retrieval depth.

    Returns:
        The generated, evidence-grounded answer with citations, a sentence-level
        grounding readout, and a full explainability trace (which passage
        supported which sentence, source file/page, confidence, ranking).

    Raises:
        HTTPException: 503 if the pipeline isn't initialized, 500 on internal failure.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    cfg = _pipeline["retrieval_cfg"]

    try:
        candidates = _pipeline["hybrid_retriever"].retrieve(
            request.question,
            top_k_dense=cfg.top_k_dense,
            top_k_bm25=cfg.top_k_bm25,
            top_k_graph=cfg.top_k_graph,
            final_top_k=request.top_k * 2,
        )
        candidate_dicts = [
            {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in candidates
        ]
        if not candidate_dicts:
            raise HTTPException(status_code=404, detail="No relevant evidence found for this query")

        ranked = _pipeline["reranker"].rerank(request.question, candidate_dicts, top_k=request.top_k)
        system_prompt, user_prompt = _pipeline["prompt_builder"].build(request.question, ranked)
        answer = _pipeline["generator"].generate(system_prompt, user_prompt)
        # Ensure any raw JSON wrapper from the LLM output is parsed into clean prose
        def clean_llm_prose(text: str) -> str:
            t = text.strip()
            if t.startswith("```"):
                lines = t.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                t = "\n".join(lines).strip()
            if t.startswith("{") and t.endswith("}"):
                try:
                    parsed = json.loads(t)
                    for k in ["answer", "Answer", "response", "Response", "output", "Output", "result", "Result"]:
                        if k in parsed and isinstance(parsed[k], str):
                            return parsed[k].strip()
                except Exception:
                    pass
            return t

        answer = clean_llm_prose(answer)
        answer, model_confidence = extract_confidence(answer)
        category = "Oncology"
        treatment_type = None

        # Sentence-level grounding (hybrid lexical+semantic) + explainability trace
        sentence_report = _pipeline["sentence_grounder"].check(answer, ranked, text_key="text")
        explainability_report = build_explainability_report(sentence_report, ranked)

        evidence_items = [
            EvidenceItem(
                citation=f"P{i}",
                source_file=chunk.get("metadata", {}).get("source_file", "unknown"),
                page_number=chunk.get("metadata", {}).get("page_number", "?"),
                text=chunk["text"],
            )
            for i, chunk in enumerate(ranked, start=1)
        ]

        evidence_explanation_items = [
            EvidenceExplanationItem(
                passage_tag=e.passage_tag,
                chunk_id=e.chunk_id,
                source_file=e.source_file,
                page_number=e.page_number,
                rerank_position=e.rerank_position,
                used_by_sentences=e.used_by_sentences,
                confidence=e.confidence,
            )
            for e in explainability_report.evidence_explanations
        ]

        cleaned_final_answer = sentence_report.cleaned_answer

        response = QueryResponse(
            question=request.question,
            answer=cleaned_final_answer,
            evidence=evidence_items,
            is_well_grounded=sentence_report.grounded_ratio >= 0.7,
            avg_sentence_overlap=sentence_report.faithfulness_score,
            explainability_score=explainability_report.explainability_score,
            evidence_explanations=evidence_explanation_items,
            unsupported_sentences=explainability_report.unsupported_sentences,
            low_confidence_sentences=explainability_report.low_confidence_sentences,
            model_confidence=model_confidence,
            category=category,
            treatment_type=treatment_type,
        )
        
        # Log the answer to the JSON file
        try:
            log_file = "result_generated.json"
            existing_results = []
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    try:
                        existing_results = json.load(f)
                    except json.JSONDecodeError:
                        pass
            
            # Find matching sources
            used_sources = [e.citation for e in evidence_items]

            existing_results.append({
                "id": f"Q{len(existing_results)+1:03d}",
                "question": request.question,
                "answer": cleaned_final_answer,
                "difficulty": "moderate",
                "category": category or "Oncology",
                "sources": used_sources,
                "supporting_chunks": len(evidence_items),
                "is_well_grounded": sentence_report.grounded_ratio >= 0.7,
                "grounding_score": round(sentence_report.faithfulness_score, 3),
                "explainability_score": round(explainability_report.explainability_score, 3),
                "model_confidence": model_confidence or "High"
            })
            
            with open(log_file, "w") as f:
                json.dump(existing_results, f, indent=2)
            logger.info(f"Successfully saved query and answer to {log_file}")
        except Exception as log_exc:
            logger.error(f"Failed to log query to {log_file}: {log_exc}")

        return response
    except MedGraphRAGError as exc:
        logger.error(f"Pipeline error handling query: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Mounted LAST, after every @app.get/@app.post route above, so /health and
# /query are matched first -- Starlette checks routes in registration order,
# and a mount's path prefix ("/") would otherwise shadow them. Visiting "/"
# in a browser now serves the chat UI instead of returning 404; the UI's own
# fetch('/query') call stays same-origin, so no CORS setup is needed.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
