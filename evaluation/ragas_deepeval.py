"""
evaluation/ragas_deepeval.py
------------------------------
Module: RAGAS + DeepEval integration
Adds: framework-standard Faithfulness, Answer Relevance, Groundedness,
Hallucination scores, computed by established third-party libraries
alongside your custom metrics.py implementations.

RESEARCH RATIONALE
-------------------
Reporting BOTH your own faithfulness/groundedness pipeline (grounding_
checker.py, generator.py's retry gate) AND independent RAGAS/DeepEval
scores is important for a peer-reviewed submission: it shows your
internal metric isn't just measuring itself favorably (a common reviewer
objection to self-reported LLM pipeline metrics), and it lets you report
metric AGREEMENT (e.g. Pearson/Spearman correlation between your
grounding_score and RAGAS faithfulness) as an additional validity
argument for your custom grounding architecture.

DEPENDENCIES (add to requirements.txt):
    ragas>=0.1.9
    deepeval>=1.0
    datasets>=2.14        (ragas dependency)

Both RAGAS and DeepEval default to calling OpenAI models for their LLM-
judge metrics. To keep this fully local/offline (matching your
Ollama-based architecture) and avoid uncontrolled API cost/latency,
this wrapper configures both frameworks to use your LOCAL Qwen2.5 model
(or a larger local judge model, recommended: a 7B+ model if available,
since small models are noisier as judges) via their respective
"custom LLM" integration points.

FILES TO MODIFY
----------------
- evaluation/ragas_deepeval.py   (NEW - this file)
- evaluation/run_evaluation.py   (call run_ragas_eval() / run_deepeval_eval()
                                   once per benchmark run, not per-item --
                                   both frameworks batch internally and are
                                   far more efficient that way)

FUNCTIONS TO ADD
-----------------
- build_ragas_ollama_llm(model_name, base_url) -> LangchainLLMWrapper
- run_ragas_eval(dataset_rows) -> pandas.DataFrame
- run_deepeval_eval(test_cases) -> dict
- correlate_with_internal_scores(ragas_df, internal_scores) -> dict
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# RAGAS (local Ollama judge, no OpenAI dependency)
# --------------------------------------------------------------------------


def build_ragas_ollama_llm(model_name: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
    """
    Wraps your existing local Ollama deployment as a RAGAS-compatible
    judge LLM, so no external API key / cost is introduced.
    NOTE: for judge-model tasks (grading faithfulness/relevance), a
    larger local model than the 3B generator (e.g. qwen2.5:14b or
    similar, if your hardware allows) will produce materially more
    reliable RAGAS scores; using the same 3B model as both generator and
    judge is acceptable for iteration but should be flagged as a
    limitation in the paper if used for final reported numbers.
    """
    try:
        from langchain_community.chat_models import ChatOllama
        from ragas.llms import LangchainLLMWrapper
    except ImportError as e:
        raise ImportError(
            "pip install ragas langchain-community  (required for RAGAS local judge)"
        ) from e

    chat = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)
    return LangchainLLMWrapper(chat)


def build_ragas_embeddings(embedder=None):
    """
    Reuses your EXISTING BGE-base embedder for RAGAS's embedding-based
    metrics (e.g. answer relevance uses embedding similarity between the
    question and a set of LLM-generated hypothetical questions from the
    answer). Falls back to RAGAS's default HuggingFace embeddings wrapper
    around the same BGE checkpoint if a raw embedder instance isn't passed.
    """
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
    except ImportError as e:
        raise ImportError("pip install ragas langchain-community") from e

    if embedder is not None and hasattr(embedder, "as_langchain_embeddings"):
        return LangchainEmbeddingsWrapper(embedder.as_langchain_embeddings())

    hf_bge = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return LangchainEmbeddingsWrapper(hf_bge)


def run_ragas_eval(dataset_rows: List[dict], llm=None, embeddings=None):
    """
    dataset_rows: list of dicts, each with keys matching RAGAS's schema:
        {
          "question": str,
          "answer": str,               # your final assembled answer text
          "contexts": List[str],       # the evidence texts actually used
          "ground_truth": str,         # reference answer from your 200-QA set
        }
    Returns a pandas.DataFrame with per-row + aggregate scores for:
        faithfulness, answer_relevancy, context_precision, context_recall
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as e:
        raise ImportError("pip install ragas datasets") from e

    llm = llm or build_ragas_ollama_llm()
    embeddings = embeddings or build_ragas_embeddings()

    hf_dataset = Dataset.from_list(dataset_rows)
    result = evaluate(
        hf_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )
    return result.to_pandas()


# --------------------------------------------------------------------------
# DeepEval (local Ollama judge via its generic LLM interface)
# --------------------------------------------------------------------------


def build_deepeval_ollama_model(model_name: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
    try:
        from deepeval.models.base_model import DeepEvalBaseLLM
        from langchain_community.chat_models import ChatOllama
    except ImportError as e:
        raise ImportError("pip install deepeval langchain-community") from e

    class OllamaDeepEvalModel(DeepEvalBaseLLM):
        def __init__(self, model_name: str, base_url: str):
            self.chat = ChatOllama(model=model_name, base_url=base_url, temperature=0.0)

        def load_model(self):
            return self.chat

        def generate(self, prompt: str) -> str:
            return self.chat.invoke(prompt).content

        async def a_generate(self, prompt: str) -> str:
            resp = await self.chat.ainvoke(prompt)
            return resp.content

        def get_model_name(self) -> str:
            return f"ollama/{model_name}"

    return OllamaDeepEvalModel(model_name, base_url)


def run_deepeval_eval(test_cases: List[dict], model=None) -> Dict[str, float]:
    """
    test_cases: list of dicts with keys:
        {
          "input": str,                # question
          "actual_output": str,        # generated answer
          "retrieval_context": List[str],  # evidence texts used
          "expected_output": str,      # reference answer
        }
    Returns aggregate scores for Faithfulness, Answer Relevancy,
    Contextual Precision/Recall, and a Hallucination metric (DeepEval's
    HallucinationMetric expects a `context` field distinct from
    retrieval_context -- we pass the same evidence texts as the
    grounding-truth context to check against).
    """
    try:
        from deepeval import evaluate
        from deepeval.test_case import LLMTestCase
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            HallucinationMetric,
        )
    except ImportError as e:
        raise ImportError("pip install deepeval") from e

    model = model or build_deepeval_ollama_model()

    cases = []
    for tc in test_cases:
        cases.append(LLMTestCase(
            input=tc["input"],
            actual_output=tc["actual_output"],
            retrieval_context=tc.get("retrieval_context", []),
            expected_output=tc.get("expected_output", ""),
            context=tc.get("retrieval_context", []),
        ))

    metrics = [
        FaithfulnessMetric(model=model, threshold=0.5),
        AnswerRelevancyMetric(model=model, threshold=0.5),
        ContextualPrecisionMetric(model=model, threshold=0.5),
        ContextualRecallMetric(model=model, threshold=0.5),
        HallucinationMetric(model=model, threshold=0.5),
    ]

    results = evaluate(cases, metrics)

    aggregate: Dict[str, List[float]] = {}
    for test_result in results.test_results:
        for m in test_result.metrics_data:
            aggregate.setdefault(m.name, []).append(m.score)

    return {name: (sum(scores) / len(scores) if scores else 0.0) for name, scores in aggregate.items()}


# --------------------------------------------------------------------------
# Cross-framework validity check
# --------------------------------------------------------------------------


def correlate_with_internal_scores(ragas_df, internal_groundedness_scores: Sequence[float]) -> Dict[str, float]:
    """
    Computes Pearson correlation between your internal grounding_checker
    per-item groundedness_rate (sentence_level_report()['groundedness_rate'])
    and RAGAS's faithfulness column, as evidence for the paper that the
    lightweight, latency-cheap internal check (embedding similarity +
    lexical overlap) is a valid proxy for the more expensive LLM-judge
    metric -- justifying using the internal check at INFERENCE time
    (generator.py's retry gate) while reserving RAGAS/DeepEval for
    periodic offline evaluation.
    """
    try:
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
    except ImportError as e:
        raise ImportError("pip install scipy numpy") from e

    ragas_faithfulness = ragas_df["faithfulness"].to_numpy()
    internal = np.array(internal_groundedness_scores)
    if len(ragas_faithfulness) != len(internal):
        raise ValueError("Row count mismatch between RAGAS results and internal scores.")

    pearson_r, pearson_p = pearsonr(ragas_faithfulness, internal)
    spearman_r, spearman_p = spearmanr(ragas_faithfulness, internal)
    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }
