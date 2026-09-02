"""Stage 17-18: Benchmarking & Ablation Studies.

Runs every baseline method (Vanilla RAG / Dense, BM25, Hybrid, GraphRAG,
Our Proposed) over a fixed evaluation question set, generates an answer for
each via the shared LLM generator, scores each with RAGAS + DeepEval +
hallucination analysis, and logs everything to MLflow for experiment
tracking and later comparison across runs. Results are also written to
``outputs/benchmark_results/`` as CSV for direct inclusion in the paper's
results tables/figures.

Ablation studies are implemented as additional method variants with one
component toggled off relative to the proposed method (e.g. proposed
without reranking, proposed without graph expansion) — see
``build_ablation_methods``.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pandas as pd

from medgraphrag.benchmark.baselines import RetrievalMethod
from medgraphrag.evaluation.deepeval_evaluator import DeepEvalEvaluator
from medgraphrag.evaluation.hallucination_analysis import HallucinationAnalyzer
from medgraphrag.evaluation.ragas_evaluator import RagasEvaluator
from medgraphrag.generation.llm_generator import OllamaGenerator
from medgraphrag.generation.prompts import PromptBuilder
from medgraphrag.utils.exceptions import EvaluationError
from medgraphrag.utils.io import ensure_dir
from medgraphrag.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalQuestion:
    """A single benchmark evaluation question, with an optional reference answer."""

    question: str
    ground_truth: str | None = None


class BenchmarkRunner:
    """Orchestrates end-to-end benchmarking of multiple retrieval methods."""

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        generator: OllamaGenerator,
        ragas_evaluator: RagasEvaluator,
        deepeval_evaluator: DeepEvalEvaluator,
        hallucination_analyzer: HallucinationAnalyzer,
        results_dir: str = "outputs/benchmark_results",
        mlflow_tracking_uri: str = "file:./outputs/mlruns",
    ) -> None:
        """Wire together the evaluation stack used to score every method identically.

        Args:
            prompt_builder: Shared prompt construction module.
            generator: Shared local LLM generator.
            ragas_evaluator: RAGAS metric evaluator.
            deepeval_evaluator: DeepEval metric evaluator.
            hallucination_analyzer: Lexical-grounding hallucination analyzer.
            results_dir: Directory where per-method CSV results are written.
            mlflow_tracking_uri: MLflow tracking backend URI.
        """
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.ragas_evaluator = ragas_evaluator
        self.deepeval_evaluator = deepeval_evaluator
        self.hallucination_analyzer = hallucination_analyzer
        self.results_dir = ensure_dir(results_dir)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("MedGraphRAG_Benchmark")

    def _run_method_on_questions(
        self, method: RetrievalMethod, questions: list[EvalQuestion], top_k: int
    ) -> pd.DataFrame:
        """Retrieve, generate, and record raw outputs for one method across all questions."""
        rows = []
        for eq in questions:
            retrieved = method.run(eq.question, top_k=top_k)
            if not retrieved:
                logger.warning(f"[{method.name}] No chunks retrieved for: {eq.question!r}")
                rows.append(
                    {
                        "question": eq.question,
                        "ground_truth": eq.ground_truth,
                        "answer": "",
                        "contexts": [],
                        "evidence_chunks": [],
                    }
                )
                continue

            system_prompt, user_prompt = self.prompt_builder.build(eq.question, retrieved)
            answer = self.generator.generate(system_prompt, user_prompt)

            rows.append(
                {
                    "question": eq.question,
                    "ground_truth": eq.ground_truth,
                    "answer": answer,
                    "contexts": [c["text"] for c in retrieved],
                    "evidence_chunks": retrieved,
                }
            )
        return pd.DataFrame(rows)

    def run_method(
        self, method: RetrievalMethod, questions: list[EvalQuestion], top_k: int = 5
    ) -> dict:
        """Run full retrieve -> generate -> evaluate pipeline for a single method.

        Args:
            method: A ``RetrievalMethod``-compatible baseline or proposed method.
            questions: Evaluation question set.
            top_k: Number of chunks to retrieve per question.

        Returns:
            Dict of aggregated metrics (RAGAS + DeepEval + hallucination) for this method.

        Raises:
            EvaluationError: If metric computation fails for this method.
        """
        logger.info(f"Running benchmark method: {method.name}")
        df = self._run_method_on_questions(method, questions, top_k)

        has_ground_truth = df["ground_truth"].notna().all() and len(df) > 0
        ground_truths = df["ground_truth"].tolist() if has_ground_truth else None

        try:
            ragas_scores = self.ragas_evaluator.evaluate_batch(
                questions=df["question"].tolist(),
                answers=df["answer"].tolist(),
                contexts=df["contexts"].tolist(),
                ground_truths=ground_truths,
            )
            deepeval_scores = self.deepeval_evaluator.evaluate_batch(
                questions=df["question"].tolist(),
                answers=df["answer"].tolist(),
                contexts=df["contexts"].tolist(),
            )
            hallucination_report = self.hallucination_analyzer.analyze_batch(
                answers=df["answer"].tolist(),
                evidence_chunks_per_answer=df["evidence_chunks"].tolist(),
            )
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Evaluation failed for method '{method.name}': {exc}") from exc

        metrics = {
            "method": method.name,
            **ragas_scores.as_dict(),
            **{f"deepeval_{k}": v for k, v in deepeval_scores.as_dict().items()},
            "hallucination_rate": hallucination_report.hallucination_rate,
            "mean_sentence_overlap": hallucination_report.mean_sentence_overlap,
        }

        with mlflow.start_run(run_name=method.name):
            mlflow.log_params({"method": method.name, "top_k": top_k, "n_questions": len(questions)})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})

        out_path = self.results_dir / f"{method.name.replace(' ', '_').replace('/', '-')}.csv"
        df.drop(columns=["evidence_chunks"]).to_csv(out_path, index=False)
        logger.info(f"Saved per-question results for '{method.name}' to {out_path}")

        return metrics

    def run_all(
        self, methods: list[RetrievalMethod], questions: list[EvalQuestion], top_k: int = 5
    ) -> pd.DataFrame:
        """Run every method and return a summary comparison table.

        Args:
            methods: All methods to benchmark (baselines + proposed + ablations).
            questions: Shared evaluation question set.
            top_k: Chunks retrieved per question, applied uniformly across methods.

        Returns:
            DataFrame with one row per method and all aggregated metric columns,
            also written to ``outputs/benchmark_results/summary.csv``.
        """
        all_metrics = [self.run_method(m, questions, top_k) for m in methods]
        summary_df = pd.DataFrame(all_metrics)
        summary_path = self.results_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Benchmark summary written to {summary_path}")
        return summary_df
