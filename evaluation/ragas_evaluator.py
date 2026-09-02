"""Stage 16: Evaluation (RAGAS metrics).

Computes the four core RAG evaluation metrics required by the research
design using the RAGAS library, which itself uses an LLM-as-judge. To stay
within hardware constraints and avoid cloud API dependency, the judge LLM
is configured to be the same local Ollama model used for generation
(Qwen2.5 3B Instruct) rather than GPT-4 — a documented trade-off (a smaller
judge model is noisier) that we report explicitly in the paper's
limitations section.

Metrics computed:
  - faithfulness: does the answer's content follow from the retrieved context?
  - answer_relevancy: does the answer actually address the question?
  - context_precision: are the retrieved passages relevant to the question?
  - context_recall: does the retrieved context cover what's needed to answer?
"""

from __future__ import annotations

from dataclasses import dataclass

from datasets import Dataset
from langchain_community.llms import Ollama as LangchainOllama
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RagasScores:
    """Aggregate RAGAS metric scores for one evaluation run."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


class RagasEvaluator:
    """Runs RAGAS metric evaluation using a local Ollama model as judge."""

    def __init__(self, judge_model: str = "qwen2.5:3b-instruct", host: str = "http://localhost:11434") -> None:
        """Configure the RAGAS evaluator with a local LLM judge.

        Args:
            judge_model: Ollama model name used as the LLM judge.
            host: Ollama daemon endpoint.
        """
        self.judge_llm = LangchainOllama(model=judge_model, base_url=host, temperature=0.0)
        self.metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> RagasScores:
        """Evaluate a batch of (question, answer, retrieved contexts) triples.

        Args:
            questions: List of clinical questions.
            answers: Corresponding generated answers.
            contexts: For each question, the list of retrieved passage texts
                used to generate the answer.
            ground_truths: Optional reference answers, required for
                ``context_recall``. If omitted, context_recall is skipped.

        Returns:
            Aggregated (mean) ``RagasScores`` across the batch.

        Raises:
            EvaluationError: If inputs are malformed or RAGAS evaluation fails.
        """
        if not (len(questions) == len(answers) == len(contexts)):
            raise EvaluationError("questions, answers, and contexts must have equal length")

        data: dict[str, list] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths is not None:
            data["ground_truth"] = ground_truths
            metrics.append(context_recall)

        try:
            dataset = Dataset.from_dict(data)
            result = evaluate(dataset, metrics=metrics, llm=self.judge_llm)
            df = result.to_pandas()
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"RAGAS evaluation failed: {exc}") from exc

        scores = RagasScores(
            faithfulness=float(df["faithfulness"].mean()),
            answer_relevancy=float(df["answer_relevancy"].mean()),
            context_precision=float(df["context_precision"].mean()),
            context_recall=float(df["context_recall"].mean()) if ground_truths is not None else float("nan"),
        )
        logger.info(f"RAGAS scores: {scores.as_dict()}")
        return scores
