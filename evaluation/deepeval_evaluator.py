"""Stage 16: Evaluation (DeepEval metrics).

Complements RAGAS with DeepEval's metric suite, providing a second,
independently-implemented evaluation library — using two frameworks
guards against metric-implementation idiosyncrasies skewing the paper's
reported results, and DeepEval's pytest-style assertion API is convenient
for CI-style regression testing of the RAG pipeline over time.
"""

from __future__ import annotations

from dataclasses import dataclass

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

from utils.exceptions import EvaluationError
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaDeepEvalModel(DeepEvalBaseLLM):
    """Adapter exposing a local Ollama model through DeepEval's judge-LLM interface."""

    def __init__(self, model_name: str = "qwen2.5:3b-instruct", host: str = "http://localhost:11434") -> None:
        import ollama

        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def load_model(self):  # noqa: D102 — required by DeepEvalBaseLLM interface
        return self.client

    def generate(self, prompt: str) -> str:
        """Generate a judge response for a given evaluation prompt."""
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        """Async variant required by the interface; delegates to sync generate."""
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"ollama/{self.model_name}"


@dataclass
class DeepEvalScores:
    """Aggregate DeepEval metric scores for one evaluation run."""

    faithfulness: float
    answer_relevancy: float

    def as_dict(self) -> dict[str, float]:
        return {"faithfulness": self.faithfulness, "answer_relevancy": self.answer_relevancy}


class DeepEvalEvaluator:
    """Runs DeepEval faithfulness and answer-relevancy metrics using a local judge."""

    def __init__(self, judge_model: str = "qwen2.5:3b-instruct", host: str = "http://localhost:11434") -> None:
        self.judge = OllamaDeepEvalModel(model_name=judge_model, host=host)
        self.faithfulness_metric = FaithfulnessMetric(model=self.judge, threshold=0.5)
        self.relevancy_metric = AnswerRelevancyMetric(model=self.judge, threshold=0.5)

    def evaluate_batch(
        self, questions: list[str], answers: list[str], contexts: list[list[str]]
    ) -> DeepEvalScores:
        """Evaluate a batch of (question, answer, context) triples with DeepEval.

        Args:
            questions: List of clinical questions.
            answers: Corresponding generated answers.
            contexts: For each question, the retrieved passage texts.

        Returns:
            Aggregated (mean) ``DeepEvalScores`` across the batch.

        Raises:
            EvaluationError: If inputs are malformed or evaluation fails.
        """
        if not (len(questions) == len(answers) == len(contexts)):
            raise EvaluationError("questions, answers, and contexts must have equal length")

        test_cases = [
            LLMTestCase(input=q, actual_output=a, retrieval_context=c)
            for q, a, c in zip(questions, answers, contexts, strict=True)
        ]

        try:
            deepeval_evaluate(test_cases, [self.faithfulness_metric, self.relevancy_metric])
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"DeepEval evaluation failed: {exc}") from exc

        faithfulness_scores = [self.faithfulness_metric.measure(tc) for tc in test_cases]
        relevancy_scores = [self.relevancy_metric.measure(tc) for tc in test_cases]

        scores = DeepEvalScores(
            faithfulness=sum(faithfulness_scores) / len(faithfulness_scores),
            answer_relevancy=sum(relevancy_scores) / len(relevancy_scores),
        )
        logger.info(f"DeepEval scores: {scores.as_dict()}")
        return scores
