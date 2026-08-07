"""Stage 14: LLM Generation.

Wraps the Ollama Python client to call a locally-hosted small instruction
model (Qwen2.5 3B Instruct per project configuration). Using Ollama keeps
the model weights entirely outside Python's process memory (the Ollama
daemon manages loading/unloading and quantization), which is the key
enabler for running a capable instruction-tuned LLM at all on an 8GB
laptop alongside the rest of the pipeline.
"""

from __future__ import annotations

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from utils.exceptions import GenerationError
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaGenerator:
    """Generates grounded answers via a local Ollama-hosted LLM."""

    def __init__(
        self,
        model_name: str = "qwen2.5:3b-instruct",
        host: str = "http://localhost:11434",
        temperature: float = 0.05,
        max_tokens: int = 512,
        top_p: float = 0.7,
    ) -> None:
        """Configure the Ollama client and generation parameters.

        Args:
            model_name: Name of the pulled Ollama model (e.g. "qwen2.5:3b-instruct").
                Must be pulled beforehand via `ollama pull qwen2.5:3b-instruct`.
            host: Ollama daemon HTTP endpoint.
            temperature: Sampling temperature; kept low (0.2) for clinical
                factual consistency rather than creative variation.
            max_tokens: Maximum tokens to generate per response.
            top_p: Nucleus sampling parameter.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.client = ollama.Client(host=host)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Call the Ollama chat endpoint with retry on transient failures."""
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
    "temperature": self.temperature,
    "num_predict": self.max_tokens,
    "num_ctx": 8192,
    "top_p": self.top_p,
    "repeat_penalty": 1.15,
},
        )
        return response["message"]["content"]

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a grounded answer from a (system, user) prompt pair.

        Args:
            system_prompt: The fixed clinical-assistant system prompt.
            user_prompt: The evidence-grounded user prompt built by ``PromptBuilder``.

        Returns:
            The model's generated answer text.

        Raises:
            GenerationError: If the Ollama daemon is unreachable or generation
                fails after retries (e.g. model not pulled, daemon not running).
        """
        try:
            answer = self._call_ollama(system_prompt, user_prompt)
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(
                f"LLM generation failed via Ollama model '{self.model_name}'. "
                f"Ensure the Ollama daemon is running and the model is pulled "
                f"(`ollama pull {self.model_name}`). Original error: {exc}"
            ) from exc

        logger.info(f"Generated answer ({len(answer)} chars) using {self.model_name}")
        return answer.strip()
