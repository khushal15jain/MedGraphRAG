"""Stage 13: Prompt Construction.

Assembles the final LLM prompt from (a) a fixed system prompt establishing
the clinical-assistant persona and grounding rules, and (b) a dynamically
built evidence block from the reranked, retrieved chunks, each numbered
[P1], [P2], ... so the LLM can produce inline citations that
``evidence_grounding.py`` can later verify against the actual source text.
"""

from __future__ import annotations

from pathlib import Path

from utils.exceptions import GenerationError
from utils.logger import get_logger

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class PromptBuilder:
    """Builds grounded QA prompts from retrieved evidence chunks."""

    def __init__(
        self,
        system_prompt_path: str | Path = _PROMPTS_DIR / "system_prompt.txt",
        qa_template_path: str | Path = _PROMPTS_DIR / "qa_prompt_template.txt",
    ) -> None:
        """Load prompt templates from disk.

        Args:
            system_prompt_path: Path to the system prompt text file.
            qa_template_path: Path to the QA user-prompt template file.

        Raises:
            GenerationError: If either template file is missing.
        """
        try:
            self.system_prompt = Path(system_prompt_path).read_text(encoding="utf-8").strip()
            self.qa_template = Path(qa_template_path).read_text(encoding="utf-8")
        except OSError as exc:
            raise GenerationError(f"Failed to load prompt templates: {exc}") from exc

    @staticmethod
    def _format_evidence_block(chunks: list[dict], text_key: str = "text") -> str:
        """Format retrieved chunks into a numbered evidence block.

        Args:
            chunks: Reranked chunk dicts, each with at least ``text_key``
                and ideally ``metadata`` (source_file, page_number).
            text_key: Dict key holding chunk text.

        Returns:
            Multi-line string with one "[Pn] (source, page X): text" entry per chunk.
        """
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {}) or {}
            source = meta.get("source_file", "unknown source")
            page = meta.get("page_number", "?")
            text = chunk.get(text_key, "").strip()
            lines.append(f"[P{i}] ({source}, p.{page}): {text}")
        return "\n\n".join(lines)

    def build(self, question: str, ranked_chunks: list[dict]) -> tuple[str, str]:
        """Build the (system_prompt, user_prompt) pair to send to the LLM.

        Args:
            question: The user's clinical question.
            ranked_chunks: Final reranked evidence chunks (post-Stage 12),
                in citation order (chunk i -> [P{i+1}]).

        Returns:
            Tuple of (system_prompt, user_prompt) strings.

        Raises:
            GenerationError: If ``ranked_chunks`` is empty (nothing to ground on).
        """
        if not ranked_chunks:
            raise GenerationError("Cannot build a grounded prompt with zero evidence chunks")

        evidence_block = self._format_evidence_block(ranked_chunks)
        user_prompt = self.qa_template.format(evidence_block=evidence_block, question=question)

        logger.debug(f"Built prompt with {len(ranked_chunks)} evidence passages")
        return self.system_prompt, user_prompt
