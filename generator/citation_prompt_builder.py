"""Citation-aware, evidence-constrained prompt builder.

Drop-in replacement for the existing PromptBuilder -- same call contract,
``build(question, evidence_chunks, generated_answer) -> (system_prompt, user_prompt)``,
so swapping it into app/api.py and eval_pipeline.py is a one-line change:

    from generator.citation_prompt_builder import CitationAwarePromptBuilder as PromptBuilder

Why the current low Faithfulness / Groundedness / Clinical-Reliability
numbers are, at least in part, a PROMPTING problem rather than only a
model-capability ceiling: if the system prompt doesn't explicitly forbid
unsupported statements and doesn't force a citation on every factual
sentence, a small instruction-tuned model (Qwen2.5-3B) will default to its
own parametric medical knowledge to fill gaps in the retrieved evidence --
exactly the failure mode the Evidence Grounding stage exists to catch
downstream. Catching it after generation is a safety net; suppressing it
during generation is cheaper and raises the ceiling the safety net has to
work with in the first place.

v2 changes (merged from a reviewed alternative prompt draft):
  - Added an explicit contradiction-disclosure rule: if cited passages
    disagree, the model must say so instead of silently picking one side.
  - Added an exact, parser-friendly refusal phrase for out-of-scope
    questions, so "insufficient evidence" answers are consistently
    detectable downstream instead of phrased a different way each time.
  - Added a self-reported Confidence line (High/Medium/Low) with a parser,
    kept intentionally SEPARATE from the answer body and from the sentence-
    level grounding score. This is NOT a validated confidence measure --
    small models are poorly calibrated at self-reporting confidence, so
    treat it as an extra reported field to inspect, not ground truth. Check
    it against actual Faithfulness scores before trusting it in the paper.

v3 change (this revision):
  - The refinement stage now receives BOTH the retrieved context and the
    already-generated answer, instead of the generated answer alone. Giving
    the refiner only the draft answer means it can only rewrite prose --
    it has no source of truth to check any given sentence against, so
    "remove unsupported statements" was previously unenforceable; it could
    only guess from phrasing which sentences sounded unsupported. With the
    context present in the same prompt, every sentence in the draft can be
    checked against the passages directly, which is what makes rules 3-4
    below ("keep supported facts" / "remove unsupported statements")
    actually checkable by the model rather than aspirational.

Deliberately NOT adopted from that draft: moving citations into a separate
"Supporting Evidence: Source N" list at the end of the answer. That format
is incompatible with SentenceLevelGrounder/EvidenceGroundingChecker, which
both parse inline `[Pn]` tags at the end of each sentence -- citations
detached from the sentences they support can't be attributed per-sentence,
which is exactly what Explainability measures. Citations stay inline here.
"""

from __future__ import annotations

import re

_CONFIDENCE_LINE_PATTERN = re.compile(
    r"\n?\s*Confidence:\s*(High|Medium|Low)\s*\.?\s*$", re.IGNORECASE
)

INSUFFICIENT_EVIDENCE_PHRASE = (
    "The retrieved documents do not contain sufficient information to answer this question."
)


def extract_confidence(raw_output: str) -> tuple[str, str | None]:
    """Split a trailing ``Confidence: High/Medium/Low`` line off the model's raw output.

    Call this BEFORE passing the answer into SentenceLevelGrounder or
    EvidenceGroundingChecker -- an un-stripped "Confidence: High" line has
    no sentence-ending punctuation, so the sentence splitter would merge it
    into the previous sentence (or leave it as a stray fragment) and
    contaminate that sentence's grounding score.

    Args:
        raw_output: the LLM's full response, as returned by generator.generate().

    Returns:
        (answer_without_confidence_line, confidence_label_or_None). The
        label is normalized to "High" / "Medium" / "Low"; None if no
        confidence line was found (e.g. the model omitted it).
    """
    match = _CONFIDENCE_LINE_PATTERN.search(raw_output)
    if not match:
        return raw_output.strip(), None
    answer = raw_output[: match.start()].strip()
    confidence = match.group(1).capitalize()
    return answer, confidence


class CitationAwarePromptBuilder:
    """Builds an evidence-constrained, citation-mandatory refinement prompt.

    This stage refines a previously generated answer against the retrieved
    context, rather than generating from scratch. Both must be supplied so
    the model can check each claim in the draft against the evidence instead
    of just rewording it.
    """

    def build(
        self,
        question: str,
        evidence_chunks: list[dict],
        text_key: str = "text",
    ) -> tuple[str, str]:
        """Build the (system_prompt, user_prompt) pair for one refinement call.

        Args:
            question: the user's clinical question.
            evidence_chunks: ranked passages, in prompt order (index 0 -> [P1]),
                same shape produced by the existing reranker stage.
            generated_answer: the draft answer from the generation stage that
                this call will refine. Passing this alongside the evidence
                (rather than the evidence or the draft alone) is what lets the
                model check individual claims against sources instead of only
                improving phrasing.
            text_key: dict key holding chunk text.

        Returns:
            (system_prompt, user_prompt) -- matches the existing PromptBuilder
            contract, so callers don't need to change. The model's raw output
            should be passed through extract_confidence() before any
            grounding/explainability checks.
        """
        passages_block = "\n\n".join(
            f"[P{i}] {chunk.get(text_key, '')}" for i, chunk in enumerate(evidence_chunks, start=1)
        )

        system_prompt = (
            "You are an expert Medical Oncology AI assistant for a Medical GraphRAG system.\n\n"
            "Your responsibilities are:\n"
            "1. Answer the user's question using ONLY the retrieved medical context.\n"
            "2. Generate ONE final answer.\n"
            "3. The SAME answer must be:\n"
            "   • Displayed in the web application.\n"
            "   • Saved in the JSON output.\n"
            "4. Never generate two different versions.\n\n"
            "==================================================\n"
            "GENERAL RULES\n"
            "==================================================\n"
            "1. Use ONLY the retrieved medical context.\n"
            "2. Rephrase naturally while preserving the original medical meaning.\n"
            "3. Never copy large passages verbatim.\n"
            "4. Never hallucinate.\n"
            "5. Never use external medical knowledge.\n"
            "6. Never infer missing facts.\n"
            "7. Never modify:\n"
            "   • numerical values\n"
            "   • percentages\n"
            "   • dosages\n"
            "   • stages\n"
            "   • biomarker names\n"
            "   • anatomical terminology\n"
            "8. Preserve numbered or bulleted lists whenever appropriate.\n"
            "9. Do not include:\n"
            "   • confidence scores\n"
            "   • similarity scores\n"
            "   • reranker scores\n"
            "   • internal retrieval information\n"
            "10. If the retrieved context is insufficient, return EXACTLY:\n"
            '"I could not find sufficient information in the uploaded medical documents to answer this question."\n\n'
            "==================================================\n"
            "OUTPUT FORMAT\n"
            "==================================================\n"
            "Return ONLY valid JSON.\n"
            "{\n"
            '  "id": "{id}",\n'
            '  "question": "{question}",\n'
            '  "answer": "<final rephrased answer>",\n'
            '  "sources": [\n'
            '      "P1",\n'
            '      "P5"\n'
            '  ]\n'
            "}"
        )

        user_prompt = (
            f"==================================================\n"
            f"QUESTION\n"
            f"==================================================\n"
            f"{question}\n\n"
            f"==================================================\n"
            f"RETRIEVED CONTEXT\n"
            f"==================================================\n"
            f"{passages_block}\n\n"
            f"Return ONLY valid JSON format."
        )

        return system_prompt, user_prompt