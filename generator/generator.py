"""generator.py
------------
Structured Grounded Answer Generation Module.

Executes low-temperature, constrained LLM generation over reranked clinical evidence,
parses claim-level evidence citations, and validates answer grounding.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from generation.prompt_builder import EvidenceItem, build_prompt

logger = logging.getLogger(__name__)

ANSWER_CACHE = {}
@dataclass
class Claim:
    text: str
    evidence_ids: List[str]
    confidence: str  # "high" | "medium" | "low" | "unsupported"
    grounded: Optional[bool] = None       # filled in by grounding_checker
    grounding_score: Optional[float] = None


@dataclass
class GroundedAnswer:
    question: str
    claims: List[Claim] = field(default_factory=list)
    unanswerable_aspects: List[str] = field(default_factory=list)
    evidence_used: List[EvidenceItem] = field(default_factory=list)
    retried: bool = False
    raw_model_output: str = ""


# --------------------------------------------------------------------------
# JSON extraction with repair
# --------------------------------------------------------------------------


def _extract_json(raw: str) -> Optional[dict]:
    raw = raw.strip()
    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass
    # strip markdown fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    # grab first {...} block greedily
    brace = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except Exception:
            pass
    return None


def call_llm_json(prompt: str, llm_client, temperature: float = 0.05,
                   max_tokens: int = 600) -> Dict[str, Any]:
    raw = llm_client.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    parsed = _extract_json(raw)
    if parsed is None:
        logger.warning("Failed to parse JSON from model output; treating as zero claims.")
        return {"claims": [], "unanswerable_aspects": [], "_raw": raw, "_parse_failed": True}
    parsed["_raw"] = raw
    parsed["_parse_failed"] = False
    return parsed


def _to_claims(payload: dict) -> List[Claim]:
    claims = []
    for c in payload.get("claims", []):
        claims.append(Claim(
            text=str(c.get("text", "")).strip(),
            evidence_ids=list(c.get("evidence_ids", []) or []),
            confidence=str(c.get("confidence", "low")),
        ))
    return claims


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def generate_grounded_answer(
    question: str,
    evidence_list: List[EvidenceItem],
    llm_client,
    grounding_check_fn,  # from grounding_checker.py: (claims, evidence_list, embedder) -> List[Claim] with .grounded set
    embedder=None,
    retry_threshold: float = 0.20,
    chat_history: Optional[str] = None,
) -> GroundedAnswer:
    prompt = build_prompt(question, evidence_list, chat_history=chat_history)
    payload = call_llm_json(prompt, llm_client)
    claims = _to_claims(payload)

    # First-pass grounding check (cheap embedding similarity -- see
    # grounding_checker.py). This is the SAME check used at evaluation
    # time, so gating on it here directly targets the metric.
    claims = grounding_check_fn(claims, evidence_list, embedder)

    ungrounded_frac = (
        sum(1 for c in claims if c.grounded is False) / len(claims) if claims else 0.0
    )

    retried = False
    if ungrounded_frac > retry_threshold:
        retried = True
        correction_prompt = _build_correction_prompt(prompt, claims)
        payload2 = call_llm_json(correction_prompt, llm_client)
        claims2 = _to_claims(payload2)
        claims2 = grounding_check_fn(claims2, evidence_list, embedder)
        # only accept the retry if it's actually better
        new_ungrounded_frac = (
            sum(1 for c in claims2 if c.grounded is False) / len(claims2) if claims2 else 1.0
        )
        if claims2 and new_ungrounded_frac <= ungrounded_frac:
            claims = claims2
            payload = payload2
        
    return GroundedAnswer(
        question=question,
        claims=claims,
        unanswerable_aspects=list(payload.get("unanswerable_aspects", []) or []),
        evidence_used=evidence_list,
        retried=retried,
        raw_model_output=payload.get("_raw", ""),
    )


def _build_correction_prompt(original_prompt: str, claims: List[Claim]) -> str:
    ungrounded = [c for c in claims if c.grounded is False]
    listing = "\n".join(f'- "{c.text}" (cited {c.evidence_ids})' for c in ungrounded)
    correction = f"""
Your previous answer contained claims that are NOT actually supported by
the cited evidence blocks (verified by similarity check):
{listing}

Re-generate your answer. For each of these claims, either:
(a) find the CORRECT evidence_id(s) that truly support it, or
(b) change confidence to "unsupported" and set evidence_ids to [].
Do not introduce any new unsupported claims. Respond with ONLY the JSON object.
"""
    return original_prompt + "\n\n" + correction


# --------------------------------------------------------------------------
# Human-readable assembly (post grounding-filter -- see grounding_checker.py
# for how unsupported/ungrounded claims get dropped or flagged before this
# is called)
# --------------------------------------------------------------------------


def assemble_answer_text(claims: List[Claim], include_low_confidence: bool = True) -> str:
    lines = []
    for c in claims:
        if c.grounded is False and c.confidence != "low":
            continue  # dropped by grounding checker; never shown to user
        marker = "".join(f"[{eid}]" for eid in c.evidence_ids) if c.evidence_ids else ""
        if c.confidence == "unsupported":
            if include_low_confidence:
                lines.append(f"{c.text} (low confidence — not directly supported by retrieved evidence)")
            continue
        lines.append(f"{c.text} {marker}".strip())
    return " ".join(lines)
