"""LLM-as-judge for faithfulness / grounding evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.chunking import Chunk
from src.config import OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class FaithfulnessResult:
    faithfulness_score: float          # 0.0 – 1.0
    supported_claims: list[str]
    unsupported_claims: list[str]
    verdict: str                       # "grounded" | "partially_grounded" | "unsupported"
    explanation: str
    judge_model: str                   # "openai" | "lexical_fallback"


_JUDGE_SYSTEM = """\
You are a faithfulness judge for a RAG system.
Given an answer and a set of source context chunks, evaluate whether the answer
is fully supported by the retrieved context.

Respond ONLY with valid JSON in this exact structure:
{
  "claims": [
    {"claim": "<sentence>", "supported": true, "reason": "<brief reason>"},
    ...
  ],
  "faithfulness_score": 0.85,
  "verdict": "grounded",
  "explanation": "<one paragraph>"
}

verdict must be one of: "grounded", "partially_grounded", "unsupported"
faithfulness_score is the fraction of claims that are supported (0.0 to 1.0).
"""


def judge_faithfulness(
    answer: str,
    chunks: list[tuple[Chunk, float]],
) -> FaithfulnessResult:
    """Route to OpenAI judge or lexical fallback."""
    if OPENAI_API_KEY:
        return _openai_judge(answer, chunks)
    return _lexical_fallback_judge(answer, chunks)


# ── OpenAI judge ──────────────────────────────────────────────────────────────

def _openai_judge(
    answer: str,
    chunks: list[tuple[Chunk, float]],
) -> FaithfulnessResult:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    context = "\n\n---\n\n".join(
        f"[{c.chunk_id}]\n{c.text}" for c, _ in chunks
    )

    user_msg = (
        f"Context chunks:\n{context}\n\n"
        f"Answer to evaluate:\n{answer}"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return _parse_judge_response(raw, judge_model="openai")


# ── Lexical fallback judge ────────────────────────────────────────────────────

def _lexical_fallback_judge(
    answer: str,
    chunks: list[tuple[Chunk, float]],
) -> FaithfulnessResult:
    """
    Score faithfulness by measuring token overlap between each answer sentence
    and the union of retrieved context tokens. Not semantic, but deterministic
    and always runnable without an API key.
    """
    context_tokens = set()
    for chunk, _ in chunks:
        context_tokens.update(_tokenize(chunk.text))

    sentences = _split_sentences(answer)
    supported: list[str] = []
    unsupported: list[str] = []

    for sent in sentences:
        sent_tokens = _tokenize(sent)
        if not sent_tokens:
            continue
        overlap = len(sent_tokens & context_tokens) / len(sent_tokens)
        if overlap >= 0.35:   # tuneable threshold
            supported.append(sent)
        else:
            unsupported.append(sent)

    total = len(supported) + len(unsupported)
    score = len(supported) / total if total > 0 else 0.0

    if score >= 0.75:
        verdict = "grounded"
    elif score >= 0.35:
        verdict = "partially_grounded"
    else:
        verdict = "unsupported"

    return FaithfulnessResult(
        faithfulness_score=round(score, 3),
        supported_claims=supported,
        unsupported_claims=unsupported,
        verdict=verdict,
        explanation=(
            f"Lexical overlap judge: {len(supported)}/{total} sentences "
            f"share ≥35% token overlap with the retrieved context."
        ),
        judge_model="lexical_fallback",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_judge_response(raw: str, judge_model: str) -> FaithfulnessResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to extract JSON block from markdown fences
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if match:
            data = json.loads(match.group(1))
        else:
            return FaithfulnessResult(
                faithfulness_score=0.0,
                supported_claims=[],
                unsupported_claims=["Could not parse judge response"],
                verdict="unsupported",
                explanation=raw[:500],
                judge_model=judge_model,
            )

    claims = data.get("claims", [])
    supported = [c["claim"] for c in claims if c.get("supported")]
    unsupported = [c["claim"] for c in claims if not c.get("supported")]

    return FaithfulnessResult(
        faithfulness_score=float(data.get("faithfulness_score", 0.0)),
        supported_claims=supported,
        unsupported_claims=unsupported,
        verdict=data.get("verdict", "unsupported"),
        explanation=data.get("explanation", ""),
        judge_model=judge_model,
    )


def _tokenize(text: str) -> set[str]:
    # Lowercase, strip punctuation, filter stopwords
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "to",
        "of", "in", "on", "at", "by", "for", "with", "and", "or",
        "but", "not", "it", "its", "this", "that", "as", "if",
    }
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if t not in _STOP and len(t) > 2}


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]
