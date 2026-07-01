"""Token counting and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import COST_PER_1K_INPUT, COST_PER_1K_OUTPUT


@dataclass
class CostEstimate:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


def estimate_cost(input_tokens: int, output_tokens: int) -> CostEstimate:
    """Estimate USD cost based on the configured price model."""
    cost = (
        input_tokens / 1000 * COST_PER_1K_INPUT
        + output_tokens / 1000 * COST_PER_1K_OUTPUT
    )
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=round(cost, 6),
    )


def count_words_as_tokens(text: str) -> int:
    """Approximate token count using word count (1 word ≈ 1.3 tokens)."""
    return int(len(text.split()) * 1.3)


def try_tiktoken_count(text: str, model: str = "gpt-4o-mini") -> int:
    """Use tiktoken if available, else fall back to word approximation."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return count_words_as_tokens(text)
