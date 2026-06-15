"""Token and cost calculation utilities.

Provides per-request cost estimation and aggregate cost reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime

from neuralops.core.logging import get_logger

logger = get_logger(__name__)

# USD per 1M tokens (input, output)
_MODEL_COSTS: dict[str, dict[str, tuple[float, float]]] = {
    "groq": {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "llama3-8b-8192": (0.05, 0.08),
        "llama3-70b-8192": (0.59, 0.79),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "gemma2-9b-it": (0.20, 0.20),
    },
    "mistral": {
        "mistral-small-latest": (0.20, 0.60),
        "mistral-medium-latest": (2.70, 8.10),
        "mistral-large-latest": (2.00, 6.00),
        "open-mistral-7b": (0.25, 0.25),
        "open-mixtral-8x7b": (0.70, 0.70),
    },
    "ollama": {},  # Local, no cost
}


@dataclass
class CostRecord:
    """A single request's cost record.

    Attributes:
        model: Model identifier.
        provider: Provider name.
        prompt_tokens: Input tokens used.
        completion_tokens: Output tokens generated.
        total_tokens: Sum of above.
        cost_usd: Estimated USD cost.
        latency_ms: Request latency.
        timestamp: When the request was made.
    """

    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CostSummary:
    """Aggregate cost summary over a collection of records.

    Attributes:
        total_requests: Number of API calls.
        total_input_tokens: Total input tokens.
        total_output_tokens: Total output tokens.
        total_tokens: Combined token count.
        total_cost_usd: Total estimated cost.
        avg_latency_ms: Mean latency across requests.
        by_model: Per-model breakdown dict.
        by_provider: Per-provider breakdown dict.
    """

    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    avg_latency_ms: float
    by_model: dict[str, dict] = field(default_factory=dict)
    by_provider: dict[str, dict] = field(default_factory=dict)


def calculate_cost(
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Calculate USD cost for a single LLM request.

    Args:
        model: Model identifier string.
        provider: Provider name ("groq" | "mistral" | "ollama").
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Estimated cost in USD, rounded to 8 decimal places.
    """
    provider_costs = _MODEL_COSTS.get(provider, {})
    costs = provider_costs.get(model)
    if costs is None:
        logger.debug("Unknown model for cost calculation, using zeros", model=model, provider=provider)
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * costs[0]
    output_cost = (completion_tokens / 1_000_000) * costs[1]
    return round(input_cost + output_cost, 8)


def summarize_costs(records: list[CostRecord]) -> CostSummary:
    """Aggregate a list of CostRecord objects into a summary.

    Args:
        records: List of individual request cost records.

    Returns:
        CostSummary with totals and per-model/provider breakdowns.
    """
    if not records:
        return CostSummary(
            total_requests=0,
            total_input_tokens=0,
            total_output_tokens=0,
            total_tokens=0,
            total_cost_usd=0.0,
            avg_latency_ms=0.0,
        )

    total_cost = sum(r.cost_usd for r in records)
    total_input = sum(r.prompt_tokens for r in records)
    total_output = sum(r.completion_tokens for r in records)
    avg_latency = sum(r.latency_ms for r in records) / len(records)

    by_model: dict[str, dict] = {}
    by_provider: dict[str, dict] = {}

    for r in records:
        if r.model not in by_model:
            by_model[r.model] = {"requests": 0, "tokens": 0, "cost_usd": 0.0}
        by_model[r.model]["requests"] += 1
        by_model[r.model]["tokens"] += r.total_tokens
        by_model[r.model]["cost_usd"] += r.cost_usd

        if r.provider not in by_provider:
            by_provider[r.provider] = {"requests": 0, "tokens": 0, "cost_usd": 0.0}
        by_provider[r.provider]["requests"] += 1
        by_provider[r.provider]["tokens"] += r.total_tokens
        by_provider[r.provider]["cost_usd"] += r.cost_usd

    return CostSummary(
        total_requests=len(records),
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_tokens=total_input + total_output,
        total_cost_usd=round(total_cost, 8),
        avg_latency_ms=round(avg_latency, 2),
        by_model=by_model,
        by_provider=by_provider,
    )
