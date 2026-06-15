"""Unit tests for the cost tracker utilities."""

from neuralops.engine.cost_tracker import (
    CostRecord,
    CostSummary,
    calculate_cost,
    summarize_costs,
)

# ── calculate_cost ────────────────────────────────────────────────────────────


def test_calculate_cost_groq_known_model():
    cost = calculate_cost("llama-3.1-8b-instant", "groq", 1_000_000, 1_000_000)
    assert abs(cost - 0.13) < 0.001  # 0.05 input + 0.08 output


def test_calculate_cost_groq_large_model():
    cost = calculate_cost("llama-3.3-70b-versatile", "groq", 1_000_000, 1_000_000)
    assert abs(cost - 1.38) < 0.001  # 0.59 + 0.79


def test_calculate_cost_mistral_known_model():
    cost = calculate_cost("mistral-small-latest", "mistral", 1_000_000, 1_000_000)
    assert abs(cost - 0.80) < 0.001  # 0.20 + 0.60


def test_calculate_cost_unknown_model_returns_zero():
    cost = calculate_cost("gpt-99-turbo", "groq", 100_000, 50_000)
    assert cost == 0.0


def test_calculate_cost_unknown_provider_returns_zero():
    cost = calculate_cost("some-model", "unknown-provider", 100_000, 50_000)
    assert cost == 0.0


def test_calculate_cost_ollama_is_free():
    # ollama provider costs dict is empty, so unknown model path → 0.0
    cost = calculate_cost("llama3", "ollama", 100_000, 50_000)
    assert cost == 0.0


def test_calculate_cost_zero_tokens():
    cost = calculate_cost("llama-3.1-8b-instant", "groq", 0, 0)
    assert cost == 0.0


def test_calculate_cost_rounding():
    # Small request shouldn't return negative or wild floats
    cost = calculate_cost("llama-3.1-8b-instant", "groq", 100, 50)
    assert 0.0 <= cost < 0.01


# ── summarize_costs ───────────────────────────────────────────────────────────


def test_summarize_costs_empty_list():
    summary = summarize_costs([])
    assert summary.total_requests == 0
    assert summary.total_cost_usd == 0.0
    assert summary.avg_latency_ms == 0.0
    assert summary.by_model == {}
    assert summary.by_provider == {}


def _make_record(model="llama-3.1-8b-instant", provider="groq",
                 prompt_tokens=100, completion_tokens=50,
                 cost=0.001, latency=120.0) -> CostRecord:
    return CostRecord(
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost,
        latency_ms=latency,
    )


def test_summarize_costs_single_record():
    rec = _make_record(prompt_tokens=100, completion_tokens=50, cost=0.001, latency=200.0)
    summary = summarize_costs([rec])
    assert summary.total_requests == 1
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 50
    assert summary.total_tokens == 150
    assert abs(summary.total_cost_usd - 0.001) < 1e-9
    assert abs(summary.avg_latency_ms - 200.0) < 0.01


def test_summarize_costs_multiple_records_aggregated():
    records = [
        _make_record(cost=0.001, latency=100.0),
        _make_record(cost=0.002, latency=200.0),
        _make_record(cost=0.003, latency=300.0),
    ]
    summary = summarize_costs(records)
    assert summary.total_requests == 3
    assert abs(summary.total_cost_usd - 0.006) < 1e-9
    assert abs(summary.avg_latency_ms - 200.0) < 0.01


def test_summarize_costs_by_model_breakdown():
    records = [
        _make_record(model="model-a", provider="groq", cost=0.01),
        _make_record(model="model-a", provider="groq", cost=0.02),
        _make_record(model="model-b", provider="groq", cost=0.03),
    ]
    summary = summarize_costs(records)
    assert "model-a" in summary.by_model
    assert "model-b" in summary.by_model
    assert summary.by_model["model-a"]["requests"] == 2
    assert summary.by_model["model-b"]["requests"] == 1
    assert abs(summary.by_model["model-a"]["cost_usd"] - 0.03) < 1e-9


def test_summarize_costs_by_provider_breakdown():
    records = [
        _make_record(provider="groq", cost=0.01),
        _make_record(provider="groq", cost=0.02),
        _make_record(provider="mistral", cost=0.05),
    ]
    summary = summarize_costs(records)
    assert "groq" in summary.by_provider
    assert "mistral" in summary.by_provider
    assert summary.by_provider["groq"]["requests"] == 2
    assert summary.by_provider["mistral"]["requests"] == 1


def test_summarize_costs_token_totals():
    records = [
        _make_record(prompt_tokens=100, completion_tokens=50),
        _make_record(prompt_tokens=200, completion_tokens=100),
    ]
    summary = summarize_costs(records)
    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 150
    assert summary.total_tokens == 450


# ── CostRecord dataclass ──────────────────────────────────────────────────────


def test_cost_record_has_timestamp():
    rec = _make_record()
    assert rec.timestamp is not None


def test_cost_record_fields():
    rec = _make_record(model="test-model", provider="groq", prompt_tokens=10,
                       completion_tokens=5, cost=0.0001, latency=50.0)
    assert rec.model == "test-model"
    assert rec.provider == "groq"
    assert rec.prompt_tokens == 10
    assert rec.completion_tokens == 5
    assert rec.total_tokens == 15
    assert rec.cost_usd == 0.0001
    assert rec.latency_ms == 50.0


# ── CostSummary dataclass ─────────────────────────────────────────────────────


def test_cost_summary_default_dicts():
    summary = CostSummary(
        total_requests=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_tokens=0,
        total_cost_usd=0.0,
        avg_latency_ms=0.0,
    )
    assert summary.by_model == {}
    assert summary.by_provider == {}
