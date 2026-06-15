"""Unit tests for the unified LLM client."""

import httpx
import pytest
import respx

from neuralops.core.exceptions import LLMProviderError, LLMRateLimitError
from neuralops.services.llm_client import (
    LLMBackend,
    LLMClient,
    LLMResponse,
    _calculate_cost,
    _count_tokens,
)

# ── _count_tokens ─────────────────────────────────────────────────────────────


def test_count_tokens_returns_positive_int():
    count = _count_tokens("Hello, world!")
    assert isinstance(count, int)
    assert count > 0


def test_count_tokens_scales_with_length():
    short = _count_tokens("Hi")
    long = _count_tokens("This is a much longer sentence with many tokens in it.")
    assert long > short


def test_count_tokens_unknown_model_fallback():
    # Should not raise — falls back to cl100k_base encoding
    count = _count_tokens("test text", model="completely-unknown-model")
    assert count > 0


# ── _calculate_cost ───────────────────────────────────────────────────────────


def test_calculate_cost_groq_known():
    cost = _calculate_cost("llama-3.1-8b-instant", LLMBackend.GROQ, 1_000_000, 1_000_000)
    assert abs(cost - 0.13) < 0.001


def test_calculate_cost_mistral_known():
    cost = _calculate_cost("mistral-small-latest", LLMBackend.MISTRAL, 1_000_000, 1_000_000)
    assert abs(cost - 0.80) < 0.001


def test_calculate_cost_unknown_model_uses_default():
    # Unknown model falls back to default pricing (not 0.0)
    cost = _calculate_cost("unknown-model", LLMBackend.GROQ, 1_000_000, 1_000_000)
    assert cost > 0.0


def test_calculate_cost_ollama_is_free():
    cost = _calculate_cost("llama3", LLMBackend.OLLAMA, 1_000_000, 1_000_000)
    assert cost == 0.0


def test_calculate_cost_zero_tokens():
    cost = _calculate_cost("llama-3.1-8b-instant", LLMBackend.GROQ, 0, 0)
    assert cost == 0.0


# ── LLMBackend enum ───────────────────────────────────────────────────────────


def test_llm_backend_string_values():
    assert LLMBackend.GROQ == "groq"
    assert LLMBackend.MISTRAL == "mistral"
    assert LLMBackend.OLLAMA == "ollama"


def test_llm_backend_from_string():
    backend = LLMBackend("groq")
    assert backend == LLMBackend.GROQ


# ── LLMResponse dataclass ─────────────────────────────────────────────────────


def test_llm_response_fields():
    resp = LLMResponse(
        content="Hello",
        model="llama-3.1-8b-instant",
        backend=LLMBackend.GROQ,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cost_usd=0.00001,
        latency_ms=150.0,
    )
    assert resp.content == "Hello"
    assert resp.total_tokens == 15
    assert resp.backend == LLMBackend.GROQ


# ── LLMClient.complete (Groq via respx mock) ──────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_groq_success():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Test response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "model": "llama-3.1-8b-instant",
            },
        )
    )
    client = LLMClient()
    resp = await client.complete("Hello", backend=LLMBackend.GROQ)
    assert resp.content == "Test response"
    assert resp.backend == LLMBackend.GROQ
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 5
    assert resp.latency_ms > 0


@pytest.mark.asyncio
@respx.mock
async def test_complete_groq_with_system_prompt():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
                "model": "llama-3.1-8b-instant",
            },
        )
    )
    client = LLMClient()
    resp = await client.complete("Hi", system="You are helpful.", backend=LLMBackend.GROQ)
    assert resp.content == "OK"


@pytest.mark.asyncio
@respx.mock
async def test_complete_groq_rate_limit_raises():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limit"}})
    )
    client = LLMClient()
    with pytest.raises((LLMRateLimitError, LLMProviderError)):
        await client.complete("Hello", backend=LLMBackend.GROQ)


@pytest.mark.asyncio
@respx.mock
async def test_complete_groq_server_error_raises():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": {"message": "server error"}})
    )
    client = LLMClient()
    with pytest.raises((LLMProviderError, Exception)):
        await client.complete("Hello", backend=LLMBackend.GROQ)


# ── complete_with_fallback ────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_with_fallback_primary_succeeds():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Primary response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                "model": "llama-3.1-8b-instant",
            },
        )
    )
    client = LLMClient()
    resp = await client.complete_with_fallback(
        "Hello",
        primary=LLMBackend.GROQ,
        fallback=LLMBackend.GROQ,
    )
    assert resp.content == "Primary response"


# ── _get_mistral_client lazy init ────────────────────────────────────────────


def test_mistral_client_lazy_init():
    client = LLMClient()
    assert client._mistral_client is None
    mc = client._get_mistral_client()
    assert mc is not None
    # Second call returns same cached instance
    mc2 = client._get_mistral_client()
    assert mc is mc2


# ── Ollama backend via respx mock ─────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_complete_ollama_success():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Ollama says hello"}, "done": True},
        )
    )
    client = LLMClient()
    resp = await client.complete("Hello", backend=LLMBackend.OLLAMA)
    assert resp.content == "Ollama says hello"
    assert resp.backend == LLMBackend.OLLAMA
    assert resp.cost_usd == 0.0


@pytest.mark.asyncio
@respx.mock
async def test_complete_ollama_timeout_raises():
    respx.post("http://localhost:11434/api/chat").mock(
        side_effect=httpx.TimeoutException("timed out")
    )
    client = LLMClient()
    from neuralops.core.exceptions import LLMTimeoutError
    with pytest.raises(LLMTimeoutError):
        await client.complete("Hello", backend=LLMBackend.OLLAMA)


@pytest.mark.asyncio
@respx.mock
async def test_complete_ollama_http_error_raises():
    respx.post("http://localhost:11434/api/chat").mock(
        return_value=httpx.Response(503, json={"error": "service unavailable"})
    )
    client = LLMClient()
    with pytest.raises(LLMProviderError):
        await client.complete("Hello", backend=LLMBackend.OLLAMA)


# ── close ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_client_close_does_not_raise():
    client = LLMClient()
    await client.close()  # Should not raise
