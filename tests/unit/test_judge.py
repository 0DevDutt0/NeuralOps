"""Unit tests for the LLM-as-Judge scoring engine."""

import json

import httpx
import pytest
import respx

from neuralops.engine.judge import _WEIGHTS, LLMJudge, _parse_judge_response


def _make_judge_response(data: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": json.dumps(data)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "model": "llama-3.3-70b-versatile",
        },
    )


@pytest.mark.asyncio
@respx.mock
async def test_judge_score_returns_correct_composite():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=_make_judge_response({
            "relevance": 8.0,
            "accuracy": 7.0,
            "clarity": 9.0,
            "safety": 10.0,
            "reasoning": "Good response",
            "confidence": 0.9,
        })
    )
    judge = LLMJudge()
    score = await judge.score("Test response", user_input="Test input")
    expected = 8.0 * 0.30 + 7.0 * 0.30 + 9.0 * 0.20 + 10.0 * 0.20
    assert abs(score.composite - expected) < 0.01
    assert score.reasoning == "Good response"
    assert score.confidence == 0.9


@pytest.mark.asyncio
@respx.mock
async def test_judge_score_pair_returns_two_scores():
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=_make_judge_response({
            "relevance": 7.0,
            "accuracy": 7.0,
            "clarity": 7.0,
            "safety": 7.0,
            "reasoning": "Average",
            "confidence": 0.7,
        })
    )
    judge = LLMJudge()
    score_a, score_b = await judge.score_pair("Response A", "Response B")
    assert score_a.composite > 0
    assert score_b.composite > 0


def test_parse_judge_response_clamps_scores():
    raw = json.dumps({
        "relevance": 15.0,  # Over 10 — should clamp to 10
        "accuracy": -2.0,   # Under 1 — should clamp to 1
        "clarity": 8.0,
        "safety": 9.0,
        "reasoning": "test",
        "confidence": 0.5,
    })
    result = _parse_judge_response(raw, "test-model", ["relevance", "accuracy"])
    assert result.relevance == 10.0
    assert result.accuracy == 1.0


def test_parse_judge_response_uses_composite_if_provided():
    raw = json.dumps({
        "relevance": 5.0,
        "accuracy": 5.0,
        "clarity": 5.0,
        "safety": 5.0,
        "composite": 9.99,  # Explicit override
        "reasoning": "Custom composite",
        "confidence": 1.0,
    })
    result = _parse_judge_response(raw, "test-model", [])
    assert result.composite == 9.99


@pytest.mark.asyncio
@respx.mock
async def test_judge_returns_default_on_json_parse_error():
    """If the judge returns non-JSON, we get default 5.0 scores without raising."""
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "not valid json!"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "model": "llama-3.3-70b-versatile",
            },
        )
    )
    judge = LLMJudge()
    score = await judge.score("some response")
    assert score.composite == 5.0
    assert score.confidence == 0.0


def test_judge_weights_sum_to_one():
    total = sum(_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9
