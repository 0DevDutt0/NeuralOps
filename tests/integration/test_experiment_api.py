"""Integration tests for the A/B experiment API endpoints."""

import json

import httpx
import pytest
import respx
from httpx import AsyncClient


async def _setup_prompt_with_versions(api_client: AsyncClient) -> dict:
    """Helper: create a prompt with two versions and return prompt data."""
    p = await api_client.post("/api/v1/prompts/", json={"name": "exp-prompt"})
    prompt = p.json()
    for ver in ["1.0.0", "2.0.0"]:
        await api_client.post(
            f"/api/v1/prompts/{prompt['id']}/versions/",
            json={"version": ver, "content": f"Version {ver}: {{input}}"},
        )
    return prompt


@pytest.mark.asyncio
async def test_create_experiment(api_client: AsyncClient):
    prompt = await _setup_prompt_with_versions(api_client)
    resp = await api_client.post(
        "/api/v1/experiments/",
        json={
            "name": "test-exp",
            "prompt_id": prompt["id"],
            "version_a": "1.0.0",
            "version_b": "2.0.0",
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "test-exp"
    assert data["status"] == "running"
    assert data["trial_count"] == 0


@pytest.mark.asyncio
async def test_list_experiments_empty(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/experiments/")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_experiment_not_found(api_client: AsyncClient):
    resp = await api_client.get("/api/v1/experiments/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_experiment_invalid_version(api_client: AsyncClient):
    p = await api_client.post("/api/v1/prompts/", json={"name": "inv-exp"})
    resp = await api_client.post(
        "/api/v1/experiments/",
        json={
            "name": "bad-exp",
            "prompt_id": p.json()["id"],
            "version_a": "9.9.9",
            "version_b": "8.8.8",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_significance_no_trials(api_client: AsyncClient):
    prompt = await _setup_prompt_with_versions(api_client)
    exp_resp = await api_client.post(
        "/api/v1/experiments/",
        json={
            "name": "sig-exp",
            "prompt_id": prompt["id"],
            "version_a": "1.0.0",
            "version_b": "2.0.0",
        },
    )
    exp_id = exp_resp.json()["id"]
    resp = await api_client.get(f"/api/v1/experiments/{exp_id}/significance")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_significant"] is False
    assert data["p_value"] is None


@pytest.mark.asyncio
@respx.mock
async def test_run_trial(api_client: AsyncClient):
    """Run a trial with mocked LLM calls."""
    # Mock both completion calls and judge calls
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "Response A"}}],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
                    "model": "llama-3.1-8b-instant",
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "Response B"}}],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 12, "total_tokens": 27},
                    "model": "llama-3.1-8b-instant",
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": json.dumps({
                        "relevance": 8.0, "accuracy": 7.0, "clarity": 8.0, "safety": 9.0,
                        "composite": 8.0, "reasoning": "Good", "confidence": 0.85,
                    })}}],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
                    "model": "llama-3.3-70b-versatile",
                },
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": json.dumps({
                        "relevance": 7.0, "accuracy": 6.0, "clarity": 7.0, "safety": 9.0,
                        "composite": 7.0, "reasoning": "Decent", "confidence": 0.75,
                    })}}],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
                    "model": "llama-3.3-70b-versatile",
                },
            ),
        ]
    )

    prompt = await _setup_prompt_with_versions(api_client)
    exp_resp = await api_client.post(
        "/api/v1/experiments/",
        json={
            "name": "trial-test",
            "prompt_id": prompt["id"],
            "version_a": "1.0.0",
            "version_b": "2.0.0",
        },
    )
    exp_id = exp_resp.json()["id"]

    trial_resp = await api_client.post(
        f"/api/v1/experiments/{exp_id}/trials/",
        json={"user_input": "What is machine learning?"},
    )
    assert trial_resp.status_code == 201
    trial = trial_resp.json()
    assert "output_a" in trial
    assert "output_b" in trial
    assert trial["score_a"] >= 0
    assert trial["score_b"] >= 0
