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


@pytest.mark.asyncio
@respx.mock
async def test_fixed_evaluation_excludes_failed_pairs_and_tracks_usage(api_client: AsyncClient):
    def complete(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        message = payload["messages"][-1]["content"]
        if "response_format" in payload:
            if "Trial fail" in message and "Version 2.0.0" in message:
                answer = "invalid json"
            else:
                score = 3 if "Version 1.0.0" in message else (9 if "Trial 3" in message else 8)
                answer = json.dumps(dict.fromkeys(("relevance", "accuracy", "clarity", "safety"), score))
        else:
            answer = message
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            "model": payload["model"],
        })

    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(side_effect=complete)
    prompt = await _setup_prompt_with_versions(api_client)
    active = await api_client.post(f"/api/v1/prompts/{prompt['id']}/activate/1.0.0")
    assert active.status_code == 200
    experiment = await api_client.post("/api/v1/experiments/", json={
        "name": "fixed evaluation",
        "prompt_id": prompt["id"],
        "version_a": "1.0.0",
        "version_b": "2.0.0",
        "planned_trials": 3,
    })
    assert experiment.status_code == 201
    experiment_id = experiment.json()["id"]
    path = f"/api/v1/experiments/{experiment_id}"
    for user_input in ("Trial 1", "Trial 2", "Trial fail"):
        trial = await api_client.post(f"{path}/trials/", json={"user_input": user_input})
        assert trial.status_code == 201
    assert trial.json()["score_b"] is None
    assert trial.json()["scoring_error"]

    midway = (await api_client.get(f"{path}/significance")).json()
    assert (midway["trial_count"], midway["scored_trial_count"]) == (3, 2)
    assert midway["p_value"] is None
    assert midway["mean_score_b"] == 8.0

    assert (await api_client.post(f"{path}/trials/", json={"user_input": "Trial 3"})).status_code == 201
    finished = (await api_client.get(path)).json()
    assert finished["status"] == "completed"
    assert finished["winner"] == "B"
    final = (await api_client.get(f"{path}/significance")).json()
    assert final["scored_trial_count"] == 3
    assert final["confidence_interval_low"] > 0
    assert final["mean_difference_b_minus_a"] == pytest.approx(5.3333)
    assert (await api_client.post(f"{path}/trials/", json={"user_input": "Extra trial"})).status_code == 409

    versions = (await api_client.get(f"/api/v1/prompts/{prompt['id']}/versions/")).json()
    assert [version["version"] for version in versions if version["is_active"]] == ["1.0.0"]
    usage = (await api_client.get("/api/v1/experiments/usage", params={"experiment_id": experiment_id})).json()
    assert usage["trials"] == usage["trials_with_usage"] == 4
    assert {row["stage"] for row in usage["usage"]} == {"generation_a", "generation_b", "judge_a", "judge_b"}
    assert sum(row["requests"] for row in usage["usage"]) == 16
    assert sum(row["prompt_tokens"] for row in usage["usage"]) == 320
    assert sum(row["estimated_cost_usd"] for row in usage["usage"]) > 0
