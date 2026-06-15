"""Unit tests for the A/B experiment runner."""

import json

import httpx
import pytest
import respx

from neuralops.engine.ab_runner import ABRunner, _run_significance_test


def _make_llm_response(content: str = "Mock output") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50},
            "model": "llama-3.1-8b-instant",
        },
    )


def _make_judge_response(score: float = 7.5) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "relevance": score,
                        "accuracy": score,
                        "clarity": score,
                        "safety": score,
                        "composite": score,
                        "reasoning": "Test reasoning",
                        "confidence": 0.8,
                    }),
                }
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
            "model": "llama-3.3-70b-versatile",
        },
    )


# ── Significance test unit tests (pure logic, no mocking needed) ──────────────


def test_significance_insufficient_data():
    result = _run_significance_test([5.0], [6.0], 0.05)
    assert result.p_value is None
    assert result.is_significant is False
    assert result.winner is None


def test_significance_clearly_different_groups():
    # Group A: consistently low scores
    scores_a = [4.0, 4.1, 3.9, 4.2, 4.0] * 8  # 40 samples
    # Group B: consistently high scores
    scores_b = [8.0, 8.1, 7.9, 8.2, 8.0] * 8  # 40 samples
    result = _run_significance_test(scores_a, scores_b, 0.05)
    assert result.is_significant is True
    assert result.winner == "B"
    assert result.p_value is not None
    assert result.p_value < 0.05


def test_significance_similar_groups():
    # Groups with the same distribution — should not be significant
    scores = [7.0, 7.1, 6.9, 7.0, 7.1] * 8
    result = _run_significance_test(scores, scores, 0.05)
    # With identical data, t-test returns NaN p-value — handle gracefully
    assert result.winner is None or result.is_significant is False


def test_significance_mean_calculation():
    scores_a = [6.0, 8.0]  # mean 7.0
    scores_b = [4.0, 6.0]  # mean 5.0
    result = _run_significance_test(scores_a, scores_b, 0.05)
    assert abs(result.mean_a - 7.0) < 0.01
    assert abs(result.mean_b - 5.0) < 0.01


# ── ABRunner integration test with mocked LLM ─────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_ab_runner_single_trial(db_session):
    """Run a single trial end-to-end with mocked LLM calls."""
    # Mock all Groq calls — both completions and judge
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=[
            _make_llm_response("Response from version A"),
            _make_llm_response("Response from version B"),
            _make_judge_response(8.0),  # score for A
            _make_judge_response(7.0),  # score for B
        ]
    )

    # Create experiment and version records in DB
    from neuralops.schemas.experiment import ExperimentCreate
    from neuralops.schemas.prompt import PromptCreate, PromptVersionCreate
    from neuralops.services import prompt_service
    from neuralops.services.experiment_service import create_experiment

    prompt = await prompt_service.create_prompt(db_session, PromptCreate(name="test-prompt-ab"))
    await prompt_service.create_version(
        db_session, prompt.id, PromptVersionCreate(version="1.0.0", content="v1: {input}")
    )
    await prompt_service.create_version(
        db_session, prompt.id, PromptVersionCreate(version="2.0.0", content="v2: {input}")
    )

    exp = await create_experiment(
        db_session,
        ExperimentCreate(
            name="test-exp",
            prompt_id=prompt.id,
            version_a="1.0.0",
            version_b="2.0.0",
        ),
    )

    # Fetch the ORM object for runner
    from sqlalchemy import select

    from neuralops.models.experiment import Experiment as ExperimentModel
    result = await db_session.execute(
        select(ExperimentModel).where(ExperimentModel.id == exp.id)
    )
    orm_exp = result.scalar_one()

    runner = ABRunner()
    trial = await runner.run_trial(
        db=db_session,
        experiment=orm_exp,
        user_input="What is Python?",
        content_a="v1: {input}",
        content_b="v2: {input}",
    )

    assert trial.output_a == "Response from version A"
    assert trial.output_b == "Response from version B"
    assert trial.score_a >= 0
    assert trial.score_b >= 0
    assert trial.winner_this_trial in ("A", "B", "tie")
