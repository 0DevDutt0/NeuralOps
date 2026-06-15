"""Unit tests for the experiment service layer (direct calls, no HTTP)."""

import json

import httpx
import pytest
import respx
from sqlalchemy import delete, select

from neuralops.core.exceptions import ExperimentNotFoundError, VersionNotFoundError
from neuralops.models.prompt import PromptVersion
from neuralops.schemas.experiment import ExperimentCreate
from neuralops.schemas.prompt import PromptCreate, PromptVersionCreate
from neuralops.services import experiment_service, prompt_service


async def _make_experiment(db, prompt_name="exp-prompt"):
    """Helper: create prompt with two versions and a new experiment."""
    prompt = await prompt_service.create_prompt(db, PromptCreate(name=prompt_name))
    await prompt_service.create_version(
        db, prompt.id, PromptVersionCreate(version="1.0.0", content="Version 1: {input}")
    )
    await prompt_service.create_version(
        db, prompt.id, PromptVersionCreate(version="2.0.0", content="Version 2: {input}")
    )
    exp = await experiment_service.create_experiment(
        db,
        ExperimentCreate(name="direct-test", prompt_id=prompt.id, version_a="1.0.0", version_b="2.0.0"),
    )
    return prompt, exp


# ── create_experiment ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_experiment_missing_version_raises(db_session):
    prompt = await prompt_service.create_prompt(db_session, PromptCreate(name="no-ver"))
    with pytest.raises(VersionNotFoundError):
        await experiment_service.create_experiment(
            db_session,
            ExperimentCreate(
                name="bad-exp", prompt_id=prompt.id, version_a="9.9.9", version_b="8.8.8"
            ),
        )


# ── list_experiments ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_experiments_empty(db_session):
    result = await experiment_service.list_experiments(db_session)
    assert result == []


@pytest.mark.asyncio
async def test_list_experiments_with_status_filter(db_session):
    _, exp = await _make_experiment(db_session)
    running = await experiment_service.list_experiments(db_session, status="running")
    assert len(running) == 1
    completed = await experiment_service.list_experiments(db_session, status="completed")
    assert len(completed) == 0


@pytest.mark.asyncio
async def test_list_experiments_pagination(db_session):
    _, _ = await _make_experiment(db_session, "p1")
    _, _ = await _make_experiment(db_session, "p2")
    first = await experiment_service.list_experiments(db_session, skip=0, limit=1)
    assert len(first) == 1


# ── get_experiment ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_experiment_not_found(db_session):
    with pytest.raises(ExperimentNotFoundError):
        await experiment_service.get_experiment(db_session, "ghost-id")


@pytest.mark.asyncio
async def test_get_experiment_detail(db_session):
    _, exp = await _make_experiment(db_session)
    detail = await experiment_service.get_experiment(db_session, exp.id)
    assert detail.id == exp.id
    assert detail.recent_trials == []


# ── get_significance ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_significance_not_found(db_session):
    with pytest.raises(ExperimentNotFoundError):
        await experiment_service.get_significance(db_session, "ghost-id")


@pytest.mark.asyncio
async def test_get_significance_no_trials(db_session):
    _, exp = await _make_experiment(db_session)
    sig = await experiment_service.get_significance(db_session, exp.id)
    assert sig.experiment_id == exp.id
    assert sig.is_significant is False
    assert sig.p_value is None
    assert sig.trial_count == 0


# ── run_trial ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_run_trial_direct(db_session):
    """Run a trial via the service layer with mocked LLM calls."""

    def _llm_resp(text):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
                "model": "llama-3.1-8b-instant",
            },
        )

    def _judge_resp(score):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": json.dumps({
                    "relevance": score, "accuracy": score, "clarity": score, "safety": score,
                    "composite": score, "reasoning": "OK", "confidence": 0.8,
                })}}],
                "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
                "model": "llama-3.3-70b-versatile",
            },
        )

    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        side_effect=[_llm_resp("A answer"), _llm_resp("B answer"), _judge_resp(8.0), _judge_resp(7.0)]
    )

    _, exp = await _make_experiment(db_session)
    trial = await experiment_service.run_trial(db_session, exp.id, "What is AI?")

    assert trial.output_a == "A answer"
    assert trial.output_b == "B answer"
    assert trial.score_a > 0
    assert trial.score_b > 0


@pytest.mark.asyncio
async def test_run_trial_not_found(db_session):
    with pytest.raises(ExperimentNotFoundError):
        await experiment_service.run_trial(db_session, "ghost-id", "input")


@pytest.mark.asyncio
async def test_run_trial_version_deleted_raises(db_session):
    """VersionNotFoundError if prompt version rows are deleted after experiment creation."""
    _, exp = await _make_experiment(db_session, "deleted-ver-p")

    # Delete the versions so the lookup inside run_trial fails
    await db_session.execute(delete(PromptVersion))
    await db_session.commit()

    with pytest.raises(VersionNotFoundError):
        await experiment_service.run_trial(db_session, exp.id, "any input")


@pytest.mark.asyncio
async def test_run_trial_on_completed_experiment_raises(db_session):
    """Running a trial on a completed experiment should raise."""
    from neuralops.core.exceptions import ExperimentAlreadyCompletedError
    from neuralops.models.experiment import Experiment as ExperimentModel

    _, exp = await _make_experiment(db_session, "completed-p")

    # Mark the experiment as completed
    row = (await db_session.execute(
        select(ExperimentModel).where(ExperimentModel.id == exp.id)
    )).scalar_one()
    row.status = "completed"
    row.winner = "A"
    await db_session.commit()

    with pytest.raises(ExperimentAlreadyCompletedError):
        await experiment_service.run_trial(db_session, exp.id, "some input")
