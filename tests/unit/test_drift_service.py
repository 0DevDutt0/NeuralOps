"""Unit tests for the drift monitoring service (status/summary functions)."""

import pytest

from neuralops.schemas.prompt import PromptCreate, PromptVersionCreate
from neuralops.services import prompt_service
from neuralops.services.drift_service import get_drift_status, get_drift_summary

# ── get_drift_status ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_drift_status_unknown_prompt(db_session):
    """Unknown prompt ID returns a status with no score and not drifting."""
    status = await get_drift_status(db_session, "nonexistent-prompt-id")
    assert status.prompt_id == "nonexistent-prompt-id"
    assert status.latest_score is None
    assert status.is_drifting is False
    assert status.active_version is None
    assert status.recent_logs == []


@pytest.mark.asyncio
async def test_get_drift_status_with_prompt_no_logs(db_session):
    """Prompt with no drift logs shows None score and no drifting."""
    prompt = await prompt_service.create_prompt(
        db_session, PromptCreate(name="monitored-prompt")
    )
    status = await get_drift_status(db_session, prompt.id)
    assert status.prompt_id == prompt.id
    assert status.prompt_name == "monitored-prompt"
    assert status.latest_score is None
    assert status.is_drifting is False


@pytest.mark.asyncio
async def test_get_drift_status_with_active_version(db_session):
    """Prompt with active version shows it in status."""
    prompt = await prompt_service.create_prompt(
        db_session, PromptCreate(name="versioned-prompt")
    )
    await prompt_service.create_version(
        db_session,
        prompt.id,
        PromptVersionCreate(version="1.0.0", content="You are helpful."),
    )
    await prompt_service.activate_version(db_session, prompt.id, "1.0.0")

    status = await get_drift_status(db_session, prompt.id)
    assert status.active_version == "1.0.0"


# ── get_drift_summary ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_drift_summary_no_prompts(db_session):
    """Empty database returns zero totals."""
    summary = await get_drift_summary(db_session)
    assert summary.total_prompts == 0
    assert summary.drifting_prompts == 0
    assert summary.healthy_prompts == 0
    assert summary.alerts_last_24h == 0
    assert summary.last_global_check is None


@pytest.mark.asyncio
async def test_get_drift_summary_with_active_versions(db_session):
    """Prompts with active versions are tracked in summary total."""
    for name in ["p1", "p2"]:
        prompt = await prompt_service.create_prompt(
            db_session, PromptCreate(name=name)
        )
        await prompt_service.create_version(
            db_session,
            prompt.id,
            PromptVersionCreate(version="1.0.0", content="content"),
        )
        await prompt_service.activate_version(db_session, prompt.id, "1.0.0")

    summary = await get_drift_summary(db_session)
    # Both prompts have active versions — they appear in the total
    assert summary.total_prompts == 2
    # No drift logs yet, so neither is classified as drifting or healthy
    assert summary.drifting_prompts == 0
    assert summary.healthy_prompts == 0


@pytest.mark.asyncio
async def test_get_drift_summary_alert_threshold_present(db_session):
    """Summary always includes the configured alert threshold."""
    summary = await get_drift_summary(db_session)
    assert summary.alert_threshold > 0


@pytest.mark.asyncio
async def test_get_drift_summary_classifies_drifting_and_healthy(db_session):
    """Prompts with drift logs should be classified as drifting or healthy."""
    from datetime import datetime

    from neuralops.core.config import settings
    from neuralops.models.drift_log import DriftLog

    # Create two prompts with active versions
    prompts = []
    for name in ["dp1", "dp2"]:
        p = await prompt_service.create_prompt(db_session, PromptCreate(name=name))
        await prompt_service.create_version(
            db_session, p.id, PromptVersionCreate(version="1.0.0", content="test")
        )
        await prompt_service.activate_version(db_session, p.id, "1.0.0")
        prompts.append(p)

    now = datetime.utcnow()

    # First prompt: score well above threshold → healthy
    db_session.add(DriftLog(
        prompt_id=prompts[0].id,
        prompt_version="1.0.0",
        window_start=now,
        window_end=now,
        sample_count=5,
        mean_composite_score=settings.drift_alert_threshold + 1.0,
        alert_fired=False,
    ))

    # Second prompt: score below threshold → drifting
    db_session.add(DriftLog(
        prompt_id=prompts[1].id,
        prompt_version="1.0.0",
        window_start=now,
        window_end=now,
        sample_count=5,
        mean_composite_score=settings.drift_alert_threshold - 1.0,
        alert_fired=True,
        alert_message="Score dropped",
    ))
    await db_session.commit()

    summary = await get_drift_summary(db_session)
    assert summary.total_prompts == 2
    assert summary.healthy_prompts == 1
    assert summary.drifting_prompts == 1
    assert summary.alerts_last_24h >= 1
