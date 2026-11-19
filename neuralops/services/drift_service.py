"""Drift detection service.

Computes rolling quality metrics for active prompt versions and alerts
via Telegram when scores fall below the configured threshold.
Designed to be called by the APScheduler job in api/main.py.
"""

import uuid
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.core.config import settings
from neuralops.core.logging import get_logger
from neuralops.engine.judge import LLMJudge
from neuralops.models.drift_log import DriftLog
from neuralops.models.prompt import Prompt, PromptVersion
from neuralops.schemas.drift import DriftLogResponse, DriftStatusResponse, DriftSummaryResponse
from neuralops.services.llm_client import LLMClient

logger = get_logger(__name__)

# Synthetic test cases for drift evaluation
_TEST_PROMPTS = [
    "Explain what you do in one sentence.",
    "What is the capital of France?",
    "Summarize the key benefits of regular exercise.",
    "Write a professional email subject line for a product launch.",
    "What are three best practices for secure password management?",
]


async def run_drift_check(db: AsyncSession, judge: LLMJudge | None = None) -> list[DriftLogResponse]:
    """Run drift evaluation for all prompts with an active version.

    For each active prompt version, runs a set of test cases through
    the LLM, scores the outputs with the judge, and stores a DriftLog
    snapshot. Fires a Telegram alert if the score drops below threshold.

    Args:
        db: Async database session.

    Returns:
        List of DriftLogResponse for all snapshots created in this run.
    """
    logger.info("Starting drift check run")
    window_end = datetime.utcnow()
    window_start = window_end - timedelta(minutes=settings.drift_check_interval_minutes)

    # Get all prompts that have an active version
    result = await db.execute(
        select(PromptVersion).where(PromptVersion.is_active == True)  # noqa: E712
    )
    active_versions = result.scalars().all()

    if not active_versions:
        logger.info("No active prompt versions found for drift check")
        return []

    judge = judge or LLMJudge(LLMClient())
    logs_created = []

    for pv in active_versions:
        try:
            log = await _evaluate_prompt_version(
                db, pv, judge, window_start, window_end
            )
            logs_created.append(DriftLogResponse.model_validate(log))
        except Exception as exc:
            logger.error(
                "Drift check failed for prompt version",
                prompt_id=pv.prompt_id,
                version=pv.version,
                error=str(exc),
            )

    logger.info("Drift check complete", snapshots_created=len(logs_created))
    return logs_created


async def _evaluate_prompt_version(
    db: AsyncSession,
    pv: PromptVersion,
    judge: LLMJudge,
    window_start: datetime,
    window_end: datetime,
) -> DriftLog:
    """Evaluate a single prompt version and store a drift snapshot.

    Args:
        db: Async database session.
        pv: The active PromptVersion to evaluate.
        judge: LLMJudge instance for scoring.
        window_start: Start of evaluation window.
        window_end: End of evaluation window.

    Returns:
        The stored DriftLog ORM object.
    """
    from neuralops.services.llm_client import LLMBackend

    llm = judge._client
    scores: list[dict] = []

    for test_input in _TEST_PROMPTS:
        try:
            response = await llm.complete(
                prompt=test_input,
                system=pv.system_prompt,
                backend=LLMBackend(settings.llm_provider),
                temperature=0.1,
            )
            score = await judge.score(response.content, user_input=test_input)
            if score.composite is None:
                continue
            scores.append({
                "input": test_input,
                "relevance": score.relevance,
                "accuracy": score.accuracy,
                "clarity": score.clarity,
                "safety": score.safety,
                "composite": score.composite,
            })
        except Exception as exc:
            logger.warning(
                "Failed to evaluate test case during drift check",
                prompt_id=pv.prompt_id,
                error=str(exc),
            )

    if not scores:
        mean_composite = 0.0
        mean_relevance = mean_accuracy = mean_clarity = mean_safety = 0.0
    else:
        mean_composite = sum(s["composite"] for s in scores) / len(scores)
        mean_relevance = sum(s["relevance"] for s in scores) / len(scores)
        mean_accuracy = sum(s["accuracy"] for s in scores) / len(scores)
        mean_clarity = sum(s["clarity"] for s in scores) / len(scores)
        mean_safety = sum(s["safety"] for s in scores) / len(scores)

    alert_fired = mean_composite < settings.drift_alert_threshold and bool(scores)
    alert_message = None
    if alert_fired:
        alert_message = (
            f"DRIFT ALERT: Prompt '{pv.prompt_id}' version {pv.version} "
            f"scored {mean_composite:.2f} (threshold: {settings.drift_alert_threshold})"
        )
        await _send_telegram_alert(alert_message)

    log = DriftLog(
        id=str(uuid.uuid4()),
        prompt_id=pv.prompt_id,
        prompt_version=pv.version,
        window_start=window_start,
        window_end=window_end,
        sample_count=len(scores),
        mean_composite_score=round(mean_composite, 4),
        mean_relevance=round(mean_relevance, 4),
        mean_accuracy=round(mean_accuracy, 4),
        mean_clarity=round(mean_clarity, 4),
        mean_safety=round(mean_safety, 4),
        mean_toxicity=0.0,
        pii_hit_rate=0.0,
        alert_fired=alert_fired,
        alert_message=alert_message,
        metrics_detail={"samples": scores},
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)

    logger.info(
        "Drift snapshot created",
        prompt_id=pv.prompt_id,
        version=pv.version,
        mean_score=mean_composite,
        alert_fired=alert_fired,
    )
    return log


async def _send_telegram_alert(message: str) -> None:
    """Send a drift alert via Telegram bot.

    Args:
        message: Alert text to send.
    """
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.debug("Telegram not configured, skipping alert")
        return

    try:
        from telegram import Bot  # type: ignore[import]

        bot = Bot(token=settings.telegram_bot_token)
        await bot.send_message(chat_id=settings.telegram_chat_id, text=f"🚨 {message}")
        logger.info("Telegram drift alert sent")
    except Exception as exc:
        logger.error("Failed to send Telegram alert", error=str(exc))


async def get_drift_status(
    db: AsyncSession, prompt_id: str
) -> DriftStatusResponse:
    """Get the current drift status for a specific prompt.

    Args:
        db: Async database session.
        prompt_id: UUID of the prompt.

    Returns:
        DriftStatusResponse with latest score and alert status.
    """
    prompt_result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = prompt_result.scalar_one_or_none()
    prompt_name = prompt.name if prompt else prompt_id

    active_version_result = await db.execute(
        select(PromptVersion).where(
            PromptVersion.prompt_id == prompt_id,
            PromptVersion.is_active == True,  # noqa: E712
        )
    )
    active_version = active_version_result.scalar_one_or_none()

    logs_result = await db.execute(
        select(DriftLog)
        .where(DriftLog.prompt_id == prompt_id)
        .order_by(DriftLog.created_at.desc())
        .limit(10)
    )
    logs = logs_result.scalars().all()

    latest_score = logs[0].mean_composite_score if logs else None
    is_drifting = (
        latest_score is not None and latest_score < settings.drift_alert_threshold
    )
    last_checked = logs[0].created_at if logs else None

    return DriftStatusResponse(
        prompt_id=prompt_id,
        prompt_name=prompt_name,
        active_version=active_version.version if active_version else None,
        latest_score=latest_score,
        alert_threshold=settings.drift_alert_threshold,
        is_drifting=is_drifting,
        last_checked=last_checked,
        recent_logs=[DriftLogResponse.model_validate(log) for log in logs],
    )


async def get_drift_summary(db: AsyncSession) -> DriftSummaryResponse:
    """Get a summary of drift status across all monitored prompts.

    Args:
        db: Async database session.

    Returns:
        DriftSummaryResponse with aggregate stats.
    """
    active_result = await db.execute(
        select(PromptVersion.prompt_id).where(PromptVersion.is_active == True)  # noqa: E712
    )
    active_prompt_ids = [r[0] for r in active_result.all()]

    drifting = 0
    healthy = 0
    for pid in active_prompt_ids:
        logs_result = await db.execute(
            select(DriftLog)
            .where(DriftLog.prompt_id == pid)
            .order_by(DriftLog.created_at.desc())
            .limit(1)
        )
        latest = logs_result.scalar_one_or_none()
        if latest:
            if latest.mean_composite_score < settings.drift_alert_threshold:
                drifting += 1
            else:
                healthy += 1

    cutoff = datetime.utcnow() - timedelta(hours=24)
    alerts_result = await db.execute(
        select(DriftLog).where(DriftLog.alert_fired == True, DriftLog.created_at >= cutoff)  # noqa: E712
    )
    alerts_24h = len(alerts_result.scalars().all())

    last_log = await db.execute(
        select(DriftLog).order_by(DriftLog.created_at.desc()).limit(1)
    )
    last = last_log.scalar_one_or_none()

    return DriftSummaryResponse(
        total_prompts=len(active_prompt_ids),
        drifting_prompts=drifting,
        healthy_prompts=healthy,
        alert_threshold=settings.drift_alert_threshold,
        last_global_check=last.created_at if last else None,
        alerts_last_24h=alerts_24h,
    )
