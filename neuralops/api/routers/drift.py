"""Drift monitoring endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.api.dependencies import get_db
from neuralops.models.drift_log import DriftLog
from neuralops.schemas.drift import DriftLogResponse, DriftStatusResponse, DriftSummaryResponse
from neuralops.services import drift_service

router = APIRouter(tags=["drift"])


@router.get("/summary", response_model=DriftSummaryResponse, summary="Drift summary")
async def get_summary(db: AsyncSession = Depends(get_db)) -> DriftSummaryResponse:
    """Get aggregate drift status across all monitored prompts.

    Returns:
        Summary with total/drifting/healthy counts and alert stats.
    """
    return await drift_service.get_drift_summary(db)


@router.get("/prompts/{prompt_id}", response_model=DriftStatusResponse, summary="Prompt drift status")
async def get_prompt_drift(
    prompt_id: str,
    db: AsyncSession = Depends(get_db),
) -> DriftStatusResponse:
    """Get the current drift status for a specific prompt.

    Args:
        prompt_id: UUID of the prompt.
        db: Database session.

    Returns:
        Drift status with latest score and recent logs.
    """
    return await drift_service.get_drift_status(db, prompt_id)


@router.get("/logs", response_model=list[DriftLogResponse], summary="Recent drift logs")
async def list_drift_logs(
    prompt_id: str | None = Query(None, description="Filter by prompt ID"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[DriftLogResponse]:
    """List recent drift log snapshots.

    Args:
        prompt_id: Optional prompt filter.
        limit: Max records to return.
        db: Database session.

    Returns:
        List of drift log snapshots, newest first.
    """
    query = select(DriftLog).order_by(DriftLog.created_at.desc()).limit(limit)
    if prompt_id:
        query = query.where(DriftLog.prompt_id == prompt_id)
    result = await db.execute(query)
    logs = result.scalars().all()
    return [DriftLogResponse.model_validate(log) for log in logs]


@router.post("/trigger", response_model=list[DriftLogResponse], summary="Trigger drift check")
async def trigger_drift_check(
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> list[DriftLogResponse]:
    """Manually trigger a drift check run for all active prompts.

    This runs synchronously for immediate feedback. Use the background
    scheduler for automated periodic checks.

    Args:
        background_tasks: FastAPI background task runner.
        db: Database session.

    Returns:
        List of drift log snapshots created in this run.
    """
    return await drift_service.run_drift_check(db)
