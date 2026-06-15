"""A/B experiment management endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.api.dependencies import get_ab_runner, get_db
from neuralops.core.exceptions import NeuralOpsError, to_http_exception
from neuralops.engine.ab_runner import ABRunner
from neuralops.schemas.experiment import (
    ExperimentCreate,
    ExperimentDetailResponse,
    ExperimentResponse,
    SignificanceResponse,
    TrialCreate,
    TrialResponse,
)
from neuralops.services import experiment_service

router = APIRouter(tags=["experiments"])


@router.post("/", response_model=ExperimentResponse, status_code=201, summary="Create experiment")
async def create_experiment(
    data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """Create a new A/B experiment comparing two prompt versions.

    Args:
        data: Experiment name, prompt ID, versions, and judge criteria.
        db: Database session.

    Returns:
        The newly created experiment.
    """
    try:
        return await experiment_service.create_experiment(db, data)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get("/", response_model=list[ExperimentResponse], summary="List experiments")
async def list_experiments(
    status: str | None = Query(None, description="Filter: 'running' | 'completed' | 'paused'"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
) -> list[ExperimentResponse]:
    """List all experiments with optional status filter.

    Args:
        status: Optional status filter.
        skip: Pagination offset.
        limit: Max records.
        db: Database session.

    Returns:
        List of experiments.
    """
    return await experiment_service.list_experiments(db, status=status, skip=skip, limit=limit)


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse, summary="Get experiment")
async def get_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> ExperimentDetailResponse:
    """Fetch an experiment with its 20 most recent trials.

    Args:
        experiment_id: UUID of the experiment.
        db: Database session.

    Returns:
        Experiment with recent trial history.
    """
    try:
        return await experiment_service.get_experiment(db, experiment_id)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.post(
    "/{experiment_id}/trials/",
    response_model=TrialResponse,
    status_code=201,
    summary="Run a trial",
)
async def run_trial(
    experiment_id: str,
    data: TrialCreate,
    db: AsyncSession = Depends(get_db),
    runner: ABRunner = Depends(get_ab_runner),
) -> TrialResponse:
    """Execute a single A/B trial for an experiment.

    Runs the input through both prompt versions concurrently, scores
    with LLM-as-Judge, stores the result, and checks for a winner.

    Args:
        experiment_id: UUID of the experiment.
        data: User input to send to both versions.
        db: Database session.
        runner: Injected ABRunner.

    Returns:
        Trial result with outputs and scores.
    """
    try:
        return await experiment_service.run_trial(db, experiment_id, data.user_input, runner)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)


@router.get(
    "/{experiment_id}/significance",
    response_model=SignificanceResponse,
    summary="Check statistical significance",
)
async def get_significance(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
) -> SignificanceResponse:
    """Run significance test on all trials for an experiment.

    Uses Welch's t-test (unequal variance) to determine if the score
    difference between versions A and B is statistically significant.

    Args:
        experiment_id: UUID of the experiment.
        db: Database session.

    Returns:
        Significance test result with p-value and winner.
    """
    try:
        return await experiment_service.get_significance(db, experiment_id)
    except NeuralOpsError as exc:
        raise to_http_exception(exc)
