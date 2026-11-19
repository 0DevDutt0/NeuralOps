"""Experiment service — manages A/B experiment lifecycle.

Creates experiments, runs trials via ABRunner, and provides
aggregated statistics including significance test results.
"""

import uuid

from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.core.config import settings
from neuralops.core.exceptions import (
    ExperimentAlreadyCompletedError,
    ExperimentNotFoundError,
    VersionNotFoundError,
)
from neuralops.core.logging import get_logger
from neuralops.engine.ab_runner import ABRunner, _run_significance_test, get_paired_scores
from neuralops.models.experiment import Experiment, ExperimentTrial, TrialUsage
from neuralops.models.prompt import PromptVersion
from neuralops.schemas.experiment import (
    ExperimentCreate,
    ExperimentDetailResponse,
    ExperimentResponse,
    SignificanceResponse,
    TrialResponse,
    UsageSummary,
    UsageSummaryRow,
)

logger = get_logger(__name__)


async def create_experiment(
    db: AsyncSession, data: ExperimentCreate
) -> ExperimentResponse:
    """Create a new A/B experiment.

    Args:
        db: Async database session.
        data: ExperimentCreate request body.

    Returns:
        ExperimentResponse for the newly created experiment.

    Raises:
        VersionNotFoundError: If either specified version doesn't exist.
    """
    for ver in (data.version_a, data.version_b):
        result = await db.execute(
            select(PromptVersion).where(
                PromptVersion.prompt_id == data.prompt_id,
                PromptVersion.version == ver,
            )
        )
        if result.scalar_one_or_none() is None:
            raise VersionNotFoundError(
                f"Version '{ver}' not found on prompt '{data.prompt_id}'"
            )

    experiment = Experiment(
        id=str(uuid.uuid4()),
        name=data.name,
        prompt_id=data.prompt_id,
        version_a=data.version_a,
        version_b=data.version_b,
        planned_trials=data.planned_trials,
        significance_level=settings.experiment_significance_level,
        judge_criteria=data.judge_criteria,
        status="running",
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    logger.info("Experiment created", experiment_id=experiment.id, name=experiment.name)
    return ExperimentResponse.model_validate(experiment)


async def get_experiment(db: AsyncSession, experiment_id: str) -> ExperimentDetailResponse:
    """Fetch an experiment with its 20 most recent trials.

    Args:
        db: Async database session.
        experiment_id: UUID of the experiment.

    Returns:
        ExperimentDetailResponse.

    Raises:
        ExperimentNotFoundError: If the experiment doesn't exist.
    """
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found")

    trials_result = await db.execute(
        select(ExperimentTrial)
        .where(ExperimentTrial.experiment_id == experiment_id)
        .order_by(ExperimentTrial.created_at.desc())
        .limit(20)
    )
    trials = trials_result.scalars().all()

    resp = ExperimentDetailResponse.model_validate(experiment)
    resp.recent_trials = [TrialResponse.model_validate(t) for t in reversed(trials)]
    return resp


async def list_experiments(
    db: AsyncSession, status: str | None = None, skip: int = 0, limit: int = 50
) -> list[ExperimentResponse]:
    """List experiments with optional status filter.

    Args:
        db: Async database session.
        status: Filter by "running" | "completed" | "paused".
        skip: Pagination offset.
        limit: Max records.

    Returns:
        List of ExperimentResponse.
    """
    query = select(Experiment).offset(skip).limit(limit).order_by(Experiment.created_at.desc())
    if status:
        query = query.where(Experiment.status == status)
    result = await db.execute(query)
    experiments = result.scalars().all()
    return [ExperimentResponse.model_validate(e) for e in experiments]


async def run_trial(
    db: AsyncSession,
    experiment_id: str,
    user_input: str,
    runner: ABRunner | None = None,
) -> TrialResponse:
    """Execute a single trial for an experiment.

    Args:
        db: Async database session.
        experiment_id: UUID of the experiment to trial.
        user_input: Input text to send to both prompt versions.
        runner: Optional pre-created ABRunner instance.

    Returns:
        TrialResponse with scores and outputs.

    Raises:
        ExperimentNotFoundError: If the experiment doesn't exist.
        ExperimentAlreadyCompletedError: If the experiment already has a winner.
        VersionNotFoundError: If the prompt versions can't be loaded.
    """
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found")
    if experiment.status == "completed":
        raise ExperimentAlreadyCompletedError(
            f"Experiment '{experiment_id}' is already completed (winner: {experiment.winner})"
        )

    # Load prompt versions
    pv_a_result = await db.execute(
        select(PromptVersion).where(
            PromptVersion.prompt_id == experiment.prompt_id,
            PromptVersion.version == experiment.version_a,
        )
    )
    pv_b_result = await db.execute(
        select(PromptVersion).where(
            PromptVersion.prompt_id == experiment.prompt_id,
            PromptVersion.version == experiment.version_b,
        )
    )
    pv_a = pv_a_result.scalar_one_or_none()
    pv_b = pv_b_result.scalar_one_or_none()

    if pv_a is None:
        raise VersionNotFoundError(f"Version {experiment.version_a} not found")
    if pv_b is None:
        raise VersionNotFoundError(f"Version {experiment.version_b} not found")

    ab = runner or ABRunner()
    trial_result = await ab.run_trial(
        db=db,
        experiment=experiment,
        user_input=user_input,
        content_a=pv_a.content,
        content_b=pv_b.content,
        system_a=pv_a.system_prompt,
        system_b=pv_b.system_prompt,
    )

    # Reload trial from DB for accurate response
    trial_db_result = await db.execute(
        select(ExperimentTrial).where(ExperimentTrial.id == trial_result.trial_id)
    )
    trial_db = trial_db_result.scalar_one()
    return TrialResponse.model_validate(trial_db)


async def get_significance(
    db: AsyncSession, experiment_id: str
) -> SignificanceResponse:
    """Run significance test against all trials for an experiment.

    Args:
        db: Async database session.
        experiment_id: UUID of the experiment.

    Returns:
        SignificanceResponse with p-value and winner.

    Raises:
        ExperimentNotFoundError: If the experiment doesn't exist.
    """
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()
    if experiment is None:
        raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found")

    scores_a, scores_b = await get_paired_scores(db, experiment_id)
    significance_level = experiment.significance_level or settings.experiment_significance_level
    sig = _run_significance_test(
        scores_a, scores_b, significance_level,
        evaluate=experiment.status == "completed" and experiment.planned_trials == len(scores_a),
    )

    return SignificanceResponse(
        experiment_id=experiment_id,
        trial_count=experiment.trial_count,
        scored_trial_count=sig.n_trials,
        planned_trials=experiment.planned_trials,
        mean_score_a=sig.mean_a,
        mean_score_b=sig.mean_b,
        p_value=sig.p_value,
        is_significant=sig.is_significant,
        winner=sig.winner,
        confidence_level=1.0 - significance_level,
        mean_difference_b_minus_a=sig.mean_difference,
        confidence_interval_low=sig.interval_low,
        confidence_interval_high=sig.interval_high,
    )


async def get_usage(db: AsyncSession, experiment_id: str | None = None) -> UsageSummary:
    """Aggregate recorded usage without treating historical trials as free."""
    trial_count = select(func.count()).select_from(ExperimentTrial)
    covered_count = select(func.count(distinct(TrialUsage.trial_id))).join_from(
        TrialUsage, ExperimentTrial
    )
    group = (TrialUsage.stage, TrialUsage.provider, TrialUsage.model)
    usage = (
        select(
            *group,
            func.count().label("requests"),
            func.sum(TrialUsage.prompt_tokens).label("prompt_tokens"),
            func.sum(TrialUsage.completion_tokens).label("completion_tokens"),
            func.sum(TrialUsage.estimated_cost_usd).label("estimated_cost_usd"),
        )
        .join_from(TrialUsage, ExperimentTrial)
        .group_by(*group)
        .order_by(*group)
    )
    if experiment_id is not None:
        where = ExperimentTrial.experiment_id == experiment_id
        trial_count = trial_count.where(where)
        covered_count = covered_count.where(where)
        usage = usage.where(where)

    totals = (await db.execute(select(trial_count.scalar_subquery(), covered_count.scalar_subquery()))).one()
    rows = (await db.execute(usage)).mappings()
    return UsageSummary(
        trials=totals[0],
        trials_with_usage=totals[1],
        usage=[UsageSummaryRow.model_validate(row) for row in rows],
    )
