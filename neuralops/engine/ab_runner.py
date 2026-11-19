"""Async A/B test executor.

Runs the same input through two prompt versions concurrently, scores
both outputs with LLM-as-Judge and evaluates a fixed number of valid pairs.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, variance

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.core.config import settings
from neuralops.core.exceptions import ExperimentAlreadyCompletedError
from neuralops.core.logging import get_logger
from neuralops.engine.judge import JudgeScore, LLMJudge
from neuralops.models.experiment import Experiment, ExperimentTrial, TrialUsage
from neuralops.services.llm_client import LLMBackend, LLMClient

logger = get_logger(__name__)


@dataclass
class TrialResult:
    """Result of a single A/B trial.

    Attributes:
        trial_id: UUID of the stored trial record.
        experiment_id: UUID of the parent experiment.
        user_input: Input that was sent to both versions.
        output_a: Response from version A.
        output_b: Response from version B.
        score_a: Composite judge score for A.
        score_b: Composite judge score for B.
        score_details_a: Full JudgeScore for A.
        score_details_b: Full JudgeScore for B.
        latency_a_ms: Latency for version A LLM call.
        latency_b_ms: Latency for version B LLM call.
        winner_this_trial: "A" | "B" | "tie" for this single trial.
        experiment_winner: Overall winner if significance reached, else None.
    """

    trial_id: str
    experiment_id: str
    user_input: str
    output_a: str
    output_b: str
    score_a: float | None
    score_b: float | None
    score_details_a: JudgeScore
    score_details_b: JudgeScore
    latency_a_ms: float
    latency_b_ms: float
    winner_this_trial: str | None
    experiment_winner: str | None = None


@dataclass
class SignificanceTestResult:
    """Result of the statistical significance test.

    Attributes:
        p_value: Two-sided paired t-test p-value when estimable.
        is_significant: True if p < significance_level.
        winner: "A" | "B" | None.
        mean_a: Mean score for version A.
        mean_b: Mean score for version B.
        n_trials: Number of trials included.
    """

    p_value: float | None
    is_significant: bool
    winner: str | None
    mean_a: float
    mean_b: float
    n_trials: int
    mean_difference: float | None
    interval_low: float | None = None
    interval_high: float | None = None


def _run_significance_test(
    scores_a: list[float],
    scores_b: list[float],
    significance_level: float,
    *,
    evaluate: bool = True,
) -> SignificanceTestResult:
    """Summarize paired scores and compute inference at the planned endpoint."""
    differences = [b - a for a, b in zip(scores_a, scores_b, strict=True)]
    n = len(differences)
    result = SignificanceTestResult(
        p_value=None,
        is_significant=False,
        winner=None,
        mean_a=round(mean(scores_a), 4) if n else 0.0,
        mean_b=round(mean(scores_b), 4) if n else 0.0,
        n_trials=n,
        mean_difference=round(mean(differences), 4) if n else None,
    )
    if not evaluate or n < 2 or variance(round(delta, 10) for delta in differences) == 0:
        return result

    from scipy import stats

    paired = stats.ttest_rel(scores_b, scores_a)
    interval = paired.confidence_interval(confidence_level=1 - significance_level)
    result.p_value = float(paired.pvalue)
    result.interval_low = float(interval.low)
    result.interval_high = float(interval.high)
    result.is_significant = result.p_value < significance_level
    if result.is_significant:
        result.winner = "B" if mean(differences) > 0 else "A"
    return result


async def get_paired_scores(db: AsyncSession, experiment_id: str) -> tuple[list[float], list[float]]:
    """Read scores from trials where both arms scored successfully."""
    rows = await db.execute(
        select(ExperimentTrial.score_a, ExperimentTrial.score_b).where(
            ExperimentTrial.experiment_id == experiment_id,
            ExperimentTrial.scoring_error.is_(None),
        )
    )
    scores_a, scores_b = [], []
    for score_a, score_b in rows:
        if score_a is not None and score_b is not None:
            scores_a.append(score_a)
            scores_b.append(score_b)
    return scores_a, scores_b


class ABRunner:
    """Executes A/B trials and manages experiment statistics.

    Attributes:
        _llm: LLMClient for calling prompt versions.
        _judge: LLMJudge for scoring outputs.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._judge = judge or LLMJudge(self._llm)

    async def run_trial(
        self,
        db: AsyncSession,
        experiment: Experiment,
        user_input: str,
        content_a: str,
        content_b: str,
        system_a: str | None = None,
        system_b: str | None = None,
    ) -> TrialResult:
        """Execute a single A/B trial and persist results.

        Runs both prompt versions concurrently, then scores both outputs
        with the judge concurrently. Updates experiment statistics and
        checks for statistical significance.

        Args:
            db: Async database session.
            experiment: The Experiment ORM object.
            user_input: The input to inject into both prompt versions.
            content_a: Prompt content for version A (may contain {variables}).
            content_b: Prompt content for version B.
            system_a: Optional system prompt for version A.
            system_b: Optional system prompt for version B.

        Returns:
            TrialResult with outputs, scores, and winner information.
        """
        # Inject user_input into prompt templates
        prompt_a = content_a.replace("{input}", user_input).replace("{user_input}", user_input)
        prompt_b = content_b.replace("{input}", user_input).replace("{user_input}", user_input)

        # Run both LLM calls concurrently
        backend = LLMBackend(settings.llm_provider)
        resp_a, resp_b = await asyncio.gather(
            self._llm.complete(prompt=prompt_a, system=system_a, backend=backend),
            self._llm.complete(prompt=prompt_b, system=system_b, backend=backend),
        )

        latency_a = resp_a.latency_ms
        latency_b = resp_b.latency_ms

        # Score both outputs concurrently
        score_a, score_b = await self._judge.score_pair(
            response_a=resp_a.content,
            response_b=resp_b.content,
            user_input=user_input,
            criteria=experiment.judge_criteria or None,
        )

        winner_this = None
        valid_pair = score_a.composite is not None and score_b.composite is not None
        if score_a.composite is not None and score_b.composite is not None:
            winner_this = "A" if score_a.composite > score_b.composite else (
                "B" if score_b.composite > score_a.composite else "tie"
            )

        # Serialize writes for this experiment after the provider calls finish.
        updated = await db.execute(
            update(Experiment)
            .where(Experiment.id == experiment.id, Experiment.status != "completed")
            .values(trial_count=Experiment.trial_count + 1)
            .returning(Experiment.id)
            .execution_options(synchronize_session=False)
        )
        if updated.scalar_one_or_none() is None:
            raise ExperimentAlreadyCompletedError("The experiment no longer accepts trials")
        await db.refresh(experiment)

        # Store trial
        trial = ExperimentTrial(
            id=str(uuid.uuid4()),
            experiment_id=experiment.id,
            user_input=user_input,
            output_a=resp_a.content,
            output_b=resp_b.content,
            score_a=score_a.composite,
            score_b=score_b.composite,
            scoring_error=None if valid_pair else "The judge could not score both outputs",
            judge_reasoning={
                "a": {
                    "relevance": score_a.relevance,
                    "accuracy": score_a.accuracy,
                    "clarity": score_a.clarity,
                    "safety": score_a.safety,
                    "reasoning": score_a.reasoning,
                    "confidence": score_a.confidence,
                    "model": score_a.model,
                    "error": score_a.error,
                },
                "b": {
                    "relevance": score_b.relevance,
                    "accuracy": score_b.accuracy,
                    "clarity": score_b.clarity,
                    "safety": score_b.safety,
                    "reasoning": score_b.reasoning,
                    "confidence": score_b.confidence,
                    "model": score_b.model,
                    "error": score_b.error,
                },
            },
            latency_a_ms=latency_a,
            latency_b_ms=latency_b,
        )
        db.add(trial)
        await db.flush()

        for stage, response in (
            ("generation_a", resp_a), ("generation_b", resp_b),
            ("judge_a", score_a.usage), ("judge_b", score_b.usage),
        ):
            if response is not None:
                db.add(TrialUsage(
                    trial_id=trial.id,
                    stage=stage,
                    model=response.model,
                    provider=response.backend.value,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    estimated_cost_usd=response.cost_usd,
                ))

        experiment_winner = None
        if score_a.composite is not None and score_b.composite is not None:
            experiment.scored_trial_count += 1
            n = experiment.scored_trial_count
            experiment.mean_score_a += (score_a.composite - experiment.mean_score_a) / n
            experiment.mean_score_b += (score_b.composite - experiment.mean_score_b) / n
            if experiment.planned_trials == n:
                scores_a, scores_b = await get_paired_scores(db, experiment.id)
                sig = _run_significance_test(
                    scores_a, scores_b, experiment.significance_level or settings.experiment_significance_level
                )
                experiment.mean_score_a = sig.mean_a
                experiment.mean_score_b = sig.mean_b
                experiment.winner = sig.winner
                experiment.status = "completed"
                experiment.completed_at = datetime.utcnow()
                experiment_winner = sig.winner
                logger.info(
                    "Fixed evaluation completed",
                    experiment_id=experiment.id,
                    winner=sig.winner,
                    p_value=sig.p_value,
                )

        await db.commit()
        await db.refresh(trial)

        return TrialResult(
            trial_id=trial.id,
            experiment_id=experiment.id,
            user_input=user_input,
            output_a=resp_a.content,
            output_b=resp_b.content,
            score_a=score_a.composite,
            score_b=score_b.composite,
            score_details_a=score_a,
            score_details_b=score_b,
            latency_a_ms=latency_a,
            latency_b_ms=latency_b,
            winner_this_trial=winner_this,
            experiment_winner=experiment_winner,
        )
