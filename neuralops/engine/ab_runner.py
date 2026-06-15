"""Async A/B test executor.

Runs the same input through two prompt versions concurrently, scores
both outputs with LLM-as-Judge, stores the trial, and checks for
statistical significance to auto-promote a winner.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from neuralops.core.config import settings
from neuralops.core.logging import get_logger
from neuralops.engine.judge import JudgeScore, LLMJudge
from neuralops.models.experiment import Experiment, ExperimentTrial
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
    score_a: float
    score_b: float
    score_details_a: JudgeScore
    score_details_b: JudgeScore
    latency_a_ms: float
    latency_b_ms: float
    winner_this_trial: str
    experiment_winner: str | None = None


@dataclass
class SignificanceTestResult:
    """Result of the statistical significance test.

    Attributes:
        p_value: p-value from Welch's t-test (None if n < 2).
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


def _run_significance_test(
    scores_a: list[float],
    scores_b: list[float],
    significance_level: float,
) -> SignificanceTestResult:
    """Run Welch's t-test to check if the score difference is significant.

    Args:
        scores_a: All scores for version A.
        scores_b: All scores for version B.
        significance_level: Alpha threshold (e.g. 0.05).

    Returns:
        SignificanceTestResult with p-value and winner.
    """
    from scipy import stats  # type: ignore[import]

    n = min(len(scores_a), len(scores_b))
    mean_a = sum(scores_a) / len(scores_a) if scores_a else 0.0
    mean_b = sum(scores_b) / len(scores_b) if scores_b else 0.0

    if n < 2:
        return SignificanceTestResult(
            p_value=None, is_significant=False, winner=None,
            mean_a=mean_a, mean_b=mean_b, n_trials=n
        )

    _, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    is_significant = float(p_value) < significance_level
    winner = None
    if is_significant:
        winner = "A" if mean_a > mean_b else "B"

    return SignificanceTestResult(
        p_value=round(float(p_value), 6),
        is_significant=is_significant,
        winner=winner,
        mean_a=round(mean_a, 4),
        mean_b=round(mean_b, 4),
        n_trials=n,
    )


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
        t0_a = t0_b = time.monotonic()

        async def call_a():
            nonlocal t0_a
            t0_a = time.monotonic()
            return await self._llm.complete(
                prompt=prompt_a,
                system=system_a,
                backend=LLMBackend(settings.llm_provider),
            )

        async def call_b():
            nonlocal t0_b
            t0_b = time.monotonic()
            return await self._llm.complete(
                prompt=prompt_b,
                system=system_b,
                backend=LLMBackend(settings.llm_provider),
            )

        resp_a, resp_b = await asyncio.gather(call_a(), call_b())

        latency_a = resp_a.latency_ms
        latency_b = resp_b.latency_ms

        # Score both outputs concurrently
        score_a, score_b = await self._judge.score_pair(
            response_a=resp_a.content,
            response_b=resp_b.content,
            user_input=user_input,
            criteria=experiment.judge_criteria or None,
        )

        winner_this = "A" if score_a.composite > score_b.composite else (
            "B" if score_b.composite > score_a.composite else "tie"
        )

        # Store trial
        trial = ExperimentTrial(
            id=str(uuid.uuid4()),
            experiment_id=experiment.id,
            user_input=user_input,
            output_a=resp_a.content,
            output_b=resp_b.content,
            score_a=score_a.composite,
            score_b=score_b.composite,
            judge_reasoning={
                "a": {
                    "relevance": score_a.relevance,
                    "accuracy": score_a.accuracy,
                    "clarity": score_a.clarity,
                    "safety": score_a.safety,
                    "reasoning": score_a.reasoning,
                    "confidence": score_a.confidence,
                },
                "b": {
                    "relevance": score_b.relevance,
                    "accuracy": score_b.accuracy,
                    "clarity": score_b.clarity,
                    "safety": score_b.safety,
                    "reasoning": score_b.reasoning,
                    "confidence": score_b.confidence,
                },
            },
            latency_a_ms=latency_a,
            latency_b_ms=latency_b,
        )
        db.add(trial)

        # Update experiment statistics
        n = experiment.trial_count + 1
        # Incremental mean update
        new_mean_a = (experiment.mean_score_a * experiment.trial_count + score_a.composite) / n
        new_mean_b = (experiment.mean_score_b * experiment.trial_count + score_b.composite) / n
        experiment.trial_count = n
        experiment.mean_score_a = round(new_mean_a, 4)
        experiment.mean_score_b = round(new_mean_b, 4)

        # Check significance (requires fetching all trial scores for a proper t-test)
        experiment_winner = None
        if n >= 30:
            from sqlalchemy import select

            rows = await db.execute(
                select(ExperimentTrial.score_a, ExperimentTrial.score_b).where(
                    ExperimentTrial.experiment_id == experiment.id
                )
            )
            all_scores = rows.all()
            all_a = [r[0] for r in all_scores]
            all_b = [r[1] for r in all_scores]
            sig = _run_significance_test(all_a, all_b, settings.experiment_significance_level)
            if sig.is_significant and sig.winner:
                experiment.winner = sig.winner
                experiment.status = "completed"
                experiment.completed_at = datetime.utcnow()
                experiment_winner = sig.winner
                logger.info(
                    "Experiment winner declared",
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
