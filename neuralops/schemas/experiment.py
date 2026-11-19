"""Pydantic v2 schemas for A/B experiment endpoints."""

from datetime import datetime

from pydantic import BaseModel, Field

# ── Requests ──────────────────────────────────────────────────────────────────


class ExperimentCreate(BaseModel):
    """Request body for creating a new A/B experiment."""

    name: str = Field(..., min_length=1, max_length=255)
    prompt_id: str = Field(..., description="ID of the prompt to experiment on")
    version_a: str = Field(..., description="Semver of version A")
    version_b: str = Field(..., description="Semver of version B")
    planned_trials: int = Field(
        default=30, ge=2, le=10_000, description="Number of valid paired inputs set before the run"
    )
    judge_criteria: list[str] = Field(
        default_factory=lambda: ["relevance", "accuracy", "clarity", "safety"],
        description="List of criteria the judge should score on",
    )


class TrialCreate(BaseModel):
    """Request body for running a single trial in an experiment."""

    user_input: str = Field(..., min_length=1, description="Input to send to both versions")


# ── Responses ─────────────────────────────────────────────────────────────────


class TrialResponse(BaseModel):
    """Response schema for a single experiment trial."""

    id: str
    experiment_id: str
    user_input: str
    output_a: str
    output_b: str
    score_a: float | None
    score_b: float | None
    scoring_error: str | None = None
    judge_reasoning: dict
    latency_a_ms: float
    latency_b_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


class ExperimentResponse(BaseModel):
    """Response schema for an experiment."""

    id: str
    name: str
    prompt_id: str
    version_a: str
    version_b: str
    judge_criteria: list[str]
    status: str
    winner: str | None
    planned_trials: int | None
    significance_level: float | None
    trial_count: int
    scored_trial_count: int
    mean_score_a: float
    mean_score_b: float
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class ExperimentDetailResponse(ExperimentResponse):
    """Experiment response including recent trials."""

    recent_trials: list[TrialResponse] = Field(default_factory=list)


class SignificanceResponse(BaseModel):
    """Statistical significance test result."""

    experiment_id: str
    trial_count: int
    scored_trial_count: int
    planned_trials: int | None
    mean_score_a: float
    mean_score_b: float
    p_value: float | None
    is_significant: bool
    winner: str | None
    confidence_level: float
    mean_difference_b_minus_a: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None


class UsageSummaryRow(BaseModel):
    """Tracked activity grouped by stage, provider and model."""

    stage: str
    provider: str
    model: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


class UsageSummary(BaseModel):
    """Recorded usage and coverage across all or one experiment's trials."""

    trials: int
    trials_with_usage: int
    usage: list[UsageSummaryRow]
