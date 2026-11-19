"""Pydantic v2 schemas for drift monitoring endpoints."""

from datetime import datetime

from pydantic import BaseModel


class DriftLogResponse(BaseModel):
    """Response schema for a drift log snapshot."""

    id: str
    prompt_id: str
    prompt_version: str
    window_start: datetime
    window_end: datetime
    sample_count: int
    mean_composite_score: float
    mean_relevance: float
    mean_accuracy: float
    mean_clarity: float
    mean_safety: float
    mean_toxicity: float
    pii_hit_rate: float
    alert_fired: bool
    alert_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DriftStatusResponse(BaseModel):
    """Current drift status for a prompt."""

    prompt_id: str
    prompt_name: str
    active_version: str | None
    latest_score: float | None
    alert_threshold: float
    is_drifting: bool
    last_checked: datetime | None
    recent_logs: list[DriftLogResponse] = []


class DriftSummaryResponse(BaseModel):
    """Summary of drift status across all monitored prompts."""

    total_prompts: int
    drifting_prompts: int
    healthy_prompts: int
    alert_threshold: float
    last_global_check: datetime | None
    alerts_last_24h: int
