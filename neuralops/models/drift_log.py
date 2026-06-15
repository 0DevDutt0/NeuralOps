"""SQLAlchemy ORM model for drift detection logs.

Table:
    drift_logs — time-series snapshots of prompt quality metrics.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from neuralops.core.database import Base


class DriftLog(Base):
    """A snapshot of quality metrics for a prompt at a point in time.

    Attributes:
        id: UUID primary key.
        prompt_id: FK reference to the prompt being monitored.
        prompt_version: The active version at the time of measurement.
        window_start: Start of the measurement window (UTC).
        window_end: End of the measurement window (UTC).
        sample_count: Number of test cases evaluated.
        mean_composite_score: Average composite judge score (0–10).
        mean_relevance: Average relevance score.
        mean_accuracy: Average accuracy score.
        mean_clarity: Average clarity score.
        mean_safety: Average safety score.
        mean_toxicity: Average toxicity score from guardrails.
        pii_hit_rate: Fraction of samples with PII detected.
        alert_fired: Whether a drift alert was sent for this snapshot.
        alert_message: The alert message if one was sent.
        metrics_detail: JSON with full per-sample breakdown.
        created_at: UTC timestamp when this snapshot was recorded.
    """

    __tablename__ = "drift_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    prompt_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    sample_count: Mapped[int] = mapped_column(default=0, nullable=False)
    mean_composite_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_relevance: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_accuracy: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_clarity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_safety: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_toxicity: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    pii_hit_rate: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    alert_fired: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    alert_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_detail: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
