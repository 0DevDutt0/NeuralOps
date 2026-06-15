"""SQLAlchemy ORM models for A/B experiments and trials.

Tables:
    experiments      — experiment configuration (two versions to compare).
    experiment_trials — individual trial results with judge scores.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from neuralops.core.database import Base


class Experiment(Base):
    """An A/B experiment comparing two prompt versions.

    Attributes:
        id: UUID primary key.
        name: Human-readable experiment name.
        prompt_id: FK to the parent Prompt.
        version_a: Semver of prompt version A.
        version_b: Semver of prompt version B.
        judge_criteria: JSON list of scoring criteria strings.
        status: "running" | "completed" | "paused".
        winner: "A" | "B" | None — set when significance reached.
        trial_count: Total number of trials run.
        mean_score_a: Running mean score for version A.
        mean_score_b: Running mean score for version B.
        created_at: UTC timestamp of creation.
        completed_at: UTC timestamp when winner was declared.
        trials: Relationship to ExperimentTrial records.
    """

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version_a: Mapped[str] = mapped_column(String(20), nullable=False)
    version_b: Mapped[str] = mapped_column(String(20), nullable=False)
    judge_criteria: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False)
    winner: Mapped[str | None] = mapped_column(String(1), nullable=True)
    trial_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    mean_score_a: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    mean_score_b: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    trials: Mapped[list["ExperimentTrial"]] = relationship(
        "ExperimentTrial", back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentTrial(Base):
    """A single trial within an A/B experiment.

    Attributes:
        id: UUID primary key.
        experiment_id: FK to the parent Experiment.
        user_input: The input sent to both prompt versions.
        output_a: Raw LLM output from version A.
        output_b: Raw LLM output from version B.
        score_a: Composite judge score for output A (0–10).
        score_b: Composite judge score for output B (0–10).
        judge_reasoning: JSON with per-criterion scores and reasoning.
        latency_a_ms: Latency in ms for version A.
        latency_b_ms: Latency in ms for version B.
        created_at: UTC timestamp.
        experiment: Back-reference to parent Experiment.
    """

    __tablename__ = "experiment_trials"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    experiment_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    output_a: Mapped[str] = mapped_column(Text, nullable=False)
    output_b: Mapped[str] = mapped_column(Text, nullable=False)
    score_a: Mapped[float] = mapped_column(Float, nullable=False)
    score_b: Mapped[float] = mapped_column(Float, nullable=False)
    judge_reasoning: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    latency_a_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_b_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="trials")
