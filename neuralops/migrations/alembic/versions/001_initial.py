"""Initial migration — creates all NeuralOps tables.

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all tables."""
    op.create_table(
        "prompts",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_prompts_name", "prompts", ["name"])

    op.create_table(
        "prompt_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("prompt_id", sa.String(36), sa.ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("system_prompt", sa.Text, nullable=True),
        sa.Column("variables", sa.JSON, nullable=False),
        sa.Column("meta", sa.JSON, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, default=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_prompt_versions_prompt_id", "prompt_versions", ["prompt_id"])

    op.create_table(
        "experiments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("prompt_id", sa.String(36), sa.ForeignKey("prompts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_a", sa.String(20), nullable=False),
        sa.Column("version_b", sa.String(20), nullable=False),
        sa.Column("judge_criteria", sa.JSON, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, default="running"),
        sa.Column("winner", sa.String(1), nullable=True),
        sa.Column("trial_count", sa.Integer, nullable=False, default=0),
        sa.Column("mean_score_a", sa.Float, nullable=False, default=0.0),
        sa.Column("mean_score_b", sa.Float, nullable=False, default=0.0),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("completed_at", sa.DateTime, nullable=True),
    )
    op.create_index("ix_experiments_prompt_id", "experiments", ["prompt_id"])

    op.create_table(
        "experiment_trials",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("experiment_id", sa.String(36), sa.ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_input", sa.Text, nullable=False),
        sa.Column("output_a", sa.Text, nullable=False),
        sa.Column("output_b", sa.Text, nullable=False),
        sa.Column("score_a", sa.Float, nullable=False),
        sa.Column("score_b", sa.Float, nullable=False),
        sa.Column("judge_reasoning", sa.JSON, nullable=False),
        sa.Column("latency_a_ms", sa.Float, nullable=False, default=0.0),
        sa.Column("latency_b_ms", sa.Float, nullable=False, default=0.0),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_experiment_trials_experiment_id", "experiment_trials", ["experiment_id"])

    op.create_table(
        "registered_models",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("context_window", sa.Integer, nullable=False, default=8192),
        sa.Column("cost_per_1k_input_tokens", sa.Float, nullable=False, default=0.0),
        sa.Column("cost_per_1k_output_tokens", sa.Float, nullable=False, default=0.0),
        sa.Column("avg_latency_ms", sa.Float, nullable=False, default=0.0),
        sa.Column("is_active", sa.Boolean, nullable=False, default=True),
        sa.Column("routing_priority", sa.Integer, nullable=False, default=100),
        sa.Column("capabilities", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("updated_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_registered_models_name", "registered_models", ["name"])

    op.create_table(
        "drift_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("prompt_id", sa.String(36), nullable=False),
        sa.Column("prompt_version", sa.String(20), nullable=False),
        sa.Column("window_start", sa.DateTime, nullable=False),
        sa.Column("window_end", sa.DateTime, nullable=False),
        sa.Column("sample_count", sa.Integer, nullable=False, default=0),
        sa.Column("mean_composite_score", sa.Float, nullable=False, default=0.0),
        sa.Column("mean_relevance", sa.Float, nullable=False, default=0.0),
        sa.Column("mean_accuracy", sa.Float, nullable=False, default=0.0),
        sa.Column("mean_clarity", sa.Float, nullable=False, default=0.0),
        sa.Column("mean_safety", sa.Float, nullable=False, default=0.0),
        sa.Column("mean_toxicity", sa.Float, nullable=False, default=0.0),
        sa.Column("pii_hit_rate", sa.Float, nullable=False, default=0.0),
        sa.Column("alert_fired", sa.Boolean, nullable=False, default=False),
        sa.Column("alert_message", sa.Text, nullable=True),
        sa.Column("metrics_detail", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_drift_logs_prompt_id", "drift_logs", ["prompt_id"])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table("drift_logs")
    op.drop_table("registered_models")
    op.drop_table("experiment_trials")
    op.drop_table("experiments")
    op.drop_table("prompt_versions")
    op.drop_table("prompts")
