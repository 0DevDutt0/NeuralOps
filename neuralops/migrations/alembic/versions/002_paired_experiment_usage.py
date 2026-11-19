"""Add fixed evaluation sizes, explicit scoring failures, and trial usage."""

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("experiments", sa.Column("planned_trials", sa.Integer(), nullable=True))
    op.add_column("experiments", sa.Column("significance_level", sa.Float(), nullable=True))
    op.add_column(
        "experiments", sa.Column("scored_trial_count", sa.Integer(), server_default="0", nullable=False)
    )
    with op.batch_alter_table("experiment_trials") as batch:
        batch.alter_column("score_a", existing_type=sa.Float(), nullable=True)
        batch.alter_column("score_b", existing_type=sa.Float(), nullable=True)
        batch.add_column(sa.Column("scoring_error", sa.Text(), nullable=True))

    trials = sa.table(
        "experiment_trials",
        sa.column("experiment_id", sa.String()),
        sa.column("score_a", sa.Float()),
        sa.column("score_b", sa.Float()),
        sa.column("judge_reasoning", sa.JSON()),
        sa.column("scoring_error", sa.Text()),
    )
    for arm in ("a", "b"):
        invalid = trials.c.judge_reasoning[arm]["reasoning"].as_string().like("Parse error%")
        op.execute(trials.update().where(invalid).values(
            {f"score_{arm}": None, "scoring_error": "Invalid judge response in historical trial"}
        ))

    experiments = sa.table(
        "experiments",
        sa.column("id", sa.String()),
        sa.column("status", sa.String()),
        sa.column("trial_count", sa.Integer()),
        sa.column("planned_trials", sa.Integer()),
        sa.column("significance_level", sa.Float()),
        sa.column("scored_trial_count", sa.Integer()),
        sa.column("mean_score_a", sa.Float()),
        sa.column("mean_score_b", sa.Float()),
    )
    valid = (trials.c.experiment_id == experiments.c.id) & trials.c.scoring_error.is_(None)
    op.execute(experiments.update().values(
        scored_trial_count=sa.select(sa.func.count()).where(valid).scalar_subquery(),
        mean_score_a=sa.func.coalesce(sa.select(sa.func.avg(trials.c.score_a)).where(valid).scalar_subquery(), 0),
        mean_score_b=sa.func.coalesce(sa.select(sa.func.avg(trials.c.score_b)).where(valid).scalar_subquery(), 0),
    ))
    op.execute(experiments.update().where(
        experiments.c.trial_count == 0, experiments.c.status == "running"
    ).values(planned_trials=30, significance_level=0.05))

    op.create_table(
        "trial_usage",
        sa.Column("trial_id", sa.String(36), sa.ForeignKey("experiment_trials.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("stage", sa.String(12), primary_key=True),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("provider", sa.String(30), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("trial_usage")
    op.execute(sa.text("UPDATE experiment_trials SET score_a = 5.0 WHERE score_a IS NULL"))
    op.execute(sa.text("UPDATE experiment_trials SET score_b = 5.0 WHERE score_b IS NULL"))
    with op.batch_alter_table("experiment_trials") as batch:
        batch.drop_column("scoring_error")
        batch.alter_column("score_a", existing_type=sa.Float(), nullable=False)
        batch.alter_column("score_b", existing_type=sa.Float(), nullable=False)
    with op.batch_alter_table("experiments") as batch:
        batch.drop_column("planned_trials")
        batch.drop_column("significance_level")
        batch.drop_column("scored_trial_count")
