"""dataset versions, scheduled analyses, dashboard verdict_state

Purely additive: two new tables and four new columns (three on questions, one
on dashboards). Existing rows keep working -- questions default to trigger
'manual', dashboards get a NULL verdict_state that readers resolve, and
dataset_versions is backfilled by app.dataset_versions on startup.

Revision ID: a3f1c2d4e5b6
Revises: 1e67feee1649
Create Date: 2026-09-29 12:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a3f1c2d4e5b6"
down_revision: Union[str, Sequence[str], None] = "1e67feee1649"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSONVariant = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("filepath", sa.String(), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("col_count", sa.Integer(), nullable=True),
        sa.Column("profile_json", JSONVariant, nullable=True),
        sa.Column("uploaded_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_dataset_versions_id"), "dataset_versions", ["id"], unique=False)
    op.create_index(op.f("ix_dataset_versions_dataset_id"), "dataset_versions", ["dataset_id"], unique=False)

    op.create_table(
        "scheduled_analyses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("interval", sa.String(), nullable=False),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("last_dashboard_id", sa.Integer(), nullable=True),
        sa.Column("change_threshold_pct", sa.Float(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("last_trend", sa.String(), nullable=True),
        sa.Column("last_change_summary", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["workspace_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["last_dashboard_id"], ["dashboards.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_scheduled_analyses_id"), "scheduled_analyses", ["id"], unique=False)
    op.create_index(op.f("ix_scheduled_analyses_workspace_id"), "scheduled_analyses", ["workspace_id"], unique=False)

    op.add_column("questions", sa.Column("dataset_version_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_questions_dataset_version_id", "questions", "dataset_versions", ["dataset_version_id"], ["id"]
    )
    op.add_column(
        "questions",
        sa.Column("trigger", sa.String(), nullable=False, server_default="manual"),
    )
    op.add_column("questions", sa.Column("scheduled_analysis_id", sa.Integer(), nullable=True))
    op.create_index(
        op.f("ix_questions_scheduled_analysis_id"), "questions", ["scheduled_analysis_id"], unique=False
    )
    op.add_column("dashboards", sa.Column("verdict_state", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("dashboards", "verdict_state")
    op.drop_index(op.f("ix_questions_scheduled_analysis_id"), table_name="questions")
    op.drop_column("questions", "scheduled_analysis_id")
    op.drop_column("questions", "trigger")
    op.drop_constraint("fk_questions_dataset_version_id", "questions", type_="foreignkey")
    op.drop_column("questions", "dataset_version_id")
    op.drop_index(op.f("ix_scheduled_analyses_workspace_id"), table_name="scheduled_analyses")
    op.drop_index(op.f("ix_scheduled_analyses_id"), table_name="scheduled_analyses")
    op.drop_table("scheduled_analyses")
    op.drop_index(op.f("ix_dataset_versions_dataset_id"), table_name="dataset_versions")
    op.drop_index(op.f("ix_dataset_versions_id"), table_name="dataset_versions")
    op.drop_table("dataset_versions")
