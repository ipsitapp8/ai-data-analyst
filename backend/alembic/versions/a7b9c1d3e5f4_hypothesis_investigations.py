"""investigation_hypotheses, investigation_analyses

Additive only: two new tables for hypothesis-driven investigations.

Revision ID: a7b9c1d3e5f4
Revises: f1a3b5c7d9e2
Create Date: 2026-10-11 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a7b9c1d3e5f4"
down_revision: Union[str, Sequence[str], None] = "f1a3b5c7d9e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "investigation_hypotheses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("investigation_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("spec_json", _json(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("assessment_json", _json(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["investigation_id"], ["investigations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_investigation_hypotheses_id"), "investigation_hypotheses", ["id"])
    op.create_index(op.f("ix_investigation_hypotheses_team_id"), "investigation_hypotheses", ["team_id"])
    op.create_index(op.f("ix_investigation_hypotheses_investigation_id"), "investigation_hypotheses", ["investigation_id"])

    op.create_table(
        "investigation_analyses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("investigation_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("signature", sa.String(), nullable=False),
        sa.Column("op", sa.String(), nullable=False),
        sa.Column("params_json", _json(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("result_json", _json(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("hypothesis_keys_json", _json(), nullable=True),
        sa.Column("requested_by", sa.String(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["investigation_id"], ["investigations.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_investigation_analyses_id"), "investigation_analyses", ["id"])
    op.create_index(op.f("ix_investigation_analyses_team_id"), "investigation_analyses", ["team_id"])
    op.create_index(op.f("ix_investigation_analyses_investigation_id"), "investigation_analyses", ["investigation_id"])
    op.create_index(op.f("ix_investigation_analyses_signature"), "investigation_analyses", ["signature"])


def downgrade() -> None:
    op.drop_table("investigation_analyses")
    op.drop_table("investigation_hypotheses")
