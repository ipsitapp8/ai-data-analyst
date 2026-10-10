"""knowledge_notes (dataset knowledge + lessons for the Planner)

Purely additive: one new table.

Revision ID: c8d0e2f3a4b5
Revises: b7c9d1e2f3a4
Create Date: 2026-10-09 12:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "c8d0e2f3a4b5"
down_revision: Union[str, Sequence[str], None] = "b7c9d1e2f3a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768  # must match app.config.EMBEDDING_DIM


def upgrade() -> None:
    op.create_table(
        "knowledge_notes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=True),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_knowledge_notes_id"), "knowledge_notes", ["id"])
    op.create_index(op.f("ix_knowledge_notes_team_id"), "knowledge_notes", ["team_id"])
    op.create_index(op.f("ix_knowledge_notes_dataset_id"), "knowledge_notes", ["dataset_id"])


def downgrade() -> None:
    op.drop_table("knowledge_notes")
