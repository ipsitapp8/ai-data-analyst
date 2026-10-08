"""analysis_memories (RAG over past analyses)

Purely additive: one new table.

Revision ID: b7c9d1e2f3a4
Revises: a3f1c2d4e5b6
Create Date: 2026-10-03 12:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "b7c9d1e2f3a4"
down_revision: Union[str, Sequence[str], None] = "a3f1c2d4e5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 768  # must match app.config.EMBEDDING_DIM


def upgrade() -> None:
    # Needs a pgvector-enabled server (pgvector/pgvector image, Neon, Supabase...).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "analysis_memories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=True),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("question_id", sa.Integer(), nullable=False),
        sa.Column("question_text", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("verdict_state", sa.String(), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["question_id"], ["questions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("question_id"),
    )
    op.create_index(op.f("ix_analysis_memories_id"), "analysis_memories", ["id"])
    op.create_index(op.f("ix_analysis_memories_team_id"), "analysis_memories", ["team_id"])
    op.create_index(op.f("ix_analysis_memories_dataset_id"), "analysis_memories", ["dataset_id"])
    op.execute(
        "CREATE INDEX ix_analysis_memories_embedding ON analysis_memories "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.drop_table("analysis_memories")
