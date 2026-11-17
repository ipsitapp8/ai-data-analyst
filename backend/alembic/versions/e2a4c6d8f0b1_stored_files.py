"""stored_files: uploaded file bytes, so a wiped disk can be restored

Additive only: one new table.

Revision ID: e2a4c6d8f0b1
Revises: d5f7b9c1e3a2
Create Date: 2026-11-17 06:00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e2a4c6d8f0b1"
down_revision: Union[str, Sequence[str], None] = "d5f7b9c1e3a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "stored_files",
        sa.Column("filepath", sa.String(), nullable=False),
        sa.Column("content", sa.LargeBinary(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("filepath"),
    )


def downgrade() -> None:
    op.drop_table("stored_files")
