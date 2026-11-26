"""copilot_conversations, copilot_messages: saved Silt Copilot chats

Additive only: two new tables. A table that is already there is left alone, so
startup cannot fail on a database where one was created outside Alembic.

Revision ID: f4b6d8a0c2e3
Revises: e2a4c6d8f0b1
Create Date: 2026-11-26 10:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "f4b6d8a0c2e3"
down_revision: Union[str, Sequence[str], None] = "e2a4c6d8f0b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _json():
    return sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    existing = set(sa.inspect(op.get_bind()).get_table_names())

    if "copilot_conversations" not in existing:
        op.create_table(
            "copilot_conversations",
            sa.Column("id", sa.String(length=64), nullable=False),
            sa.Column("team_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("title", sa.String(length=200), nullable=False),
            sa.Column("dataset_id", sa.Integer(), nullable=True),
            sa.Column("document_id", sa.String(length=64), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["team_id"], ["teams.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_copilot_conversations_team_id"), "copilot_conversations", ["team_id"])
        op.create_index(op.f("ix_copilot_conversations_user_id"), "copilot_conversations", ["user_id"])
        op.create_index(op.f("ix_copilot_conversations_updated_at"), "copilot_conversations", ["updated_at"])

    if "copilot_messages" not in existing:
        op.create_table(
            "copilot_messages",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("conversation_id", sa.String(length=64), nullable=False),
            sa.Column("position", sa.Integer(), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("answer_text", sa.Text(), nullable=True),
            sa.Column("response_json", _json(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["conversation_id"], ["copilot_conversations.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("conversation_id", "position", name="uq_copilot_message_position"),
        )
        op.create_index(op.f("ix_copilot_messages_conversation_id"), "copilot_messages", ["conversation_id"])


def downgrade() -> None:
    op.drop_table("copilot_messages")
    op.drop_table("copilot_conversations")
