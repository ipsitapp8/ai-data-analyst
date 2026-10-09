"""dashboards.view_overrides_json (Chart Studio overrides)

Purely additive: one nullable column. Readers treat NULL as "no overrides".

Revision ID: e0f2a4b6c8d1
Revises: d9e1f3a4b5c6
Create Date: 2026-10-10 12:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e0f2a4b6c8d1"
down_revision: Union[str, Sequence[str], None] = "d9e1f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("dashboards", sa.Column("view_overrides_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("dashboards", "view_overrides_json")
