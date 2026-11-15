"""scenario models: versioned custom what-if model definitions

Additive only: two new tables. No existing table or column is touched and no row is rewritten, so it is safe on a
populated database. Definitions are immutable per version (a change is a new version) and saved scenarios pin the
version they used, so editing a model never changes an earlier result.

Revision ID: c4e6a8b0d2f1
Revises: b3d5f7a9c1e2
Create Date: 2026-11-13 21:10:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'c4e6a8b0d2f1'
down_revision: Union[str, Sequence[str], None] = 'b3d5f7a9c1e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('scenario_models',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('slug', sa.String(), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('current_version', sa.Integer(), nullable=False),
    sa.Column('created_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('team_id', 'slug', name='uq_scenario_model_slug')
    )
    op.create_index(op.f('ix_scenario_models_id'), 'scenario_models', ['id'], unique=False)
    op.create_index(op.f('ix_scenario_models_team_id'), 'scenario_models', ['team_id'], unique=False)
    op.create_index(op.f('ix_scenario_models_slug'), 'scenario_models', ['slug'], unique=False)
    op.create_table('scenario_model_versions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('model_id', sa.Integer(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('definition_json', JSON_TYPE, nullable=True),
    sa.Column('definition_sha256', sa.String(), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('created_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['model_id'], ['scenario_models.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('model_id', 'version', name='uq_scenario_model_version')
    )
    op.create_index(op.f('ix_scenario_model_versions_id'), 'scenario_model_versions', ['id'], unique=False)
    op.create_index(op.f('ix_scenario_model_versions_model_id'), 'scenario_model_versions', ['model_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_scenario_model_versions_model_id'), table_name='scenario_model_versions')
    op.drop_index(op.f('ix_scenario_model_versions_id'), table_name='scenario_model_versions')
    op.drop_table('scenario_model_versions')
    op.drop_index(op.f('ix_scenario_models_slug'), table_name='scenario_models')
    op.drop_index(op.f('ix_scenario_models_team_id'), table_name='scenario_models')
    op.drop_index(op.f('ix_scenario_models_id'), table_name='scenario_models')
    op.drop_table('scenario_models')
