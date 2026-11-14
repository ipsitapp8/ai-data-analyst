"""scenario evidence: sealed, append-only records of what-if calculations

Additive only: one new table. Nothing existing is touched and no row is rewritten. Each row stores the request, the
computed result, the verification outcome, the model version and engine versions, the dataset provenance and a content
hash, so a calculation can be inspected and reproduced later. Rows are never updated.

Revision ID: d5f7b9c1e3a2
Revises: c4e6a8b0d2f1
Create Date: 2026-11-14 10:20:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd5f7b9c1e3a2'
down_revision: Union[str, Sequence[str], None] = 'c4e6a8b0d2f1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('scenario_evidence',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('team_id', sa.Integer(), nullable=False),
    sa.Column('scenario_id', sa.Integer(), nullable=True),
    sa.Column('kind', sa.String(), nullable=False),
    sa.Column('model_ref', sa.String(), nullable=False),
    sa.Column('model_sha256', sa.String(), nullable=True),
    sa.Column('dataset_id', sa.Integer(), nullable=True),
    sa.Column('dataset_version_id', sa.Integer(), nullable=True),
    sa.Column('dataset_sha256', sa.String(), nullable=True),
    sa.Column('request_json', JSON_TYPE, nullable=True),
    sa.Column('result_json', JSON_TYPE, nullable=True),
    sa.Column('verification_status', sa.String(), nullable=False),
    sa.Column('verification_json', JSON_TYPE, nullable=True),
    sa.Column('engine_versions_json', JSON_TYPE, nullable=True),
    sa.Column('seed', sa.BigInteger(), nullable=True),
    sa.Column('content_sha256', sa.String(), nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('created_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], ),
    sa.ForeignKeyConstraint(['scenario_id'], ['scenarios.id'], ),
    sa.ForeignKeyConstraint(['team_id'], ['teams.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_scenario_evidence_id'), 'scenario_evidence', ['id'], unique=False)
    op.create_index(op.f('ix_scenario_evidence_team_id'), 'scenario_evidence', ['team_id'], unique=False)
    op.create_index(op.f('ix_scenario_evidence_scenario_id'), 'scenario_evidence', ['scenario_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_scenario_evidence_scenario_id'), table_name='scenario_evidence')
    op.drop_index(op.f('ix_scenario_evidence_team_id'), table_name='scenario_evidence')
    op.drop_index(op.f('ix_scenario_evidence_id'), table_name='scenario_evidence')
    op.drop_table('scenario_evidence')
