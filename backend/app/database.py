"""SQLAlchemy engine/session setup.

Dialect-branched: SQLite (default, local dev/tests) keeps the exact behavior
this app has always had -- single-file, additive hand-rolled migrations below.
Postgres (set DATABASE_URL) gets real connection pooling and Alembic-managed
migrations instead. Nothing about the SQLite path changes when DATABASE_URL
is unset, which is the backward-compatibility guarantee for this module.
"""
from __future__ import annotations

import json
from functools import partial

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

_dialect = make_url(DATABASE_URL).get_backend_name()
_IS_POSTGRES = _dialect == "postgresql"

# Same fallback the old json.dumps(profile, default=str) call relied on for
# stray non-JSON-native values (e.g. numpy scalars) -- applied engine-wide now
# that JSON columns serialize automatically instead of at each call site.
_json_serializer = partial(json.dumps, default=str)

if _IS_POSTGRES:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # managed Postgres closes idle connections; avoids a stale-connection error on first use after idle
        pool_recycle=1800,
        json_serializer=_json_serializer,
    )
else:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        json_serializer=_json_serializer,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Tables that predate multi-tenancy and need a backfilled team_id.
_TEAM_SCOPED_TABLES = ("datasets", "questions", "dashboards", "audit_trail")

# Simple additive columns (no backfill needed -- NULL/default is a fine value
# for pre-existing rows, which just aren't inspectable via the newer features).
_ADDITIVE_COLUMNS = (
    ("execution_logs", "formula_explanation", "TEXT"),
    ("execution_logs", "data_slice_json", "TEXT"),
    ("audit_trail", "element_id", "TEXT"),
    ("questions", "dataset_version_id", "INTEGER REFERENCES dataset_versions(id)"),
    ("questions", "trigger", "TEXT NOT NULL DEFAULT 'manual'"),
    ("questions", "scheduled_analysis_id", "INTEGER"),
    ("dashboards", "verdict_state", "TEXT"),
    ("dashboards", "view_overrides_json", "TEXT"),
)


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:t"), {"t": table}
    ).first()
    return row is not None


def _has_column(conn, table: str, column: str) -> bool:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return any(r[1] == column for r in rows)


def _run_migrations() -> None:
    """Idempotent, additive-only migration for the team_id multi-tenancy columns.

    SQLite can't cheaply add a NOT NULL FK to a table that already has rows
    (that requires a full table rebuild), so team_id is added nullable here
    and backfilled to a "Legacy" community/team -- the application layer is
    what actually enforces it from then on (see Dataset.team_id in models.py).
    Safe to run on every startup: every step checks before acting.
    """
    with engine.begin() as conn:
        for table in _TEAM_SCOPED_TABLES:
            if _table_exists(conn, table) and not _has_column(conn, table, "team_id"):
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN team_id INTEGER REFERENCES teams(id)"))

        for table, column, sql_type in _ADDITIVE_COLUMNS:
            if _table_exists(conn, table) and not _has_column(conn, table, column):
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}"))

        orphaned = [
            table for table in _TEAM_SCOPED_TABLES
            if _table_exists(conn, table)
            and conn.execute(text(f"SELECT 1 FROM {table} WHERE team_id IS NULL LIMIT 1")).first()
        ]
        if not orphaned:
            return

        legacy_team_id = conn.execute(
            text("SELECT id FROM teams WHERE name = 'Legacy' LIMIT 1")
        ).scalar()
        if legacy_team_id is None:
            legacy_community_id = conn.execute(
                text("INSERT INTO communities (name, created_by, created_at) "
                     "VALUES ('Legacy', NULL, CURRENT_TIMESTAMP)")
            ).lastrowid
            legacy_team_id = conn.execute(
                text("INSERT INTO teams (community_id, name, created_by, created_at) "
                     "VALUES (:cid, 'Legacy', NULL, CURRENT_TIMESTAMP)"),
                {"cid": legacy_community_id},
            ).lastrowid

        for table in orphaned:
            conn.execute(
                text(f"UPDATE {table} SET team_id = :tid WHERE team_id IS NULL"),
                {"tid": legacy_team_id},
            )


def _run_alembic_upgrade() -> None:
    """Postgres path: apply pending Alembic migrations up to head. Runs on every
    startup, same operational shape as the SQLite path's auto-migration below --
    no separate manual migration step needed to deploy."""
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    backend_dir = Path(__file__).resolve().parents[1]
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    command.upgrade(cfg, "head")


def init_db() -> None:
    """Bring the schema up to date. SQLite: create-all + the additive migrations
    below (unchanged). Postgres: Alembic upgrade to head."""
    from app import models  # noqa: F401  (ensure models are registered on Base)

    if _IS_POSTGRES:
        _run_alembic_upgrade()
    else:
        Base.metadata.create_all(bind=engine)
        _run_migrations()

    from app.dataset_versions import backfill_dataset_versions

    backfill_dataset_versions()


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
