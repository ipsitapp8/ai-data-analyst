"""SQLAlchemy engine/session setup for the SQLite store."""
from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Tables that predate multi-tenancy and need a backfilled team_id.
_TEAM_SCOPED_TABLES = ("datasets", "questions", "dashboards", "audit_trail")


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


def init_db() -> None:
    """Create all tables if they don't already exist, then migrate older ones."""
    from app import models  # noqa: F401  (ensure models are registered on Base)

    Base.metadata.create_all(bind=engine)
    _run_migrations()


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
