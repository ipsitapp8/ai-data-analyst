"""The team_id backfill migration -- this is what runs against the real,
populated app.db on every deploy.

Swaps app.database's module-level `engine` for a throwaway one pointed at a
scratch file for the duration of the test, rather than deleting/reimporting
`app.*` from sys.modules: this test suite shares one FastAPI app/TestClient
across the whole session (see conftest.py), and reimporting modules like
app.rate_limit out from under it would silently detach its internal state
(its rate-limit dict, specifically) from the one the running app actually
uses -- exactly what happened the first time this test was written.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app import database


def test_legacy_backfill_on_pre_existing_data():
    """Simulates upgrading a pre-multi-tenancy database: rows exist with no
    team_id, and the migration must backfill them into an auto-created
    "Legacy" team rather than leaving them orphaned or crashing."""
    scratch_db = Path(tempfile.mkdtemp(prefix="silt_migration_test_")) / "legacy.db"
    scratch_engine = create_engine(f"sqlite:///{scratch_db.as_posix()}", connect_args={"check_same_thread": False})

    # Pre-migration schema by hand: just the old datasets table, no team_id
    # column at all, exactly like a database that predates this migration.
    with scratch_engine.begin() as conn:
        conn.execute(text(
            "CREATE TABLE datasets (id INTEGER PRIMARY KEY, filename TEXT, "
            "filepath TEXT, row_count INTEGER, col_count INTEGER, "
            "profile_json TEXT, uploaded_at TEXT)"
        ))
        conn.execute(text(
            "INSERT INTO datasets (id, filename, filepath, row_count, col_count) "
            "VALUES (1, 'old.csv', '/x/old.csv', 10, 2)"
        ))

    original_engine = database.engine
    database.engine = scratch_engine
    try:
        database.init_db()

        with Session(scratch_engine) as session:
            row = session.execute(text("SELECT team_id FROM datasets WHERE id = 1")).first()
            assert row is not None
            assert row[0] is not None, "pre-existing row should have been backfilled with a team_id"

            team = session.execute(
                text("SELECT id, name FROM teams WHERE id = :tid"), {"tid": row[0]}
            ).first()
            assert team is not None
            assert team[1] == "Legacy"
    finally:
        database.engine = original_engine
        scratch_engine.dispose()
