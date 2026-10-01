"""One-time copy of an existing SQLite app.db into Postgres.

Usage:
    DATABASE_URL=postgresql://user:pass@host/db \
        python backend/scripts/migrate_sqlite_to_postgres.py [--sqlite-path PATH]

Safety:
    - The source SQLite file is opened in the driver's own read-only mode
      (sqlite3 URI "mode=ro") -- any attempted write raises immediately,
      rather than relying on this script simply choosing not to write.
    - The script refuses to run unless DATABASE_URL resolves to postgresql --
      it will not silently copy sqlite-to-sqlite or overwrite the source.
    - All target-side inserts happen in a single transaction: either every
      table copies cleanly, or nothing is committed at all.
    - Existing ids are preserved (so cross-table foreign keys stay valid),
      and Postgres's auto-increment sequences are reset to match afterward.

This does not delete or modify the source file in any way, and does not run
automatically as part of app startup -- it's a deliberate, manual, one-time
operation you run yourself when you're ready to cut over.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import MetaData, create_engine, select, text  # noqa: E402

from app import models  # noqa: E402, F401 (registers tables on Base.metadata)
from app.config import DATABASE_URL  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402

# FK-safe order -- every table's dependencies precede it.
TABLE_ORDER = [
    "users",
    "communities",
    "teams",
    "team_members",
    "datasets",
    "questions",
    "plans",
    "execution_logs",
    "critic_reviews",
    "dashboards",
    "audit_trail",
]

# Columns that are JSON-serialized TEXT on the source SQLite file (pre-Part-A)
# and must be parsed into real Python objects before writing to the target's
# native JSON/JSONB columns.
JSON_COLUMNS = {
    "datasets": ["profile_json"],
    "plans": ["steps_json"],
    "execution_logs": ["result_json", "chart_paths_json", "data_slice_json"],
    "critic_reviews": ["issues_json", "checks_json"],
    "dashboards": ["kpis_json", "charts_json"],
}


def _parse_json_columns(table: str, row: dict) -> dict:
    for col in JSON_COLUMNS.get(table, []):
        raw = row.get(col)
        if isinstance(raw, str):
            row[col] = json.loads(raw) if raw else None
    return row


def _reset_sequence(session, table: str) -> None:
    session.execute(
        text(
            f"SELECT setval(pg_get_serial_sequence(:t, 'id'), "
            f"COALESCE((SELECT MAX(id) FROM {table}), 1), true)"
        ),
        {"t": table},
    )


def migrate(sqlite_path: Path) -> None:
    target_dialect = engine.url.get_backend_name()
    if target_dialect != "postgresql":
        raise SystemExit(
            f"DATABASE_URL resolves to '{target_dialect}', not postgresql -- "
            "refusing to run. Set DATABASE_URL to the Postgres target first."
        )
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite file not found: {sqlite_path}")

    # sqlite3's own read-only URI mode: any write attempt raises, this isn't
    # just "we choose not to call commit". Built via Path.as_uri() + a
    # `creator` callable rather than a bare sqlite:/// connection string --
    # correctly handles a Windows drive letter, which SQLAlchemy's own
    # sqlite URL parser does not.
    ro_uri = sqlite_path.resolve().as_uri() + "?mode=ro"
    source_engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(ro_uri, uri=True, check_same_thread=False),
    )
    source_meta = MetaData()
    source_meta.reflect(bind=source_engine)

    Base.metadata.create_all(bind=engine)  # no-op if Alembic already applied it

    session = SessionLocal()
    try:
        with source_engine.connect() as source_conn:
            for table_name in TABLE_ORDER:
                if table_name not in source_meta.tables:
                    print(f"  (skip: {table_name} not present in source)")
                    continue
                source_table = source_meta.tables[table_name]
                rows = source_conn.execute(select(source_table)).mappings().all()
                if not rows:
                    print(f"  {table_name}: 0 rows")
                    continue

                model_class = next(
                    m for m in Base.registry.mappers if m.local_table.name == table_name
                ).class_
                for row in rows:
                    data = _parse_json_columns(table_name, dict(row))
                    session.add(model_class(**data))
                # Flushed per table (not once at the end) so INSERTs execute in
                # exactly TABLE_ORDER -- relying on SQLAlchemy's own cross-table
                # dependency sort here isn't necessary and isn't worth the risk.
                session.flush()
                print(f"  {table_name}: {len(rows)} rows")

            for table_name in TABLE_ORDER:
                if table_name in source_meta.tables:
                    _reset_sequence(session, table_name)

        session.commit()
        print("Done -- all tables committed in one transaction.")
    except Exception:
        session.rollback()
        print("FAILED -- rolled back, nothing was committed to Postgres.", file=sys.stderr)
        raise
    finally:
        session.close()
        source_engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=None,
        help="Path to the source SQLite file (defaults to app.config.DB_PATH)",
    )
    args = parser.parse_args()

    if args.sqlite_path is None:
        from app.config import DB_PATH

        args.sqlite_path = DB_PATH

    print(f"Source (read-only): {args.sqlite_path}")
    print(f"Target: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL}")
    migrate(args.sqlite_path)
