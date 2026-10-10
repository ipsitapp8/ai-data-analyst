"""Schema upgrades for the platform tables and columns, on a disposable
database, and the startup path that recovers interrupted work."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from app import database, models

NEW_TABLES = {"analysis_jobs", "run_events", "evidence_records", "run_manifests", "investigations", "dq_snapshots",
              "dq_configs", "dq_incidents", "copilot_sessions", "copilot_turns", "scenarios", "semantic_metrics",
              "semantic_metric_versions", "semantic_terms", "semantic_relationships", "eval_runs", "eval_results"}


@pytest.mark.skipif(database._IS_POSTGRES, reason="exercises the SQLite additive-migration path")
def test_upgrading_an_existing_sqlite_database_adds_columns_and_keeps_rows():
    """A database created before this work: old tables, with rows, and none of
    the new columns. init_db must add what is missing and lose nothing."""
    path = Path(tempfile.mkdtemp(prefix="silt_platform_migration_")) / "old.db"
    engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False})
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE communities (id INTEGER PRIMARY KEY, name TEXT, created_by INTEGER, created_at DATETIME)"))
        conn.execute(text("CREATE TABLE teams (id INTEGER PRIMARY KEY, community_id INTEGER, name TEXT, created_by INTEGER, created_at DATETIME)"))
        conn.execute(text("INSERT INTO communities (id, name) VALUES (1, 'C')"))
        conn.execute(text("INSERT INTO teams (id, community_id, name) VALUES (1, 1, 'T')"))
        conn.execute(text("CREATE TABLE datasets (id INTEGER PRIMARY KEY, team_id INTEGER, filename TEXT, filepath TEXT, "
                          "row_count INTEGER, col_count INTEGER, profile_json TEXT, uploaded_at DATETIME)"))
        conn.execute(text("INSERT INTO datasets VALUES (1, 1, 'old.csv', '/nowhere/old.csv', 3, 2, '{}', '2025-01-01 00:00:00')"))
        conn.execute(text("CREATE TABLE dataset_versions (id INTEGER PRIMARY KEY, dataset_id INTEGER, version_number INTEGER, "
                          "filename TEXT, filepath TEXT, row_count INTEGER, col_count INTEGER, profile_json TEXT, uploaded_at DATETIME)"))
        conn.execute(text("INSERT INTO dataset_versions VALUES (1, 1, 1, 'old.csv', '/nowhere/old.csv', 3, 2, '{}', '2025-01-01 00:00:00')"))
        conn.execute(text("CREATE TABLE questions (id INTEGER PRIMARY KEY, team_id INTEGER, dataset_id INTEGER, text TEXT, status TEXT, "
                          "current_stage TEXT, stage_detail TEXT, error TEXT, dataset_version_id INTEGER, trigger TEXT NOT NULL DEFAULT 'manual', "
                          "scheduled_analysis_id INTEGER, created_at DATETIME, updated_at DATETIME)"))
        conn.execute(text("INSERT INTO questions (id, team_id, dataset_id, text, status, current_stage) VALUES (1, 1, 1, 'old question', 'verified', 'done')"))
        conn.execute(text("INSERT INTO questions (id, team_id, dataset_id, text, status, current_stage) VALUES (2, 1, 1, 'cut off by a restart', 'running', 'executing')"))
        conn.execute(text("CREATE TABLE scheduled_analyses (id INTEGER PRIMARY KEY, workspace_id INTEGER, dataset_id INTEGER, question_text TEXT, "
                          "interval TEXT, created_by INTEGER, created_at DATETIME, last_run_at DATETIME, last_dashboard_id INTEGER, "
                          "change_threshold_pct FLOAT, is_active BOOLEAN, last_trend TEXT, last_change_summary TEXT)"))
        conn.execute(text("INSERT INTO scheduled_analyses (id, workspace_id, dataset_id, question_text, interval, change_threshold_pct, is_active) "
                          "VALUES (1, 1, 1, 'Revenue?', 'daily', 10.0, 1)"))
        conn.execute(text("CREATE TABLE alerts (id INTEGER PRIMARY KEY, team_id INTEGER, scheduled_analysis_id INTEGER, question_id INTEGER, "
                          "kind TEXT, title TEXT, detail TEXT, created_at DATETIME, read_at DATETIME)"))
        conn.execute(text("INSERT INTO alerts (id, team_id, kind, title, created_at) VALUES (1, 1, 'change', 'Old alert', '2025-01-01 00:00:00')"))

    original = database.engine
    database.engine = engine
    try:
        database.init_db()
        database.init_db()  # idempotent: a second start changes nothing and does not fail
        insp = inspect(engine)
        assert NEW_TABLES <= set(insp.get_table_names())
        cols = lambda t: {c["name"] for c in insp.get_columns(t)}  # noqa: E731
        assert {"route", "rerun_of_question_id", "clarification_json"} <= cols("questions")
        assert "content_sha256" in cols("dataset_versions")
        assert {"status", "dedup_key", "occurrences", "delivery_state", "explanation_json", "resolved_at"} <= cols("alerts")
        assert {"min_effect_abs", "comparison", "suppress_on_dq", "notify_email", "last_processed_question_id"} <= cols("scheduled_analyses")

        with Session(engine) as s:
            old_alert = s.get(models.Alert, 1)
            assert old_alert.title == "Old alert" and old_alert.status is None and old_alert.occurrences is None
            sa = s.get(models.ScheduledAnalysis, 1)
            assert sa.change_threshold_pct == 10.0 and sa.comparison is None and sa.notify_email is None
            q = s.get(models.Question, 1)
            assert q.text == "old question" and q.route is None and q.status == "verified"
            assert s.get(models.DatasetVersion, 1).content_sha256 is None
            assert s.query(models.Question).count() == 2 and s.query(models.Dataset).count() == 1
            # every new table is usable
            s.add(models.AnalysisJob(team_id=1, question_id=1))
            s.add(models.EvalRun(suite_version="x", mode="offline"))
            s.commit()
            with pytest.raises(Exception):  # one job per question
                s.add(models.AnalysisJob(team_id=1, question_id=1))
                s.commit()
    finally:
        database.engine = original
        engine.dispose()


def test_constraints_that_the_job_and_evidence_logic_rely_on_exist():
    tables = models.Base.metadata.tables
    unique = lambda t: {tuple(c.name for c in col.columns) for col in tables[t].constraints if col.__class__.__name__ == "UniqueConstraint"} | \
        {(c.name,) for c in tables[t].columns if c.unique}  # noqa: E731
    assert ("question_id",) in unique("analysis_jobs") and ("idempotency_key",) in unique("analysis_jobs")
    assert ("question_id",) in unique("run_manifests") and ("claim_id",) in unique("evidence_records")
    assert ("dataset_version_id",) in unique("dq_snapshots") and ("dataset_id",) in unique("dq_configs")
    team_scoped = ["analysis_jobs", "investigations", "dq_incidents", "dq_configs", "copilot_sessions", "scenarios",
                   "semantic_metrics", "semantic_terms", "semantic_relationships"]
    for t in team_scoped:
        col = tables[t].columns["team_id"]
        assert not col.nullable and col.index, f"{t}.team_id must be required and indexed"
        assert any(fk.column.table.name == "teams" for fk in col.foreign_keys), t


def test_alembic_revisions_form_one_chain_ending_at_the_platform_revision():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    backend = Path(database.__file__).resolve().parents[1]
    cfg = Config(str(backend / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend / "alembic"))
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"migration history has branched: {heads}"
    head = script.get_revision(heads[0])
    source = Path(head.path).read_text()
    for table in NEW_TABLES:
        assert f"'{table}'" in source or f'"{table}"' in source, f"{table} is missing from the head migration"
    assert "ix_analysis_memories_embedding" not in source.split("def upgrade")[1], "must not drop the vector index"
    for column in ("content_sha256", "dedup_key", "last_processed_question_id", "clarification_json"):
        assert column in source


@pytest.mark.skipif(not database._IS_POSTGRES, reason="needs TEST_DATABASE_URL pointing at Postgres")
def test_postgres_schema_matches_the_models_after_alembic_upgrade(client):
    """After `alembic upgrade head` (run by init_db at startup) the live schema
    has every table and column the models declare."""
    insp = inspect(database.engine)
    live = set(insp.get_table_names())
    for name, table in models.Base.metadata.tables.items():
        assert name in live, f"table {name} was not created by the migrations"
        live_cols = {c["name"] for c in insp.get_columns(name)}
        missing = {c.name for c in table.columns} - live_cols
        assert not missing, f"{name} is missing columns {missing}"
