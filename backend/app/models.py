"""SQLAlchemy ORM models — the audit-trail-first schema.

Tables: users, communities, teams, team_members, datasets, questions, plans,
execution_logs, critic_reviews, dashboards, audit_trail.

Every dashboard output is linked, via audit_trail rows, back to the exact
execution_log (code + stdout/result) and critic_review that produced /
verified it. That linkage is what the Audit Trail UI page renders.

datasets/questions/dashboards/audit_trail carry a `team_id` for multi-tenant
isolation -- see the nullability note on Dataset.team_id for why it isn't a
DB-level NOT NULL constraint.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from app.config import EMBEDDING_DIM
from app.database import Base

# Portable JSON column type: native JSONB on Postgres, SQLite's built-in JSON
# handling everywhere else. Either way SQLAlchemy (de)serializes automatically --
# callers get/set a plain dict/list, no manual json.loads/json.dumps needed.
JSONVariant = JSON().with_variant(JSONB, "postgresql")


def utcnow() -> dt.datetime:
    return dt.datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)


class Community(Base):
    __tablename__ = "communities"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    teams = relationship("Team", back_populates="community")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    community_id = Column(Integer, ForeignKey("communities.id"), nullable=False)
    name = Column(String, nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow)

    community = relationship("Community", back_populates="teams")
    members = relationship("TeamMember", back_populates="team")


class TeamMember(Base):
    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    # Nullable + invited_email: lets an owner invite someone by email before
    # that person has an account. Claimed (user_id set, status->active) at signup.
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    invited_email = Column(String, nullable=True)
    role = Column(String, default="member")  # owner|admin|member
    status = Column(String, default="active")  # active|pending
    joined_at = Column(DateTime, default=utcnow)

    team = relationship("Team", back_populates="members")
    user = relationship("User")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    # Nullable at the DB level only because SQLite can't cheaply add a NOT NULL
    # FK to an existing table with rows -- the migration backfills every existing
    # row to a "Legacy" team, and the app layer treats this as required from here
    # on (always set on write, always filtered on read).
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    profile_json = Column(JSONVariant, default=dict)  # column types, missing values, stats
    uploaded_at = Column(DateTime, default=utcnow)

    questions = relationship("Question", back_populates="dataset")
    versions = relationship(
        "DatasetVersion", back_populates="dataset", order_by="DatasetVersion.version_number"
    )


class DatasetVersion(Base):
    """One uploaded file within a dataset "slot". The Dataset row keeps its id
    (and mirrors the latest version's file/profile so existing readers keep
    working); each "Replace data" upload adds a row here instead of creating an
    unrelated dataset. Questions pin the version they ran against, so an old
    dashboard never silently changes when newer data arrives."""

    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    version_number = Column(Integer, nullable=False)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    profile_json = Column(JSONVariant, default=dict)
    uploaded_at = Column(DateTime, default=utcnow)

    dataset = relationship("Dataset", back_populates="versions")


class ScheduledAnalysis(Base):
    """A question saved to be re-run on an interval; see app/scheduler.py."""

    __tablename__ = "scheduled_analyses"

    id = Column(Integer, primary_key=True, index=True)
    workspace_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    question_text = Column(Text, nullable=False)
    interval = Column(String, nullable=False, default="daily")  # daily|weekly
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    last_run_at = Column(DateTime, nullable=True)
    last_dashboard_id = Column(Integer, ForeignKey("dashboards.id"), nullable=True)
    change_threshold_pct = Column(Float, nullable=False, default=10.0)
    is_active = Column(Boolean, nullable=False, default=True)
    last_trend = Column(String, nullable=True)  # up|down|flat, since the previous run
    last_change_summary = Column(Text, default="")


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)  # see Dataset.team_id note
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    text = Column(Text, nullable=False)
    status = Column(String, default="pending")  # pending|running|verified|failed
    current_stage = Column(String, default="queued")
    stage_detail = Column(Text, default="")  # short human-readable status line
    error = Column(Text, nullable=True)
    # Additive: the exact dataset version this run analyzed (NULL on legacy
    # rows), and whether a person or the scheduler started it.
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=True)
    trigger = Column(String, nullable=False, default="manual", server_default="manual")  # manual|scheduled
    # Deliberately not an FK: scheduled_analyses.last_dashboard_id -> dashboards
    # -> questions would make the FK graph circular, which SQLite can't create.
    scheduled_analysis_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    dataset = relationship("Dataset", back_populates="questions")
    plans = relationship("Plan", back_populates="question")
    execution_logs = relationship("ExecutionLog", back_populates="question")
    critic_reviews = relationship("CriticReview", back_populates="question")
    dashboards = relationship("Dashboard", back_populates="question")
    audit_entries = relationship("AuditTrail", back_populates="question")


class Plan(Base):
    __tablename__ = "plans"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    steps_json = Column(JSONVariant, nullable=False)  # list[{id, description, goal}]
    reasoning = Column(Text, default="")  # planner's rationale for this breakdown
    revision = Column(Integer, default=0)  # 0 = initial plan, 1+ = critic-triggered revision
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="plans")


class ExecutionLog(Base):
    __tablename__ = "execution_logs"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    step_index = Column(Integer, nullable=False)
    step_description = Column(Text, nullable=False)
    attempt_number = Column(Integer, default=1)  # 1..max_retries
    code = Column(Text, nullable=False)
    stdout = Column(Text, default="")
    stderr = Column(Text, default="")
    success = Column(Boolean, default=False)
    result_json = Column(JSONVariant, default=dict)  # structured result the code printed
    chart_paths_json = Column(JSONVariant, default=list)  # plotly json files produced
    reasoning = Column(Text, default="")  # executor's explanation of the approach
    formula_explanation = Column(Text, default="")  # plain-English calculation, for click-to-inspect
    data_slice_json = Column(JSONVariant, default=dict)  # {columns, rows} this step's result was computed from
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="execution_logs")


class CriticReview(Base):
    __tablename__ = "critic_reviews"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    verdict = Column(String, nullable=False)  # verified|rejected
    confidence = Column(Float, default=0.0)
    issues_json = Column(JSONVariant, default=list)  # list of flagged problems
    checks_json = Column(JSONVariant, default=list)  # independent re-checks the critic ran
    summary = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="critic_reviews")


class Dashboard(Base):
    __tablename__ = "dashboards"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)  # see Dataset.team_id note
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    kpis_json = Column(JSONVariant, default=list)
    charts_json = Column(JSONVariant, default=list)  # [{title, plotly_json, source_step_index}]
    narrative = Column(Text, default="")
    verified = Column(Boolean, default=False)
    verification_summary = Column(Text, default="")
    # VERIFIED | VERIFIED_WITH_CAVEATS | UNVERIFIED (see app/verdict.py). NULL on
    # dashboards created before this column existed; readers resolve those.
    verdict_state = Column(String, nullable=True)
    # Chart Studio overrides, keyed by chart element_id (or "idx<n>" for charts
    # that predate element ids): {"chart_type": str | None, "style": {...}}.
    # Shared by the whole team, like the dashboard itself.
    view_overrides_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="dashboards")


class AnalysisMemory(Base):
    """One finished analysis, embedded for retrieval by the Planner (see
    app/memory.py). Scoped by team_id + dataset_id; never read across teams."""

    __tablename__ = "analysis_memories"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, unique=True)
    question_text = Column(Text, nullable=False)
    summary = Column(Text, default="")  # KPIs + narrative, what the Planner sees
    verdict_state = Column(String, nullable=True)
    # pgvector column on Postgres (indexed, searched in SQL); JSON list on SQLite.
    embedding = Column(JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql"), nullable=False)
    created_at = Column(DateTime, default=utcnow)


class KnowledgeNote(Base):
    """A short, team-owned fact the Planner is given about one dataset.

    kind="knowledge": what the data means (column definitions, units, quirks) --
    always included in planning. kind="lesson": a past mistake to avoid, written
    by a user or auto-captured from a Critic rejection (source="critic") --
    retrieved by similarity. Scoped by team_id + dataset_id; never read across
    teams. See app/knowledge.py.
    """

    __tablename__ = "knowledge_notes"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    kind = Column(String, nullable=False)  # "knowledge" | "lesson"
    source = Column(String, nullable=False, default="user")  # "user" | "critic"
    text = Column(Text, nullable=False)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=True)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    # Nullable: a note is still saved (and, for knowledge, still used) if the
    # embedding call fails; only lesson retrieval needs the vector.
    embedding = Column(JSON().with_variant(Vector(EMBEDDING_DIM), "postgresql"), nullable=True)
    created_at = Column(DateTime, default=utcnow)


class ShareLink(Base):
    """A revocable, expiring read-only link to one finished dashboard (see
    app/routers/sharing.py). The token is the only credential."""

    __tablename__ = "share_links"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String, nullable=False, unique=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False, index=True)
    dashboard_id = Column(Integer, ForeignKey("dashboards.id"), nullable=False)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)


class DatasetInsight(Base):
    """Auto-generated starter questions + data-quality warnings for one dataset
    version (see app/insights.py)."""

    __tablename__ = "dataset_insights"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False, index=True)
    dataset_version_id = Column(Integer, ForeignKey("dataset_versions.id"), nullable=True)
    questions_json = Column(JSONVariant, default=list)  # [{question, why}]
    warnings_json = Column(JSONVariant, default=list)  # [{severity, column, message}]
    created_at = Column(DateTime, default=utcnow)


class Alert(Base):
    """In-app alert raised by a scheduled run (see app/scheduler.py), shown on the
    Scheduled page whether or not email is configured. Read state is shared by
    the whole team."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False, index=True)
    scheduled_analysis_id = Column(Integer, ForeignKey("scheduled_analyses.id"), nullable=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=True)
    kind = Column(String, nullable=False)  # change | anomaly | unverified
    title = Column(String, nullable=False)
    detail = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)
    read_at = Column(DateTime, nullable=True)


class AuditTrail(Base):
    __tablename__ = "audit_trail"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)  # see Dataset.team_id note
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    element_label = Column(String, nullable=False)  # e.g. "KPI: Revenue Drop %"
    element_type = Column(String, default="kpi")  # kpi|chart|narrative
    # Matches the "element_id" embedded in the dashboard's kpis_json/charts_json,
    # so the frontend's click-to-inspect can look this row up by that id.
    element_id = Column(String, nullable=True, index=True)
    execution_log_id = Column(Integer, ForeignKey("execution_logs.id"), nullable=True)
    critic_review_id = Column(Integer, ForeignKey("critic_reviews.id"), nullable=True)
    reasoning = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="audit_entries")
    execution_log = relationship("ExecutionLog")
    critic_review = relationship("CriticReview")
