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
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.database import Base


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
    profile_json = Column(Text, default="{}")  # column types, missing values, stats
    uploaded_at = Column(DateTime, default=utcnow)

    questions = relationship("Question", back_populates="dataset")


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
    steps_json = Column(Text, nullable=False)  # list[{id, description, goal}]
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
    result_json = Column(Text, default="{}")  # structured result the code printed
    chart_paths_json = Column(Text, default="[]")  # plotly json files produced
    reasoning = Column(Text, default="")  # executor's explanation of the approach
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="execution_logs")


class CriticReview(Base):
    __tablename__ = "critic_reviews"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    verdict = Column(String, nullable=False)  # verified|rejected
    confidence = Column(Float, default=0.0)
    issues_json = Column(Text, default="[]")  # list of flagged problems
    checks_json = Column(Text, default="[]")  # independent re-checks the critic ran
    summary = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="critic_reviews")


class Dashboard(Base):
    __tablename__ = "dashboards"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)  # see Dataset.team_id note
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    kpis_json = Column(Text, default="[]")
    charts_json = Column(Text, default="[]")  # [{title, plotly_json, source_step_index}]
    narrative = Column(Text, default="")
    verified = Column(Boolean, default=False)
    verification_summary = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="dashboards")


class AuditTrail(Base):
    __tablename__ = "audit_trail"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)  # see Dataset.team_id note
    question_id = Column(Integer, ForeignKey("questions.id"), nullable=False)
    element_label = Column(String, nullable=False)  # e.g. "KPI: Revenue Drop %"
    element_type = Column(String, default="kpi")  # kpi|chart|narrative
    execution_log_id = Column(Integer, ForeignKey("execution_logs.id"), nullable=True)
    critic_review_id = Column(Integer, ForeignKey("critic_reviews.id"), nullable=True)
    reasoning = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow)

    question = relationship("Question", back_populates="audit_entries")
    execution_log = relationship("ExecutionLog")
    critic_review = relationship("CriticReview")
