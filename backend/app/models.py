"""SQLAlchemy ORM models — the audit-trail-first schema.

Tables: datasets, questions, plans, execution_logs, critic_reviews,
dashboards, audit_trail.

Every dashboard output is linked, via audit_trail rows, back to the exact
execution_log (code + stdout/result) and critic_review that produced /
verified it. That linkage is what the Audit Trail UI page renders.
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


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
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
