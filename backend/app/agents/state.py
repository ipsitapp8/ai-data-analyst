"""LangGraph state shared across Planner -> Executor -> Critic -> Dashboard nodes."""
from __future__ import annotations

from typing import Any, TypedDict

from app.database import SessionLocal
from app.models import Question


class StepResult(TypedDict, total=False):
    step_index: int
    description: str
    code: str
    stdout: str
    stderr: str
    result: dict[str, Any] | None
    chart_paths: list[str]
    success: bool
    reasoning: str
    attempts: int
    execution_log_id: int


class AgentState(TypedDict, total=False):
    question_id: int
    dataset_id: int
    question_text: str
    csv_path: str
    profile: dict[str, Any]

    plan: list[dict[str, Any]]
    plan_reasoning: str
    plan_id: int

    step_index: int
    retry_count: int
    last_error: str | None
    step_results: list[StepResult]

    critic_verdict: str | None
    critic_summary: str
    critic_issues: list[str]
    critic_checks: list[dict[str, Any]]
    critic_review_id: int | None

    revision_count: int
    revision_feedback: str | None

    dashboard: dict[str, Any] | None
    failed: bool
    failure_reason: str | None

    # triage gate (runs before the planner)
    rejected: bool
    rejection_reason: str | None
    suggestions: list[str]


def update_stage(question_id: int, stage: str, detail: str = "") -> None:
    db = SessionLocal()
    try:
        q = db.get(Question, question_id)
        if q:
            q.current_stage = stage
            q.stage_detail = detail
            if stage not in ("done", "failed"):
                q.status = "running"
            db.commit()
    finally:
        db.close()


def mark_terminal(question_id: int, status: str, error: str | None = None) -> None:
    db = SessionLocal()
    try:
        q = db.get(Question, question_id)
        if q:
            q.status = status
            if status in ("verified", "unverified"):
                q.current_stage = "done"
            elif status == "rejected":
                q.current_stage = "rejected"
                q.stage_detail = "Question not analyzable for this dataset"
            else:
                q.current_stage = "failed"
            q.error = error
            db.commit()
    finally:
        db.close()
