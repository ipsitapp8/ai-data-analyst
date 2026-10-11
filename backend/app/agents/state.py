"""LangGraph state shared across Planner -> Executor -> Critic -> Dashboard nodes."""
from __future__ import annotations

from typing import Any, TypedDict

from app import runtime
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
    formula_explanation: str


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

    # router (runs first)
    route: str
    route_reasons: list[str]
    route_spec: dict[str, Any]
    approved_metrics: list[dict[str, Any]]
    planner_hint: str
    clarification: dict[str, Any]

    # compile -> verify -> publish
    draft: dict[str, Any]
    evidence: list[dict[str, Any]]
    dataset_evidence: dict[str, Any]
    verification_failed: bool
    verification_feedback: str | None
    investigation_id: int


def update_stage(question_id: int, stage: str, detail: str = "") -> None:
    """Record progress. Also the run's main checkpoint: every node calls this
    first, so a cancelled, overdue or over-budget run stops here."""
    runtime.checkpoint()
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
    from app import jobs

    jobs.set_progress(question_id, stage, detail)


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
            elif status == "needs_clarification":
                q.current_stage = "needs_clarification"
                q.stage_detail = "This question has more than one possible meaning"
            elif status == "cancelled":
                q.current_stage = "cancelled"
                q.stage_detail = "Cancelled"
            else:
                q.current_stage = "failed"
            q.error = error
            db.commit()
    finally:
        db.close()
