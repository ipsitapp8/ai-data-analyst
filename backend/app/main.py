"""FastAPI app: upload/profile, ask-question (kicks off the LangGraph run),
status polling, dashboard retrieval, and audit trail retrieval."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import config
from app.agents.graph import run_question_graph
from app.database import get_db, init_db
from app.models import (
    AuditTrail,
    CriticReview,
    Dashboard,
    Dataset,
    ExecutionLog,
    Plan,
    Question,
    Team,
    User,
)
from app.profiling import load_and_profile_csv
from app.routers.auth import router as auth_router
from app.routers.workspaces import router as workspaces_router
from app.schemas import (
    AuditEntry,
    AuditTrailResponse,
    DashboardResponse,
    DatasetProfile,
    QuestionCreate,
    QuestionCreated,
    StatusResponse,
)
from app.security import get_current_team, get_current_user

app = FastAPI(title="AI Data Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(workspaces_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/api/health")
def health():
    from app.sandbox.runner import docker_image_available
    return {
        "status": "ok",
        "sandbox_backend": config.SANDBOX_BACKEND,
        "sandbox_image_ready": docker_image_available(),
        "gemini_key_configured": bool(config.GEMINI_API_KEY),
        "llama_key_configured": bool(config.LLAMA_API_KEY),
    }


# ---------------------------------------------------------------- datasets --

@app.post("/api/datasets/upload", response_model=DatasetProfile)
def upload_dataset(
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported right now.")

    dest_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest_path = config.UPLOADS_DIR / dest_name
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        df, profile = load_and_profile_csv(str(dest_path))
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse CSV: {e}") from e

    dataset = Dataset(
        team_id=team.id,
        filename=file.filename,
        filepath=str(dest_path),
        row_count=profile["row_count"],
        col_count=profile["col_count"],
        profile_json=json.dumps(profile, default=str),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return DatasetProfile(
        id=dataset.id,
        filename=dataset.filename,
        row_count=dataset.row_count,
        col_count=dataset.col_count,
        profile=profile,
        uploaded_at=dataset.uploaded_at,
    )


@app.get("/api/datasets/{dataset_id}", response_model=DatasetProfile)
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    return DatasetProfile(
        id=dataset.id,
        filename=dataset.filename,
        row_count=dataset.row_count,
        col_count=dataset.col_count,
        profile=json.loads(dataset.profile_json),
        uploaded_at=dataset.uploaded_at,
    )


@app.get("/api/datasets", response_model=list[DatasetProfile])
def list_datasets(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    datasets = (
        db.query(Dataset)
        .filter(Dataset.team_id == team.id)
        .order_by(Dataset.uploaded_at.desc())
        .all()
    )
    return [
        DatasetProfile(
            id=d.id, filename=d.filename, row_count=d.row_count, col_count=d.col_count,
            profile=json.loads(d.profile_json), uploaded_at=d.uploaded_at,
        )
        for d in datasets
    ]


# ---------------------------------------------------------------- questions --

def _humanize_age(created: dt.datetime) -> str:
    delta = dt.datetime.utcnow() - created
    mins = int(delta.total_seconds() // 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins}m ago"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def _get_team_question(db: Session, team: Team, question_id: int) -> Question:
    """Fetch a question and verify it belongs to the caller's active team.

    This re-check (not just trusting the X-Team-Id header) is the actual
    cross-team isolation guarantee for every question-scoped endpoint.
    """
    question = db.get(Question, question_id)
    if not question or question.team_id != team.id:
        raise HTTPException(404, "Question not found")
    return question


@app.get("/api/questions")
def list_questions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Recent analyses, newest first — feeds the Overview and Reports pages."""
    questions = (
        db.query(Question)
        .filter(Question.team_id == team.id)
        .order_by(Question.created_at.desc())
        .limit(50)
        .all()
    )
    out = []
    for q in questions:
        dash = (
            db.query(Dashboard)
            .filter(Dashboard.question_id == q.id)
            .order_by(Dashboard.created_at.desc())
            .first()
        )
        kpi_count = len(json.loads(dash.kpis_json)) if dash else 0
        out.append({
            "id": q.id,
            "dataset_id": q.dataset_id,
            "text": q.text,
            "status": q.status,
            "current_stage": q.current_stage,
            "created_at": q.created_at,
            "age": _humanize_age(q.created_at),
            "weekday": q.created_at.weekday(),
            "kpi_count": kpi_count,
            "has_dashboard": dash is not None,
        })
    return out


@app.post("/api/questions", response_model=QuestionCreated)
def ask_question(
    payload: QuestionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset or dataset.team_id != team.id:
        raise HTTPException(404, "Dataset not found")

    question = Question(
        team_id=team.id,
        dataset_id=payload.dataset_id,
        text=payload.question,
        status="running",
        current_stage="queued",
        stage_detail="Starting analysis",
    )
    db.add(question)
    db.commit()
    db.refresh(question)

    thread = threading.Thread(target=run_question_graph, args=(question.id,), daemon=True)
    thread.start()

    return QuestionCreated(question_id=question.id, status=question.status)


def _plan_steps_for_question(db: Session, question_id: int) -> tuple[list[dict], int | None]:
    plan = (
        db.query(Plan)
        .filter(Plan.question_id == question_id)
        .order_by(Plan.created_at.desc())
        .first()
    )
    if not plan:
        return [], None
    return json.loads(plan.steps_json), plan.id


@app.get("/api/questions/{question_id}/status", response_model=StatusResponse)
def get_status(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    question = _get_team_question(db, team, question_id)

    steps, plan_id = _plan_steps_for_question(db, question_id)
    plan_created_at = None
    if plan_id:
        plan_created_at = db.get(Plan, plan_id).created_at

    logs: list[ExecutionLog] = []
    if plan_created_at:
        logs = (
            db.query(ExecutionLog)
            .filter(ExecutionLog.question_id == question_id, ExecutionLog.created_at >= plan_created_at)
            .order_by(ExecutionLog.step_index, ExecutionLog.attempt_number)
            .all()
        )

    logs_by_step: dict[int, list[ExecutionLog]] = {}
    for log in logs:
        logs_by_step.setdefault(log.step_index, []).append(log)

    step_status = []
    retry_count = 0
    for i, s in enumerate(steps):
        step_logs = logs_by_step.get(i, [])
        if any(l.success for l in step_logs):
            status = "done"
        elif step_logs:
            status = "failed" if len(step_logs) >= config.MAX_EXECUTOR_RETRIES and question.current_stage in ("done", "failed") else "running"
            retry_count = max(retry_count, len(step_logs) - 1)
        else:
            status = "pending"
        step_status.append({"id": i, "description": s["description"], "status": status})

    # Mark the first non-done step as "running" if the pipeline is actively executing.
    if question.current_stage == "executing":
        for s in step_status:
            if s["status"] == "pending":
                s["status"] = "running"
                break

    return StatusResponse(
        question_id=question.id,
        question_text=question.text,
        status=question.status,
        current_stage=question.current_stage,
        stage_detail=question.stage_detail or "",
        steps=step_status,
        retry_count=retry_count,
        error=question.error,
    )


@app.get("/api/questions/{question_id}/dashboard", response_model=DashboardResponse)
def get_dashboard(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _get_team_question(db, team, question_id)
    dash = (
        db.query(Dashboard)
        .filter(Dashboard.question_id == question_id)
        .order_by(Dashboard.created_at.desc())
        .first()
    )
    if not dash:
        raise HTTPException(404, "Dashboard not ready yet")

    return DashboardResponse(
        question_id=question_id,
        verified=dash.verified,
        verification_summary=dash.verification_summary,
        kpis=json.loads(dash.kpis_json),
        charts=json.loads(dash.charts_json),
        narrative=dash.narrative,
        created_at=dash.created_at,
    )


@app.get("/api/questions/{question_id}/audit-trail", response_model=AuditTrailResponse)
def get_audit_trail(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    question = _get_team_question(db, team, question_id)

    steps, _ = _plan_steps_for_question(db, question_id)
    entries = (
        db.query(AuditTrail)
        .filter(AuditTrail.question_id == question_id)
        .order_by(AuditTrail.created_at)
        .all()
    )

    out_entries = []
    for e in entries:
        code = e.execution_log.code if e.execution_log else None
        stdout = e.execution_log.stdout if e.execution_log else None
        result = json.loads(e.execution_log.result_json) if e.execution_log else None
        critic_verdict = e.critic_review.verdict if e.critic_review else None
        critic_summary = e.critic_review.summary if e.critic_review else None
        out_entries.append(
            AuditEntry(
                id=e.id,
                element_label=e.element_label,
                element_type=e.element_type,
                reasoning=e.reasoning,
                code=code,
                stdout=stdout,
                result=result,
                critic_verdict=critic_verdict,
                critic_summary=critic_summary,
                created_at=e.created_at,
            )
        )

    return AuditTrailResponse(
        question_id=question_id,
        question_text=question.text,
        plan=steps,
        entries=out_entries,
    )


@app.get("/api/questions/{question_id}/critic-reviews")
def get_critic_reviews(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _get_team_question(db, team, question_id)
    reviews = (
        db.query(CriticReview)
        .filter(CriticReview.question_id == question_id)
        .order_by(CriticReview.created_at)
        .all()
    )
    return [
        {
            "id": r.id,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "issues": json.loads(r.issues_json),
            "checks": json.loads(r.checks_json),
            "summary": r.summary,
            "created_at": r.created_at,
        }
        for r in reviews
    ]
