"""FastAPI app: upload/profile, ask-question (kicks off the LangGraph run),
status polling, dashboard retrieval, and audit trail retrieval."""
from __future__ import annotations

import datetime as dt
import json
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app import config, data_quality, jobs, provenance
from app.database import get_db, init_db
from app.dataset_versions import add_version, latest_version, version_count
from app.logging_config import setup_logging
from app.models import (
    AnalysisJob,
    AuditTrail,
    CriticReview,
    Dashboard,
    Dataset,
    DatasetVersion,
    EvidenceRecord,
    ExecutionLog,
    Plan,
    Question,
    Team,
    User,
)
from app.profiling import load_and_profile_csv
from app.routers.auth import router as auth_router
from app.routers.alerts import router as alerts_router
from app.routers.chat import router as chat_router
from app.routers.insights import router as insights_router
from app.routers.correlations import router as correlations_router
from app.routers.chart_views import router as chart_views_router
from app.routers.copilot import router as copilot_router
from app.routers.evals import router as evals_router
from app.routers.evidence import router as evidence_router
from app.routers.investigations import router as investigations_router
from app.routers.jobs import router as jobs_router
from app.routers.quality import router as quality_router
from app.routers.scenarios import router as scenarios_router
from app.routers.semantic import router as semantic_router
from app.routers.inspect import router as inspect_router
from app.routers.knowledge import router as knowledge_router
from app.routers.sharing import router as sharing_router
from app.routers.scheduled import router as scheduled_router
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
from app.verdict import (
    flagged_element_ids,
    flagged_item_count,
    narrative_audit,
    rejected_reviews,
    rejection_payload,
    resolve_verdict_state,
)

app = FastAPI(title="AI Data Analyst API")

# The Streamlit frontend talks to this API server-side (plain `requests`
# calls), which browsers -- and therefore CORS -- never enter into; this
# middleware only matters if something calls the API directly from a
# browser. Configurable so a standalone/exposed deployment can lock it down;
# defaults to today's open behavior so nothing breaks for the shipped topology.
_cors_origins = [o.strip() for o in config.CORS_ALLOWED_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(workspaces_router)
app.include_router(inspect_router)
app.include_router(knowledge_router)
app.include_router(sharing_router)
app.include_router(insights_router)
app.include_router(correlations_router)
app.include_router(alerts_router)
app.include_router(chat_router)
app.include_router(scheduled_router)
app.include_router(chart_views_router)
app.include_router(jobs_router)
app.include_router(evidence_router)
app.include_router(investigations_router)
app.include_router(quality_router)
app.include_router(copilot_router)
app.include_router(scenarios_router)
app.include_router(semantic_router)
app.include_router(evals_router)


@app.on_event("startup")
def on_startup() -> None:
    setup_logging()
    init_db()
    # Recovery first, then workers: jobs cut off by the last shutdown are back
    # in the queue (or failed, if out of attempts) before anything new starts.
    jobs.start_workers()
    from app.scheduler import start_scheduler
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown() -> None:
    from app.scheduler import stop_scheduler
    stop_scheduler()
    jobs.stop_workers()


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    from sqlalchemy import text
    from app.sandbox.runner import docker_image_available, sandbox_status
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:  # noqa: BLE001 - health check must report, not raise
        db_ok = False
    sandbox = sandbox_status()
    try:
        queue = jobs.queue_metrics(db) if db_ok else {}
    except Exception:  # noqa: BLE001
        queue = {}
    return {
        # "degraded" also when the sandbox is not usable: analyses that need
        # generated code will be refused (fail closed) until it is.
        "status": "ok" if db_ok and sandbox["ready"] else "degraded",
        "database_ok": db_ok,
        "sandbox_backend": config.SANDBOX_BACKEND,
        "sandbox_image_ready": docker_image_available() if config.SANDBOX_BACKEND == "docker" else False,
        "sandbox": sandbox,
        "gemini_key_configured": bool(config.GEMINI_API_KEY),
        "llama_key_configured": bool(config.LLAMA_API_KEY),
        "llm_failover_enabled": config.LLM_FAILOVER_ENABLED,
        "app_version": config.APP_VERSION,
        "jobs": {"queue_depth": queue.get("queue_depth", 0), "running": queue.get("running", 0),
                 "workers": config.JOB_WORKERS},
    }


# ---------------------------------------------------------------- datasets --

def _save_and_profile_upload(file: UploadFile) -> tuple[Path, dict]:
    """Stream an uploaded CSV to disk (size-capped) and profile it."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are supported right now.")

    dest_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest_path = config.UPLOADS_DIR / dest_name
    written = 0
    chunk_size = 1024 * 1024
    with open(dest_path, "wb") as f:
        while chunk := file.file.read(chunk_size):
            written += len(chunk)
            if written > config.MAX_UPLOAD_BYTES:
                f.close()
                dest_path.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"File exceeds the {config.MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
                )
            f.write(chunk)

    try:
        _, profile = load_and_profile_csv(str(dest_path))
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse CSV: {e}") from e
    return dest_path, profile


def _after_new_version(db: Session, dataset: Dataset, version: DatasetVersion) -> None:
    """Fingerprint the file and run data-quality checks against the baseline.
    Neither can fail the upload."""
    try:
        provenance.fingerprint(db, version)
    except Exception:  # noqa: BLE001
        db.rollback()
    data_quality.run_safely(db, dataset, version)


def _dataset_out(db: Session, d: Dataset) -> DatasetProfile:
    latest = latest_version(db, d)
    return DatasetProfile(
        id=d.id,
        filename=d.filename,
        row_count=d.row_count,
        col_count=d.col_count,
        profile=d.profile_json,
        uploaded_at=latest.uploaded_at if latest else d.uploaded_at,
        version=latest.version_number if latest else 1,
        version_count=version_count(db, d.id),
    )


@app.post("/api/datasets/upload", response_model=DatasetProfile)
def upload_dataset(
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dest_path, profile = _save_and_profile_upload(file)

    dataset = Dataset(
        team_id=team.id,
        filename=file.filename,
        filepath=str(dest_path),
        row_count=profile["row_count"],
        col_count=profile["col_count"],
        profile_json=profile,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    version = add_version(db, dataset, filename=file.filename, filepath=str(dest_path), profile=profile)
    _after_new_version(db, dataset, version)
    return _dataset_out(db, dataset)


@app.post("/api/datasets/{dataset_id}/versions", response_model=DatasetProfile)
def replace_dataset_data(
    dataset_id: int,
    file: UploadFile,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Upload a new version into an existing dataset slot ("Replace data")."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset or dataset.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    dest_path, profile = _save_and_profile_upload(file)
    version = add_version(db, dataset, filename=file.filename, filepath=str(dest_path), profile=profile)
    _after_new_version(db, dataset, version)
    return _dataset_out(db, dataset)


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
    return _dataset_out(db, dataset)


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
    return [_dataset_out(db, d) for d in datasets]


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
        kpi_count = len(dash.kpis_json) if dash else 0
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
            "trigger": q.trigger or "manual",
            "route": q.route,
            "rerun_of_question_id": q.rerun_of_question_id,
            "verdict_state": (
                resolve_verdict_state(dash, rejected_reviews(db, q.id)) if dash else None
            ),
        })
    return out


@app.post("/api/questions", response_model=QuestionCreated)
def ask_question(
    payload: QuestionCreate,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key", max_length=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Queue an analysis. Returns at once; the run is a durable job (app/jobs.py).

    Send an `Idempotency-Key` header to make retries safe: the same key from
    the same team returns the first submission instead of starting another."""
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset or dataset.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    text = (payload.question or "").strip()
    if not text:
        raise HTTPException(400, "Question text is required")
    if len(text) > 2000:
        raise HTTPException(400, "Question is longer than 2000 characters")
    try:
        question, job, created = jobs.submit_question(
            db, team_id=team.id, dataset=dataset, text=text, created_by=user.id,
            idempotency_key=idempotency_key or payload.idempotency_key, stage_detail="Waiting for a worker",
        )
    except jobs.AdmissionRejected as e:
        raise HTTPException(429, str(e)) from e
    return QuestionCreated(question_id=question.id, status=question.status, job_id=job.id,
                           job_state=job.state, created=created)


def _plan_steps_for_question(db: Session, question_id: int) -> tuple[list[dict], int | None]:
    plan = (
        db.query(Plan)
        .filter(Plan.question_id == question_id)
        .order_by(Plan.created_at.desc())
        .first()
    )
    if not plan:
        return [], None
    return plan.steps_json, plan.id


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

    job = db.query(AnalysisJob).filter_by(question_id=question_id).first()
    return StatusResponse(
        question_id=question.id,
        question_text=question.text,
        status=question.status,
        current_stage=question.current_stage,
        stage_detail=question.stage_detail or "",
        steps=step_status,
        retry_count=retry_count,
        error=question.error,
        route=question.route,
        clarification=question.clarification_json,
        job=jobs.job_out(job) if job else None,
    )


@app.get("/api/questions/{question_id}/dashboard", response_model=DashboardResponse)
def get_dashboard(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    question = _get_team_question(db, team, question_id)
    dash = (
        db.query(Dashboard)
        .filter(Dashboard.question_id == question_id)
        .order_by(Dashboard.created_at.desc())
        .first()
    )
    if not dash:
        raise HTTPException(404, "Dashboard not ready yet")

    rejections = rejected_reviews(db, question_id)
    flagged = flagged_element_ids(db, question_id)
    narr_flagged, narr_element_id = narrative_audit(db, question_id)
    version = db.get(DatasetVersion, question.dataset_version_id) if question.dataset_version_id else None
    evidence_rows = db.query(EvidenceRecord).filter_by(dashboard_id=dash.id).all()
    evidence_by_element = {e.element_id: e.status for e in evidence_rows if e.element_id}
    evidence_counts: dict[str, int] = {}
    for e in evidence_rows:
        evidence_counts[e.status] = evidence_counts.get(e.status, 0) + 1
    return DashboardResponse(
        id=dash.id,
        question_id=question_id,
        verified=dash.verified,
        verdict_state=resolve_verdict_state(dash, rejections),
        flagged_count=flagged_item_count(rejections),
        rejections=rejection_payload(rejections),
        narrative_flagged=narr_flagged,
        narrative_element_id=narr_element_id,
        dataset_version=version.version_number if version else None,
        verification_summary=dash.verification_summary,
        kpis=[{**k, "flagged": k.get("element_id") in flagged,
               "evidence_status": evidence_by_element.get(k.get("element_id"))} for k in dash.kpis_json],
        charts=[{**c, "flagged": c.get("element_id") in flagged,
                 "evidence_status": evidence_by_element.get(c.get("element_id"))} for c in dash.charts_json],
        evidence_summary={"claims": len(evidence_rows), **evidence_counts},
        narrative_evidence_status=evidence_by_element.get(narr_element_id),
        route=question.route,
        dataset_fingerprint=version.content_sha256 if version else None,
        narrative=dash.narrative,
        view_overrides=_load_json_object(dash.view_overrides_json),
        created_at=dash.created_at,
    )


def _load_json_object(raw: str | None) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


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
        result = e.execution_log.result_json if e.execution_log else None
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
            "issues": r.issues_json,
            "checks": r.checks_json,
            "summary": r.summary,
            "created_at": r.created_at,
        }
        for r in reviews
    ]
