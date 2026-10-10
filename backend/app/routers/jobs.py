"""Job status, progress, cancellation, retry and the run trace."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app import jobs
from app.database import get_db
from app.models import AnalysisJob, RunEvent, Team, User
from app.routers._common import team_question
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api", tags=["jobs"])


def _team_job(db: Session, team: Team, job_id: int) -> AnalysisJob:
    job = db.get(AnalysisJob, job_id)
    if job is None or job.team_id != team.id:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs")
def list_jobs(
    state: str | None = Query(default=None, pattern="^(queued|running|succeeded|failed|cancelled|timed_out|active)$"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    q = db.query(AnalysisJob).filter(AnalysisJob.team_id == team.id)
    if state == "active":
        q = q.filter(AnalysisJob.state.in_(jobs.ACTIVE))
    elif state:
        q = q.filter(AnalysisJob.state == state)
    total = q.count()
    rows = q.order_by(AnalysisJob.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "jobs": [jobs.job_out(j) for j in rows]}


@router.get("/jobs/metrics")
def job_metrics(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Queue depth and state counts for this team."""
    return jobs.queue_metrics(db, team.id)


@router.get("/jobs/{job_id}")
def get_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return jobs.job_out(_team_job(db, team, job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    job = _team_job(db, team, job_id)
    jobs.request_cancel(db, job)
    db.refresh(job)
    return jobs.job_out(job)


@router.post("/jobs/{job_id}/retry")
def retry_job(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    job = _team_job(db, team, job_id)
    if jobs.active_count(db, team.id) >= jobs.config.JOB_TEAM_MAX_ACTIVE:
        raise HTTPException(429, "This team already has the maximum number of analyses queued or running.")
    if not jobs.retry(db, job):
        raise HTTPException(409, "Only a failed, cancelled or timed-out job with no published dashboard can be retried.")
    db.refresh(job)
    return jobs.job_out(job)


@router.get("/questions/{question_id}/job")
def question_job(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    team_question(db, team, question_id)
    job = db.query(AnalysisJob).filter_by(question_id=question_id).first()
    if job is None:
        raise HTTPException(404, "This analysis has no job record")
    return jobs.job_out(job)


@router.post("/questions/{question_id}/cancel")
def cancel_question(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    team_question(db, team, question_id)
    job = db.query(AnalysisJob).filter_by(question_id=question_id).first()
    if job is None:
        raise HTTPException(404, "This analysis has no job record")
    jobs.request_cancel(db, job)
    db.refresh(job)
    return jobs.job_out(job)


@router.get("/questions/{question_id}/trace")
def question_trace(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Routing decision, node transitions, model calls and code runs of one
    analysis, in order, with a usage total."""
    question = team_question(db, team, question_id)
    events = (db.query(RunEvent).filter_by(question_id=question_id)
              .order_by(RunEvent.id).limit(1000).all())
    usage = {"llm_calls": 0, "sandbox_runs": 0, "tokens_in": 0, "tokens_out": 0, "estimated_cost_usd": 0.0,
             "provider_fallbacks": 0, "sandbox_timeouts": 0}
    for e in events:
        d = e.detail_json or {}
        if e.kind == "llm_call":
            usage["llm_calls"] += 1
            usage["tokens_in"] += int(d.get("tokens_in") or 0)
            usage["tokens_out"] += int(d.get("tokens_out") or 0)
            usage["estimated_cost_usd"] += float(d.get("estimated_cost_usd") or 0)
            usage["provider_fallbacks"] += 1 if d.get("fallback") else 0
        elif e.kind == "sandbox_run":
            usage["sandbox_runs"] += 1
            usage["sandbox_timeouts"] += 1 if d.get("timed_out") else 0
    usage["estimated_cost_usd"] = round(usage["estimated_cost_usd"], 6)
    route = next((e for e in events if e.kind == "route"), None)
    return {
        "question_id": question_id, "route": question.route,
        "route_reasons": (route.detail_json or {}).get("reasons", []) if route else [],
        "budget": (route.detail_json or {}).get("budget", {}) if route else {},
        "usage": usage,
        "events": [{"kind": e.kind, "name": e.name, "detail": e.detail_json or {}, "duration_ms": e.duration_ms,
                    "at": e.created_at} for e in events],
    }
