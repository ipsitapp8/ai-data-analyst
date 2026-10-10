"""Scheduled (tracked) analyses: save a question, pause/resume it, delete it."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dashboard, Dataset, ScheduledAnalysis, Team, User
from app.scheduler import INTERVALS, next_run_at
from app.schemas import ScheduledCreate, ScheduledOut, ScheduledUpdate
from app.security import get_current_team, get_current_user
from app.verdict import rejected_reviews, resolve_verdict_state

router = APIRouter(prefix="/api/scheduled-analyses", tags=["scheduled"])


def _validate(interval: str | None, threshold: float | None) -> None:
    if interval is not None and interval not in INTERVALS:
        raise HTTPException(400, f"interval must be one of: {', '.join(INTERVALS)}")
    if threshold is not None and threshold < 0:
        raise HTTPException(400, "change_threshold_pct must be >= 0")


def _out(db: Session, sa: ScheduledAnalysis) -> ScheduledOut:
    dash = db.get(Dashboard, sa.last_dashboard_id) if sa.last_dashboard_id else None
    return ScheduledOut(
        id=sa.id, dataset_id=sa.dataset_id, question_text=sa.question_text, interval=sa.interval,
        change_threshold_pct=sa.change_threshold_pct, is_active=sa.is_active,
        created_at=sa.created_at, last_run_at=sa.last_run_at, next_run_at=next_run_at(sa),
        last_dashboard_id=sa.last_dashboard_id,
        last_question_id=dash.question_id if dash else None,
        last_verdict_state=resolve_verdict_state(dash, rejected_reviews(db, dash.question_id)) if dash else None,
        last_trend=sa.last_trend, last_change_summary=sa.last_change_summary or "",
    )


def _get_owned(db: Session, team: Team, sa_id: int) -> ScheduledAnalysis:
    sa = db.get(ScheduledAnalysis, sa_id)
    if not sa or sa.workspace_id != team.id:
        raise HTTPException(404, "Scheduled analysis not found")
    return sa


@router.post("", response_model=ScheduledOut)
def create_scheduled(
    payload: ScheduledCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _validate(payload.interval, payload.change_threshold_pct)
    text = payload.question.strip()
    if not text:
        raise HTTPException(400, "Question text is required")
    dataset = db.get(Dataset, payload.dataset_id)
    if not dataset or dataset.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    sa = ScheduledAnalysis(
        workspace_id=team.id, dataset_id=dataset.id, question_text=text,
        interval=payload.interval, change_threshold_pct=payload.change_threshold_pct,
        created_by=user.id,
    )
    db.add(sa)
    db.commit()
    db.refresh(sa)
    return _out(db, sa)


@router.get("", response_model=list[ScheduledOut])
def list_scheduled(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    rows = (
        db.query(ScheduledAnalysis)
        .filter(ScheduledAnalysis.workspace_id == team.id)
        .order_by(ScheduledAnalysis.created_at.desc())
        .all()
    )
    return [_out(db, sa) for sa in rows]


@router.patch("/{sa_id}", response_model=ScheduledOut)
def update_scheduled(
    sa_id: int,
    payload: ScheduledUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _validate(payload.interval, payload.change_threshold_pct)
    sa = _get_owned(db, team, sa_id)
    if payload.is_active is not None:
        sa.is_active = payload.is_active
    if payload.interval is not None:
        sa.interval = payload.interval
    if payload.change_threshold_pct is not None:
        sa.change_threshold_pct = payload.change_threshold_pct
    db.commit()
    db.refresh(sa)
    return _out(db, sa)


@router.delete("/{sa_id}")
def delete_scheduled(
    sa_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Stops future runs. Past runs stay in history (their questions keep
    scheduled_analysis_id, which is deliberately not a foreign key)."""
    sa = _get_owned(db, team, sa_id)
    db.delete(sa)
    db.commit()
    return {"deleted": True}
