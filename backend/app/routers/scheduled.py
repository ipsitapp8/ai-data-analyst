"""Scheduled (tracked) analyses: save a question, pause/resume it, delete it."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Dashboard, Dataset, ScheduledAnalysis, Team, User
from app.routers._common import require_admin
from app.scheduler import COMPARISONS, INTERVALS, next_run_at
from app.schemas import ScheduledCreate, ScheduledOut, ScheduledUpdate
from app.security import get_current_team, get_current_user
from app.verdict import rejected_reviews, resolve_verdict_state

router = APIRouter(prefix="/api/scheduled-analyses", tags=["scheduled"])


def _validate(interval: str | None, threshold: float | None) -> None:
    if interval is not None and interval not in INTERVALS:
        raise HTTPException(400, f"interval must be one of: {', '.join(INTERVALS)}")
    if threshold is not None and threshold < 0:
        raise HTTPException(400, "change_threshold_pct must be >= 0")


def _validate_monitoring(payload) -> None:
    if payload.min_effect_abs is not None and payload.min_effect_abs < 0:
        raise HTTPException(400, "min_effect_abs must be >= 0")
    if payload.comparison is not None and payload.comparison not in COMPARISONS:
        raise HTTPException(400, f"comparison must be one of: {', '.join(COMPARISONS)}")
    if payload.window_runs is not None and not 2 <= payload.window_runs <= 30:
        raise HTTPException(400, "window_runs must be between 2 and 30")


def _apply_monitoring(sa: ScheduledAnalysis, payload) -> None:
    for field in ("min_effect_abs", "comparison", "window_runs", "suppress_on_dq", "notify_email"):
        value = getattr(payload, field)
        if value is not None:
            setattr(sa, field, value)


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
        min_effect_abs=sa.min_effect_abs, comparison=sa.comparison or "previous", window_runs=sa.window_runs,
        suppress_on_dq=sa.suppress_on_dq is not False, notify_email=sa.notify_email is not False,
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
    _validate_monitoring(payload)
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
    _apply_monitoring(sa, payload)
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
    _validate_monitoring(payload)
    sa = _get_owned(db, team, sa_id)
    # Turning off the data-quality gate or email delivery changes what the whole
    # team is (not) told, so it is an owner/admin decision.
    if payload.suppress_on_dq is not None or payload.notify_email is not None:
        require_admin(db, team, user.id, "change alert delivery or the data-quality gate")
    _apply_monitoring(sa, payload)
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
