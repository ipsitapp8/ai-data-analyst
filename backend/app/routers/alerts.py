"""In-app alerts raised by scheduled runs (see app/scheduler.py)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Alert, Team, User
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _out(a: Alert, full: bool = False) -> dict:
    out = {"id": a.id, "kind": a.kind, "title": a.title, "detail": a.detail or "",
           "question_id": a.question_id, "created_at": a.created_at.isoformat(),
           "read": a.read_at is not None,
           "status": a.status or "open", "severity": a.severity or "warn",
           "occurrences": a.occurrences or 1,
           "last_seen_at": (a.last_seen_at or a.created_at).isoformat(),
           "delivery_state": a.delivery_state or "none",
           "scheduled_analysis_id": a.scheduled_analysis_id}
    if full:
        out.update({"explanation": a.explanation_json or {}, "delivery_attempts": a.delivery_attempts or 0,
                    "acknowledged_at": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None})
    return out


@router.get("")
def list_alerts(
    unread_only: bool = Query(default=False),
    status: str | None = Query(default=None, pattern="^(open|acknowledged|resolved)$"),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    q = db.query(Alert).filter_by(team_id=team.id)
    unread = q.filter(Alert.read_at.is_(None)).count()
    if unread_only:
        q = q.filter(Alert.read_at.is_(None))
    if status == "open":
        q = q.filter((Alert.status.is_(None)) | (Alert.status == "open"))
    elif status:
        q = q.filter(Alert.status == status)
    rows = q.order_by(Alert.created_at.desc(), Alert.id.desc()).limit(limit).all()
    return {"unread": unread, "alerts": [_out(a) for a in rows]}


@router.post("/read-all", status_code=204)
def mark_all_read(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    db.query(Alert).filter(Alert.team_id == team.id, Alert.read_at.is_(None)).update(
        {Alert.read_at: dt.datetime.utcnow()}, synchronize_session=False)
    db.commit()


@router.post("/{alert_id}/read", status_code=204)
def mark_read(
    alert_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    a = db.get(Alert, alert_id)
    if not a or a.team_id != team.id:
        raise HTTPException(404, "Alert not found")
    if a.read_at is None:
        a.read_at = dt.datetime.utcnow()
        db.commit()


def _owned(db: Session, team: Team, alert_id: int) -> Alert:
    a = db.get(Alert, alert_id)
    if not a or a.team_id != team.id:
        raise HTTPException(404, "Alert not found")
    return a


@router.get("/{alert_id}")
def get_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """One alert with its explanation: what it was compared with, the changes,
    the KPI's history and the evidence it points at."""
    return _out(_owned(db, team, alert_id), full=True)


@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Someone is looking at it. Still the same incident: a recurrence updates
    this alert instead of raising a new one."""
    a = _owned(db, team, alert_id)
    if (a.status or "open") == "open":
        now = dt.datetime.utcnow()
        a.status, a.acknowledged_by, a.acknowledged_at = "acknowledged", user.id, now
        a.read_at = a.read_at or now
        db.commit()
    return _out(a, full=True)


@router.post("/{alert_id}/resolve")
def resolve_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Close the incident. If the same condition appears again it is a new alert."""
    a = _owned(db, team, alert_id)
    if a.status != "resolved":
        now = dt.datetime.utcnow()
        a.status, a.resolved_by, a.resolved_at = "resolved", user.id, now
        a.read_at = a.read_at or now
        db.commit()
    return _out(a, full=True)
