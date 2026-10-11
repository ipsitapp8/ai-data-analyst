"""In-app alerts raised by scheduled runs (see app/scheduler.py)."""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Alert, Team, User
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


def _out(a: Alert) -> dict:
    return {"id": a.id, "kind": a.kind, "title": a.title, "detail": a.detail or "",
            "question_id": a.question_id, "created_at": a.created_at.isoformat(),
            "read": a.read_at is not None}


@router.get("")
def list_alerts(
    unread_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    q = db.query(Alert).filter_by(team_id=team.id)
    unread = q.filter(Alert.read_at.is_(None)).count()
    if unread_only:
        q = q.filter(Alert.read_at.is_(None))
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
