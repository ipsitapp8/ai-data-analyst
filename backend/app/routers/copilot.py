"""Analytical copilot sessions (see app/copilot.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import copilot
from app.database import get_db
from app.models import CopilotSession, CopilotTurn, Team, User
from app.rate_limit import enforce as rate_limit
from app.routers._common import team_dataset
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/copilot", tags=["copilot"])


class SessionCreate(BaseModel):
    dataset_id: int
    title: str = Field(default="", max_length=120)


class MessageIn(BaseModel):
    message: str = Field(min_length=1, max_length=copilot.MAX_MESSAGE_CHARS)


def _session(db: Session, team: Team, user: User, session_id: int) -> CopilotSession:
    try:
        return copilot.get_session(db, team.id, user.id, session_id)
    except copilot.CopilotError as e:
        raise HTTPException(404, str(e)) from e


@router.post("/sessions", status_code=201)
def create_session(
    payload: SessionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dataset = team_dataset(db, team, payload.dataset_id)
    s = copilot.create_session(db, team_id=team.id, user_id=user.id, dataset=dataset, title=payload.title.strip())
    return copilot.session_out(db, s, with_turns=True)


@router.get("/sessions")
def list_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    rows = (db.query(CopilotSession).filter_by(team_id=team.id, user_id=user.id)
            .order_by(CopilotSession.updated_at.desc(), CopilotSession.id.desc()).limit(limit).all())
    return [copilot.session_out(db, s) for s in rows]


@router.get("/sessions/{session_id}")
def get_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return copilot.session_out(db, _session(db, team, user, session_id), with_turns=True)


@router.post("/sessions/{session_id}/messages")
def post_message(
    session_id: int,
    payload: MessageIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    rate_limit(f"copilot:{user.id}", max_attempts=40, window_seconds=60)
    s = _session(db, team, user, session_id)
    try:
        return copilot.handle_message(db, s, payload.message)
    except copilot.CopilotError as e:
        db.rollback()
        raise HTTPException(400, str(e)) from e


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    s = _session(db, team, user, session_id)
    db.query(CopilotTurn).filter_by(session_id=s.id).delete(synchronize_session=False)
    db.delete(s)
    db.commit()
