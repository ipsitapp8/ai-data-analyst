"""Shareable read-only dashboard links.

A link points at one finished, non-UNVERIFIED dashboard and carries an unguessable
token (192 bits) that is the only credential. It expires and can be revoked. The
public payload deliberately omits everything internal (team/user ids, code,
audit-trail ids, execution logs): a viewer sees the same KPIs, charts, narrative
and Critic verdict a team member does, and nothing more.
"""
from __future__ import annotations

import datetime as dt
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import config
from app.database import get_db
from app.models import Dashboard, DatasetVersion, Question, ShareLink, Team, User
from app.rate_limit import enforce as rate_limit
from app.security import get_current_team, get_current_user
from app.verdict import (
    UNVERIFIED,
    flagged_element_ids,
    flagged_item_count,
    narrative_audit,
    rejected_reviews,
    rejection_payload,
    resolve_verdict_state,
)

router = APIRouter(tags=["sharing"])

MAX_DAYS = 90
MAX_ACTIVE_PER_QUESTION = 10


class ShareCreate(BaseModel):
    expires_in_days: int = Field(default=7, ge=1, le=MAX_DAYS)


class ShareOut(BaseModel):
    id: int
    url: str
    token: str
    created_at: dt.datetime
    expires_at: dt.datetime
    active: bool


def _active(link: ShareLink, now: dt.datetime) -> bool:
    return link.revoked_at is None and link.expires_at > now


def _out(link: ShareLink) -> ShareOut:
    return ShareOut(
        id=link.id,
        url=f"{config.APP_URL.rstrip('/')}/Shared?t={link.token}",
        token=link.token,
        created_at=link.created_at,
        expires_at=link.expires_at,
        active=_active(link, dt.datetime.utcnow()),
    )


def _team_question(db: Session, team: Team, question_id: int) -> Question:
    q = db.get(Question, question_id)
    if not q or q.team_id != team.id:
        raise HTTPException(404, "Question not found")
    return q


def _latest_dashboard(db: Session, question_id: int) -> Dashboard | None:
    return (
        db.query(Dashboard)
        .filter(Dashboard.question_id == question_id)
        .order_by(Dashboard.created_at.desc())
        .first()
    )


@router.post("/api/questions/{question_id}/share", response_model=ShareOut, status_code=201)
def create_share(
    question_id: int,
    body: ShareCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _team_question(db, team, question_id)
    dash = _latest_dashboard(db, question_id)
    if dash is None:
        raise HTTPException(404, "Dashboard not ready yet")
    if resolve_verdict_state(dash, rejected_reviews(db, question_id)) == UNVERIFIED:
        raise HTTPException(409, "The Critic could not verify this analysis, so it can't be shared publicly")

    now = dt.datetime.utcnow()
    existing = db.query(ShareLink).filter_by(question_id=question_id, team_id=team.id).all()
    if sum(_active(l, now) for l in existing) >= MAX_ACTIVE_PER_QUESTION:
        raise HTTPException(409, "Too many active links for this analysis; revoke some first")

    link = ShareLink(
        token=secrets.token_urlsafe(24),
        team_id=team.id,
        question_id=question_id,
        dashboard_id=dash.id,
        created_by=user.id,
        expires_at=now + dt.timedelta(days=body.expires_in_days),
    )
    db.add(link)
    db.commit()
    db.refresh(link)
    return _out(link)


@router.get("/api/questions/{question_id}/shares", response_model=list[ShareOut])
def list_shares(
    question_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    _team_question(db, team, question_id)
    now = dt.datetime.utcnow()
    links = (
        db.query(ShareLink)
        .filter_by(question_id=question_id, team_id=team.id)
        .order_by(ShareLink.created_at.desc(), ShareLink.id.desc())
        .all()
    )
    return [_out(l) for l in links if _active(l, now)]


@router.delete("/api/shares/{share_id}", status_code=204)
def revoke_share(
    share_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    link = db.get(ShareLink, share_id)
    if not link or link.team_id != team.id:
        raise HTTPException(404, "Share link not found")
    if link.revoked_at is None:
        link.revoked_at = dt.datetime.utcnow()
        db.commit()


@router.get("/api/public/shared/{token}")
def public_dashboard(token: str, request: Request, db: Session = Depends(get_db)):
    """No auth. Unknown, revoked and expired tokens all look identical (404)."""
    rate_limit(f"public_share:{request.client.host}", max_attempts=60, window_seconds=60)
    link = db.query(ShareLink).filter_by(token=token).first()
    if link is None or not _active(link, dt.datetime.utcnow()):
        raise HTTPException(404, "This link is invalid or has expired")

    dash = db.get(Dashboard, link.dashboard_id)
    question = db.get(Question, link.question_id)
    if dash is None or question is None:
        raise HTTPException(404, "This link is invalid or has expired")

    rejections = rejected_reviews(db, question.id)
    verdict = resolve_verdict_state(dash, rejections)
    if verdict == UNVERIFIED:  # re-checked at read time: a dashboard can be re-judged
        raise HTTPException(404, "This link is invalid or has expired")
    flagged = flagged_element_ids(db, question.id)
    narr_flagged, _ = narrative_audit(db, question.id)
    version = db.get(DatasetVersion, question.dataset_version_id) if question.dataset_version_id else None
    return {
        "question_text": question.text,
        "verdict_state": verdict,
        "flagged_count": flagged_item_count(rejections),
        "rejections": [{"summary": r["summary"], "issues": r["issues"]} for r in rejection_payload(rejections)],
        "narrative_flagged": narr_flagged,
        "dataset_version": version.version_number if version else None,
        "kpis": [{"label": k.get("label"), "value": k.get("value"),
                  "flagged": k.get("element_id") in flagged} for k in dash.kpis_json or []],
        "charts": [{"title": c.get("title"), "plotly_json": c.get("plotly_json"),
                    "flagged": c.get("element_id") in flagged} for c in dash.charts_json or []],
        "narrative": dash.narrative,
        "created_at": dash.created_at,
        "expires_at": link.expires_at,
    }
