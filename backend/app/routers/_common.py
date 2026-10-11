"""Shared lookups for routers: fetch a team-owned row or answer 404.

"Not found" and "belongs to another team" are deliberately the same response,
so an id from another tenant reveals nothing -- not even that it exists.
"""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models import Dataset, DatasetVersion, Question, Team, TeamMember


def team_dataset(db: Session, team: Team, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if ds is None or ds.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    return ds


def team_question(db: Session, team: Team, question_id: int) -> Question:
    q = db.get(Question, question_id)
    if q is None or q.team_id != team.id:
        raise HTTPException(404, "Question not found")
    return q


def dataset_version(db: Session, dataset: Dataset, version_id: int) -> DatasetVersion:
    v = db.get(DatasetVersion, version_id)
    if v is None or v.dataset_id != dataset.id:
        raise HTTPException(404, "Dataset version not found")
    return v


def role_of(db: Session, team_id: int, user_id: int) -> str | None:
    m = (db.query(TeamMember)
         .filter(TeamMember.team_id == team_id, TeamMember.user_id == user_id, TeamMember.status == "active")
         .first())
    return m.role if m else None


def require_admin(db: Session, team: Team, user_id: int, what: str) -> None:
    if role_of(db, team.id, user_id) not in ("owner", "admin"):
        raise HTTPException(403, f"Only a team owner or admin can {what}")
