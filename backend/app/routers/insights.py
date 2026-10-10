"""Auto-insights endpoints (see app/insights.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import insights
from app.database import get_db
from app.dataset_versions import latest_version
from app.models import Dataset, DatasetInsight, KnowledgeNote, Team, User
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/datasets/{dataset_id}/insights", tags=["insights"])


def _team_dataset(db: Session, team: Team, dataset_id: int) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if not ds or ds.team_id != team.id:
        raise HTTPException(404, "Dataset not found")
    return ds


def _out(row: DatasetInsight | None) -> dict:
    if row is None:
        return {"generated": False, "questions": [], "warnings": [], "created_at": None}
    return {"generated": True, "questions": row.questions_json or [], "warnings": row.warnings_json or [],
            "created_at": row.created_at.isoformat()}


def _latest(db: Session, team: Team, dataset_id: int, version_id: int | None) -> DatasetInsight | None:
    q = db.query(DatasetInsight).filter_by(team_id=team.id, dataset_id=dataset_id)
    q = q.filter(DatasetInsight.dataset_version_id == version_id) if version_id else q
    return q.order_by(DatasetInsight.created_at.desc(), DatasetInsight.id.desc()).first()


@router.get("")
def get_insights(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Cached insights for the dataset's latest version (not generated yet -> generated: false)."""
    ds = _team_dataset(db, team, dataset_id)
    version = latest_version(db, ds)
    return _out(_latest(db, team, dataset_id, version.id if version else None))


@router.post("")
def generate_insights(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """(Re)generate insights for the latest version. Warnings are always returned;
    questions come back empty if the LLM is unavailable."""
    ds = _team_dataset(db, team, dataset_id)
    version = latest_version(db, ds)
    profile = (version.profile_json if version else None) or ds.profile_json or {}
    notes = [n.text for n in db.query(KnowledgeNote).filter_by(
        team_id=team.id, dataset_id=dataset_id, kind="knowledge").limit(10).all()]
    row = DatasetInsight(
        team_id=team.id, dataset_id=dataset_id,
        dataset_version_id=version.id if version else None,
        questions_json=insights.suggest_questions(ds.filename, profile, notes),
        warnings_json=insights.data_warnings(profile),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _out(row)
