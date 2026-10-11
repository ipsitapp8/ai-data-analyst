"""Data-quality snapshots, baselines, thresholds and incidents."""
from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app import data_quality
from app.database import get_db
from app.dataset_versions import latest_version
from app.models import DQConfig, DQIncident, Team, User
from app.routers._common import dataset_version, require_admin, team_dataset
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api", tags=["data-quality"])


class QualityConfigIn(BaseModel):
    baseline_version_id: int | None = None
    thresholds: dict[str, Any] = {}


@router.get("/datasets/{dataset_id}/quality")
def dataset_quality(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return data_quality.overview(db, team_dataset(db, team, dataset_id))


@router.post("/datasets/{dataset_id}/quality/run")
def run_quality(
    dataset_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Snapshot and check the latest version now (uploads do this automatically;
    this is for datasets uploaded before data-quality checks existed)."""
    dataset = team_dataset(db, team, dataset_id)
    version = latest_version(db, dataset)
    if version is None:
        raise HTTPException(404, "This dataset has no versions")
    try:
        result = data_quality.run_for_version(db, dataset, version)
    except OSError as e:
        raise HTTPException(410, "The file of this dataset version is no longer available") from e
    return {**result, "version": version.version_number}


@router.put("/datasets/{dataset_id}/quality/config")
def set_quality_config(
    dataset_id: int,
    payload: QualityConfigIn,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    dataset = team_dataset(db, team, dataset_id)
    require_admin(db, team, user.id, "change data-quality thresholds or the baseline")
    try:
        thresholds = data_quality.validate_thresholds(payload.thresholds)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    if payload.baseline_version_id is not None:
        dataset_version(db, dataset, payload.baseline_version_id)
    cfg = data_quality.get_config(db, dataset)
    if cfg is None:
        cfg = DQConfig(team_id=team.id, dataset_id=dataset.id)
        db.add(cfg)
    cfg.baseline_version_id = payload.baseline_version_id
    cfg.thresholds_json = thresholds
    cfg.updated_by, cfg.updated_at = user.id, dt.datetime.utcnow()
    db.commit()
    return data_quality.overview(db, dataset)


def _incident(db: Session, team: Team, incident_id: int) -> DQIncident:
    i = db.get(DQIncident, incident_id)
    if i is None or i.team_id != team.id:
        raise HTTPException(404, "Incident not found")
    return i


@router.post("/quality/incidents/{incident_id}/acknowledge")
def acknowledge_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    i = _incident(db, team, incident_id)
    if i.status == "open":
        i.status = "acknowledged"
        db.commit()
    return data_quality.incident_out(i)


@router.post("/quality/incidents/{incident_id}/resolve")
def resolve_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    i = _incident(db, team, incident_id)
    if i.status != "resolved":
        i.status, i.resolved_by, i.resolved_at = "resolved", user.id, dt.datetime.utcnow()
        db.commit()
    return data_quality.incident_out(i)
