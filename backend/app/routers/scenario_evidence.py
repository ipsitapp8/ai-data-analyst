"""Sealed records of what-if calculations (see app/scenario_evidence.py).

Team-scoped like every other team resource: another team's record is answered with 404, the same as a missing one. Records are
append-only: there is no update or delete. The server recomputes every result from the request; a client cannot supply one.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import scenario_evidence as ev
from app import scenario_model_store, scenarios
from app.database import get_db
from app.models import Scenario, ScenarioEvidence, Team, User
from app.routers.scenarios import _persist, resolve_model
from app.routers._text import refuse_nul
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/scenario-evidence", tags=["scenario-evidence"], dependencies=[Depends(refuse_nul)])
MAX_REQUEST_BYTES = 100_000


class EvidenceCreate(BaseModel):
    kind: str = Field(max_length=30)
    model: str = Field(max_length=100)
    request: dict[str, Any]
    dataset_id: int | None = None
    dataset_version: int | None = Field(default=None, ge=1)
    note: str = Field(default="", max_length=500)


def _owned(db: Session, team: Team, evidence_id: int) -> ScenarioEvidence:
    row = db.get(ScenarioEvidence, evidence_id)
    if row is None or row.team_id != team.id:
        raise HTTPException(404, "Evidence record not found")
    return row


@router.post("", status_code=201)
def create(payload: EvidenceCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user),
           team: Team = Depends(get_current_team)):
    """Seal a calculation. The result is recomputed from `request` here and stored with its checks and provenance."""
    if len(json.dumps(payload.request, default=str)) > MAX_REQUEST_BYTES:
        raise HTTPException(413, "The request is too large")
    model = resolve_model(db, team, payload.model)
    return _persist(db, team, user, payload.kind, model, payload.request, payload.dataset_id, payload.dataset_version, payload.note)


@router.get("")
def list_records(scenario_id: int | None = Query(default=None), kind: str | None = Query(default=None, max_length=30),
                 limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0), db: Session = Depends(get_db),
                 user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    q = db.query(ScenarioEvidence).filter_by(team_id=team.id)
    if scenario_id is not None:
        saved = db.get(Scenario, scenario_id)
        if saved is None or saved.team_id != team.id:
            raise HTTPException(404, "Scenario not found")
        q = q.filter_by(scenario_id=scenario_id)
    if kind:
        q = q.filter_by(kind=kind)
    total = q.count()
    rows = q.order_by(ScenarioEvidence.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "records": [ev.out(r) for r in rows]}


@router.get("/{evidence_id}")
def get_record(evidence_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
               team: Team = Depends(get_current_team)):
    return ev.out(_owned(db, team, evidence_id), full=True)


@router.post("/{evidence_id}/reproduce")
def reproduce(evidence_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
              team: Team = Depends(get_current_team)):
    """Run the stored request again now and say whether it gives the same sealed result."""
    row = _owned(db, team, evidence_id)
    try:
        model = resolve_model(db, team, row.model_ref)
        return ev.reproduce(row, model)
    except scenario_model_store.ModelNotFound as e:
        raise HTTPException(404, str(e)) from e
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e
