"""Custom what-if models (see app/scenario_models.py and scenario_model_store.py).

Team-scoped like every other team resource: another team's model is answered with 404, the same as a missing one.
Definitions are immutable per version; editing adds a version.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import scenario_model_store as store
from app import scenario_models
from app.database import get_db
from app.models import Team, User
from app.scenarios import ScenarioError
from app.routers._text import refuse_nul
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/scenario-models", tags=["scenario-models"], dependencies=[Depends(refuse_nul)])
MAX_DEFINITION_BYTES = 60_000


class DefinitionRequest(BaseModel):
    definition: dict[str, Any]


class ModelCreate(DefinitionRequest):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=1000)
    note: str = Field(default="", max_length=500)


class VersionCreate(DefinitionRequest):
    note: str = Field(default="", max_length=500)


def _guard(definition: dict) -> dict:
    if len(json.dumps(definition, default=str)) > MAX_DEFINITION_BYTES:
        raise HTTPException(413, "The model definition is too large")
    return definition


def _fail(e: ScenarioError) -> HTTPException:
    if isinstance(e, store.ModelNotFound):
        return HTTPException(404, str(e))
    return HTTPException(400, str(e))


@router.post("/validate")
def validate(payload: DefinitionRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
             team: Team = Depends(get_current_team)):
    """Every problem with a definition at once, the order its outputs are computed in, their units, and whether the
    metrics it is bound to are approved. Nothing is saved."""
    analysis = scenario_models.analyze(_guard(payload.definition))
    errors = list(analysis["errors"])
    if analysis["valid"]:
        try:
            store.check_sources(db, team.id, analysis["definition"])
        except ScenarioError as e:
            errors.extend(getattr(e, "problems", [str(e)]))
    return {"valid": not errors, "errors": errors, "warnings": analysis["warnings"], "order": analysis["order"],
            "dependencies": analysis["dependencies"], "units": analysis["units"], "definition": analysis["definition"]}


@router.post("", status_code=201)
def create(payload: ModelCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user),
           team: Team = Depends(get_current_team)):
    try:
        model, _ = store.create_model(db, team_id=team.id, user_id=user.id, name=payload.name, description=payload.description,
                                      definition=_guard(payload.definition), note=payload.note)
    except ScenarioError as e:
        raise _fail(e) from e
    return store.model_out(db, model, full=True)


@router.get("")
def list_models(include_archived: bool = Query(default=False), db: Session = Depends(get_db),
                user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    return {"models": [store.model_out(db, m) for m in store.team_models(db, team.id, include_archived)]}


@router.get("/{model_id}")
def get_model(model_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
              team: Team = Depends(get_current_team)):
    try:
        return store.model_out(db, store.owned(db, team.id, model_id), full=True)
    except ScenarioError as e:
        raise _fail(e) from e


@router.get("/{model_id}/versions/{version}")
def get_version(model_id: int, version: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                team: Team = Depends(get_current_team)):
    try:
        return store.version_definition(db, store.owned(db, team.id, model_id), version)
    except ScenarioError as e:
        raise _fail(e) from e


@router.post("/{model_id}/versions", status_code=201)
def add_version(model_id: int, payload: VersionCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                team: Team = Depends(get_current_team)):
    try:
        model = store.owned(db, team.id, model_id)
        store.add_version(db, model, user.id, _guard(payload.definition), payload.note)
    except ScenarioError as e:
        raise _fail(e) from e
    return store.model_out(db, model, full=True)


@router.post("/{model_id}/archive")
def archive(model_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
            team: Team = Depends(get_current_team)):
    try:
        return store.model_out(db, store.set_status(db, store.owned(db, team.id, model_id), "archived"))
    except ScenarioError as e:
        raise _fail(e) from e


@router.post("/{model_id}/restore")
def restore(model_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user),
            team: Team = Depends(get_current_team)):
    try:
        return store.model_out(db, store.set_status(db, store.owned(db, team.id, model_id), "active"))
    except ScenarioError as e:
        raise _fail(e) from e
