"""Universal what-if API (app/whatif/). A thin layer: team scoping, dataset loading and persistence around app.whatif.service,
the same service the Copilot adapter calls. Saved universal scenarios live in the existing `scenarios` table under a
`universal:<domain>:<template>` model key, so no migration is needed and business scenarios are untouched."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.analyst.data import load_table
from app.database import get_db
from app.dataset_versions import latest_version
from app.models import Scenario, Team, User
from app.routers._common import team_dataset
from app.routers._text import refuse_nul
from app.security import get_current_team, get_current_user
from app.whatif import service
from app.whatif.domains import get_domain

router = APIRouter(prefix="/api/whatif", tags=["whatif"], dependencies=[Depends(refuse_nul)])
PREFIX = "universal:"
MAX_SCENARIO_BYTES = 200_000


class ProfileRequest(BaseModel):
    dataset_id: int
    domain: str | None = Field(default=None, max_length=40)


class SuggestRequest(BaseModel):
    dataset_id: int | None = None
    domain: str | None = Field(default=None, max_length=40)
    objective: str | None = Field(default=None, max_length=1000)
    use_model: bool = False


class ScenarioRequest(BaseModel):
    scenario: dict[str, Any]
    dataset_id: int | None = None


class SaveRequest(ScenarioRequest):
    name: str | None = Field(default=None, max_length=120)


class QuestionRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    dataset_id: int | None = None
    domain: str | None = Field(default=None, max_length=40)
    inputs: dict[str, float] = Field(default_factory=dict, max_length=20)


class CompareRequest(BaseModel):
    ids: list[int] = Field(default_factory=list, max_length=8)
    scenarios: list[dict[str, Any]] = Field(default_factory=list, max_length=8)


def _frame(db: Session, team: Team, dataset_id: int | None):
    """(dataset info, DataFrame) for a dataset of this team, or (None, None). Another team's dataset is a 404."""
    if dataset_id is None:
        return None, None
    ds = team_dataset(db, team, dataset_id)
    version = latest_version(db, ds)
    try:
        df = load_table(version.filepath if version is not None else ds.filepath)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"The dataset could not be read ({type(e).__name__}).") from e
    return {"dataset_id": ds.id, "filename": version.filename if version else ds.filename, "version": version.version_number if version else None}, df


def _check(scenario: dict) -> dict:
    if len(json.dumps(scenario, default=str)) > MAX_SCENARIO_BYTES:
        raise HTTPException(413, "The scenario is too large")
    for k in ("domain", "template_id"):
        if not isinstance(scenario.get(k), str):
            raise HTTPException(422, f"The scenario needs a '{k}'")
    try:
        get_domain(scenario["domain"])
    except KeyError as e:
        raise HTTPException(422, str(e).strip("'\"")) from e
    # the result is always recomputed on the server; whatever the client sent is not trusted
    return {**scenario, "results": None, "explanation": None}


def _bind(scenario: dict, info: dict | None) -> dict:
    return {**scenario, "dataset": info}


def _owned(db: Session, team: Team, scenario_id: int) -> Scenario:
    row = db.get(Scenario, scenario_id)
    if row is None or row.team_id != team.id or not (row.model or "").startswith(PREFIX):
        raise HTTPException(404, "Scenario not found")
    return row


def _out(row: Scenario, full: bool = True) -> dict:
    sc = (row.results_json or {}).get("scenario") or {}
    base = {"id": row.id, "name": row.name, "domain": sc.get("domain"), "template_id": sc.get("template_id"), "created_at": row.created_at,
            "dataset_id": row.dataset_id, "status": (sc.get("validation") or {}).get("status"), "question": sc.get("question"),
            "evidence_kind": (sc.get("results") or {}).get("evidence_kind")}
    return {**base, "scenario": sc} if full else base


@router.get("/domains")
def domains(user: User = Depends(get_current_user)):
    return {"domains": service.domains()}


@router.post("/profile")
def profile(payload: ProfileRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    info, df = _frame(db, team, payload.dataset_id)
    try:
        return service.profile_table(df, info["filename"], payload.domain)
    except KeyError as e:
        raise HTTPException(422, str(e).strip("'\"")) from e


@router.post("/suggest")
def suggest(payload: SuggestRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    info, df = _frame(db, team, payload.dataset_id)
    ranker = None
    if payload.use_model and payload.objective:
        from app.whatif import llm
        ranker = llm.ranker
    try:
        return service.suggest(df, payload.domain, filename=info["filename"] if info else None, objective=payload.objective, ranker=ranker)
    except KeyError as e:
        raise HTTPException(422, str(e).strip("'\"")) from e


@router.post("/from-question")
def from_question(payload: QuestionRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    """A structured scenario from a sentence (the same reading the Copilot uses). Nothing is calculated here."""
    info, df = _frame(db, team, payload.dataset_id)
    out = service.from_question(payload.message, df, filename=info["filename"] if info else None, domain=payload.domain, inputs=payload.inputs)
    if out["scenario"]:
        out["scenario"] = _bind(out["scenario"], info)
    return out


@router.post("/validate")
def validate(payload: ScenarioRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    info, df = _frame(db, team, payload.dataset_id)
    s = _bind(_check(payload.scenario), info)
    return service.validate(s, df)


@router.post("/run")
def run(payload: ScenarioRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    """Validate and calculate. A scenario that is not valid comes back with its reasons and no results (HTTP 200: it is an answer, not a fault)."""
    info, df = _frame(db, team, payload.dataset_id)
    return service.run(_bind(_check(payload.scenario), info), df)


@router.post("/scenarios", status_code=201)
def save(payload: SaveRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    info, df = _frame(db, team, payload.dataset_id)
    s = service.run(_bind(_check(payload.scenario), info), df)
    if s["validation"]["status"] != "valid":
        raise HTTPException(422, {"message": "Only a valid scenario can be saved", "validation": s["validation"]})
    s["meta"]["created_by"] = user.id
    row = Scenario(team_id=team.id, created_by=user.id, dataset_id=payload.dataset_id, name=(payload.name or s["name"]).strip() or s["template_id"],
                   model=f"{PREFIX}{s['domain']}:{s['template_id']}", baseline_json=s["baseline"], adjustments_json={"request": {**s, "results": None, "explanation": None}},
                   results_json={"universal": True, "scenario": s})
    s["name"] = row.name
    db.add(row)
    db.commit()
    db.refresh(row)
    return _out(row)


@router.get("/scenarios")
def list_saved(limit: int = Query(default=50, ge=1, le=200), offset: int = Query(default=0, ge=0), db: Session = Depends(get_db),
               user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    q = db.query(Scenario).filter(Scenario.team_id == team.id, Scenario.model.like(f"{PREFIX}%"))
    return {"total": q.count(), "scenarios": [_out(r, full=False) for r in q.order_by(Scenario.id.desc()).offset(offset).limit(limit).all()]}


@router.get("/scenarios/{scenario_id}")
def get_saved(scenario_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    return _out(_owned(db, team, scenario_id))


@router.put("/scenarios/{scenario_id}")
def update_saved(scenario_id: int, payload: SaveRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                 team: Team = Depends(get_current_team)):
    row = _owned(db, team, scenario_id)
    info, df = _frame(db, team, payload.dataset_id if payload.dataset_id is not None else row.dataset_id)
    s = service.run(_bind(_check(payload.scenario), info), df)
    if s["validation"]["status"] != "valid":
        raise HTTPException(422, {"message": "Only a valid scenario can be saved", "validation": s["validation"]})
    row.name = (payload.name or row.name).strip() or row.name
    s["name"] = row.name
    row.dataset_id = payload.dataset_id if payload.dataset_id is not None else row.dataset_id
    row.model, row.baseline_json = f"{PREFIX}{s['domain']}:{s['template_id']}", s["baseline"]
    row.adjustments_json = {"request": {**s, "results": None, "explanation": None}}
    row.results_json = {"universal": True, "scenario": s}
    db.commit()
    db.refresh(row)
    return _out(row)


@router.post("/scenarios/{scenario_id}/rerun")
def rerun_saved(scenario_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    """Recalculate a saved scenario against the dataset's latest version (the stored result is left as it was)."""
    row = _owned(db, team, scenario_id)
    info, df = _frame(db, team, row.dataset_id)
    return service.run(_bind(_check((row.adjustments_json or {}).get("request") or {}), info), df)


@router.delete("/scenarios/{scenario_id}", status_code=204)
def delete_saved(scenario_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    db.delete(_owned(db, team, scenario_id))
    db.commit()


@router.post("/compare")
def compare(payload: CompareRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    items = [(_owned(db, team, i).results_json or {}).get("scenario") for i in payload.ids]
    items += [s for s in payload.scenarios]
    try:
        return service.compare([i for i in items if i])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
