"""What-if scenarios (see app/scenarios.py)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import metric_engine as me
from app import scenarios
from app.database import get_db
from app.dataset_versions import latest_version
from app.models import Scenario, Team, User
from app.routers._common import team_dataset
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


class BaselineRequest(BaseModel):
    dataset_id: int
    model: str
    mapping: dict[str, str | None] = {}
    filters: list[dict] = Field(default_factory=list, max_length=20)


class EvaluateRequest(BaseModel):
    model: str
    baseline: dict[str, float]
    scenarios: dict[str, dict[str, dict[str, Any]]] = {}
    assumptions: dict[str, float] = {}
    sensitivity_pct: float = 10.0
    spread_pct: float | None = None  # when set, adds standard optimistic / pessimistic scenarios


class ScenarioSave(EvaluateRequest):
    name: str = Field(min_length=1, max_length=120)
    dataset_id: int | None = None


def _evaluate(req: EvaluateRequest) -> dict:
    try:
        named = dict(req.scenarios)
        if req.spread_pct is not None:
            named = {**scenarios.standard_scenarios(req.model, req.spread_pct), **named}
        return scenarios.evaluate(req.model, req.baseline, named, req.assumptions, req.sensitivity_pct)
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


def _out(s: Scenario, full: bool = True) -> dict:
    out = {"id": s.id, "name": s.name, "model": s.model, "dataset_id": s.dataset_id,
           "dataset_version_id": s.dataset_version_id, "created_at": s.created_at, "kind": "scenario"}
    if full:
        out.update({"baseline": s.baseline_json or {}, "request": s.adjustments_json or {},
                    "results": s.results_json or {}})
    else:
        r = s.results_json or {}
        out["primary_output"] = r.get("primary_output")
        out["baseline_primary"] = ((r.get("baseline") or {}).get("outputs") or {}).get(r.get("primary_output"))
    return out


@router.get("/models")
def list_models(user: User = Depends(get_current_user)):
    return {"models": scenarios.describe_models(), "disclaimer": scenarios.DISCLAIMER}


@router.post("/baseline")
def derive_baseline(
    payload: BaselineRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Baseline inputs computed from dataset columns, each with its formula."""
    dataset = team_dataset(db, team, payload.dataset_id)
    version = latest_version(db, dataset)
    try:
        df = me.load_frame(version.filepath if version else dataset.filepath)
        out = scenarios.baseline_from_dataset(df, dataset.profile_json or {}, payload.model, payload.mapping,
                                              payload.filters)
    except (scenarios.ScenarioError, me.FormulaError) as e:
        raise HTTPException(400, str(e)) from e
    except OSError as e:
        raise HTTPException(410, "The dataset file is no longer available") from e
    return {**out, "dataset_id": dataset.id, "dataset_version": version.version_number if version else None}


@router.post("/evaluate")
def evaluate(payload: EvaluateRequest, user: User = Depends(get_current_user),
             team: Team = Depends(get_current_team)):
    return _evaluate(payload)


@router.post("", status_code=201)
def save_scenario(
    payload: ScenarioSave,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    results = _evaluate(payload)
    version_id = None
    if payload.dataset_id is not None:
        dataset = team_dataset(db, team, payload.dataset_id)
        version = latest_version(db, dataset)
        version_id = version.id if version else None
    row = Scenario(team_id=team.id, dataset_id=payload.dataset_id, dataset_version_id=version_id,
                   name=payload.name.strip(), model=payload.model, baseline_json=results["baseline"]["inputs"],
                   adjustments_json={"scenarios": payload.scenarios, "assumptions": results["assumptions"],
                                     "sensitivity_pct": payload.sensitivity_pct, "spread_pct": payload.spread_pct},
                   results_json=results, created_by=user.id)
    db.add(row)
    db.commit()
    db.refresh(row)
    return _out(row)


@router.get("")
def list_scenarios(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    q = db.query(Scenario).filter_by(team_id=team.id)
    total = q.count()
    rows = q.order_by(Scenario.id.desc()).offset(offset).limit(limit).all()
    return {"total": total, "limit": limit, "offset": offset, "scenarios": [_out(s, full=False) for s in rows]}


def _owned(db: Session, team: Team, scenario_id: int) -> Scenario:
    s = db.get(Scenario, scenario_id)
    if s is None or s.team_id != team.id:
        raise HTTPException(404, "Scenario not found")
    return s


@router.get("/compare")
def compare_scenarios(
    ids: str = Query(..., description="Comma-separated saved scenario ids (2 to 6)", max_length=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    try:
        wanted = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError as e:
        raise HTTPException(400, "ids must be integers") from e
    if not 2 <= len(wanted) <= 6:
        raise HTTPException(400, "Compare between 2 and 6 scenarios")
    rows = [_owned(db, team, i) for i in wanted]
    if len({r.model for r in rows}) > 1:
        raise HTTPException(400, "Only scenarios of the same model can be compared")
    table = []
    for r in rows:
        res = r.results_json or {}
        table.append({"id": r.id, "name": r.name, "baseline_inputs": (res.get("baseline") or {}).get("inputs"),
                      "baseline_outputs": (res.get("baseline") or {}).get("outputs"),
                      "scenarios": {k: v.get("outputs") for k, v in (res.get("scenarios") or {}).items()},
                      "assumptions": res.get("assumptions")})
    return {"kind": "scenario", "disclaimer": scenarios.DISCLAIMER, "model": rows[0].model, "items": table}


@router.get("/{scenario_id}")
def get_scenario(
    scenario_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    return _out(_owned(db, team, scenario_id))


@router.delete("/{scenario_id}", status_code=204)
def delete_scenario(
    scenario_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    db.delete(_owned(db, team, scenario_id))
    db.commit()
