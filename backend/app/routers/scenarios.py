"""What-if scenarios (see app/scenarios.py)."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app import metric_engine as me
from app import (provenance, scenario_evidence, scenario_nl, scenario_service, scenario_elasticity, scenario_model_store, scenario_models, scenario_segments, scenario_sensitivity,
                 scenario_sim, scenario_solve, scenario_time, scenario_verify, scenarios)
from app.database import get_db
from app.dataset_versions import latest_version
from app.models import DatasetVersion, Scenario, Team, User
from app.routers._common import team_dataset
from app import rate_limit
from app.security import get_current_team, get_current_user

router = APIRouter(prefix="/api/scenarios", tags=["scenarios"])


class BaselineRequest(BaseModel):
    dataset_id: int
    model: str
    mapping: dict[str, str | None] = {}
    filters: list[dict] = Field(default_factory=list, max_length=20)
    overrides: dict[str, float] = Field(default_factory=dict, max_length=20)   # values you enter yourself for inputs the data lacks
    dataset_version: int | None = Field(default=None, ge=1)
    date_column: str | None = Field(default=None, max_length=200)
    period: dict[str, Any] | None = None


class EvaluateRequest(BaseModel):
    model: str
    baseline: dict[str, float]
    scenarios: dict[str, dict[str, dict[str, Any]]] = {}
    assumptions: dict[str, float] = {}
    # Where an assumption came from when it was not simply typed (for example a chosen elasticity estimate). Kept with
    # the scenario so its provenance survives; it never changes a calculation.
    assumption_sources: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=5)
    sensitivity_pct: float = 10.0
    spread_pct: float | None = None  # when set, adds standard optimistic / pessimistic scenarios


class SimulationOptions(BaseModel):
    """Monte Carlo settings (app/scenario_sim.py). `uncertainty` maps an input to a distribution; inputs left out are
    held fixed. A missing seed is drawn by the server and returned, so every run can be repeated."""
    adjustments: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=20)
    uncertainty: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=20)
    correlation: dict[str, Any] | None = None
    iterations: int = Field(default=scenario_sim.DEFAULT_ITERATIONS, ge=scenario_sim.MIN_ITERATIONS,
                            le=scenario_sim.MAX_ITERATIONS)
    seed: int | None = Field(default=None, ge=0, lt=2 ** 32)
    target_output: str | None = Field(default=None, max_length=60)
    target_value: float | None = None
    target_direction: str = Field(default="at_least", max_length=10)


class SimulateRequest(SimulationOptions):
    model: str
    baseline: dict[str, float]
    assumptions: dict[str, float] = {}


class ConstraintIn(BaseModel):
    metric: str = Field(max_length=60)
    op: str = Field(max_length=2)
    value: float


class GoalSeekRequest(BaseModel):
    model: str
    baseline: dict[str, float]
    assumptions: dict[str, float] = {}
    adjustments: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=20)
    target_output: str = Field(max_length=60)
    target_value: float
    solve_for: str = Field(max_length=60)
    lower: float | None = None
    upper: float | None = None
    constraints: list[ConstraintIn] = Field(default_factory=list, max_length=scenario_solve.MAX_CONSTRAINTS)


class ScenarioSave(EvaluateRequest):
    name: str = Field(min_length=1, max_length=120)
    dataset_id: int | None = None
    simulation: SimulationOptions | None = None
    # Where the baseline came from (dataset version, period, filters, mapping), kept with the scenario.
    baseline_source: dict[str, Any] | None = None


def resolve_model(db: Session, team: Team, ref: str):
    """A built-in model name, or the compiled spec of this team's custom model (pinned to a version)."""
    if not scenario_model_store.is_custom(ref):
        return ref
    try:
        return scenario_model_store.load_spec(db, team.id, ref)
    except scenario_model_store.ModelNotFound as e:
        raise HTTPException(404, str(e)) from e
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


def _evaluate(req: EvaluateRequest, model) -> dict:
    try:
        return scenario_service.evaluate(model, req.baseline, req.scenarios, req.assumptions, sensitivity_pct=req.sensitivity_pct,
                                         spread_pct=req.spread_pct, assumption_sources=req.assumption_sources)
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


def _versioned_frame(db: Session, team: Team, dataset_id: int, version_number: int | None):
    """The dataset's rows at a chosen version (latest by default), with the provenance to record beside any result."""
    dataset = team_dataset(db, team, dataset_id)
    if version_number is None:
        version = latest_version(db, dataset)
    else:
        version = (db.query(DatasetVersion).filter(DatasetVersion.dataset_id == dataset.id,
                                                   DatasetVersion.version_number == version_number).first())
        if version is None:
            raise HTTPException(404, "Dataset version not found")
    try:
        df = me.load_frame(version.filepath if version else dataset.filepath)
    except me.FormulaError as e:
        raise HTTPException(400, str(e)) from e
    except OSError as e:
        raise HTTPException(410, "The dataset file is no longer available") from e
    latest = latest_version(db, dataset)
    return dataset, version, df, {
        "dataset_id": dataset.id, "dataset_filename": version.filename if version else dataset.filename,
        "dataset_version": version.version_number if version else None, "dataset_version_id": version.id if version else None,
        "latest_version": latest.version_number if latest else None,
        "content_sha256": provenance.fingerprint(db, version) if version else None,
        "rows_in_dataset": int(len(df))}


def _simulate(model, baseline: dict, assumptions: dict, opts: SimulationOptions) -> dict:
    try:
        return scenario_service.simulate(model, baseline, assumptions, adjustments=opts.adjustments, uncertainty=opts.uncertainty,
                                         correlation=opts.correlation, iterations=opts.iterations, seed=opts.seed,
                                         target_output=opts.target_output, target_value=opts.target_value,
                                         target_direction=opts.target_direction)
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


def _baseline_for(db: Session, team: Team, payload, model, dataset, df, version) -> dict:
    """A baseline for one period of the data (all of it by default), with its provenance and an independent check."""
    try:
        period = scenario_time.parse_period(payload.period)
        if period["start"] is not None:
            if not payload.date_column:
                raise scenario_time.TimeError("Choose the date column that the period applies to")
            scenario_time.date_column_of(df, payload.date_column)
        filters = [*scenario_time.period_filters(payload.date_column, period), *payload.filters]
        if isinstance(model, dict):
            out = scenario_model_store.derive_baseline(db, team.id, model, dataset.id, dataset.profile_json or {}, df, filters,
                                                       payload.overrides)
        else:
            out = scenarios.baseline_from_dataset(df, dataset.profile_json or {}, model, payload.mapping, filters)
            for name in list(out["missing"]):
                if name in payload.overrides:
                    out["inputs"][name] = float(payload.overrides[name])
                    out["sources"][name] = {"source": "override", "formula": None, "note": "entered by you"}
                    out["missing"].remove(name)
        scenario_time.require_rows(period, out["rows_used"], df, payload.date_column)
    except scenario_models.AmbiguousMetric as e:
        raise HTTPException(409, {"message": str(e), "candidates": e.candidates}) from e
    except (scenarios.ScenarioError, me.FormulaError) as e:
        raise HTTPException(400, str(e)) from e
    out["verification"] = scenario_verify.safe(scenario_verify.verify_baseline, version.filepath if version else dataset.filepath,
                                               out, filters)
    return {**out, "data_kind": "historical_actual", "data_kinds": scenario_time.DATA_KINDS,
            "period": scenario_time.describe_period(period, payload.date_column, out["rows_used"], df)}


@router.post("/baseline")
def derive_baseline(
    payload: BaselineRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    """Baseline inputs computed from dataset columns (built-in models) or from a custom model's bindings, for all of
    the data or one period of it, each with its formula, where it came from, the dataset version and an independent
    recomputation. This is recorded data (historical actuals), not a scenario or a forecast."""
    dataset, version, df, prov = _versioned_frame(db, team, payload.dataset_id, payload.dataset_version)
    model = resolve_model(db, team, payload.model)
    out = _baseline_for(db, team, payload, model, dataset, df, version)
    return {**out, "dataset_id": dataset.id, "dataset_version": prov["dataset_version"], "provenance": prov}


@router.get("/periods")
def periods(dataset_id: int = Query(...), date_column: str = Query(..., max_length=200),
            dataset_version: int | None = Query(default=None, ge=1), db: Session = Depends(get_db),
            user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    """The periods of a dataset that have rows, from the dataset's own calendar."""
    _, _, df, prov = _versioned_frame(db, team, dataset_id, dataset_version)
    try:
        return {**scenario_time.available_periods(df, date_column), "provenance": prov}
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


class SegmentsRequest(BaseModel):
    dataset_id: int
    dataset_version: int | None = Field(default=None, ge=1)
    model: str
    mapping: dict[str, str | None] = {}
    segment_column: str = Field(max_length=200)
    date_column: str | None = Field(default=None, max_length=200)
    period: dict[str, Any] | None = None
    filters: list[dict] = Field(default_factory=list, max_length=20)
    assumptions: dict[str, float] = {}
    shared_adjustments: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=20)
    segment_adjustments: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict, max_length=scenario_segments.MAX_SEGMENTS)
    shared_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=20)     # totals to allocate across segments
    constant_inputs: dict[str, float] = Field(default_factory=dict, max_length=20)             # the same value in every segment


@router.get("/segment-values")
def segment_values(dataset_id: int = Query(...), segment_column: str = Query(..., max_length=200),
                   dataset_version: int | None = Query(default=None, ge=1), db: Session = Depends(get_db),
                   user: User = Depends(get_current_user), team: Team = Depends(get_current_team)):
    """The segments of a column (value and row count) so they can be chosen."""
    _, _, df, prov = _versioned_frame(db, team, dataset_id, dataset_version)
    try:
        return {**scenario_segments.segment_counts(df, segment_column), "provenance": prov}
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/segments")
def segments(payload: SegmentsRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
             team: Team = Depends(get_current_team)):
    """Change inputs for chosen segments (a region, product, campaign) and see each segment and the total. Segments
    partition the rows, so nothing is counted twice; the total is the sum of the segments."""
    dataset, version, df, prov = _versioned_frame(db, team, payload.dataset_id, payload.dataset_version)
    model = resolve_model(db, team, payload.model)
    try:
        period = scenario_time.parse_period(payload.period)
        if period["start"] is not None:
            if not payload.date_column:
                raise scenario_time.TimeError("Choose the date column that the period applies to")
            scenario_time.date_column_of(df, payload.date_column)
        filters = [*scenario_time.period_filters(payload.date_column, period), *payload.filters]
        scope = me.apply_filters(df, filters)
        scenario_time.require_rows(period, int(len(scope)), df, payload.date_column)
        profile = dataset.profile_json or {}
        if isinstance(model, dict):
            fn = lambda frame: scenario_model_store.derive_baseline(db, team.id, model, dataset.id, profile, frame)  # noqa: E731
        else:
            fn = lambda frame: scenarios.baseline_from_dataset(frame, profile, model, payload.mapping)  # noqa: E731
        out = scenario_segments.run(model, scope, fn, segment_column=payload.segment_column, assumptions=payload.assumptions,
                                    shared_adjustments=payload.shared_adjustments, segment_adjustments=payload.segment_adjustments,
                                    shared_inputs=payload.shared_inputs, constant_inputs=payload.constant_inputs,
                                    reference={"csv_path": version.filepath if version else dataset.filepath, "filters": filters})
    except scenario_models.AmbiguousMetric as e:
        raise HTTPException(409, {"message": str(e), "candidates": e.candidates}) from e
    except (scenarios.ScenarioError, me.FormulaError) as e:
        raise HTTPException(400, str(e)) from e
    return {**scenario_service.explained(out), "period": scenario_time.describe_period(period, payload.date_column, int(len(scope)), df), "provenance": prov,
            "data_kinds": scenario_time.DATA_KINDS}


class ComparePeriodsRequest(BaseModel):
    dataset_id: int
    dataset_version: int | None = Field(default=None, ge=1)
    model: str
    mapping: dict[str, str | None] = {}
    date_column: str = Field(max_length=200)
    period_a: dict[str, Any]
    period_b: dict[str, Any]
    filters: list[dict] = Field(default_factory=list, max_length=20)
    assumptions: dict[str, float] = {}
    overrides: dict[str, float] = Field(default_factory=dict, max_length=20)


@router.post("/compare-periods")
def compare_periods(payload: ComparePeriodsRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                    team: Team = Depends(get_current_team)):
    """What a model's inputs and outputs were in two recorded periods, the change, and what drove it. Historical
    actuals only: it neither adjusts anything nor looks forward."""
    dataset, version, df, prov = _versioned_frame(db, team, payload.dataset_id, payload.dataset_version)
    model = resolve_model(db, team, payload.model)
    sides = {}
    for key, spec in (("a", payload.period_a), ("b", payload.period_b)):
        req = BaselineRequest(dataset_id=payload.dataset_id, model=payload.model, mapping=payload.mapping, filters=payload.filters,
                              overrides=payload.overrides, dataset_version=payload.dataset_version, date_column=payload.date_column,
                              period=spec)
        sides[key] = _baseline_for(db, team, req, model, dataset, df, version)
    try:
        out = scenario_time.compare_periods(model, sides["a"]["inputs"], sides["b"]["inputs"], payload.assumptions)
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e
    out["verification"] = scenario_verify.safe(scenario_verify.verify_period_comparison, out, scenarios._model(model))
    out["periods"] = {k: {"period": v["period"], "sources": v["sources"], "rows_used": v["rows_used"],
                          "baseline_verification": v["verification"]} for k, v in sides.items()}
    return {**scenario_service.explained(out), "provenance": prov}


@router.post("/evaluate")
def evaluate(payload: EvaluateRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
             team: Team = Depends(get_current_team)):
    return _evaluate(payload, resolve_model(db, team, payload.model))


class ElasticityRequest(BaseModel):
    dataset_id: int
    dataset_version: int | None = Field(default=None, ge=1)
    price_column: str = Field(max_length=200)
    quantity_column: str = Field(max_length=200)
    date_column: str | None = Field(default=None, max_length=200)
    segment_column: str | None = Field(default=None, max_length=200)
    granularity: str = Field(default="auto", max_length=10)
    filters: list[dict] = Field(default_factory=list, max_length=20)
    level: float = Field(default=0.95, ge=0.5, lt=1)
    exclude_outliers: bool = False


@router.post("/elasticity")
def estimate_elasticity(payload: ElasticityRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                        team: Team = Depends(get_current_team)):
    """A suggested price elasticity from the dataset's own history. A suggestion only: it changes nothing until the
    user chooses to use it."""
    _, _, df, prov = _versioned_frame(db, team, payload.dataset_id, payload.dataset_version)
    try:
        out = scenario_elasticity.estimate(df, payload.price_column, payload.quantity_column,
                                           date_column=payload.date_column, segment_column=payload.segment_column,
                                           granularity=payload.granularity, filters=payload.filters,
                                           exclude_outliers=payload.exclude_outliers, level=payload.level)
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e
    return {**scenario_service.explained(out), "provenance": prov}


class GridRequest(BaseModel):
    model: str
    baseline: dict[str, float]
    assumptions: dict[str, float] = {}
    adjustments: dict[str, dict[str, Any]] = Field(default_factory=dict, max_length=20)
    x_input: str = Field(max_length=60)
    y_input: str = Field(max_length=60)
    output: str | None = Field(default=None, max_length=60)
    x_range: dict[str, Any] | None = None
    y_range: dict[str, Any] | None = None
    resolution: int = Field(default=scenario_sensitivity.DEFAULT_RESOLUTION, ge=scenario_sensitivity.MIN_RESOLUTION,
                            le=scenario_sensitivity.MAX_RESOLUTION)
    threshold: float | None = 0.0


@router.post("/sensitivity-grid")
def sensitivity_grid(payload: GridRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                     team: Team = Depends(get_current_team)):
    """One output over a grid of two inputs, with the best and worst cell, the current position and where the output
    crosses a threshold (break-even for profit at 0), independently verified."""
    model = resolve_model(db, team, payload.model)
    try:
        return scenario_service.grid(model, payload.baseline, payload.x_input, payload.y_input, payload.output,
                                     adjustments=payload.adjustments, assumptions=payload.assumptions, x_range=payload.x_range,
                                     y_range=payload.y_range, resolution=payload.resolution, threshold=payload.threshold)
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


class InspectRequest(EvaluateRequest):
    scenario: str = Field(default="baseline", max_length=60)
    output: str | None = Field(default=None, max_length=60)
    persist: bool = False
    dataset_id: int | None = None
    dataset_version: int | None = Field(default=None, ge=1)
    note: str = Field(default="", max_length=500)


@router.post("/inspect")
def inspect_calculation(payload: InspectRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
                        team: Team = Depends(get_current_team)):
    """"How was this calculated?" for one output of the baseline or one scenario: formula, substituted values, every step,
    inputs and their sources. The calculation is recomputed here from the request. With `persist`, a sealed evidence record
    is stored as well."""
    model = resolve_model(db, team, payload.model)
    result = _evaluate(payload, model)
    spec = scenarios._model(model)
    output = payload.output or result["primary_output"]
    if payload.scenario == "baseline":
        inputs, adjustments = result["baseline"]["inputs"], {}
    elif payload.scenario in result["scenarios"]:
        inputs, adjustments = result["scenarios"][payload.scenario]["inputs"], result["scenarios"][payload.scenario]["adjustments"]
    else:
        raise HTTPException(400, f"There is no scenario '{payload.scenario}'. Available: baseline, {', '.join(result['scenarios'])}")
    try:
        tr = scenario_evidence.trace(model, result["baseline"]["inputs"], inputs, result["assumptions"], output, scenario=payload.scenario,
                                     adjustments=adjustments, assumption_sources=result["assumption_sources"])
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e
    out = {"trace": tr, "verification": result["verification"], "explanation": result["explanation"]}
    if payload.persist:
        out["evidence"] = _persist(db, team, user, "scenario", model, _evaluation_request(payload), payload.dataset_id, payload.dataset_version, payload.note)
    return out


def _evaluation_request(p: EvaluateRequest) -> dict:
    return {"baseline": p.baseline, "scenarios": p.scenarios, "assumptions": p.assumptions, "sensitivity_pct": p.sensitivity_pct,
            "spread_pct": p.spread_pct, "assumption_sources": p.assumption_sources}


def _persist(db: Session, team: Team, user: User, kind: str, model, request: dict, dataset_id: int | None, dataset_version: int | None,
             note: str = "", scenario_id: int | None = None) -> dict:
    prov = None
    if dataset_id is not None:
        _, _, _, prov = _versioned_frame(db, team, dataset_id, dataset_version)
        prov = {"dataset_id": prov["dataset_id"], "dataset_version_id": prov["dataset_version_id"], "content_sha256": prov["content_sha256"]}
    try:
        row = scenario_evidence.create(db, team_id=team.id, user_id=user.id, kind=kind, model_ref=scenarios.model_key(model), model=model,
                                       request=request, scenario_id=scenario_id, dataset=prov, note=note)
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e
    db.commit()
    db.refresh(row)
    return scenario_evidence.out(row)


class AskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=scenario_nl.MAX_MESSAGE)
    model: str
    baseline: dict[str, float]
    assumptions: dict[str, float] = {}
    context: dict[str, Any] | None = None
    use_model: bool = False


@router.post("/ask")
def ask(payload: AskRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
        team: Team = Depends(get_current_team)):
    """A what-if question in plain words. The sentence is read into a structured request, validated, and run through the same
    engines, checks and explanation as the form. Anything unclear comes back as a question. A language model is used only
    if `use_model` is set, only to translate a sentence the rules could not read, and never supplies a number."""
    model = resolve_model(db, team, payload.model)
    if payload.use_model:
        rate_limit.enforce(f"scenario-ask:{team.id}:{user.id}", 15, 60)
    try:
        return scenario_service.ask(model, payload.baseline, payload.assumptions, payload.message, payload.context,
                                    use_model=payload.use_model)
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


@router.post("/simulate")
def simulate(payload: SimulateRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
             team: Team = Depends(get_current_team)):
    """Monte Carlo over one scenario: percentiles, probabilities, histogram, and an independent verification."""
    return _simulate(resolve_model(db, team, payload.model), payload.baseline, payload.assumptions, payload)


@router.post("/goal-seek")
def goal_seek(payload: GoalSeekRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user),
              team: Team = Depends(get_current_team)):
    """The value of one input that reaches a target output, found on the deterministic engine and verified."""
    model = resolve_model(db, team, payload.model)
    try:
        return scenario_service.goal_seek(model, payload.baseline, payload.target_output, payload.target_value, payload.solve_for,
                                          adjustments=payload.adjustments, assumptions=payload.assumptions, lower=payload.lower,
                                          upper=payload.upper, constraints=[c.model_dump() for c in payload.constraints])
    except scenarios.ScenarioError as e:
        raise HTTPException(400, str(e)) from e


@router.post("", status_code=201)
def save_scenario(
    payload: ScenarioSave,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    team: Team = Depends(get_current_team),
):
    model = resolve_model(db, team, payload.model)
    results = _evaluate(payload, model)
    sim_config = None
    if payload.simulation is not None:
        # Run it now and keep the seed the server used: the saved scenario can then be reproduced exactly.
        sim = _simulate(model, payload.baseline, results["assumptions"], payload.simulation)
        results["simulation"] = sim
        sim_config = {**payload.simulation.model_dump(), "seed": sim["config"]["seed"]}
    version_id = None
    source = None
    if payload.dataset_id is not None:
        dataset, version, _, prov = _versioned_frame(db, team, payload.dataset_id,
                                                     (payload.baseline_source or {}).get("dataset_version"))
        version_id = version.id if version else None
        source = {**(payload.baseline_source or {}), "provenance": prov}
    if payload.baseline_source is not None and len(json.dumps(payload.baseline_source, default=str)) > 20_000:
        raise HTTPException(413, "The baseline source is too large")
    row = Scenario(team_id=team.id, dataset_id=payload.dataset_id, dataset_version_id=version_id,
                   name=payload.name.strip(), model=scenarios.model_key(model), baseline_json=results["baseline"]["inputs"],
                   adjustments_json={"scenarios": payload.scenarios, "assumptions": results["assumptions"],
                                     "sensitivity_pct": payload.sensitivity_pct, "spread_pct": payload.spread_pct,
                                     "assumption_sources": results["assumption_sources"],
                                     **({"baseline_source": source} if source else {}),
                                     **({"simulation": sim_config} if sim_config else {})},
                   results_json=results, created_by=user.id)
    db.add(row)
    db.flush()
    prov = None
    if payload.dataset_id is not None:
        prov = {"dataset_id": source["provenance"]["dataset_id"], "dataset_version_id": source["provenance"]["dataset_version_id"],
                "content_sha256": source["provenance"]["content_sha256"]}
    scenario_evidence.create(db, team_id=team.id, user_id=user.id, kind="scenario", model_ref=scenarios.model_key(model), model=model,
                             request=_evaluation_request(payload), scenario_id=row.id, dataset=prov, note="saved with the scenario")
    if payload.simulation is not None:
        scenario_evidence.create(db, team_id=team.id, user_id=user.id, kind="simulation", model_ref=scenarios.model_key(model), model=model,
                                 request={"baseline": payload.baseline, "assumptions": results["assumptions"], **{
                                     **payload.simulation.model_dump(), "seed": sim_config["seed"]}},
                                 scenario_id=row.id, dataset=prov, note="saved with the scenario")
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
