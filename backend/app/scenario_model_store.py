"""Persistence and data binding for custom what-if models.

Custom models are team-owned rows. A definition is immutable once saved: editing adds a version, and a saved scenario
refers to `custom:<id>@<version>`, so changing a model never changes an earlier result. Another team's model looks
exactly like one that does not exist.

Inputs can be bound to the team's *approved* semantic-layer metrics. A draft or deprecated metric is refused, with the
reason, at save time and again when a baseline is derived (its status may have changed since). Nothing in a dataset, a
note or a model response can approve a metric; this module only reads the approval the semantic layer already holds.
"""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from app import metric_engine as me
from app import semantic
from app.models import ScenarioModel, ScenarioModelVersion
from app.scenario_models import (AmbiguousMetric, ModelDefinitionError, analyze, canonical, compile_spec,
                                 definition_sha256, describe)
from app.scenarios import ScenarioError

_REF = re.compile(r"^custom:(\d+)(?:@(\d+))?$")


class ModelNotFound(ScenarioError):
    """The custom model or version does not exist for this team (another team's model looks exactly the same)."""


def is_custom(ref: Any) -> bool:
    return isinstance(ref, str) and ref.startswith("custom:")


def parse_ref(ref: str) -> tuple[int, int | None]:
    m = _REF.match(ref or "")
    if not m:
        raise ScenarioError("A custom model is referred to as custom:<id> or custom:<id>@<version>")
    return int(m.group(1)), int(m.group(2)) if m.group(2) else None


def owned(db: Session, team_id: int, model_id: int) -> ScenarioModel:
    row = db.get(ScenarioModel, model_id)
    if row is None or row.team_id != team_id:
        raise ModelNotFound("Custom model not found")
    return row


def _version(db: Session, model: ScenarioModel, number: int | None) -> ScenarioModelVersion:
    row = db.query(ScenarioModelVersion).filter_by(model_id=model.id, version=number or model.current_version).first()
    if row is None:
        raise ModelNotFound("Custom model version not found")
    return row


def load_spec(db: Session, team_id: int, ref: str) -> dict:
    """The compiled model for custom:<id>[@<version>], for this team only. Without a version, the current one."""
    model_id, number = parse_ref(ref)
    model = owned(db, team_id, model_id)
    v = _version(db, model, number)
    return compile_spec(v.definition_json, key=f"custom:{model.id}@{v.version}", title=model.name, model_id=model.id,
                        version=v.version)


def check_sources(db: Session, team_id: int, defn: dict) -> None:
    """Metrics a definition is bound to must exist for this team and be approved now."""
    for name, r in defn["inputs"].items():
        src = r.get("source")
        if src and src.get("type") == "metric":
            try:
                metric = semantic.get_metric(db, team_id, src["metric_id"])
            except semantic.SemanticError as e:
                raise ModelDefinitionError([f"Input '{name}': {e}"]) from e
            if metric.status != "approved":
                raise ModelDefinitionError([f"Input '{name}' is bound to the metric '{metric.name}', which is {metric.status}, "
                                            "not approved. Only an approved definition can drive a model."])
    for name, r in defn["outputs"].items():
        rec = r.get("reconcile")
        if rec:
            try:
                semantic.get_metric(db, team_id, rec["metric_id"])
            except semantic.SemanticError as e:
                raise ModelDefinitionError([f"Output '{name}': {e}"]) from e


def _store(db: Session, model: ScenarioModel, user_id: int | None, defn: dict, note: str, number: int) -> ScenarioModelVersion:
    analysis = analyze(defn)
    if not analysis["valid"]:
        raise ModelDefinitionError(analysis["errors"])
    check_sources(db, model.team_id, analysis["definition"])
    row = ScenarioModelVersion(model_id=model.id, version=number, definition_json=analysis["definition"],
                               definition_sha256=definition_sha256(analysis["definition"]), note=(note or "")[:500],
                               created_by=user_id)
    db.add(row)
    return row


def create_model(db: Session, *, team_id: int, user_id: int | None, name: str, description: str, definition: dict,
                 note: str = "") -> tuple[ScenarioModel, ScenarioModelVersion]:
    try:
        slug = semantic.slugify(name)
    except semantic.SemanticError as e:
        raise ScenarioError(str(e)) from e
    if db.query(ScenarioModel).filter_by(team_id=team_id, slug=slug).first():
        raise ScenarioError(f"This team already has a model named '{name}'; add a version to it instead")
    model = ScenarioModel(team_id=team_id, slug=slug, name=name.strip()[:120], description=(description or "")[:1000],
                          current_version=1, created_by=user_id)
    db.add(model)
    db.flush()
    try:
        version = _store(db, model, user_id, definition, note or "First version", 1)
    except ScenarioError:
        db.rollback()
        raise
    db.commit()
    db.refresh(model)
    db.refresh(version)
    return model, version


def add_version(db: Session, model: ScenarioModel, user_id: int | None, definition: dict, note: str = "") -> ScenarioModelVersion:
    if model.status != "active":
        raise ScenarioError("This model is archived; restore it before adding a version")
    number = model.current_version + 1
    version = _store(db, model, user_id, definition, note, number)
    model.current_version = number
    db.commit()
    db.refresh(version)
    return version


def set_status(db: Session, model: ScenarioModel, status: str) -> ScenarioModel:
    model.status = status
    db.commit()
    db.refresh(model)
    return model


def model_out(db: Session, model: ScenarioModel, *, full: bool = False) -> dict:
    versions = db.query(ScenarioModelVersion).filter_by(model_id=model.id).order_by(ScenarioModelVersion.version).all()
    current = next(v for v in versions if v.version == model.current_version)
    spec = compile_spec(current.definition_json, key=f"custom:{model.id}@{current.version}", title=model.name,
                        model_id=model.id, version=current.version)
    out = {"id": model.id, "slug": model.slug, "name": model.name, "description": model.description or "",
           "status": model.status, "current_version": model.current_version, "ref": spec["key"],
           "created_at": model.created_at, "updated_at": model.updated_at, "describe": describe(spec),
           "versions": [{"version": v.version, "note": v.note, "created_at": v.created_at,
                         "definition_sha256": v.definition_sha256} for v in versions]}
    if full:
        out["definition"] = canonical(current.definition_json)      # restores the author's order (JSONB does not keep it)
    return out


def version_definition(db: Session, model: ScenarioModel, number: int) -> dict:
    v = _version(db, model, number)
    return {"version": v.version, "note": v.note, "definition": canonical(v.definition_json), "definition_sha256": v.definition_sha256,
            "created_at": v.created_at}


def team_models(db: Session, team_id: int, include_archived: bool = False) -> list[ScenarioModel]:
    q = db.query(ScenarioModel).filter_by(team_id=team_id)
    if not include_archived:
        q = q.filter(ScenarioModel.status == "active")
    return q.order_by(ScenarioModel.name).all()


# ================================================================ baseline from data ==

def _approved_metric_value(db: Session, team_id: int, metric_id: int, dataset_id: int, profile: dict, df: pd.DataFrame,
                           filters: list | None) -> tuple[float | None, dict]:
    metric = semantic.get_metric(db, team_id, metric_id)
    version = semantic.active_version(db, metric)
    if metric.status != "approved" or version is None or version.approved_at is None:
        raise ScenarioError(f"The metric '{metric.name}' is {metric.status}, not approved, so it cannot supply a baseline. "
                            "Ask an owner or admin to approve it, or bind the input to a formula of your own.")
    if not semantic._applies(db, metric, version, dataset_id, profile):  # noqa: SLF001
        raise ScenarioError(f"The approved metric '{metric.name}' does not apply to this dataset (its columns are missing, "
                            "or it is defined for a different dataset).")
    try:
        out = me.evaluate(df, version.formula, [*(version.filters_json or []), *(filters or [])])
    except me.FormulaError as e:
        raise ScenarioError(f"The metric '{metric.name}' could not be computed: {e}") from e
    return out["value"], {"source": "semantic_metric", "metric_id": metric.id, "metric": metric.name,
                          "metric_version": metric.current_version, "formula": version.formula,
                          "filters": list(version.filters_json or []),
                          "note": "an approved definition from the semantic layer"}


def derive_baseline(db: Session, team_id: int, spec: dict, dataset_id: int, profile: dict, df: pd.DataFrame,
                    filters: list | None = None, overrides: dict | None = None) -> dict:
    """Baseline inputs for a custom model from its bindings, each with where it came from. Inputs with no binding
    (or whose binding gives no value) come back as missing: they are never filled in silently."""
    defn = spec["custom"]["definition"]
    inputs: dict[str, float | None] = {}
    sources: dict[str, dict] = {}
    for name, rule in defn["inputs"].items():
        if overrides and overrides.get(name) is not None:
            v = overrides[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ScenarioError(f"The value you entered for '{rule['label']}' must be a finite number")
            inputs[name], sources[name] = float(v), {"source": "override", "formula": None, "note": "entered by you"}
            continue
        src = rule["source"]
        value, info = None, None
        if src and src["type"] == "metric":
            value, info = _approved_metric_value(db, team_id, src["metric_id"], dataset_id, profile, df, filters)
        elif src and src["type"] == "term":
            found = semantic.resolve(db, team_id, dataset_id, src["term"])
            if found["ambiguous"]:
                cands = found["ambiguous"][0]["candidates"]
                raise AmbiguousMetric(f"'{src['term']}' matches more than one approved metric: "
                                      + ", ".join(c["name"] for c in cands) + f". Choose one for the input "
                                      f"'{rule['label']}' by binding it to that metric.", cands)
            if len(found["metrics"]) != 1:
                raise ScenarioError(f"No approved metric matches '{src['term']}' for the input '{rule['label']}'")
            value, info = _approved_metric_value(db, team_id, found["metrics"][0]["id"], dataset_id, profile, df, filters)
        elif src and src["type"] == "formula":
            try:
                value = me.evaluate(df, src["formula"], filters)["value"]
            except me.FormulaError as e:
                raise ScenarioError(f"The formula for '{rule['label']}' could not be computed: {e}") from e
            info = {"source": "formula", "formula": src["formula"],
                    "note": "your own formula over the dataset; it is not an approved metric"}
        if value is None:
            inputs[name] = None
            sources[name] = {"source": "assumed", "formula": None,
                             "note": ("The binding produced no value (the data are empty or the formula divides by zero); enter one."
                                      if src else "No data source is bound to this input; enter a value.")}
        else:
            inputs[name], sources[name] = float(value), info
    missing = [k for k, v in inputs.items() if v is None]
    out = {"model": spec["key"], "inputs": inputs, "sources": sources, "missing": missing,
           "rows_used": int(len(me.apply_filters(df, filters))), "reconciliation": []}
    if not missing:
        outputs = spec["compute"](inputs, {k: r["default"] for k, r in spec["assumptions"].items()})
        for name, r in defn["outputs"].items():
            rec = r.get("reconcile")
            if not rec:
                continue
            metric_value, info = _approved_metric_value(db, team_id, rec["metric_id"], dataset_id, profile, df, filters)
            model_value = outputs[name]
            agrees = (metric_value is not None and model_value is not None
                      and math.isclose(metric_value, model_value, rel_tol=1e-6, abs_tol=1e-9))
            out["reconciliation"].append({"output": name, "label": r["label"], "metric": info["metric"],
                                          "metric_id": info["metric_id"], "observed": metric_value, "model_value": model_value,
                                          "difference": None if metric_value is None or model_value is None else model_value - metric_value,
                                          "agrees": agrees})
    return out
