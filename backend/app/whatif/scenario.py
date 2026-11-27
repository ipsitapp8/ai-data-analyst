"""The standard scenario record, and the checks that run before anything is calculated.

A scenario is plain JSON so it can be saved, compared, sent by the API and built by the Copilot:

    id, name, domain, template_id, question, method,
    dataset      {dataset_id, version, filename} or None
    columns      {concept: [column, ...]}   the user's choice where a concept matches several columns
    inputs       {name: number}             every value the user supplied (parameters, baselines, assumptions, changes)
    baseline     {name: number}             what the model starts from (typed or read from the data), with baseline_sources
    changes      {input: {"type", "value"}} what is varied
    assumptions  {name: number}             model assumptions (for example price elasticity)
    uncertainty  {input: distribution}      for Monte Carlo; these are assumptions, not measurements
    constraints  [ ... ]                    custom-model constraints / goal-seek limits
    target       {output, value, direction, solve_for} for goal seeking
    custom_model definition for the Custom domain (validated by app.scenario_models)
    validation   {status, errors, missing_inputs, warnings}
    results, uncertainty_info, explanation, meta
"""
from __future__ import annotations

import datetime as dt
import math
import uuid
from typing import Any

import pandas as pd

from app.whatif import ENGINE_VERSION, operations
from app.whatif.domains import GENERIC_TEMPLATES, find_template, get_domain
from app.whatif.domains.base import Template
from app.whatif.profile import build_profile, map_concepts

METHODS_BY_OPERATION = {
    "threshold": {"deterministic"}, "shift_metric": {"deterministic"}, "association": {"regression"}, "forecast_trend": {"forecast"},
    "formula_model": {"deterministic", "monte_carlo", "sensitivity", "break_even", "goal_seek"},
    "custom_model": {"deterministic", "monte_carlo", "sensitivity", "goal_seek"},
}
SINGLE_CONCEPT_PARAMS = {"shift_metric": ("concept",), "association": ("x", "y"), "forecast_trend": ("date", "value")}
VALID, NEEDS_INPUT, INVALID, UNSUPPORTED_STATUS = "valid", "needs_input", "invalid", "unsupported"


def new_scenario(domain: str, template_id: str, *, inputs: dict | None = None, name: str | None = None, dataset: dict | None = None,
                 columns: dict | None = None, method: str | None = None, uncertainty: dict | None = None, assumptions: dict | None = None,
                 constraints: list | None = None, target: dict | None = None, custom_model: dict | None = None, seed: int | None = None,
                 created_by: str | None = None, question: str | None = None) -> dict:
    t = find_template(domain, template_id)
    return {"id": uuid.uuid4().hex[:12], "name": name or (t.title if t else template_id), "domain": domain, "template_id": template_id,
            "question": question or (t.question if t else ""), "method": method or (t.method if t else "deterministic"),
            "dataset": dataset, "columns": columns or {}, "inputs": dict(inputs or {}), "baseline": {}, "baseline_sources": {}, "changes": {},
            "assumptions": dict(assumptions or {}), "uncertainty": dict(uncertainty or {}), "constraints": list(constraints or []),
            "target": target, "custom_model": custom_model, "seed": seed,
            "validation": {"status": "unvalidated", "errors": [], "missing_inputs": [], "warnings": []}, "results": None,
            "meta": {"engine_version": ENGINE_VERSION, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "created_by": created_by}}


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def resolve_columns(scenario: dict, template: Template, profile: dict | None) -> tuple[dict, list[dict]]:
    """The concept -> columns mapping to calculate with, and the choices the user still has to make."""
    chosen = {k: ([v] if isinstance(v, str) else list(v)) for k, v in (scenario.get("columns") or {}).items()}
    if profile is None:
        return chosen, []
    names = {c["name"] for c in profile["columns"]}
    # a column-based template uses only the columns named for it, so the domain's own concept mapping is not reported as "used"
    by_domain = scenario["domain"] != "custom" and template.id not in GENERIC_TEMPLATES
    mapped = {k: v["columns"] for k, v in map_concepts(profile, scenario["domain"]).items()} if by_domain else {}
    cols = {**mapped, **{k: v for k, v in chosen.items() if v}}
    for k, v in chosen.items():
        bad = [c for c in v if c not in names]
        if bad:
            raise ValueError(f"Column(s) not in the dataset: {', '.join(bad)}")
    choices = []
    for param in SINGLE_CONCEPT_PARAMS.get(template.operation, ()):
        concept = template.params.get(param)
        if concept and len(cols.get(concept) or []) > 1:
            choices.append({"concept": concept, "candidates": cols[concept]})
    candidates = map_concepts(profile, scenario["domain"]) if by_domain else {}
    for key, m in candidates.items():
        if m["ambiguous"] and key not in chosen and key in {template.params.get(p) for p in ("concept", "x", "y", "date", "value")}:
            choices.append({"concept": key, "candidates": m["candidates"]})
    return cols, choices


def model_of(scenario: dict, template: Template) -> "str | dict":
    """The model the formula engine runs: a built-in name, or a compiled definition."""
    from app import scenario_models

    if scenario["domain"] == "custom" and template.operation == "custom_model":
        defn = scenario.get("custom_model")
        if not defn:
            raise ValueError("A custom scenario needs a model definition")
        return scenario_models.compile_spec(defn, key=f"whatif:custom:{scenario['id']}", title=scenario.get("name") or "Custom model")
    if isinstance(template.model, str):
        return template.model
    return scenario_models.compile_spec(template.model, key=f"whatif:{template.id}", title=template.title)


def validate(scenario: dict, df: pd.DataFrame | None = None, profile: dict | None = None) -> dict:
    """Check a scenario without running it. Returns {"status", "errors", "missing_inputs", "warnings", "columns", "baseline",
    "baseline_sources", "changes", "assumptions"}. Missing values are listed, never filled in."""
    errors: list[str] = []
    missing: list[dict] = []
    warnings: list[str] = []
    out = {"status": VALID, "errors": errors, "missing_inputs": missing, "warnings": warnings, "columns": {}, "baseline": {},
           "baseline_sources": {}, "changes": {}, "assumptions": {}}
    try:
        domain = get_domain(scenario.get("domain", ""))
    except KeyError as e:
        return {**out, "status": INVALID, "errors": [str(e).strip("'\"")]}
    template = find_template(domain.key, scenario.get("template_id", ""))
    if template is None:
        return {**out, "status": INVALID, "errors": [f"Unknown scenario template '{scenario.get('template_id')}' for {domain.title}"]}
    if template.unsupported or template.operation == "none":
        why = template.unsupported or f"This needs information no dataset holds: {'; '.join(template.external)}."
        return {**out, "status": UNSUPPORTED_STATUS, "errors": [why]}
    if df is not None and profile is None:
        profile = build_profile(df, domain=domain.key if domain.key != "custom" else None)
    method = scenario.get("method") or template.method
    allowed = METHODS_BY_OPERATION.get(template.operation, {"deterministic"})
    if method not in allowed:
        errors.append(f"The method '{method}' cannot be used for this question. Available: {', '.join(sorted(allowed))}.")
    if method == "break_even" and template.model != "unit_economics":
        errors.append("Break-even analysis is only available for the price / volume / cost model.")

    raw = scenario.get("inputs") or {}
    specs = {s.name: s for s in template.inputs}
    for k, v in raw.items():
        if k not in specs:
            errors.append(f"Unknown input '{k}'. This scenario takes: {', '.join(specs) or 'no inputs'}.")
        elif not _is_number(v):
            errors.append(f"{specs[k].label} must be a finite number")
        else:
            s = specs[k]
            if s.min is not None and v < s.min:
                errors.append(f"{s.label} cannot be below {s.min:g}")
            if s.max is not None and v > s.max:
                errors.append(f"{s.label} cannot be above {s.max:g}")

    # columns, for anything that reads the dataset
    needs_data = template.operation in operations.OPERATIONS
    cols: dict = {}
    if needs_data:
        if df is None or profile is None:
            return {**out, "status": NEEDS_INPUT, "missing_inputs": [{"name": "dataset", "label": "A dataset", "why": "This question is answered from your data"}]}
        try:
            cols, choices = resolve_columns(scenario, template, profile)
        except ValueError as e:
            return {**out, "status": INVALID, "errors": [str(e)]}
        out["columns"] = cols
        for ch in choices:
            missing.append({"name": f"column:{ch['concept']}", "label": f"Which column is '{ch['concept']}'?", "candidates": ch["candidates"], "kind": "column"})
        required = [template.params.get(p) for p in ("concept", "x", "y", "date")] if template.operation != "threshold" else [template.params["concept"]]
        absent = [c for c in required if c and not cols.get(c) and not any(m["name"] == f"column:{c}" for m in missing)]
        if absent:
            labels = []
            for c in absent:
                concept = domain.concept(c)
                labels.append(concept.label if concept else c)
            return {**out, "status": INVALID, "errors": [f"The dataset has no column for: {', '.join(labels)}. "
                                                         "Choose a different domain or question, or add that information to the data."]}
    # inputs
    baseline, sources, changes, assumptions = {}, {}, {}, {}
    builtin_fit = None
    if isinstance(template.model, str) and df is not None:
        from app.analyst import scenario_bridge
        builtin_fit = scenario_bridge.infer_model(df)
        if builtin_fit and builtin_fit[0] == template.model:
            from app import scenarios
            try:
                derived = scenarios.baseline_from_dataset(df, {"columns": [{"name": c, "kind": "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical"}
                                                                              for c in df.columns]}, template.model, builtin_fit[1])
                for k, v in derived["inputs"].items():
                    if v is not None:
                        baseline[k], sources[k] = float(v), (derived["sources"].get(k) or {}).get("formula") or "from your data"
            except Exception:  # noqa: BLE001 - a baseline that cannot be derived is asked for instead
                pass
    solving = ((scenario.get("target") or {}).get("solve_for") if method == "goal_seek" else None)
    for s in template.inputs:
        if s.role == "change" and s.target == solving:
            continue                       # the input being solved for is the answer, not something to vary
        value = raw.get(s.name)
        if s.role in ("baseline", "assumption") and value is None and s.name in baseline:
            value = baseline[s.name]
        if value is None and s.from_data and df is not None and profile is not None:
            value, how = operations.derive_input(s.from_data, df, cols or {k: v["columns"] for k, v in map_concepts(profile, domain.key).items()}, profile)
            if value is not None:
                sources[s.name] = f"read from your data ({how})"
        elif value is not None and s.name in raw:
            sources[s.name] = "entered by you"
        if value is None:
            value = s.default
            if value is not None:
                sources[s.name] = "default"
        if value is None:
            if not s.optional:
                missing.append({"name": s.name, "label": s.label, "unit": s.unit, "why": s.help or "Needed for this calculation and not in your data", "kind": "number"})
            continue
        if s.role == "baseline":
            baseline[s.name] = float(value)
        elif s.role == "assumption":
            assumptions[s.name] = float(value)
        elif s.role == "change":
            changes[s.target] = {"type": s.change_type, "value": float(value)}
    for k, v in (scenario.get("assumptions") or {}).items():
        if _is_number(v):
            assumptions[k] = float(v)
    if method == "monte_carlo" and not scenario.get("uncertainty"):
        names = ", ".join(template.params.get("uncertain", [])) or "an input"
        missing.append({"name": "uncertainty", "label": f"How uncertain is {names}?", "why": "A Monte Carlo run needs a distribution you state, for example normal with a 15% spread", "kind": "distribution"})
    if method == "goal_seek" and not scenario.get("target"):
        missing.append({"name": "target", "label": "The target to reach (output, value, and the input to solve for)", "kind": "target"})
    if template.operation == "threshold" and "current_threshold" in raw and "new_threshold" in raw and raw["current_threshold"] == raw["new_threshold"]:
        warnings.append("The new threshold equals the current one, so nothing changes.")
    if template.operation == "custom_model":
        try:
            from app import scenario_models
            a = scenario_models.analyze(scenario.get("custom_model"))
            if not a["valid"]:
                errors.extend(a["errors"])
            else:
                warnings.extend(a["warnings"])
                spec_inputs = a["definition"]["inputs"]
                out["baseline"] = {k: float(v) for k, v in (scenario.get("baseline") or {}).items() if k in spec_inputs and _is_number(v)}
                baseline = out["baseline"]
                for k in spec_inputs:
                    if k not in baseline:
                        missing.append({"name": k, "label": f"Baseline value of '{k}'", "kind": "number", "why": "Every variable of your model needs a starting value"})
                changes = {k: v for k, v in (scenario.get("changes") or {}).items()}
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    for step in domain_rules(domain.key, template, raw, baseline):
        warnings.append(step)
    out.update({"baseline": baseline, "baseline_sources": sources, "changes": changes if template.operation in ("formula_model", "custom_model") else {},
                "assumptions": assumptions})
    out["status"] = INVALID if errors else (NEEDS_INPUT if missing else VALID)
    return out


def domain_rules(domain_key: str, template: Template, raw: dict, baseline: dict) -> list[str]:
    """Domain-specific plausibility notes. Warnings, not refusals: the engine's own bounds refuse impossible values."""
    notes = []
    if domain_key == "education" and template.operation == "threshold":
        lo, hi = raw.get("current_threshold"), raw.get("new_threshold")
        if _is_number(lo) and _is_number(hi) and template.params.get("percent_scale") and max(lo, hi) <= 1:
            notes.append("Attendance thresholds are read as percentages; 0.85 means 0.85%, not 85%. Did you mean 85?")
    if domain_key == "healthcare" and baseline.get("beds") == 0:
        notes.append("Zero beds makes occupancy undefined.")
    return notes
