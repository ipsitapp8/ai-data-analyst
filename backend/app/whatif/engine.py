"""Runs a scenario. The only place the universal what-if layer calculates.

    dataset operations (thresholds, shifts, association, trend)   -> app.whatif.operations
    formula models (built-in business, domain templates, custom)  -> app.scenario_service, the engine the business What-If page
                                                                      and the Copilot already share (evaluate / simulate / goal_seek)

Every result has one shape (see `_standard`), whichever engine produced it, so the API, the page and the Copilot read it the same way.
No language model is involved: a model may help someone phrase or pick a question, never compute an answer.
"""
from __future__ import annotations

import copy
from typing import Any

import pandas as pd

from app import scenario_service, scenarios
from app.whatif import ENGINE_VERSION, operations, scenario as sc
from app.whatif.domains import find_template, get_domain
from app.whatif.explain import explain
from app.whatif.profile import build_profile

SENSITIVITY_PCT = 10.0
SIM_ITERATIONS = 10_000


class EngineError(ValueError):
    """The scenario could not be run; the message says why."""


def run(scenario: dict, df: pd.DataFrame | None = None, profile: dict | None = None) -> dict:
    """Validate and execute a scenario. Returns a copy with `validation`, `results`, `explanation` and `meta` filled in.
    A scenario that is not valid is returned with `results = None` and the reasons; nothing is calculated from guessed values."""
    s = copy.deepcopy(scenario)
    template = find_template(s.get("domain", "custom"), s.get("template_id", ""))
    if df is not None and profile is None:
        dom = s.get("domain")
        profile = build_profile(df, filename=(s.get("dataset") or {}).get("filename"), domain=dom if dom and dom != "custom" else None)
    v = sc.validate(s, df, profile)
    s["validation"] = {k: v[k] for k in ("status", "errors", "missing_inputs", "warnings")}
    s["results"] = None
    if v["status"] != sc.VALID:
        s["explanation"] = None
        return s
    s["columns"], s["baseline"], s["baseline_sources"], s["changes"] = v["columns"], v["baseline"], v["baseline_sources"], v["changes"]
    s["assumptions"] = v["assumptions"]
    op = template.operation
    try:
        if op in operations.OPERATIONS:
            raw = operations.OPERATIONS[op](df, s["columns"], template.params, _values(template, s))
            result = _standard(raw, template, s, engine="dataset-operation")
        else:
            result = _run_formula(s, template)
    except (operations.OperationError, scenarios.ScenarioError) as e:
        s["validation"] = {**s["validation"], "status": sc.INVALID, "errors": [str(e)]}
        s["explanation"] = None
        return s
    s["results"] = result
    if result.get("seed") is not None:
        s["seed"] = result["seed"]
    s["meta"] = {**s.get("meta", {}), "engine_version": ENGINE_VERSION, "method": s["method"], "evidence_kind": result["evidence_kind"]}
    s["explanation"] = explain(s, template, get_domain(s["domain"]))
    return s


def _values(template, s: dict) -> dict:
    out = {sp.name: sp.default for sp in template.inputs if sp.default is not None}
    out.update({k: float(v) for k, v in (s.get("inputs") or {}).items()})
    return out


def _standard(raw: dict, template, s: dict, engine: str) -> dict:
    status = raw.get("status", "ok")
    kind = template.evidence if status == "ok" else "unsupported_conclusion"
    return {"status": status, "reason": raw.get("reason"), "evidence_kind": kind, "engine": engine, "metrics": raw.get("metrics", []),
            "primary": raw.get("primary"), "details": raw.get("details", {}), "assumptions": raw.get("assumptions", []),
            "held_constant": raw.get("held_constant", []), "warnings": raw.get("warnings", []), "rows_used": raw.get("rows_used"),
            "uncertainty": raw.get("uncertainty", {"kind": "none", "note": ""}), "seed": raw.get("seed"), "verification": raw.get("verification")}


def _metrics_from_outputs(base: dict, new: dict, labels: dict, units: dict) -> list[dict]:
    return [operations._metric(k, labels[k], base.get(k), new.get(k), units.get(k, "")) for k in labels]  # noqa: SLF001


def _run_formula(s: dict, template) -> dict:
    model = sc.model_of(s, template)
    spec = scenarios._model(model)  # noqa: SLF001
    baseline, changes, assumptions, method = s["baseline"], s["changes"], s["assumptions"], s["method"]
    names = list(spec["inputs"])
    missing = [n for n in names if n not in baseline]
    if missing:
        raise EngineError(f"No starting value for: {', '.join(missing)}")
    baseline = {n: baseline[n] for n in names}
    notes = {k: "entered by you" for k in baseline}
    notes.update({k: v for k, v in (s.get("baseline_sources") or {}).items() if k in baseline})
    held = [f"{spec['inputs'][n]['label']} = {baseline[n]:,.6g}" for n in names if n not in changes]
    changed = [f"{spec['inputs'][n]['label']}: {_describe_change(baseline[n], c)}" for n, c in changes.items()]
    output_units = spec.get("output_units", {})
    common = {"assumptions": [f"{spec['assumptions'][a]['label']} = {assumptions[a]:g}" for a in assumptions if a in spec["assumptions"]],
              "held_constant": held, "changed": changed, "baseline_sources": notes}
    if method == "monte_carlo":
        target = s.get("target") or {}
        out = scenario_service.simulate(model, baseline, assumptions, adjustments=changes, uncertainty=s["uncertainty"], iterations=SIM_ITERATIONS,
                                        seed=s.get("seed"), target_output=target.get("output"), target_value=target.get("value"),
                                        target_direction=target.get("direction", "at_least"))
        t = out["target"]
        center_out = scenarios._outputs(model, scenarios.apply_adjustments(model, baseline, changes, assumptions), assumptions)  # noqa: SLF001
        base_out = scenarios._outputs(model, baseline, assumptions)  # noqa: SLF001
        metrics = _metrics_from_outputs(base_out, center_out, spec["outputs"], output_units)
        return {"status": "ok", "evidence_kind": "hypothetical_assumption", "engine": "scenario_service.simulate", "metrics": metrics,
                "primary": spec["primary_output"], "rows_used": None, "seed": out["config"]["seed"], "reason": None, "warnings": out.get("disclosures", []),
                "details": {**common, "simulation": {"output": t["output"], "label": t["label"], "p5": t["low"], "mean": t["expected"], "median": t["median"],
                                                      "p95": t["high"], "iterations": out["config"]["iterations"], "histogram": t["histogram"],
                                                      "probabilities": t["probabilities"], "uncertain_inputs": out["uncertain_inputs"]}},
                "assumptions": common["assumptions"] + ["The input distributions you chose are assumptions, not measurements."], "held_constant": held,
                "uncertainty": {"kind": "simulation_percentiles", "level": 0.90, "target": t["output"], "low": t["low"], "high": t["high"],
                                "note": "5th to 95th percentile of the simulated outcomes under the distributions you stated. These describe the simulation, not guaranteed bounds. "
                                        f"Seed {out['config']['seed']} reproduces this run exactly."},
                "verification": out.get("verification")}
    if method == "goal_seek":
        g = s["target"]
        out = scenario_service.goal_seek(model, baseline, g["output"], float(g["value"]), g["solve_for"], adjustments=changes, assumptions=assumptions,
                                         lower=g.get("lower"), upper=g.get("upper"), constraints=s.get("constraints") or None)
        solved = out.get("status") in ("solved", "multiple_solutions")
        sols = out.get("solutions") or []
        return {"status": "ok" if solved else "insufficient_evidence", "reason": None if solved else out.get("message"),
                "evidence_kind": "hypothetical_assumption" if solved else "unsupported_conclusion", "engine": "scenario_service.goal_seek",
                "metrics": [], "primary": None, "rows_used": None,
                "details": {**common, "goal_seek": {"status": out.get("status"), "message": out.get("message"), "solution": sols[0]["value"] if sols else None,
                                                    "solutions": sols, "achievable": out.get("achievable"), "solve_for": g["solve_for"]}},
                "assumptions": common["assumptions"], "held_constant": held, "warnings": out.get("disclosures", []),
                "uncertainty": {"kind": "none", "note": out.get("message") or "Solved on the model's own formulas; whether the answer is attainable in practice is not assessed."},
                "verification": out.get("verification")}
    named = {"scenario": changes} if changes else {}
    out = scenario_service.evaluate(model, baseline, named, assumptions, sensitivity_pct=SENSITIVITY_PCT)
    base_out = out["baseline"]["outputs"]
    new_out = out["scenarios"]["scenario"]["outputs"] if "scenario" in out["scenarios"] else base_out
    metrics = _metrics_from_outputs(base_out, new_out, out["output_labels"], output_units)
    details = {**common, "sensitivity": out.get("sensitivity"), "break_even": out.get("break_even"),
               "contributions": (out["scenarios"].get("scenario") or {}).get("contributions"), "engine_explanation": (out.get("explanation") or {}).get("summary")}
    warnings = [n["message"] for notes_ in (out.get("constraint_notes") or {}).values() for n in notes_]
    return {"status": "ok", "evidence_kind": template.evidence if template.evidence != "exact_calculation" else "hypothetical_assumption",
            "engine": "scenario_service.evaluate", "metrics": metrics, "primary": out["primary_output"], "rows_used": None, "reason": None, "seed": None,
            "details": details, "assumptions": common["assumptions"] + list(out.get("unsupported_assumptions", [])), "held_constant": held, "warnings": warnings,
            "uncertainty": {"kind": "none", "note": "The arithmetic is exact for the values used. Those values are your inputs, so the result is only as true as they are; real-world variation is not modelled."},
            "verification": out.get("verification")}


def _describe_change(base: float, c: dict) -> str:
    if c["type"] == "pct":
        return f"{base:,.6g} changed by {c['value']:+g}%"
    if c["type"] == "abs":
        return f"{base:,.6g} changed by {c['value']:+g}"
    return f"set to {c['value']:g} (was {base:,.6g})"


def compare(items: list[dict]) -> dict:
    """Put executed scenarios side by side. Metrics are lined up by label; a scenario without a metric shows a gap, never a guess.
    Scenarios whose baselines differ say so, because their differences are then not changes to the same starting point."""
    ran = [i for i in items if i.get("results") and i["results"].get("status") == "ok"]
    if len(ran) < 2:
        raise EngineError("Comparing needs at least two scenarios that have been run")
    labels: dict[str, dict] = {}
    for it in ran:
        for m in it["results"]["metrics"]:
            labels.setdefault(m["label"], {"label": m["label"], "unit": m["unit"], "baseline": {}, "scenario": {}, "delta": {}})
            row = labels[m["label"]]
            row["baseline"][it["id"]], row["scenario"][it["id"]], row["delta"][it["id"]] = m["baseline"], m["scenario"], m["delta"]
    shared = [r for r in labels.values() if len(r["scenario"]) == len(ran)]
    notes = []
    if not shared:
        notes.append("These scenarios share no metric, so they cannot be compared number by number.")
    if len({it["domain"] for it in ran}) > 1:
        notes.append("The scenarios come from different domains; compare their meaning, not their units.")
    for r in shared:
        if len({v for v in r["baseline"].values() if v is not None}) > 1:
            notes.append(f"'{r['label']}' starts from different baselines in these scenarios.")
    return {"scenarios": [{"id": i["id"], "name": i["name"], "domain": i["domain"], "evidence_kind": i["results"]["evidence_kind"],
                           "question": i.get("question")} for i in ran],
            "metrics": shared, "only_some": [r for r in labels.values() if len(r["scenario"]) != len(ran)], "notes": notes}
