"""How a what-if number was calculated, and a sealed record that it was.

`trace` answers "How was this calculated?" for one output of one scenario: the formula, the inputs with where they came from
and how the scenario changed them, every intermediate step with its value, the assumptions and constraints in force. Each step is
evaluated with the same safe formula parser as a custom model, and the final value is compared with the engine's own: a trace
that does not reproduce the engine's number says so (`matches_engine`).

A sealed record (table `scenario_evidence`) keeps a calculation for audit: the request, the computed result, the verification
outcome, the model definition hash, the engine versions, the dataset provenance and a SHA-256 over request and result. Records
are append-only. The server always recomputes a result from the request; it never stores a result a client sends. `reproduce`
re-runs a stored request today and says whether it gives the same result, so a changed engine, a changed model or a tampered row
is visible.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from sqlalchemy.orm import Session

from app import scenario_formula as sf
from app import scenario_service as svc
from app import scenarios
from app.models import ScenarioEvidence
from app.scenarios import ScenarioError

EVIDENCE_VERSION = "scenario-evidence.1"
KINDS = ("scenario", "simulation", "goal_seek", "sensitivity_grid")
VOLATILE = {"verification", "explanation", "runtime_ms", "data_kinds", "assumption_sources"}


# ---------------------------------------------------------------------- the trace --

def _expressions(spec: dict) -> dict[str, str]:
    if "custom" in spec:
        return {n: r["formula"] for n, r in spec["custom"]["definition"]["outputs"].items()}
    return dict(spec["expressions"])


def _fmt(v: float | None) -> str:
    if v is None:
        return "undefined"
    return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:.6g}")


def _substitute(formula: str, env: dict[str, float | None]) -> str:
    return re.sub(r"[a-z][a-z0-9_]*", lambda m: f"({_fmt(env[m.group(0)])})" if m.group(0) in env and env[m.group(0)] is not None
                  else (m.group(0) if m.group(0) not in env else "(undefined)"), formula)


def trace(model: "str | dict", baseline: dict, inputs: dict, assumptions: dict, output: str, *, scenario: str = "baseline",
          adjustments: dict | None = None, input_sources: dict | None = None, assumption_sources: dict | None = None) -> dict:
    """How `output` was calculated for these inputs: formula, substituted values, steps, inputs and their provenance."""
    spec = scenarios._model(model)  # noqa: SLF001
    if output not in spec["outputs"]:
        raise ScenarioError(f"Unknown output '{output}'. Available: {', '.join(spec['outputs'])}")
    exprs = _expressions(spec)
    parsed = {n: sf.parse(f) for n, f in exprs.items()}
    deps = {n: set(p.names) for n, p in parsed.items()}
    known = set(inputs) | set(assumptions)
    order = sf.order_outputs(deps, known)
    needed: set[str] = set()

    def walk(name: str) -> None:
        if name in needed:
            return
        needed.add(name)
        for d in deps[name]:
            if d in deps:
                walk(d)
    walk(output)
    env: dict[str, float | None] = {**inputs, **assumptions}
    steps = []
    for name in order:
        if name not in needed:
            continue
        v = sf.evaluate_scalar(parsed[name], {k: (float("nan") if x is None else x) for k, x in env.items()})
        env[name] = v
        steps.append({"name": name, "label": spec["outputs"][name], "unit": (spec.get("output_units") or {}).get(name, ""),
                      "formula": exprs[name], "substituted": _substitute(exprs[name], env), "value": v,
                      "uses": sorted(deps[name])})
    used_inputs = sorted({d for n in needed for d in deps[n] if d in spec["inputs"]}, key=list(spec["inputs"]).index)
    adjustments = adjustments or {}
    linked = (scenarios.model_key(model) == "unit_economics" and bool(assumptions.get("price_elasticity")) and "price" in adjustments
              and "volume" not in adjustments and bool(baseline.get("price")))
    rows = []
    for k in spec["inputs"]:
        if k not in used_inputs:
            continue
        rule = spec["inputs"][k]
        row = {"name": k, "label": rule["label"], "unit": rule["unit"], "baseline": baseline[k], "value": inputs[k],
               "adjustment": adjustments.get(k), "source": (input_sources or {}).get(k)}
        if k == "volume" and linked:
            row["derived_from"] = (f"baseline volume x (1 + price elasticity {assumptions['price_elasticity']:g} x the price change), "
                                   "because the price was adjusted and volume was not")
        rows.append(row)
    used_assumptions = [{"name": k, "label": spec["assumptions"][k]["label"] if k in spec.get("assumptions", {}) else k.replace("_", " "),
                         "value": assumptions[k], "source": (assumption_sources or {}).get(k, {"source": "manual"}),
                         "used_in_formulas": any(k in deps[n] for n in needed), "applied_by_engine": bool(k == "price_elasticity" and linked)}
                        for k in assumptions if any(k in deps[n] for n in needed) or (k == "price_elasticity" and linked)]
    engine = scenarios._outputs(model, inputs, assumptions)[output]  # noqa: SLF001
    final = steps[-1]["value"] if steps else None
    matches = (engine is None and final is None) or (engine is not None and final is not None and math.isclose(engine, final, rel_tol=1e-9, abs_tol=1e-9))
    notes_for = {}
    check = spec.get("constraint_violations")
    constraints = []
    for c in spec.get("constraints", []):
        constraints.append({"rule": c["text"], "message": c["message"], "severity": c["severity"]})
    if check is not None:
        notes_for = {v["constraint"] for v in check({**inputs, **assumptions, **{k: (math.nan if x is None else x) for k, x in
                                                                          scenarios._outputs(model, inputs, assumptions).items()}})}  # noqa: SLF001
        for c in constraints:
            c["satisfied"] = c["rule"] not in notes_for
    return {"kind": "calculation_trace", "evidence_version": EVIDENCE_VERSION, "scenario": scenario, "output": output,
            "label": spec["outputs"][output], "unit": (spec.get("output_units") or {}).get(output, ""), "value": engine,
            "formula": exprs[output], "steps": steps, "inputs": rows, "assumptions": used_assumptions, "constraints": constraints,
            "model": scenarios.model_key(model), "model_sha256": spec["custom"]["definition_sha256"] if "custom" in spec else None,
            "engine_version": scenarios.ENGINE_VERSION, "matches_engine": matches,
            "how_to_read": "Steps are in the order they are computed; each uses the inputs, assumptions and earlier steps it lists."}


# ----------------------------------------------------------------- sealing records --

def clean_json(value: Any) -> Any:
    """JSON-safe: numpy numbers become Python numbers and non-finite numbers become null (JSON cannot hold NaN)."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def core(result: dict) -> dict:
    """What a result is made of, without the parts that are derived or can vary (text, timing, the verification block)."""
    return clean_json({k: v for k, v in result.items() if k not in VOLATILE})


def seal(request: dict, result_core: dict) -> str:
    body = json.dumps({"request": clean_json(request), "result": result_core}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def compute(kind: str, model: "str | dict", request: dict) -> dict:
    """Run a request through the shared service. The only way a result enters a record."""
    if kind == "scenario":
        return svc.evaluate(model, request["baseline"], request.get("scenarios"), request.get("assumptions"),
                            sensitivity_pct=request.get("sensitivity_pct", 10.0), spread_pct=request.get("spread_pct"),
                            assumption_sources=request.get("assumption_sources"))
    if kind == "simulation":
        return svc.simulate(model, request["baseline"], request.get("assumptions"), adjustments=request.get("adjustments"),
                            uncertainty=request.get("uncertainty"), correlation=request.get("correlation"),
                            iterations=request.get("iterations", 10_000), seed=request.get("seed"), target_output=request.get("target_output"),
                            target_value=request.get("target_value"), target_direction=request.get("target_direction", "at_least"))
    if kind == "goal_seek":
        return svc.goal_seek(model, request["baseline"], request["target_output"], request["target_value"], request["solve_for"],
                             adjustments=request.get("adjustments"), assumptions=request.get("assumptions"), lower=request.get("lower"),
                             upper=request.get("upper"), constraints=request.get("constraints"))
    if kind == "sensitivity_grid":
        return svc.grid(model, request["baseline"], request["x_input"], request["y_input"], request.get("output"),
                        adjustments=request.get("adjustments"), assumptions=request.get("assumptions"), x_range=request.get("x_range"),
                        y_range=request.get("y_range"), resolution=request.get("resolution", 21), threshold=request.get("threshold", 0.0))
    raise ScenarioError(f"Evidence is kept for: {', '.join(KINDS)}")


def _versions(result: dict) -> dict:
    return {"scenarios": scenarios.ENGINE_VERSION, "evidence": EVIDENCE_VERSION, "service": svc.SERVICE_VERSION,
            "result_engine": result.get("engine_version"), "explain": (result.get("explanation") or {}).get("version")}


def create(db: Session, *, team_id: int, user_id: int | None, kind: str, model_ref: str, model: "str | dict", request: dict,
           scenario_id: int | None = None, dataset: dict | None = None, note: str = "") -> ScenarioEvidence:
    """Recompute the request, seal it, and store the record. `dataset` is provenance (id, version id, content hash)."""
    if kind not in KINDS:
        raise ScenarioError(f"Evidence is kept for: {', '.join(KINDS)}")
    result = compute(kind, model, request)
    rc = core(result)
    ver = result.get("verification") or {}
    spec = scenarios._model(model)  # noqa: SLF001
    seed = (result.get("config") or {}).get("seed") if kind == "simulation" else None
    stored_request = {**clean_json(request), **({"seed": seed} if seed is not None else {}), "model": scenarios.model_key(model)}
    row = ScenarioEvidence(
        team_id=team_id, scenario_id=scenario_id, kind=kind, model_ref=scenarios.model_key(model),
        model_sha256=spec["custom"]["definition_sha256"] if "custom" in spec else None,
        dataset_id=(dataset or {}).get("dataset_id"), dataset_version_id=(dataset or {}).get("dataset_version_id"),
        dataset_sha256=(dataset or {}).get("content_sha256"), request_json=stored_request, result_json=rc,
        verification_status=ver.get("status", "unverified"), verification_json=clean_json(ver),
        engine_versions_json=_versions(result), seed=seed, content_sha256=seal(stored_request, rc), note=(note or "")[:500],
        created_by=user_id)
    db.add(row)
    return row


def out(row: ScenarioEvidence, *, full: bool = False) -> dict:
    d = {"id": row.id, "scenario_id": row.scenario_id, "kind": row.kind, "model": row.model_ref, "model_sha256": row.model_sha256,
         "dataset": {"dataset_id": row.dataset_id, "dataset_version_id": row.dataset_version_id, "content_sha256": row.dataset_sha256},
         "verification_status": row.verification_status, "engine_versions": row.engine_versions_json or {}, "seed": row.seed,
         "content_sha256": row.content_sha256, "note": row.note or "", "created_at": row.created_at, "created_by": row.created_by}
    if full:
        d.update(request=row.request_json, result=row.result_json, verification=row.verification_json)
    return d


def reproduce(row: ScenarioEvidence, model: "str | dict") -> dict:
    """Run the stored request again now. `reproduced` is true only if the same request gives the same sealed result."""
    request = dict(row.request_json or {})
    request.pop("model", None)
    stored_seed = request.pop("seed", None) if row.kind == "simulation" else None
    if stored_seed is not None:
        request["seed"] = stored_seed
    now = core(compute(row.kind, model, request))
    stored_request = dict(row.request_json or {})
    sealed_now = seal(stored_request, now)
    stored_hash_ok = seal(stored_request, row.result_json or {}) == row.content_sha256          # the stored row itself is intact
    differences = [k for k in sorted(set(now) | set(row.result_json or {})) if now.get(k) != (row.result_json or {}).get(k)]
    reproduced = sealed_now == row.content_sha256 and stored_hash_ok and not differences     # a record that was altered never counts
    return {"evidence_id": row.id, "reproduced": reproduced, "record_intact": stored_hash_ok,
            "content_sha256_now": sealed_now, "content_sha256_stored": row.content_sha256, "differences": differences,
            "engine_versions_now": _versions({"engine_version": now.get("engine_version")}), "engine_versions_stored": row.engine_versions_json,
            "explanation": ("The same request gives the same result today." if reproduced else
                            ("The stored record is not what was sealed: it was altered." if not stored_hash_ok else
                             "The same request now gives a different result: the engine or the model changed since it was sealed."))}
