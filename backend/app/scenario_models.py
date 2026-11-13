"""Custom what-if models: validated, versioned definitions that run on the same engines as the built-in models.

A definition names its inputs, assumptions and outputs, gives each output a formula (app/scenario_formula.py), and may
carry units, constraints and bindings to the team's approved semantic-layer metrics. `analyze` checks a definition and
returns every problem at once; `compile_spec` turns a valid one into a model spec with the exact shape of the built-in
`unit_economics` and `funnel` specs, so evaluation, simulation, goal seek, sensitivity and verification need no special
case for custom models and there is no second calculation engine.

What a definition cannot do: run code, reach the file system or network, or change itself. A CSV, a note or a model
response cannot approve a metric: an input bound to a semantic metric is accepted only if that metric is approved,
and a draft or deprecated one is refused with the reason (see `derive_baseline`).
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from app import metric_engine as me
from app import scenario_formula as sf
from app.scenarios import ScenarioError

MAX_INPUTS = 20
MAX_ASSUMPTIONS = 5
MAX_OUTPUTS = 30
MAX_CONSTRAINTS = 10
SEVERITIES = ("error", "warning")
SOURCE_TYPES = ("metric", "formula", "term")
RESERVED = set(sf.FUNCTIONS) | {"true", "false", "none", "null"}
_INPUT_KEYS = {"label", "unit", "min", "max", "good", "source"}
_ASSUMPTION_KEYS = {"label", "unit", "default", "min", "max", "help"}
_OUTPUT_KEYS = {"label", "unit", "formula", "additive", "ratio_of", "reconcile"}
_TOP_KEYS = {"description", "strict_units", "inputs", "assumptions", "outputs", "primary_output", "constraints"}


class ModelDefinitionError(ScenarioError):
    """A definition is invalid. `.problems` lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems[:5]) + (f" (and {len(problems) - 5} more)" if len(problems) > 5 else ""))


class AmbiguousMetric(ScenarioError):
    """A term matches more than one approved metric; the user must choose. `.candidates` lists them."""

    def __init__(self, message: str, candidates: list[dict]):
        self.candidates = candidates
        super().__init__(message)


def _num(v: Any, what: str, errors: list[str], allow_none: bool = False) -> float | None:
    if v is None and allow_none:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        errors.append(f"{what} must be a finite number")
        return None
    return float(v)


def _text(v: Any, what: str, errors: list[str], limit: int, default: str = "") -> str:
    if v is None:
        return default
    if not isinstance(v, str) or len(v) > limit:
        errors.append(f"{what} must be text of at most {limit} characters")
        return default
    return v.strip()


def canonical(defn: dict) -> dict:
    """The definition with defaults filled in, in a fixed shape (what is stored and hashed)."""
    out = {"description": defn.get("description") or "", "strict_units": bool(defn.get("strict_units", False)),
           "inputs": {}, "assumptions": {}, "outputs": {}, "primary_output": defn.get("primary_output"),
           "constraints": []}
    for name, r in (defn.get("inputs") or {}).items():
        r = r if isinstance(r, dict) else {}
        out["inputs"][name] = {"label": r.get("label") or name, "unit": r.get("unit") or "", "min": r.get("min", 0.0),
                               "max": r.get("max"), "good": r.get("good", "up"), "source": r.get("source")}
    for name, r in (defn.get("assumptions") or {}).items():
        r = r if isinstance(r, dict) else {}
        out["assumptions"][name] = {"label": r.get("label") or name, "unit": r.get("unit") or "", "default": r.get("default", 0.0),
                                    "min": r.get("min", 0.0), "max": r.get("max", 1.0), "help": r.get("help") or ""}
    for name, r in (defn.get("outputs") or {}).items():
        r = r if isinstance(r, dict) else {}
        out["outputs"][name] = {"label": r.get("label") or name, "unit": r.get("unit") or "", "formula": r.get("formula"),
                                "additive": bool(r.get("additive", False)), "ratio_of": r.get("ratio_of"),
                                "reconcile": r.get("reconcile")}
    for c in defn.get("constraints") or []:
        c = c if isinstance(c, dict) else {}
        out["constraints"].append({"expr": c.get("expr"), "message": c.get("message") or c.get("expr") or "",
                                   "severity": c.get("severity", "error")})
    return out


def definition_sha256(defn: dict) -> str:
    """A fingerprint of the definition's content: the same model hashes the same whatever its key order, and whether a
    number was written 0 or 0.0 (a valid definition is normalised before it is hashed)."""
    analysis = analyze(defn)
    body = analysis["definition"] if analysis["valid"] else canonical(defn)
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def analyze(defn: Any) -> dict:
    """Check a definition. Returns {"valid", "errors", "warnings", "order", "dependencies", "units", "definition"}."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(defn, dict):
        return {"valid": False, "errors": ["A model definition must be an object"], "warnings": [], "order": [],
                "dependencies": {}, "units": {}, "definition": None}
    unknown = set(defn) - _TOP_KEYS
    if unknown:
        errors.append(f"Unknown field(s) in the definition: {', '.join(sorted(unknown))}")
    for section in ("inputs", "assumptions", "outputs"):
        if section in defn and not isinstance(defn[section], dict):
            errors.append(f"'{section}' must be an object of named entries")
    if errors:
        return {"valid": False, "errors": errors, "warnings": [], "order": [], "dependencies": {}, "units": {}, "definition": None}
    d = canonical(defn)
    strict = d["strict_units"]
    inputs, assumptions, outputs = d["inputs"], d["assumptions"], d["outputs"]

    if not 1 <= len(inputs) <= MAX_INPUTS:
        errors.append(f"A model needs 1 to {MAX_INPUTS} inputs")
    if len(assumptions) > MAX_ASSUMPTIONS:
        errors.append(f"At most {MAX_ASSUMPTIONS} assumptions")
    if not 1 <= len(outputs) <= MAX_OUTPUTS:
        errors.append(f"A model needs 1 to {MAX_OUTPUTS} outputs")
    names = list(inputs) + list(assumptions) + list(outputs)
    for n in names:
        if not isinstance(n, str) or not sf.NAME_RE.match(n):
            errors.append(f"'{n}' is not a valid name: use lowercase letters, digits and underscores, starting with a letter (at most 40)")
        elif n in RESERVED:
            errors.append(f"'{n}' is reserved (it is a function name); choose another")
    for n in {x for x in names if names.count(x) > 1}:
        errors.append(f"The name '{n}' is used more than once; inputs, assumptions and outputs share one list of names")

    for name, r in inputs.items():
        bad = set((defn["inputs"][name] if isinstance(defn["inputs"][name], dict) else {})) - _INPUT_KEYS
        if bad:
            errors.append(f"Input '{name}' has unknown field(s): {', '.join(sorted(bad))}")
        r["label"] = _text(r["label"], f"The label of input '{name}'", errors, 80, name)
        r["unit"] = _text(r["unit"], f"The unit of input '{name}'", errors, 40)
        r["min"] = _num(r["min"], f"The minimum of input '{name}'", errors)
        r["max"] = _num(r["max"], f"The maximum of input '{name}'", errors, allow_none=True)
        if r["min"] is not None and r["max"] is not None and not r["min"] <= r["max"]:
            errors.append(f"Input '{name}': minimum is above maximum")
        if r["good"] not in ("up", "down"):
            errors.append(f"Input '{name}': 'good' must be 'up' or 'down'")
        src = r["source"]
        if src is not None:
            if (not isinstance(src, dict) or src.get("type") not in SOURCE_TYPES
                    or set(src) - {"type", "metric_id", "formula", "term"}):
                errors.append(f"Input '{name}': a source is {{type: metric|formula|term, …}}")
            elif src["type"] == "metric" and (isinstance(src.get("metric_id"), bool) or not isinstance(src.get("metric_id"), int)):
                errors.append(f"Input '{name}': a metric source needs a whole-number metric_id")
            elif src["type"] == "formula":
                try:
                    me.parse_formula(src.get("formula"))
                except me.FormulaError as e:
                    errors.append(f"Input '{name}': its source formula is not valid ({e})")
            elif src["type"] == "term" and not (isinstance(src.get("term"), str) and 0 < len(src["term"].strip()) <= 100):
                errors.append(f"Input '{name}': a term source needs the words of an approved metric")
    for name, r in assumptions.items():
        bad = set((defn["assumptions"][name] if isinstance(defn["assumptions"][name], dict) else {})) - _ASSUMPTION_KEYS
        if bad:
            errors.append(f"Assumption '{name}' has unknown field(s): {', '.join(sorted(bad))}")
        r["label"] = _text(r["label"], f"The label of assumption '{name}'", errors, 80, name)
        r["unit"] = _text(r["unit"], f"The unit of assumption '{name}'", errors, 40)
        r["help"] = _text(r["help"], f"The help text of assumption '{name}'", errors, 200)
        lo, hi, dv = (_num(r[k], f"The {k} of assumption '{name}'", errors) for k in ("min", "max", "default"))
        if None not in (lo, hi, dv):
            if not lo <= hi:
                errors.append(f"Assumption '{name}': minimum is above maximum")
            elif not lo <= dv <= hi:
                errors.append(f"Assumption '{name}': the default {dv:g} is outside {lo:g} to {hi:g}")
        r["min"], r["max"], r["default"] = lo, hi, dv

    parsed: dict[str, sf.Parsed] = {}
    for name, r in outputs.items():
        bad = set((defn["outputs"][name] if isinstance(defn["outputs"][name], dict) else {})) - _OUTPUT_KEYS
        if bad:
            errors.append(f"Output '{name}' has unknown field(s): {', '.join(sorted(bad))}")
        r["label"] = _text(r["label"], f"The label of output '{name}'", errors, 80, name)
        r["unit"] = _text(r["unit"], f"The unit of output '{name}'", errors, 40)
        try:
            parsed[name] = sf.parse(r["formula"])
        except sf.FormulaError as e:
            errors.append(f"The formula for '{name}': {e}")
        ratio = r["ratio_of"]
        if ratio is not None:
            if r["additive"]:
                errors.append(f"Output '{name}' cannot be both additive and a ratio")
            if (not isinstance(ratio, dict) or set(ratio) - {"numerator", "denominator", "scale"}
                    or ratio.get("numerator") not in outputs or ratio.get("denominator") not in outputs
                    or ratio.get("numerator") == ratio.get("denominator")):
                errors.append(f"Output '{name}': ratio_of needs two different outputs as 'numerator' and 'denominator'")
            elif ratio.get("scale") is not None:
                _num(ratio["scale"], f"The scale of the ratio for '{name}'", errors)
        rec = r["reconcile"]
        if rec is not None and (not isinstance(rec, dict) or set(rec) - {"metric_id"} or isinstance(rec.get("metric_id"), bool)
                                or not isinstance(rec.get("metric_id"), int)):
            errors.append(f"Output '{name}': reconcile is {{metric_id: <approved metric id>}}")

    if d["primary_output"] not in outputs:
        errors.append("primary_output must name one of the outputs")

    known = set(inputs) | set(assumptions)
    deps = {n: set(p.names) for n, p in parsed.items()}
    order: list[str] = []
    if not errors:
        try:
            order = sf.order_outputs(deps, known)
        except sf.FormulaError as e:
            errors.append(str(e))

    units: dict[str, dict] = {}
    unit_issues: list[str] = []
    if not errors:
        try:
            for n, r in {**inputs, **assumptions}.items():
                units[n] = sf.parse_unit(r["unit"])
            declared = {n: sf.parse_unit(outputs[n]["unit"]) for n in outputs}
        except sf.FormulaError as e:
            errors.append(str(e))
        else:
            for n in order:
                for p in sf.unit_problems(parsed[n], units, declared[n]):
                    unit_issues.append(f"Output '{n}' {p}")
                units[n] = declared[n]
    constraints = []
    if not errors:
        for i, c in enumerate(d["constraints"]):
            if c["severity"] not in SEVERITIES:
                errors.append(f"Constraint {i + 1}: severity must be 'error' or 'warning'")
                continue
            try:
                op, left, right, cnames = sf.parse_constraint(c["expr"])
            except sf.FormulaError as e:
                errors.append(f"Constraint {i + 1}: {e}")
                continue
            missing = sorted(cnames - set(names))
            if missing:
                errors.append(f"Constraint {i + 1} uses unknown name(s): {', '.join(missing)}")
                continue
            problems: list[str] = []
            a, b = sf.unit_of(left, units, problems), sf.unit_of(right, units, problems)
            if a != b and not (left == ("num", 0.0) or right == ("num", 0.0)):
                unit_issues.append(f"Constraint {i + 1} compares {sf.format_unit(a)} with {sf.format_unit(b)}")
            unit_issues += [f"Constraint {i + 1} {p}" for p in problems]
            constraints.append({"text": c["expr"], "message": c["message"], "severity": c["severity"], "op": op,
                                "left": left, "right": right})
        if len(d["constraints"]) > MAX_CONSTRAINTS:
            errors.append(f"At most {MAX_CONSTRAINTS} constraints")
    (errors if strict else warnings).extend(unit_issues)
    if not errors and not unit_issues and not strict:
        pass
    result = {"valid": not errors, "errors": errors, "warnings": warnings, "order": order,
              "dependencies": {n: sorted(v) for n, v in deps.items()},
              "units": {n: sf.format_unit(u) for n, u in units.items()}, "definition": d}
    if not errors:
        result["_parsed"], result["_constraints"] = parsed, constraints
        probe = probe_warnings(d, parsed, order)
        result["warnings"] += probe
    return result


def probe_warnings(d: dict, parsed: dict, order: list[str]) -> list[str]:
    """Evaluate at a sample point (each input at max(min, 1), assumptions at their defaults) to catch outputs that
    are undefined even there, which usually means a division by something that is always zero."""
    env: dict[str, float] = {k: max(r["min"], 1.0) if r["max"] is None else min(max(r["min"], 1.0), r["max"])
                             for k, r in d["inputs"].items()}
    env.update({k: r["default"] for k, r in d["assumptions"].items()})
    notes = []
    for n in order:
        v = sf.evaluate(parsed[n].tree, env)
        v = float(np.asarray(v, dtype=float))
        env[n] = v
        if not math.isfinite(v):
            notes.append(f"Output '{n}' is undefined at a sample point (every input 1, assumptions at their defaults); "
                         "check for a division by zero")
    return notes


# ---------------------------------------------------------------- compiling --

def _compile_constraints(constraints: list[dict]):
    def violations(env: dict[str, Any]) -> list[dict]:
        out = []
        for c in constraints:
            a, b = float(np.asarray(sf.evaluate(c["left"], env), dtype=float)), float(np.asarray(sf.evaluate(c["right"], env), dtype=float))
            ok = (not (math.isnan(a) or math.isnan(b))) and {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[c["op"]]
            if not ok:
                out.append({"constraint": c["text"], "message": c["message"], "severity": c["severity"],
                            "values": None if math.isnan(a) or math.isnan(b) else [a, b]})
        return out

    def mask(env: dict[str, Any]) -> np.ndarray:
        n = next((np.size(v) for v in env.values() if np.ndim(v) > 0), 1)
        ok = np.ones(n, dtype=bool)
        for c in constraints:
            if c["severity"] != "error":
                continue
            a, b = np.broadcast_to(sf.evaluate(c["left"], env), n), np.broadcast_to(sf.evaluate(c["right"], env), n)
            with np.errstate(invalid="ignore"):
                r = {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b, "==": a == b, "!=": a != b}[c["op"]]
            ok &= r & ~(np.isnan(a) | np.isnan(b))
        return ok
    return violations, mask


def compile_spec(defn: dict, *, key: str, title: str, model_id: int | None = None, version: int | None = None) -> dict:
    """A model spec with the shape of the built-in ones, runnable by every scenario engine."""
    analysis = analyze(defn)
    if not analysis["valid"]:
        raise ModelDefinitionError(analysis["errors"])
    d, parsed, order = analysis["definition"], analysis["_parsed"], analysis["order"]
    violations, mask = _compile_constraints(analysis["_constraints"])

    def compute(i: dict, a: dict) -> dict:
        env: dict[str, float] = {**i, **a}
        for n in order:
            v = float(np.asarray(sf.evaluate(parsed[n].tree, env), dtype=float))
            env[n] = v
        return {n: (env[n] if math.isfinite(env[n]) else None) for n in d["outputs"]}

    def compute_vec(i: dict, a: dict) -> dict:
        env: dict[str, Any] = {**i, **a}
        for n in order:
            env[n] = np.asarray(sf.evaluate(parsed[n].tree, env), dtype=float)
        size = max((np.size(v) for v in i.values()), default=1)
        return {n: np.broadcast_to(np.asarray(env[n], dtype=float), (size,)).copy() if np.ndim(env[n]) == 0 else env[n]
                for n in d["outputs"]}

    inputs = {n: {"label": r["label"], "unit": r["unit"], "min": r["min"], **({"max": r["max"]} if r["max"] is not None else {}),
                  "good": r["good"]} for n, r in d["inputs"].items()}
    return {
        "key": key, "title": title, "inputs": inputs,
        "assumptions": {n: dict(r) for n, r in d["assumptions"].items()},
        "outputs": {n: r["label"] for n, r in d["outputs"].items()},
        "output_units": {n: r["unit"] for n, r in d["outputs"].items()},
        "primary_output": d["primary_output"],
        "formulas": [f"{n} = {d['outputs'][n]['formula']}" for n in d["outputs"]],
        "compute": compute, "compute_vec": compute_vec,
        "constraints": analysis["_constraints"], "constraint_violations": violations, "constraint_mask": mask,
        "additive_outputs": [n for n, r in d["outputs"].items() if r["additive"]],
        "ratio_outputs": {n: r["ratio_of"] for n, r in d["outputs"].items() if r["ratio_of"]},
        "custom": {"model_id": model_id, "version": version, "definition": d, "definition_sha256": definition_sha256(d),
                   "order": order, "units": analysis["units"], "warnings": analysis["warnings"]},
    }


def describe(spec: dict) -> dict:
    """The shape the API and page use to describe any model (built-in or custom)."""
    out = {"model": spec["key"], "title": spec["title"], "inputs": spec["inputs"], "assumptions": spec["assumptions"],
           "outputs": spec["outputs"], "primary_output": spec["primary_output"], "formulas": spec["formulas"]}
    if "custom" in spec:
        c = spec["custom"]
        out["custom"] = {"model_id": c["model_id"], "version": c["version"], "units": c["units"], "warnings": c["warnings"],
                         "constraints": [{"text": x["text"], "message": x["message"], "severity": x["severity"]} for x in spec["constraints"]],
                         "sources": {n: r.get("source") for n, r in c["definition"]["inputs"].items()}}
    return out
