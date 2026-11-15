"""Goal seek: which value of one input makes an output reach a target?

"What price gives me a 20% margin?" "How many units for a profit of 200,000?" The answer is found by
numerical root finding on the *deterministic scenario engine itself* (app/scenarios.py): the function being solved is
`output(input = x) - target`, evaluated by the same formulas, validation and price-elasticity link a scenario uses.
No language model takes part.

Method
------
1. Fix the search range: the input's physical limits, narrowed by the caller's `lower` / `upper`. An input with no
   upper limit gets a wide search ceiling, and the result says so.
2. Evaluate the function on a grid (linear and geometric spacing) and look for sign changes. Every sign change
   brackets a root; each is refined with Brent's method.
3. Every candidate is re-evaluated with the original engine and kept only if it lands on the target within a
   stated tolerance. That discards the false sign changes a division by zero can create.
4. Constraints are checked on each solution. Solutions are never invented: when none exists the result says why
   and reports the range of values the output can reach.

Limits: roots closer together than a grid step can be missed, and a root where the output only touches the target
without crossing it is not detected. The grid size is returned with every result.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import optimize

from app import scenarios
from app.scenarios import ScenarioError

SOLVE_VERSION = "scenario-solve.1"
GRID_LINEAR = 301
GRID_GEOMETRIC = 101
TOLERANCE_REL = 1e-6
UPPER_CEILING_FACTOR = 1000.0
MAX_CONSTRAINTS = 5
OPERATORS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
             "==": lambda a, b: math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)}
DISCLAIMER = ("Goal seek output: the value that reaches the target in the stated model and assumptions. It does not "
              "say customers or costs would actually respond that way.")


class GoalSeekError(ScenarioError):
    """The goal-seek request is invalid; the message says why."""


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GoalSeekError(f"{what} must be a finite number")
    return float(value)


def parse_constraints(spec: dict, constraints: Any) -> list[dict]:
    if constraints in (None, []):
        return []
    if not isinstance(constraints, list) or len(constraints) > MAX_CONSTRAINTS:
        raise GoalSeekError(f"constraints must be a list of at most {MAX_CONSTRAINTS}")
    out = []
    for c in constraints:
        if not isinstance(c, dict) or set(c) - {"metric", "op", "value"} or {"metric", "op", "value"} - set(c):
            raise GoalSeekError("Each constraint needs exactly 'metric', 'op' and 'value'")
        metric, op = c["metric"], c["op"]
        if metric in spec["inputs"]:
            kind, label = "input", spec["inputs"][metric]["label"]
        elif metric in spec["outputs"]:
            kind, label = "output", spec["outputs"][metric]
        else:
            raise GoalSeekError(f"Unknown constraint metric '{metric}'")
        if op not in OPERATORS:
            raise GoalSeekError(f"Unknown constraint operator '{op}'. Use one of: {', '.join(OPERATORS)}")
        out.append({"metric": metric, "kind": kind, "label": label, "op": op, "value": _finite(c["value"], "A constraint value")})
    return out


def _grid(lo: float, hi: float) -> np.ndarray:
    parts = [np.linspace(lo, hi, GRID_LINEAR), np.array([lo, hi])]
    start = lo if lo > 0 else hi * 1e-6
    if hi > 0 and start > 0 and hi / start > 10:
        parts.append(np.geomspace(start, hi, GRID_GEOMETRIC))
    return np.unique(np.concatenate(parts))


def goal_seek(model: "str | dict", baseline: dict, target_output: str, target_value: float, solve_for: str,
              adjustments: dict | None = None, assumptions: dict | None = None, lower: float | None = None,
              upper: float | None = None, constraints: list | None = None, verify: bool = True) -> dict:
    spec = scenarios._model(model)  # noqa: SLF001
    baseline = scenarios.validate_inputs(model, baseline)
    assumptions = scenarios.validate_assumptions(model, assumptions)
    adjustments = dict(adjustments or {})
    if target_output not in spec["outputs"]:
        raise GoalSeekError(f"Unknown target metric '{target_output}'. Available: {', '.join(spec['outputs'])}")
    target = _finite(target_value, "The target value")
    if solve_for not in spec["inputs"]:
        raise GoalSeekError(f"Unknown input '{solve_for}'. Available: {', '.join(spec['inputs'])}")
    if solve_for in adjustments:
        raise GoalSeekError(f"'{solve_for}' is the variable being solved for; remove its adjustment")
    parsed = parse_constraints(spec, constraints)
    scenarios.apply_adjustments(model, baseline, adjustments, assumptions)        # validates the other adjustments

    rule = spec["inputs"][solve_for]
    rule_lo, rule_hi = rule.get("min", 0.0), rule.get("max", math.inf)
    base_x = baseline[solve_for]
    lo = rule_lo if lower is None else _finite(lower, "lower")
    upper_assumed = upper is None and not math.isfinite(rule_hi)
    hi = (rule_hi if math.isfinite(rule_hi) else max(abs(base_x), 1.0) * UPPER_CEILING_FACTOR) if upper is None \
        else _finite(upper, "upper")
    if lo < rule_lo or hi > rule_hi:
        raise GoalSeekError(f"The search range for {rule['label']} must stay within its allowed range "
                            f"{rule_lo:g} to {rule_hi:g}")
    if not lo < hi:
        raise GoalSeekError("lower must be below upper")

    tol = TOLERANCE_REL * max(1.0, abs(target))

    def at(x: float) -> tuple[float, dict | None, dict | None]:
        try:
            inputs = scenarios.apply_adjustments(model, baseline, {**adjustments, solve_for: {"type": "set", "value": float(x)}},
                                                 assumptions)
        except ScenarioError:
            return math.nan, None, None
        outputs = scenarios._outputs(model, inputs, assumptions)                   # noqa: SLF001
        check = spec.get("constraint_violations")
        if check is not None and any(v["severity"] == "error" for v in check(
                {**inputs, **assumptions, **{k: (math.nan if o is None else o) for k, o in outputs.items()}})):
            return math.nan, None, None                    # the model does not allow this combination
        y = outputs[target_output]
        return (math.nan if y is None else float(y)), inputs, outputs

    xs = _grid(lo, hi)
    ys = np.array([at(x)[0] for x in xs])
    finite = np.isfinite(ys)
    base_out = scenarios._outputs(model, scenarios.apply_adjustments(model, baseline, adjustments, assumptions),  # noqa: SLF001
                                  assumptions)

    roots: list[float] = []
    failures: list[str] = []
    f = ys - target
    for i in range(len(xs)):
        if finite[i] and f[i] == 0.0:
            roots.append(float(xs[i]))
        if i + 1 < len(xs) and finite[i] and finite[i + 1] and f[i] * f[i + 1] < 0:
            try:
                roots.append(float(optimize.brentq(lambda x: at(x)[0] - target, xs[i], xs[i + 1], xtol=1e-14,
                                                   rtol=4 * np.finfo(float).eps, maxiter=200)))
            except (ValueError, RuntimeError) as e:
                failures.append(f"between {xs[i]:g} and {xs[i + 1]:g}: {type(e).__name__}")

    solutions, discarded = [], 0
    for x in sorted(roots):
        y, inputs, outputs = at(x)
        if inputs is None or not math.isfinite(y) or abs(y - target) > tol:
            discarded += 1                                  # a pole or an undefined point, not a solution
            continue
        if solutions and math.isclose(x, solutions[-1]["value"], rel_tol=1e-9, abs_tol=1e-12):
            continue
        checks = []
        for c in parsed:
            actual = (inputs if c["kind"] == "input" else outputs).get(c["metric"])
            checks.append({**c, "actual": actual, "ok": actual is not None and bool(OPERATORS[c["op"]](actual, c["value"]))})
        solutions.append({"value": x, "achieved": y, "error": y - target, "inputs": inputs, "outputs": outputs,
                          "change_vs_baseline": {"abs": x - base_x, "pct": (x - base_x) / abs(base_x) * 100 if base_x else None},
                          "constraints": checks, "feasible": all(c["ok"] for c in checks)})
    feasible = [i for i, s in enumerate(solutions) if s["feasible"]]
    recommended = min(feasible, key=lambda i: abs(solutions[i]["value"] - base_x)) if feasible else None

    achievable = ({"min": float(ys[finite].min()), "max": float(ys[finite].max())} if finite.any() else None)
    may_exist_beyond = False
    status, message = _status(spec, target_output, target, rule, achievable, finite, ys, solutions, feasible, failures,
                              tol)
    if status == "no_solution" and upper_assumed:
        # "Still moving towards the target at the ceiling" is not enough: a margin flattens towards 100% and never
        # reaches 150%. A curve that is flattening (its slope over the last tenth of the range well below the slope
        # over the tenth before) is treated as heading for an asymptote; one that keeps its slope may get there.
        y80, y90, y100 = at(lo + 0.8 * (hi - lo))[0], at(lo + 0.9 * (hi - lo))[0], float(ys[-1])
        if all(math.isfinite(v) for v in (y80, y90, y100)):
            s_prev, s_last, gap = y90 - y80, y100 - y90, target - y100
            may_exist_beyond = bool(s_last != 0 and s_prev != 0 and math.copysign(1, s_last) == math.copysign(1, gap)
                                    and math.copysign(1, s_prev) == math.copysign(1, s_last)
                                    and abs(s_last) >= 0.9 * abs(s_prev))
        if may_exist_beyond:
            message += (f" The output is still moving towards the target at the top of the search range "
                        f"({hi:,.6g}), so a solution may exist beyond it; widen the upper limit to look further.")
    label = spec["outputs"][target_output]
    if status in ("solved", "multiple_solutions") and recommended is not None:
        best = solutions[recommended]
        message = (f"{label} reaches {target:,.6g} when {rule['label'].lower()} is {best['value']:,.6g} "
                   f"({_pct(best['change_vs_baseline']['pct'])} from {base_x:,.6g}).")
        if status == "multiple_solutions":
            others = ", ".join(f"{solutions[i]['value']:,.6g}" for i in feasible if i != recommended)
            message += (f" {len(feasible)} values satisfy the target; this one is closest to the current value. "
                        f"The others are {others}.")
    result = {
        "kind": "goal_seek", "disclaimer": DISCLAIMER, "engine_version": SOLVE_VERSION, "model": scenarios.model_key(model),
        "title": spec["title"], "status": status, "message": message,
        "target": {"output": target_output, "label": label, "value": target,
                   "baseline_value": base_out.get(target_output)},
        "solve_for": {"input": solve_for, "label": rule["label"], "unit": rule["unit"], "baseline_value": base_x},
        "search_range": {"lower": lo, "upper": hi, "upper_assumed": upper_assumed, "grid_points": int(len(xs))},
        "tolerance": tol, "solutions": solutions, "recommended_index": recommended,
        "achievable_range": achievable, "may_exist_beyond_search_range": may_exist_beyond,
        "curve": {"x": xs.tolist(), "y": [float(v) if math.isfinite(v) else None for v in ys]},
        "constraints": parsed, "adjustments": adjustments, "assumptions": assumptions,
        "baseline_inputs": baseline, "failures": failures, "discarded_candidates": discarded,
        "disclosures": _disclosures(scenarios.model_key(model), assumptions, adjustments, solve_for, upper_assumed, hi, len(xs)),
    }
    if verify:
        from app import scenario_verify
        result["verification"] = scenario_verify.safe(scenario_verify.verify_goal_seek, result, spec)
    return result


def _pct(p: float | None) -> str:
    return "a change from zero" if p is None else f"{p:+.2f}%"


def _status(spec, target_output, target, rule, achievable, finite, ys, solutions, feasible, failures, tol):
    label = spec["outputs"][target_output]
    if not finite.any():
        return "undefined", (f"{label} is undefined at every value in the search range (it divides by zero, or the "
                             "model's constraints rule every value out), so no target can be reached.")
    span = float(ys[finite].max() - ys[finite].min())
    if span <= 1e-12 * max(1.0, abs(float(ys[finite].mean()))):
        at_target = abs(float(ys[finite][0]) - target) <= tol
        return "no_dependence", (f"{label} does not change with {rule['label'].lower()}, so no value of it can move "
                                 + ("the result; it already equals the target." if at_target else
                                    f"the result towards the target. It stays at {float(ys[finite][0]):,.6g}."))
    if solutions:
        if not feasible:
            return "no_feasible_solution", (f"{len(solutions)} value(s) reach the target, but each breaks a constraint you "
                                            "set. Relax a constraint or change the target.")
        return ("solved" if len(feasible) == 1 else "multiple_solutions"), ""
    if failures:
        return "convergence_failure", ("The solver found where the target is crossed but could not narrow it down "
                                       "(" + "; ".join(failures[:3]) + "). No solution is reported.")
    lo, hi = achievable["min"], achievable["max"]
    side = "above" if target > hi else "below" if target < lo else "between values the output reaches but never equals"
    return "no_solution", (f"No value of {rule['label'].lower()} in the search range gives {label.lower()} = "
                           f"{target:,.6g}. Over that range {label.lower()} only reaches {lo:,.6g} to {hi:,.6g}; "
                           f"the target is {side}.")


def _disclosures(model, assumptions, adjustments, solve_for, upper_assumed, hi, grid_points) -> list[str]:
    notes = ["The answer is exact for this model: it comes from the same formulas as a scenario, solved numerically "
             f"on a {grid_points}-point search grid with Brent refinement and then re-evaluated."]
    if upper_assumed:
        notes.append(f"{solve_for} has no physical upper limit, so the search stopped at an assumed ceiling of {hi:,.6g}.")
    if (model == "unit_economics" and solve_for == "price" and assumptions.get("price_elasticity")
            and "volume" not in adjustments):
        notes.append(f"Volume responds to each trial price through the stated price elasticity of "
                     f"{assumptions['price_elasticity']:g}.")
    return notes
