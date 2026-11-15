"""Independent verification of what-if results.

The scenario engines (app/scenarios.py, scenario_sim.py, scenario_solve.py) compute with numpy, scipy and the
shared model definitions. This module recomputes their results a second way, with plain Python only (`math`, lists,
loops) and its own copy of the model formulas, and compares. Two programs written separately that agree are evidence
that neither carries a bug the other does not.

What is and is not verified is stated in every result:

* verified: the arithmetic (every output from its inputs), the Shapley contributions (recomputed with the subset
  formula, not the permutation average the engine uses), the sensitivity table, break-even points, the simulation's
  draws, summary statistics, probabilities and histogram, that the draws follow the stated distributions and
  correlation, that a seed reproduces the run, and that a goal-seek solution reaches its target (and that no solution
  was missed, by an independent bisection scan).
* not verified, because it cannot be: that the inputs, ranges and assumptions the user supplied are realistic.

A check that did not run is never reported as passed. Status is `unverified` if any required check fails or did not
run, otherwise `verified`; the limitations are listed beside it, never folded into the status.
"""
from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Callable

import numpy as np

from app import scenarios

logger = logging.getLogger(__name__)
REL = 1e-9
ABS = 1e-9
SCOPE = ("Checked: the calculations were recomputed independently and agree. Not checked: whether the inputs, "
         "ranges and assumptions you supplied describe your business.")


def safe(fn: Callable[..., dict], *args: Any) -> dict:
    """Run a verifier; if it cannot run, the result is `unverified` with the reason, never silently verified."""
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - a verifier bug must not pass a result, nor take the request down
        logger.exception("scenario verification could not run")
        return {"status": "unverified", "scope": SCOPE, "checks": [], "limitations": [],
                "error": f"Verification could not run ({type(e).__name__}); this result is not verified."}


# ---------------------------------------------------- reference model (plain Python) --

def _ref_names(node) -> set:
    import ast
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _ref_eval(node, env: dict):
    """One formula node in plain floats. Division by zero, the square root or log of a number outside their domain,
    and overflow give None, and None propagates. Written separately from app/scenario_formula.py."""
    import ast
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.UnaryOp):
        v = _ref_eval(node.operand, env)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp):
        a, b = _ref_eval(node.left, env), (None if isinstance(node.op, ast.Pow) else _ref_eval(node.right, env))
        if isinstance(node.op, ast.Pow):
            e = node.right
            power = -float(e.operand.value) if isinstance(e, ast.UnaryOp) else float(e.value)
            if a is None:
                return None
            try:
                r = a ** power
            except (ZeroDivisionError, OverflowError):
                return None
            return None if isinstance(r, complex) or not math.isfinite(r) else r
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        return None if b == 0 else a / b
    if isinstance(node, ast.Call):
        args = [_ref_eval(a, env) for a in node.args]
        if any(v is None for v in args):
            return None
        fn = node.func.id
        try:
            if fn == "abs":
                return abs(args[0])
            if fn == "sqrt":
                return None if args[0] < 0 else math.sqrt(args[0])
            if fn == "log":
                return None if args[0] <= 0 else math.log(args[0])
            if fn == "exp":
                return math.exp(args[0])
            if fn == "floor":
                return float(math.floor(args[0]))
            if fn == "ceil":
                return float(math.ceil(args[0]))
            return min(args) if fn == "min" else max(args)
        except OverflowError:
            return None
    return None


def ref_custom(defn: dict, inputs: dict, assumptions: dict) -> tuple[dict, bool]:
    """A custom model's outputs, and whether every error-level constraint holds, from the definition's own text.
    Outputs are computed by repeated passes (each output once everything it uses is known), so no ordering helper is
    shared with the engine."""
    import ast
    env: dict = {**inputs, **assumptions}
    trees = {n: ast.parse(r["formula"], mode="eval").body for n, r in defn["outputs"].items()}
    pending = dict(trees)
    while pending:
        ready = [n for n, t in pending.items() if all(x in env or x not in trees for x in _ref_names(t))]
        if not ready:
            raise ValueError("circular definition")
        for n in ready:
            env[n] = _ref_eval(pending.pop(n), env)
    ok = True
    for c in defn["constraints"]:
        if c["severity"] != "error":
            continue
        node = ast.parse(c["expr"], mode="eval").body
        a, b = _ref_eval(node.left, env), _ref_eval(node.comparators[0], env)
        op = type(node.ops[0]).__name__
        holds = (a is not None and b is not None and {"Lt": a < b, "LtE": a <= b, "Gt": a > b, "GtE": a >= b,
                                                      "Eq": a == b, "NotEq": a != b}[op])
        ok = ok and holds
    return {n: env[n] for n in defn["outputs"]}, ok


def ref_outputs(model: str, i: dict, spec: dict | None = None, assumptions: dict | None = None) -> dict:
    """The model's formulas written separately from scenarios.py, in plain floats."""
    if spec is not None and "custom" in spec:
        return ref_custom(spec["custom"]["definition"], i, assumptions or {})[0]
    if model == "unit_economics":
        price, volume, cost, fixed = i["price"], i["volume"], i["unit_cost"], i["fixed_cost"]
        revenue = price * volume
        profit = (price - cost) * volume - fixed
        return {"revenue": revenue, "variable_cost": cost * volume, "gross_profit": (price - cost) * volume,
                "profit": profit, "margin_pct": None if revenue == 0 else 100.0 * profit / revenue}
    orders = i["visitors"] * i["conversion_rate"]
    revenue = orders * i["average_order_value"]
    acquisition = i["cost_per_visitor"] * i["visitors"]
    return {"orders": orders, "revenue": revenue, "acquisition_cost": acquisition, "profit": revenue - acquisition,
            "cost_per_order": None if orders == 0 else acquisition / orders}


def ref_apply(model: str, baseline: dict, adjustments: dict, assumptions: dict) -> dict:
    out = dict(baseline)
    for name, adj in adjustments.items():
        v = float(adj["value"])
        out[name] = (baseline[name] * (1 + v / 100.0) if adj["type"] == "pct"
                     else baseline[name] + v if adj["type"] == "abs" else v)
    e = assumptions.get("price_elasticity", 0.0)
    if model == "unit_economics" and e and baseline["price"] and "volume" not in adjustments:
        out["volume"] = max(0.0, baseline["volume"] * (1 + e * (out["price"] / baseline["price"] - 1.0)))
    return out


def _close(a: Any, b: Any, rel: float = REL, abs_: float = ABS) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def ref_percentile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between order statistics, the definition numpy calls 'linear'."""
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def ref_shapley(f: Callable[[dict], float], base: dict, scenario: dict) -> dict[str, float]:
    """Shapley values by the subset formula: sum over coalitions S of |S|!(n-|S|-1)!/n! x marginal contribution."""
    changed = [k for k in base if not math.isclose(base[k], scenario[k], rel_tol=1e-12, abs_tol=1e-12)]
    n = len(changed)
    out = {}
    for k in changed:
        others = [c for c in changed if c != k]
        total = 0.0
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in itertools.combinations(others, size):
                without = {**base, **{c: scenario[c] for c in subset}}
                total += weight * (f({**without, k: scenario[k]}) - f(without))
        out[k] = total
    return out


def _check(name: str, ok: bool | None, detail: str, required: bool = True) -> dict:
    status = "not_run" if ok is None else "pass" if ok else "fail"
    return {"check": name, "status": status, "required": required, "detail": detail}


def _verdict(checks: list[dict], limitations: list[str]) -> dict:
    bad = [c for c in checks if c["required"] and c["status"] in ("fail", "not_run")]
    return {"status": "unverified" if bad else "verified", "scope": SCOPE, "checks": checks,
            "limitations": limitations, "method": "recomputed with plain Python and separately written formulas"}


# --------------------------------------------------------------------- evaluate --

def verify_evaluation(result: dict, spec: dict | None = None) -> dict:
    model = result["model"]
    spec = spec or scenarios._model(model)  # noqa: SLF001
    base_in = result["baseline"]["inputs"]
    assumptions = result["assumptions"]
    primary = result["primary_output"]
    checks: list[dict] = []

    ref_base = ref_outputs(model, base_in, spec, assumptions)
    bad = [k for k, v in ref_base.items() if not _close(v, result["baseline"]["outputs"].get(k))]
    checks.append(_check("baseline_outputs", not bad, "every baseline output matches an independent recomputation"
                         if not bad else f"differs for: {', '.join(bad)}"))

    bad_in, bad_out, bad_delta = [], [], []
    for name, s in result["scenarios"].items():
        inputs = ref_apply(model, base_in, s["adjustments"], assumptions)
        if any(not _close(inputs[k], s["inputs"][k]) for k in inputs):
            bad_in.append(name)
        ref_out = ref_outputs(model, inputs, spec, assumptions)
        if any(not _close(ref_out[k], s["outputs"].get(k)) for k in ref_out):
            bad_out.append(name)
        for k, v in ref_out.items():
            d, b = s["delta_vs_baseline"].get(k), ref_base[k]
            want = None if v is None or b is None else {"abs": v - b, "pct": (v - b) / abs(b) * 100 if b else None}
            if (want is None) != (d is None) or (want and not (_close(want["abs"], d["abs"]) and _close(want["pct"], d["pct"]))):
                bad_delta.append(f"{name}.{k}")
    n = len(result["scenarios"])
    checks.append(_check("scenario_inputs", not bad_in, f"{n} scenario(s): adjustments reapplied independently, "
                         "including the elasticity link" if not bad_in else f"differs in: {', '.join(bad_in)}"))
    checks.append(_check("scenario_outputs", not bad_out, f"{n} scenario(s) recomputed from their inputs"
                         if not bad_out else f"differs in: {', '.join(bad_out)}"))
    checks.append(_check("deltas_vs_baseline", not bad_delta, "absolute and percentage differences recomputed"
                         if not bad_delta else f"differs: {', '.join(bad_delta[:5])}"))

    f = lambda inputs: ref_outputs(model, inputs, spec, assumptions)[primary] or 0.0  # noqa: E731
    bad_sh = []
    for name, s in result["scenarios"].items():
        got = s["contributions"]["by_input"]
        want = ref_shapley(f, base_in, s["inputs"])
        total = sum(got.values())
        change = s["delta_vs_baseline"][primary]["abs"] if s["delta_vs_baseline"].get(primary) else 0.0
        scale = max(1.0, abs(change))
        if (set(got) != set(want) or any(not _close(got[k], want[k], 1e-9, 1e-9 * scale) for k in want)
                or not _close(total, change, 1e-9, 1e-9 * scale)):
            bad_sh.append(name)
    checks.append(_check("shapley_contributions", not bad_sh,
                         "contributions equal the subset-formula Shapley values and sum to the total change"
                         if not bad_sh else f"differs in: {', '.join(bad_sh)}"))

    sens = result["sensitivity"]
    pct = sens["pct"]
    bad_s = []
    for row in sens["rows"]:
        rule = spec["inputs"][row["input"]]
        lo_in = max(rule.get("min", -math.inf), base_in[row["input"]] * (1 - pct / 100))
        hi_in = min(rule.get("max", math.inf), base_in[row["input"]] * (1 + pct / 100))
        low, high = f({**base_in, row["input"]: lo_in}), f({**base_in, row["input"]: hi_in})
        if not (_close(low, row["low"]) and _close(high, row["high"]) and _close(abs(high - low), row["range"])):
            bad_s.append(row["input"])
    ranges = [r["range"] for r in sens["rows"]]
    checks.append(_check("sensitivity_table", not bad_s and ranges == sorted(ranges, reverse=True),
                         "every low/high outcome recomputed and ranked correctly"
                         if not bad_s else f"differs for: {', '.join(bad_s)}"))

    if result.get("break_even"):
        bad_be = []
        for label, be, inputs in [("baseline", result["break_even"]["baseline"], base_in)] + [
                (k, result["break_even"]["scenarios"][k], v["inputs"]) for k, v in result["scenarios"].items()]:
            if not _break_even_ok(model, be, inputs):
                bad_be.append(label)
        checks.append(_check("break_even", not bad_be, "profit is zero at each break-even point and the distances agree"
                             if not bad_be else f"differs for: {', '.join(bad_be)}"))
    if "custom" in spec:
        defn = spec["custom"]["definition"]
        broken = [name for name, s_ in [("baseline", {"inputs": base_in})] + list(result["scenarios"].items())
                  if not ref_custom(defn, s_["inputs"], assumptions)[1]]
        checks.append(_check("model_constraints", not broken, "every error-level constraint holds, rechecked independently"
                             if not broken else f"constraint broken in: {', '.join(broken)}"))
    return _verdict(checks, ["Inputs and assumptions are as supplied; the model is arithmetic, not a forecast."])


def _break_even_ok(model: str, be: dict, i: dict) -> bool:
    tol = 1e-6
    if model == "unit_economics":
        contribution = i["price"] - i["unit_cost"]
        if not _close(be["contribution_per_unit"], contribution):
            return False
        if contribution > 0:
            q = i["fixed_cost"] / contribution
            at_q = contribution * q - i["fixed_cost"]
            return (_close(be["break_even_quantity"], q) and abs(at_q) <= tol * max(1.0, i["fixed_cost"])
                    and _close(be["break_even_revenue"], q * i["price"])
                    and _close(be["distance_units"], i["volume"] - q)
                    and _close(be["additional_units_needed"], max(0.0, q - i["volume"])))
        return be["status"] in ("never", "flat") and "break_even_quantity" not in be
    per_visitor = i["conversion_rate"] * i["average_order_value"] - i["cost_per_visitor"]
    if not _close(be["profit_per_visitor"], per_visitor):
        return False
    ok = _close(be["break_even_cost_per_visitor"], i["conversion_rate"] * i["average_order_value"])
    if be["break_even_conversion_rate"] is not None:
        ok = ok and abs(be["break_even_conversion_rate"] * i["average_order_value"] - i["cost_per_visitor"]) <= tol * max(1.0, i["cost_per_visitor"])
    if be["break_even_average_order_value"] is not None:
        ok = ok and abs(i["conversion_rate"] * be["break_even_average_order_value"] - i["cost_per_visitor"]) <= tol * max(1.0, i["cost_per_visitor"])
    return ok and (be["status"] == "profitable_at_any_scale") == (per_visitor > 0)


# ------------------------------------------------------------------- simulation --

def verify_simulation(result: dict, raw: dict, rerun: Callable[[], dict], spec: dict | None = None) -> dict:
    from scipy import stats as st

    from app import scenario_sim

    model = result["model"]
    spec = spec or scenarios._model(model)  # noqa: SLF001
    cfg, tgt = result["config"], result["config"]["target_output"]
    samples, outputs = raw["samples"], raw["outputs"]
    n = cfg["iterations"]
    checks: list[dict] = []

    idx = np.unique(np.linspace(0, n - 1, min(n, 300)).astype(int))
    bad = 0
    custom = "custom" in spec
    for j in idx:
        draw = {k: float(v[j]) for k, v in samples.items()}
        ref = ref_outputs(model, draw, spec, cfg["assumptions"])
        allowed = ref_custom(spec["custom"]["definition"], draw, cfg["assumptions"])[1] if custom else True
        for k, want in ref.items():
            got = float(outputs[k][j])
            if not allowed:                      # a draw that breaks a constraint is left out, so it must be NaN
                bad += 0 if math.isnan(got) else 1
            elif not _close(want, None if math.isnan(got) else got, 1e-9, 1e-9):
                bad += 1
    checks.append(_check("draw_arithmetic", bad == 0, f"{len(idx)} draws spread across the run recomputed with plain "
                         "Python: every output matches" if bad == 0 else f"{bad} output value(s) differ"))

    y = [float(v) for v in outputs[tgt] if not math.isnan(float(v))]
    ys = sorted(y)
    summ = result["outputs"][tgt]
    m = len(ys)
    if m:
        mean = math.fsum(ys) / m
        std = math.sqrt(math.fsum((v - mean) ** 2 for v in ys) / (m - 1)) if m > 1 else 0.0
        want = {"mean": mean, "median": ref_percentile(ys, 50), "std": std, "min": ys[0], "max": ys[-1],
                "p5": ref_percentile(ys, 5), "p25": ref_percentile(ys, 25), "p75": ref_percentile(ys, 75),
                "p95": ref_percentile(ys, 95)}
        scale = max(1.0, abs(mean), std)
        diff = [k for k, v in want.items() if not _close(v, summ[k], 1e-9, 1e-9 * scale)]
        ok = not diff and summ["n_valid"] == m
    else:
        diff, ok = ["n_valid"], summ["n_valid"] == 0
    checks.append(_check("summary_statistics", ok, "mean, median, standard deviation, min, max and the 5/25/75/95th "
                         "percentiles recomputed from the raw draws" if ok else f"differs: {', '.join(diff)}"))

    other = []
    for k, arr in outputs.items():
        vals = [float(v) for v in arr if not math.isnan(float(v))]
        s = result["outputs"][k]
        if len(vals) != s["n_valid"] or (vals and not _close(math.fsum(vals) / len(vals), s["mean"], 1e-9, 1e-9 * max(1.0, abs(s["mean"])))):
            other.append(k)
    checks.append(_check("all_output_means", not other, "mean and valid-draw count recomputed for every output"
                         if not other else f"differs: {', '.join(other)}"))

    p = result["target"]["probabilities"]
    base_v = result["target"]["baseline_value"]
    counts = {"positive": sum(v > 0 for v in y), "negative": sum(v < 0 for v in y),
              "exceeds_baseline": sum(v > base_v for v in y) if base_v is not None else None}
    tv = cfg["target_value"]
    counts["reaches_target"] = None if tv is None else (sum(v >= tv for v in y) if cfg["target_direction"] == "at_least"
                                                        else sum(v <= tv for v in y))
    wrong = [k for k, c in counts.items() if (p[k] is None) != (c is None) or (c is not None and (p[k]["count"] != c or not _close(p[k]["probability"], c / m if m else None)))]
    checks.append(_check("probabilities", not wrong, "each probability recounted draw by draw"
                         if not wrong else f"differs: {', '.join(wrong)}"))

    h = result["target"]["histogram"]
    ok_h = sum(h["counts"]) == m and all(b > a for a, b in zip(h["edges"], h["edges"][1:])) if m else not h["counts"]
    checks.append(_check("histogram", ok_h, "bin counts add up to the valid draws, edges increase"
                         if ok_h else "bin counts or edges are inconsistent"))

    ks_bad, worst = [], 1.0
    for name, dist in cfg["distributions"].items():
        rule = spec["inputs"][name]
        frozen = scenario_sim.frozen(dist)
        lo, hi = rule.get("min", -math.inf), rule.get("max", math.inf)
        f_lo = float(frozen.cdf(lo)) if math.isfinite(lo) else 0.0
        f_hi = float(frozen.cdf(hi)) if math.isfinite(hi) else 1.0
        u = np.clip((frozen.cdf(samples[name]) - f_lo) / (f_hi - f_lo), 0.0, 1.0)
        pvalue = float(st.kstest(u, "uniform").pvalue)
        worst = min(worst, pvalue)
        if pvalue < 1e-4:
            ks_bad.append(name)
        inside = bool(np.all((samples[name] >= lo) & (samples[name] <= hi)))
        if not inside:
            ks_bad.append(name)
    checks.append(_check("input_distributions", not ks_bad, f"draws follow the stated distributions inside the "
                         f"allowed ranges (Kolmogorov–Smirnov, smallest p = {worst:.3g})" if not ks_bad
                         else f"do not follow the stated distribution: {', '.join(ks_bad)}"))

    corr = cfg["correlation"]
    if corr is None:
        checks.append({"check": "correlation", "status": "not_applicable", "required": False,
                       "detail": "no correlation was requested"})
    else:
        names, mat = corr["inputs"], corr["matrix"]
        tol = 0.02 + 5 / math.sqrt(n)
        off = []
        for a, b in itertools.combinations(range(len(names)), 2):
            rho = float(st.spearmanr(samples[names[a]], samples[names[b]])[0])
            want_rho = 6 / math.pi * math.asin(mat[a][b] / 2)
            if abs(rho - want_rho) > tol:
                off.append(f"{names[a]}~{names[b]} ({rho:.3f} vs {want_rho:.3f})")
        checks.append(_check("correlation", not off, "rank correlations match the requested matrix "
                             "(Gaussian copula relation)" if not off else "off target: " + "; ".join(off)))

    again = rerun()
    same = all(np.array_equal(again["outputs"][k], outputs[k], equal_nan=True) for k in outputs)
    checks.append(_check("reproducible_with_seed", same, f"re-running with seed {cfg['seed']} gives identical draws"
                         if same else "a second run with the same seed gave different results"))

    follows_price = (model == "unit_economics" and bool(cfg["assumptions"].get("price_elasticity"))
                     and "price" in cfg["distributions"] and "volume" not in cfg["adjustments"])
    consts = [k for k in result["fixed_inputs"] if not np.all(samples[k] == samples[k][0])
              and not (k == "volume" and follows_price)]
    checks.append(_check("fixed_inputs_constant", not consts, "inputs without a distribution never varied"
                         if not consts else f"varied unexpectedly: {', '.join(consts)}"))

    checks.append(_check("input_ranges_supported_by_data", None, "the distributions are assumptions you supplied; "
                         "nothing here checks them against your data", required=False))
    return _verdict(checks, ["The distributions are user-stated assumptions, not estimates from data.",
                             "Probabilities carry sampling error; their standard errors are reported beside them."])


# -------------------------------------------------------------------- goal seek --

def _bisect(g: Callable[[float], float], a: float, b: float) -> float:
    fa = g(a)
    for _ in range(200):
        mid = (a + b) / 2
        fm = g(mid)
        if fm == 0 or (b - a) < 1e-15 * max(1.0, abs(mid)):
            return mid
        if fa * fm < 0:
            b = mid
        else:
            a, fa = mid, fm
    return (a + b) / 2


def verify_goal_seek(result: dict, spec: dict | None = None) -> dict:
    model, cfg = result["model"], result
    spec = spec or scenarios._model(model)  # noqa: SLF001
    tgt, target = result["target"]["output"], result["target"]["value"]
    solve_for = result["solve_for"]["input"]
    tol = result["tolerance"]
    base, assumptions, adjustments = result["baseline_inputs"], result["assumptions"], result["adjustments"]
    lo, hi = result["search_range"]["lower"], result["search_range"]["upper"]
    checks: list[dict] = []

    def ref_y(x: float) -> float:
        inputs = ref_apply(model, base, {**adjustments, solve_for: {"type": "set", "value": x}}, assumptions)
        if "custom" in spec and not ref_custom(spec["custom"]["definition"], inputs, assumptions)[1]:
            return math.nan                      # the model's constraints do not allow this value
        out = ref_outputs(model, inputs, spec, assumptions)[tgt]
        return math.nan if out is None else out

    sols = result["solutions"]
    if sols:
        miss = [s["value"] for s in sols if abs(ref_y(s["value"]) - target) > tol or not _close(ref_y(s["value"]), s["achieved"])]
        checks.append(_check("target_reached", not miss, f"{len(sols)} solution(s) re-evaluated independently: each "
                             f"reaches the target within {tol:.3g}" if not miss else f"does not reach the target at: {miss}"))
        out_of = [s["value"] for s in sols if not lo - 1e-12 <= s["value"] <= hi + 1e-12]
        checks.append(_check("within_search_range", not out_of, "every solution lies inside the search range"
                             if not out_of else f"outside the range: {out_of}"))
        if result["constraints"]:
            wrong = []
            for s in sols:
                ref_o = ref_outputs(model, ref_apply(model, base, {**adjustments, solve_for: {"type": "set", "value": s["value"]}}, assumptions), spec, assumptions)
                ref_i = ref_apply(model, base, {**adjustments, solve_for: {"type": "set", "value": s["value"]}}, assumptions)
                for c in s["constraints"]:
                    actual = (ref_i if c["kind"] == "input" else ref_o).get(c["metric"])
                    ok = actual is not None and {">=": actual >= c["value"], "<=": actual <= c["value"], ">": actual > c["value"],
                                                  "<": actual < c["value"], "==": math.isclose(actual, c["value"], rel_tol=1e-9, abs_tol=1e-9)}[c["op"]]
                    if ok != c["ok"]:
                        wrong.append((s["value"], c["metric"]))
            checks.append(_check("constraints", not wrong, "constraint results recomputed independently"
                                 if not wrong else f"disagree: {wrong}"))
    status = result["status"]
    if status in ("no_dependence", "undefined"):
        xs = [lo + (hi - lo) * k / 400 for k in range(401)]
        vals = [ref_y(x) for x in xs]
        fin = [v for v in vals if not math.isnan(v)]
        confirmed = (not fin) if status == "undefined" else (bool(fin) and max(fin) - min(fin) <= 1e-9 * max(1.0, abs(fin[0])))
        checks.append(_check("claim_confirmed", confirmed, f"the '{status}' result holds on an independent scan"
                             if confirmed else "an independent scan disagrees with the reported status"))
    else:
        xs = sorted({lo + (hi - lo) * k / 4000 for k in range(4001)} |
                    ({lo + (hi - lo) * (k / 1500) ** 3 for k in range(1501)}))
        f = [ref_y(x) - target for x in xs]
        reported = [s["value"] for s in sols]
        missing = []
        for a, b, fa, fb in zip(xs, xs[1:], f, f[1:]):
            if math.isnan(fa) or math.isnan(fb) or fa * fb >= 0:
                continue
            if any(a - 1e-9 <= r <= b + 1e-9 for r in reported):
                continue
            root = _bisect(lambda x: ref_y(x) - target, a, b)
            if abs(ref_y(root) - target) <= tol and not any(math.isclose(root, r, rel_tol=1e-6) for r in reported):
                missing.append(root)
        checks.append(_check("no_solution_missed", not missing, "an independent bisection scan of the whole range "
                             "found no further solution" if not missing else f"solutions the solver missed: {missing[:3]}"))
    return _verdict(checks, ["Roots closer together than the scan resolution, or that only touch the target without "
                             "crossing it, cannot be excluded.", "The model is arithmetic on your inputs, not a forecast."])


# ------------------------------------------------------------------- elasticity --

def _ref_period(value, granularity: str):
    """The period a timestamp belongs to, by calendar arithmetic with the standard library only."""
    import datetime as dt
    d = value.date() if hasattr(value, "date") else value
    if granularity == "day":
        return (d.year, d.month, d.day)
    if granularity == "week":                       # weeks run Monday to Sunday, labelled by their Monday
        monday = d - dt.timedelta(days=d.weekday())
        return (monday.year, monday.month, monday.day)
    if granularity == "month":
        return (d.year, d.month, 1)
    q = (d.month - 1) // 3 * 3 + 1
    return (d.year, q, 1)


def _ref_observations(df, inputs: dict) -> dict:
    """The regression's observations rebuilt row by row in plain Python: clean, then aggregate."""
    import pandas as pd

    from app import metric_engine as me
    data = me.apply_filters(df, inputs.get("filters"))
    price_c, qty_c, date_c, seg_c = (inputs["price_column"], inputs["quantity_column"], inputs.get("date_column"),
                                     inputs.get("segment_column"))
    gran = inputs["granularity"] if date_c else None
    groups: dict = {}
    singles: list = []
    cols = [price_c, qty_c] + ([date_c] if date_c else []) + ([seg_c] if seg_c else [])
    for row in data[cols].itertuples(index=False):
        price, qty = row[0], row[1]
        date = row[2] if date_c else None
        seg = str(row[-1]) if seg_c else None
        if price is None or qty is None or (date_c and pd.isna(pd.to_datetime(date, errors="coerce"))):
            continue
        price, qty = float(price), float(qty)
        if math.isnan(price) or math.isnan(qty) or math.isinf(price) or math.isinf(qty) or price <= 0 or qty <= 0:
            continue
        if date_c:
            key = (_ref_period(pd.to_datetime(date), gran), seg)
            g = groups.setdefault(key, [0.0, 0.0])
            g[0] += qty
            g[1] += price * qty
        else:
            singles.append((price, qty, seg))
    if date_c:
        return {"keyed": {k: (v[0], v[1] / v[0]) for k, v in groups.items()}}
    return {"rows": sorted(singles, key=lambda t: (t[0], t[1], str(t[2])))}


def verify_elasticity(result: dict, design: dict, fit: dict, obs, df) -> dict:
    from scipy import stats as st
    X, y = design["X"], design["y"]
    el = result["elasticity"]
    checks: list[dict] = []

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)                  # SVD, not the normal equations the engine uses
    checks.append(_check("coefficient", _close(float(beta[1]), el["estimate"], 1e-8, 1e-10),
                         f"the price coefficient {float(beta[1]):.6f} was recomputed by singular-value least squares"
                         if _close(float(beta[1]), el["estimate"], 1e-8, 1e-10) else
                         f"engine {el['estimate']:.6f} vs recomputed {float(beta[1]):.6f}"))
    e = y - X @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = None if sst == 0 else 1 - float(e @ e) / sst
    got_r2 = result["diagnostics"]["r_squared"]
    checks.append(_check("r_squared", _close(r2, got_r2, 1e-8, 1e-10), "R² recomputed from its definition"
                         if _close(r2, got_r2, 1e-8, 1e-10) else f"engine {got_r2} vs recomputed {r2}"))
    if X.shape[1] == 2:
        xs, n = X[:, 1], len(y)
        mx, my = math.fsum(xs) / n, math.fsum(y) / n
        slope = math.fsum((a - mx) * (b - my) for a, b in zip(xs, y)) / math.fsum((a - mx) ** 2 for a in xs)
        checks.append(_check("closed_form_slope", _close(slope, el["estimate"], 1e-8, 1e-10),
                             "the single-variable slope matches covariance ÷ variance, in plain Python"
                             if _close(slope, el["estimate"], 1e-8, 1e-10) else f"closed form gives {slope}"))
    if el["standard_error"] and el["ci_low"] is not None:
        half = st.t.ppf(0.5 + el["level"] / 2, el["df"]) * el["standard_error"]
        ok = _close(el["ci_low"], el["estimate"] - half, 1e-9, 1e-9) and _close(el["ci_high"], el["estimate"] + half, 1e-9, 1e-9)
        checks.append(_check("interval", ok and el["ci_low"] < el["estimate"] < el["ci_high"],
                             f"the {el['level'] * 100:g}% interval equals the estimate ± t × standard error"
                             if ok else "the interval is not estimate ± t × standard error"))
    else:
        checks.append(_check("interval", False, "no interval was produced"))
    inputs = result["inputs"]
    if df is None:
        checks.append(_check("observations_rebuilt", None, "the dataset is too large to rebuild observation by observation; "
                             "the data preparation was not independently checked", required=False))
    else:
        ref = _ref_observations(df, {**inputs, "granularity": result["inputs"]["granularity"]})
        ok = False
        detail = ""
        if "keyed" in ref:
            got = {}
            for r in obs.itertuples(index=False):
                period = _ref_period(r.period, inputs["granularity"])
                got[(period, getattr(r, "segment", None) if inputs.get("segment_column") else None)] = (float(r.qty), float(r.price))
            ok = set(got) == set(ref["keyed"]) and all(_close(got[k][0], ref["keyed"][k][0]) and _close(got[k][1], ref["keyed"][k][1])
                                                       for k in got)
            detail = f"{len(got)} period observations rebuilt row by row"
        else:
            got_rows = sorted(((float(r.price), float(r.qty), getattr(r, "segment", None) if inputs.get("segment_column") else None)
                               for r in obs.itertuples(index=False)), key=lambda t: (t[0], t[1], str(t[2])))
            ok = len(got_rows) == len(ref["rows"]) and all(_close(a[0], b[0]) and _close(a[1], b[1]) for a, b in zip(got_rows, ref["rows"]))
            detail = f"{len(got_rows)} row observations rebuilt"
        checks.append(_check("observations_rebuilt", ok, detail + " and matched the regression's inputs" if ok
                             else "the observations the regression used differ from an independent rebuild"))
    return _verdict(checks, ["The regression is observational; the check confirms the arithmetic and the data preparation, "
                             "not that price caused the change in quantity.",
                             "Controls (trend, seasonality, segment intercepts) are part of the specification and are not "
                             "independently validated as the right ones."])


# ------------------------------------------------------------ baseline from data --

def verify_baseline(csv_path: str, result: dict, request_filters: list | None) -> dict:
    """Every dataset-derived baseline input recomputed with the metric engine's separate csv + plain-Python path.
    An input that path cannot express is reported as not run, never as a pass."""
    from app import metric_engine as me
    checks: list[dict] = []
    rows_ref = None
    for name, src in result["sources"].items():
        formula = src.get("formula")
        if src.get("source") not in ("dataset", "formula", "semantic_metric") or not formula:
            continue
        filters = [*(src.get("filters") or []), *(request_filters or [])]
        try:
            ref = me.reference_evaluate(csv_path, formula, filters)
        except me.ReferenceNotApplicable as e:
            checks.append(_check(f"input_{name}", None, f"the independent path cannot express this ({e})", required=False))
            continue
        except (me.FormulaError, OSError) as e:
            checks.append(_check(f"input_{name}", False, f"the independent recomputation failed ({type(e).__name__})"))
            continue
        rows_ref = ref["rows"] if rows_ref is None else rows_ref
        got = result["inputs"][name]
        ok = me.values_agree(ref["value"], got, 1e-9, 1e-9)
        checks.append(_check(f"input_{name}", ok, f"{formula} = {got:,.6g}, recomputed independently from the file"
                             if ok else f"engine {got} vs independent {ref['value']}"))
    if rows_ref is not None and not any(s.get("filters") for s in result["sources"].values()):
        checks.append(_check("rows_in_period", rows_ref == result["rows_used"], f"{rows_ref:,} rows counted independently"
                             if rows_ref == result["rows_used"] else f"engine {result['rows_used']} vs independent {rows_ref}"))
    if not checks:
        checks.append(_check("inputs_derived_from_data", None, "no input was derived from the dataset, so there is nothing to recompute",
                             required=False))
    return _verdict(checks, ["Checks the arithmetic of the baseline over the chosen period; it does not judge whether that "
                             "period is representative."])


def verify_period_comparison(result: dict, spec: dict | None = None) -> dict:
    model = result["model"]
    spec = spec or scenarios._model(model)  # noqa: SLF001
    assumptions = result["assumptions"]
    primary = result["primary_output"]
    checks: list[dict] = []
    outs = {}
    for label in ("a", "b"):
        ref = ref_outputs(model, result[label]["inputs"], spec, assumptions)
        outs[label] = ref
        bad = [k for k, v in ref.items() if not _close(v, result[label]["outputs"].get(k))]
        checks.append(_check(f"period_{label}_outputs", not bad, f"period {label.upper()}: every output recomputed independently"
                             if not bad else f"period {label.upper()} differs for: {', '.join(bad)}"))
    bad_out, bad_in = [], []
    for k, v in outs["b"].items():
        a, b = outs["a"][k], v
        want = None if a is None or b is None else {"abs": b - a, "pct": (b - a) / abs(a) * 100 if a else None}
        got = result["output_changes"].get(k)
        if (want is None) != (got is None) or (want and not (_close(want["abs"], got["abs"]) and _close(want["pct"], got["pct"]))):
            bad_out.append(k)
    for k, x in result["a"]["inputs"].items():
        y = result["b"]["inputs"][k]
        got = result["input_changes"][k]
        if not (_close(y - x, got["abs"]) and _close((y - x) / abs(x) * 100 if x else None, got["pct"])):
            bad_in.append(k)
    checks.append(_check("changes", not bad_out and not bad_in, "input and output changes recomputed"
                         if not (bad_out or bad_in) else f"differs: {', '.join(bad_in + bad_out)}"))
    f = lambda inputs: ref_outputs(model, inputs, spec, assumptions)[primary] or 0.0  # noqa: E731
    got = result["contributions"]["by_input"]
    change = (outs["b"][primary] or 0.0) - (outs["a"][primary] or 0.0)
    scale = max(1.0, abs(change))
    if result["contributions"]["computation"]["method"] == "exact":
        want = ref_shapley(f, result["a"]["inputs"], result["b"]["inputs"])
        ok = set(got) == set(want) and all(_close(got[k], want[k], 1e-9, 1e-9 * scale) for k in want)
        checks.append(_check("attribution", ok and _close(sum(got.values()), change, 1e-9, 1e-9 * scale),
                             "the attribution equals the subset-formula Shapley values and sums to the change"
                             if ok else "the attribution differs from an independent computation"))
    else:
        checks.append(_check("attribution", _close(sum(got.values()), change, 1e-9, 1e-9 * scale),
                             "the sampled attribution sums exactly to the change (individual shares are estimates)"))
    return _verdict(checks, ["Both periods are recorded data; the check covers the arithmetic, not why the inputs moved."])


# ------------------------------------------------------------------- segments --

def verify_segments(result: dict, spec: dict | None = None, reference: dict | None = None) -> dict:
    from app import metric_engine as me
    model = result["model"]
    spec = spec or scenarios._model(model)  # noqa: SLF001
    assumptions = result["assumptions"]
    segs = result["segments"]
    checks: list[dict] = []
    additive = result["overall"]["additive_outputs"]
    ratios = result["overall"]["ratio_outputs"]

    part = result["partition"]
    ok = part["complete"] and part["rows_in_segments"] == sum(s["rows"] for s in segs) == part["rows_in_scope"]
    checks.append(_check("partition", ok, f"{part['rows_in_scope']:,} rows in scope, {part['rows_in_segments']:,} in {part['segments']} segments: "
                         "every row is in exactly one segment" if ok else "the segments do not hold exactly the rows in scope"))

    if reference is None:
        checks.append(_check("segments_rebuilt_from_file", None, "no file was available to recount from, so the segment sizes and "
                             "inputs were not independently recomputed", required=False))
    else:
        norm = lambda k: MISSING_LABEL if (k or "").strip() in me._NA_STRINGS else (k or "").strip()  # noqa: E731
        col = result["segment_column"]
        want_rows: dict = {}
        for g in me.reference_evaluate(reference["csv_path"], "count()", reference["filters"], [col])["groups"]:
            key = norm(g["key"][col])
            want_rows[key] = want_rows.get(key, 0) + g["rows"]
        got_rows = {s["value"]: s["rows"] for s in segs}
        checks.append(_check("segment_sizes", want_rows == got_rows, "every segment's row count was recounted from the file"
                             if want_rows == got_rows else f"counts differ: file {want_rows} vs engine {got_rows}"))
        bad = []
        formulas = {n: s_.get("formula") for n, s_ in segs[0]["sources"].items() if s_.get("source") in ("dataset", "formula", "semantic_metric")}
        for name, formula in formulas.items():
            try:
                groups = me.reference_evaluate(reference["csv_path"], formula, reference["filters"], [col])["groups"]
            except (me.ReferenceNotApplicable, me.FormulaError):
                continue
            by = {}
            for g in groups:
                by.setdefault(norm(g["key"][col]), []).append(g)
            if any(len(v) > 1 for v in by.values()):
                continue                       # raw keys collapse (blank vs NA spellings): cannot be recounted per segment here
            for s in segs:
                ref_v = by.get(s["value"], [{"value": None}])[0]["value"]
                if not me.values_agree(ref_v, s["baseline"]["inputs"][name], 1e-9, 1e-9):
                    bad.append(f"{s['value']}.{name}")
        checks.append(_check("segment_inputs", not bad, "each segment's inputs were recomputed from the file"
                             if not bad else f"differs: {', '.join(bad[:5])}"))

    bad = []
    for s in segs:
        for label, inputs, adj in (("baseline", s["baseline"]["inputs"], {}), ("scenario", None, s["adjustments"])):
            want_in = ref_apply(model, s["baseline"]["inputs"], adj, assumptions)
            got = s[label]
            want_out = ref_outputs(model, want_in, spec, assumptions)
            if (label == "scenario" and any(not _close(want_in[k], got["inputs"][k]) for k in want_in)) or any(
                    not _close(v, got["outputs"].get(k)) for k, v in want_out.items()):
                bad.append(f"{s['value']} {label}")
    checks.append(_check("segment_outputs", not bad, "every segment's baseline and scenario outputs were recomputed independently"
                         if not bad else f"differs: {', '.join(bad[:5])}"))

    bad = []
    for key, label in (("baseline", "baseline"), ("scenario", "scenario")):
        outs = [s[key]["outputs"] for s in segs]
        total = result["overall"][key]["outputs"]
        for k in additive:
            vals = [o[k] for o in outs]
            want = None if any(v is None for v in vals) else math.fsum(vals)
            if not _close(want, total[k], 1e-9, 1e-9 * max(1.0, abs(want or 0))):
                bad.append(f"{label}.{k}")
        for k, r in ratios.items():
            n, d = total.get(r["numerator"]), total.get(r["denominator"])
            want = None if n is None or d in (None, 0) else n / d * (r.get("scale") if r.get("scale") is not None else 1.0)
            if not _close(want, total[k], 1e-9, 1e-9):
                bad.append(f"{label}.{k}")
    checks.append(_check("totals", not bad, "totals equal the sum of the segments (additive outputs) and ratios are recomputed from those sums"
                         if not bad else f"differs: {', '.join(bad[:5])}"))

    cons = result["contributions"]
    if cons is None:
        checks.append({"check": "contributions", "status": "not_applicable", "required": False,
                       "detail": "the headline output is not additive, so there is no split of the change"})
    else:
        p = cons["output"]
        by = cons["by_segment"]
        seg_changes = {s["value"]: (s["scenario"]["outputs"][p] or 0.0) - (s["baseline"]["outputs"][p] or 0.0) for s in segs}
        total_change = (result["overall"]["scenario"]["outputs"][p] or 0.0) - (result["overall"]["baseline"]["outputs"][p] or 0.0)
        scale = max(1.0, abs(total_change))
        ok = (all(_close(by[v]["abs"], c, 1e-9, 1e-9 * scale) for v, c in seg_changes.items())
              and _close(math.fsum(seg_changes.values()), total_change, 1e-9, 1e-9 * scale))
        checks.append(_check("contributions", ok, "the segments' changes add up to the total change"
                             if ok else "the segment contributions do not add up to the total change"))

    if result["allocations"]:
        bad = []
        for name, a in result["allocations"].items():
            parts = [s["baseline"]["inputs"][name] for s in segs]
            if not (_close(math.fsum(a["shares"].values()), 1.0, 1e-12, 1e-12) and _close(math.fsum(parts), a["total"], 1e-9, 1e-9 * max(1.0, abs(a["total"])))):
                bad.append(name)
        checks.append(_check("allocations", not bad, "each shared total was split in shares adding to 100%, and the parts add back to the total"
                             if not bad else f"does not add back: {', '.join(bad)}"))

    cons_all = result["consistency"]
    bad = [k for k, c in cons_all.items() if not c["agrees"]]
    checks.append(_check("segments_add_up_to_all_rows", not bad, "the sum over segments equals the figure computed from all rows at once"
                         if not bad else "the segments do not add up to the all-rows figure for: " + ", ".join(bad)
                         + ". A mapped input is probably not additive (for example a simple average of a price column)."))

    expected = [{**result["shared_adjustments"], **(result["segment_adjustments"].get(s["value"]) or {})} for s in segs]
    once = all(s["adjustments"] == e for s, e in zip(segs, expected)) and all(
        s["scenario"]["inputs"] == s["baseline"]["inputs"] for s in segs if not s["adjustments"])
    checks.append(_check("adjustments_applied_once", once, "each input was changed at most once per segment, and unadjusted segments are unchanged"
                         if once else "an adjustment was applied twice or an unadjusted segment changed"))
    return _verdict(checks, ["The segments are a partition of the rows by one column; the check confirms the arithmetic and the "
                             "partition, not that the segments behave independently in reality."])


MISSING_LABEL = "(missing)"


# ------------------------------------------------------------- two-way sensitivity --

def verify_two_way(result: dict, spec: dict | None = None) -> dict:
    model = result["model"]
    spec = spec or scenarios._model(model)  # noqa: SLF001
    assumptions, adjustments, base = result["assumptions"], result["adjustments"], result["baseline_inputs"]
    xs, ys = result["x"]["values"], result["y"]["values"]
    xk, yk, out = result["x"]["input"], result["y"]["input"], result["output"]["key"]
    custom = "custom" in spec
    checks: list[dict] = []

    def cell(xv: float, yv: float):
        inputs = ref_apply(model, base, {**adjustments, xk: {"type": "set", "value": xv}, yk: {"type": "set", "value": yv}}, assumptions)
        if custom and not ref_custom(spec["custom"]["definition"], inputs, assumptions)[1]:
            return None
        return ref_outputs(model, inputs, spec, assumptions)[out]

    ref = [[cell(xv, yv) for xv in xs] for yv in ys]
    got = result["z"]
    bad = sum(1 for i in range(len(ys)) for j in range(len(xs))
              if not _close(ref[i][j], got[i][j], 1e-9, 1e-9 * max(1.0, abs(ref[i][j] or 0.0))))
    checks.append(_check("every_cell", bad == 0, f"all {len(xs) * len(ys):,} cells were recomputed independently and match"
                         if bad == 0 else f"{bad} cell(s) differ"))

    flat = [(v, i, j) for i, row in enumerate(ref) for j, v in enumerate(row) if v is not None]
    if flat:
        top, bottom = max(v for v, _, _ in flat), min(v for v, _, _ in flat)
        b, w = result["best"], result["worst"]
        ok = (b is not None and w is not None and _close(b["value"], top, 1e-9, 1e-9 * max(1.0, abs(top)))
              and _close(w["value"], bottom, 1e-9, 1e-9 * max(1.0, abs(bottom)))
              and _close(ref[b["i"]][b["j"]], top, 1e-9, 1e-9 * max(1.0, abs(top))) and _close(ref[w["i"]][w["j"]], bottom, 1e-9, 1e-9 * max(1.0, abs(bottom))))
        checks.append(_check("best_and_worst", ok, "the best and worst cells are the largest and smallest values of an independent scan of the grid"
                             if ok else "the reported best or worst cell is not the extreme of the recomputed grid"))
    else:
        checks.append(_check("best_and_worst", result["best"] is None and result["worst"] is None, "no cell has a value, and none is reported"))

    t = result["threshold"]
    if t is not None:
        pts = 0
        for i in range(len(ys)):
            for j in range(len(xs) - 1):
                a, c = ref[i][j], ref[i][j + 1]
                pts += 1 if a is not None and c is not None and (a - t) * (c - t) < 0 else 0
        for i in range(len(ys) - 1):
            for j in range(len(xs)):
                a, c = ref[i][j], ref[i + 1][j]
                pts += 1 if a is not None and c is not None and (a - t) * (c - t) < 0 else 0
        bnd = result["boundary"]
        above = sum(1 for v, _, _ in flat if v > t)
        below = sum(1 for v, _, _ in flat if v < t)
        on_edge = all(_close(_interp(ref, xs, ys, p, t), t, 1e-6, 1e-6 * max(1.0, abs(t))) for p in bnd["points"]) if bnd["points"] else True
        ok = len(bnd["points"]) == pts and bnd["cells_above"] == above and bnd["cells_below"] == below and on_edge
        checks.append(_check("boundary", ok, f"{pts} crossing point(s) of {t:g} found again, each lying where the output equals {t:g} on its edge"
                             if ok else "the boundary points or the cell counts differ from an independent scan"))

    cur = result["current"]
    want = ref_outputs(model, ref_apply(model, base, adjustments, assumptions), spec, assumptions)[out]
    checks.append(_check("current_position", _close(want, cur["value"], 1e-9, 1e-9 * max(1.0, abs(want or 0.0))),
                         "the current position's value was recomputed exactly (it is not read off the nearest cell)"))
    return _verdict(checks, ["Best and worst are the extremes of the grid shown, not a global optimum.",
                             "The model is arithmetic on your inputs; a cell is not a forecast."])


def _interp(ref, xs, ys, p, t):
    """The output at a boundary point, by linear interpolation along the edge it lies on (it should equal the threshold)."""
    if p["edge"] == "row":
        i = ys.index(p["y"]) if p["y"] in ys else min(range(len(ys)), key=lambda k: abs(ys[k] - p["y"]))
        j = max(k for k in range(len(xs) - 1) if xs[k] <= p["x"] + 1e-15) if p["x"] < xs[-1] else len(xs) - 2
        a, c = ref[i][j], ref[i][j + 1]
        return a + (c - a) * (p["x"] - xs[j]) / (xs[j + 1] - xs[j])
    j = xs.index(p["x"]) if p["x"] in xs else min(range(len(xs)), key=lambda k: abs(xs[k] - p["x"]))
    i = max(k for k in range(len(ys) - 1) if ys[k] <= p["y"] + 1e-15) if p["y"] < ys[-1] else len(ys) - 2
    a, c = ref[i][j], ref[i + 1][j]
    return a + (c - a) * (p["y"] - ys[i]) / (ys[i + 1] - ys[i])
