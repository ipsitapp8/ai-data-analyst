"""Monte Carlo simulation of what-if scenarios.

A scenario (app/scenarios.py) is arithmetic on stated inputs and returns one number per output. This module
answers the question that one number cannot: *how much could the result move if the inputs are uncertain?*

How it works
------------
1. The scenario's inputs are fixed by the same code the deterministic engine uses (`scenarios.apply_adjustments`),
   including the explicit price-elasticity link. Those values are the centre of every uncertain input.
2. The caller states, per input, a distribution (normal, uniform or triangular) and its spread. Nothing is invented:
   an input with no distribution is held fixed and is listed as fixed in the result, and a request with no
   uncertain input at all is refused rather than given default ranges.
3. Draws come from a Gaussian copula: correlated standard normals (the caller's correlation matrix, identity when
   none is given) are turned into uniforms and then into each input's own distribution. Each distribution is
   truncated to the input's physical limits (no negative prices, a conversion rate of at most 1) by restricting the
   uniform, so no probability mass piles up at a limit, and the mass that was cut off is reported.
4. The model's own array formulas (`compute_vec`, tested equal to the scalar formulas) give every output for
   every draw.
5. Percentiles, probabilities and a histogram summarise the simulated outcomes. "Low" and "high" are the 5th and
   95th percentiles of the draws: they describe the simulation, they are not guaranteed bounds.
6. The draws and arithmetic are then checked by an independent plain-Python implementation
   (app/scenario_verify.py) and the result carries that verification.

A fixed seed reproduces a run exactly. When no seed is given one is drawn and returned, so any run can be repeated.
Distributions are assumptions the caller supplied, not facts estimated from data, and the result says so.
"""
from __future__ import annotations

import math
import secrets
import time
from typing import Any

import numpy as np
from scipy import stats as st

from app import scenarios
from app.scenarios import ScenarioError

SIM_VERSION = "scenario-sim.1"
DEFAULT_ITERATIONS = 10_000
MIN_ITERATIONS = 100
MAX_ITERATIONS = 200_000
HISTOGRAM_BINS = 40
DIRECTIONS = ("at_least", "at_most")
DISTRIBUTIONS = ("fixed", "normal", "uniform", "triangular")
DISCLAIMER = ("Simulation output: the spread of outcomes under the input distributions you stated. Those "
              "distributions are assumptions, not measurements, and the result is not a forecast.")

_PARAM_KEYS = {
    "fixed": set(),
    "normal": {"sd_pct", "sd"},
    "uniform": {"low_pct", "high_pct", "low", "high"},
    "triangular": {"low_pct", "mode_pct", "high_pct", "low", "mode", "high"},
}


class SimulationError(ScenarioError):
    """The simulation request is invalid; the message says why."""


def _num(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise SimulationError(f"{what} must be a finite number")
    return float(value)


# ------------------------------------------------------------ distributions --

def resolve_distribution(name: str, spec: Any, center: float) -> dict:
    """Validate one input's distribution and express it in absolute units.

    Two spellings, never mixed: percentages of the scenario value (`sd_pct`, `low_pct`, `mode_pct`, `high_pct`)
    or absolute numbers (`sd`, `low`, `mode`, `high`). The centre is the scenario value of the input.
    """
    if not isinstance(spec, dict):
        raise SimulationError(f"The distribution for '{name}' must be an object with a 'type'")
    kind = spec.get("type")
    if kind not in DISTRIBUTIONS:
        raise SimulationError(f"Unknown distribution '{kind}' for '{name}'. Use one of: {', '.join(DISTRIBUTIONS)}")
    unknown = set(spec) - {"type"} - _PARAM_KEYS[kind]
    if unknown:
        raise SimulationError(f"Unknown parameter(s) for the {kind} distribution of '{name}': "
                              f"{', '.join(sorted(unknown))}")
    if kind == "fixed":
        return {"type": "fixed", "center": center}
    keys = set(spec) - {"type"}
    pct = {k for k in keys if k.endswith("_pct")}
    if pct and pct != keys:
        raise SimulationError(f"'{name}': use either percentage parameters (…_pct) or absolute ones, not both")
    use_pct = bool(pct)
    if use_pct and center <= 0:
        raise SimulationError(f"'{name}' is {center:g} in this scenario, so a percentage spread of it is zero; "
                              "give the spread in absolute units instead")

    def get(key: str, default: float | None = None) -> float:
        raw = spec.get(f"{key}_pct" if use_pct else key, default)
        if raw is None:
            raise SimulationError(f"The {kind} distribution of '{name}' needs '{key}{'_pct' if use_pct else ''}'")
        v = _num(raw, f"{key} of '{name}'")
        return center * (1 + v / 100) if (use_pct and key != "sd") else (center * v / 100 if use_pct else v)

    if kind == "normal":
        sd = get("sd")
        if sd <= 0:
            raise SimulationError(f"The standard deviation of '{name}' must be above 0")
        return {"type": "normal", "center": center, "form": "percent" if use_pct else "absolute",
                "params": {"mean": center, "sd": sd}}
    low, high = get("low"), get("high")
    if not low < high:
        raise SimulationError(f"For '{name}', low must be below high (got {low:g} and {high:g})")
    if kind == "uniform":
        return {"type": "uniform", "center": center, "form": "percent" if use_pct else "absolute",
                "params": {"low": low, "high": high}}
    mode = center if ("mode" not in spec and "mode_pct" not in spec) else get("mode", 0.0)
    if not low <= mode <= high:
        raise SimulationError(f"For '{name}', the most likely value must lie between low and high")
    return {"type": "triangular", "center": center, "form": "percent" if use_pct else "absolute",
            "params": {"low": low, "mode": mode, "high": high}}


def frozen(dist: dict):
    """The scipy distribution for a resolved (non-fixed) specification."""
    p = dist["params"]
    if dist["type"] == "normal":
        return st.norm(loc=p["mean"], scale=p["sd"])
    span = p["high"] - p["low"]
    if dist["type"] == "uniform":
        return st.uniform(loc=p["low"], scale=span)
    return st.triang(c=(p["mode"] - p["low"]) / span, loc=p["low"], scale=span)


def validate_correlation(spec: Any, uncertain: list[str]) -> tuple[list[str], np.ndarray] | None:
    """A correlation matrix over some of the uncertain inputs: square, symmetric, unit diagonal, entries in
    [-1, 1], positive semidefinite. Returns the input names and the matrix, or None when none was given."""
    if spec in (None, {}, False):
        return None
    if not isinstance(spec, dict) or set(spec) - {"inputs", "matrix"}:
        raise SimulationError("correlation must be an object with 'inputs' and 'matrix'")
    names, matrix = spec.get("inputs"), spec.get("matrix")
    if not isinstance(names, list) or len(names) < 2 or not all(isinstance(n, str) for n in names):
        raise SimulationError("correlation 'inputs' must list at least two input names")
    if len(set(names)) != len(names):
        raise SimulationError("correlation 'inputs' must not repeat an input")
    missing = [n for n in names if n not in uncertain]
    if missing:
        raise SimulationError(f"Correlation needs an uncertainty distribution for: {', '.join(missing)}")
    k = len(names)
    if not isinstance(matrix, list) or len(matrix) != k or any(not isinstance(r, list) or len(r) != k for r in matrix):
        raise SimulationError(f"The correlation matrix must be {k} × {k} to match its inputs")
    m = np.array([[_num(v, "A correlation value") for v in row] for row in matrix], dtype=float)
    if np.any(np.abs(m) > 1 + 1e-12):
        raise SimulationError("Correlation values must lie between -1 and 1")
    if not np.allclose(np.diag(m), 1.0, atol=1e-9):
        raise SimulationError("The diagonal of a correlation matrix must be 1")
    if not np.allclose(m, m.T, atol=1e-9):
        raise SimulationError("The correlation matrix must be symmetric")
    if np.linalg.eigvalsh(m).min() < -1e-8:
        raise SimulationError("The correlation matrix is not positive semidefinite: these correlations cannot all "
                              "hold at once")
    return names, m


# ------------------------------------------------------------------ context --

def _prepare(model: "str | dict", baseline: dict, adjustments: dict | None, assumptions: dict | None, uncertainty: Any,
             correlation: Any, iterations: Any, seed: Any, target_output: str | None, target_value: Any,
             target_direction: str) -> dict:
    spec = scenarios._model(model)  # noqa: SLF001 - same package, raises a clear error for an unknown model
    baseline = scenarios.validate_inputs(model, baseline)
    assumptions = scenarios.validate_assumptions(model, assumptions)
    adjustments = dict(adjustments or {})
    center = scenarios.apply_adjustments(model, baseline, adjustments, assumptions)

    if isinstance(iterations, bool) or not isinstance(iterations, int) or not MIN_ITERATIONS <= iterations <= MAX_ITERATIONS:
        raise SimulationError(f"iterations must be a whole number from {MIN_ITERATIONS:,} to {MAX_ITERATIONS:,}")
    if seed is None:
        seed = secrets.randbits(32)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise SimulationError("seed must be a whole number from 0 to 4294967295")
    if target_output is None:
        target_output = spec["primary_output"]
    if target_output not in spec["outputs"]:
        raise SimulationError(f"Unknown output '{target_output}'. Available: {', '.join(spec['outputs'])}")
    if target_direction not in DIRECTIONS:
        raise SimulationError("target_direction must be 'at_least' or 'at_most'")
    if target_value is not None:
        target_value = _num(target_value, "The target value")

    if not isinstance(uncertainty, dict) or not uncertainty:
        raise SimulationError("Say which inputs are uncertain and how. No ranges are assumed on your behalf: give at "
                              "least one input a normal, uniform or triangular distribution.")
    unknown = set(uncertainty) - set(spec["inputs"])
    if unknown:
        raise SimulationError(f"Unknown input(s) in uncertainty: {', '.join(sorted(unknown))}")
    dists = {}
    for name in spec["inputs"]:                      # model order: draws never depend on how the JSON was ordered
        if name in uncertainty:
            d = resolve_distribution(name, uncertainty[name], center[name])
            if d["type"] != "fixed":
                dists[name] = d
    if not dists:
        raise SimulationError("Every input was marked fixed: nothing is uncertain, so there is nothing to simulate. "
                              "Use the deterministic scenario instead.")
    corr = validate_correlation(correlation, list(dists))
    return {"model": scenarios.model_key(model), "model_obj": model, "spec": spec, "baseline": baseline, "assumptions": assumptions, "adjustments": adjustments,
            "center": center, "dists": dists, "corr": corr, "iterations": iterations, "seed": seed,
            "target_output": target_output, "target_value": target_value, "target_direction": target_direction}


# ----------------------------------------------------------------- sampling --

def _draw(ctx: dict) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Samples for every input (fixed ones are constant arrays) and the share of each distribution's probability
    that lay outside the input's allowed range and was cut off."""
    n, names = ctx["iterations"], list(ctx["dists"])
    rng = np.random.default_rng(ctx["seed"])
    z = rng.standard_normal((n, len(names)))
    if ctx["corr"] is not None:
        cnames, cmat = ctx["corr"]
        full = np.eye(len(names))
        idx = [names.index(c) for c in cnames]
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                full[ia, ib] = cmat[a, b]
        vals, vecs = np.linalg.eigh(full)
        z = z @ (vecs * np.sqrt(np.clip(vals, 0.0, None))).T       # rows now have covariance `full`
    u = st.norm.cdf(z)
    samples = {k: np.full(n, float(v)) for k, v in ctx["center"].items()}
    cut: dict[str, float] = {}
    for j, name in enumerate(names):
        dist = frozen(ctx["dists"][name])
        rule = ctx["spec"]["inputs"][name]
        lo, hi = rule.get("min", -math.inf), rule.get("max", math.inf)
        f_lo = float(dist.cdf(lo)) if math.isfinite(lo) else 0.0
        f_hi = float(dist.cdf(hi)) if math.isfinite(hi) else 1.0
        if f_hi - f_lo < 1e-9:
            raise SimulationError(f"The distribution of '{name}' lies outside its allowed range "
                                  f"({lo:g} to {hi:g}); choose a range that overlaps it")
        cut[name] = f_lo + (1.0 - f_hi)
        uj = np.clip(f_lo + u[:, j] * (f_hi - f_lo), 1e-12, 1 - 1e-12)
        samples[name] = np.clip(dist.ppf(uj), lo, hi)             # clip only removes float round-off at the limits
    if _elasticity_applies(ctx):
        base = ctx["baseline"]
        e = ctx["assumptions"]["price_elasticity"]
        samples["volume"] = np.maximum(0.0, base["volume"] * (1 + e * (samples["price"] / base["price"] - 1)))
    return samples, cut


def _elasticity_applies(ctx: dict) -> bool:
    """Volume follows a sampled price through the stated elasticity unless volume itself was adjusted or is uncertain."""
    return (ctx["model"] == "unit_economics" and bool(ctx["assumptions"].get("price_elasticity"))
            and bool(ctx["baseline"]["price"]) and "volume" not in ctx["adjustments"]
            and "volume" not in ctx["dists"] and "price" in ctx["dists"])


def _raw(ctx: dict) -> dict:
    samples, cut = _draw(ctx)
    outputs = {k: np.asarray(v, dtype=float).copy() for k, v in
               scenarios.outputs_vec(ctx["model_obj"], samples, ctx["assumptions"]).items()}
    rejected = 0.0
    mask_fn = ctx["spec"].get("constraint_mask")
    if mask_fn is not None:
        ok = mask_fn({**samples, **ctx["assumptions"], **outputs})
        rejected = float((~ok).mean())
        for arr in outputs.values():
            arr[~ok] = np.nan
    return {"samples": samples, "outputs": outputs, "cut": cut, "constraint_rejected": rejected}


# ------------------------------------------------------------------ summary --

def summarize(x: np.ndarray) -> dict:
    finite = x[np.isfinite(x)]
    n = int(len(finite))
    if n == 0:
        return {"n_valid": 0, "mean": None, "median": None, "std": None, "min": None, "max": None,
                "p5": None, "p25": None, "p75": None, "p95": None}
    p5, p25, p50, p75, p95 = (float(v) for v in np.percentile(finite, [5, 25, 50, 75, 95]))
    return {"n_valid": n, "mean": float(finite.mean()), "median": p50,
            "std": float(finite.std(ddof=1)) if n > 1 else 0.0, "min": float(finite.min()), "max": float(finite.max()),
            "p5": p5, "p25": p25, "p75": p75, "p95": p95}


def _probability(count: int, n: int) -> dict:
    p = count / n if n else None
    return {"probability": p, "standard_error": math.sqrt(p * (1 - p) / n) if n else None, "count": count}


def probabilities(y: np.ndarray, baseline_value: float | None, target_value: float | None, direction: str) -> dict:
    finite = y[np.isfinite(y)]
    n = int(len(finite))
    out = {"n_valid": n, "positive": _probability(int((finite > 0).sum()), n),
           "negative": _probability(int((finite < 0).sum()), n)}
    out["exceeds_baseline"] = (_probability(int((finite > baseline_value).sum()), n)
                               if baseline_value is not None else None)
    if target_value is None:
        out["reaches_target"] = None
    else:
        count = int((finite >= target_value).sum() if direction == "at_least" else (finite <= target_value).sum())
        out["reaches_target"] = {**_probability(count, n), "target": target_value, "direction": direction}
    return out


def _histogram(y: np.ndarray) -> dict:
    finite = y[np.isfinite(y)]
    if len(finite) == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(finite, bins=HISTOGRAM_BINS)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


# ------------------------------------------------------------------- public --

def simulate(model: "str | dict", baseline: dict, adjustments: dict | None = None, assumptions: dict | None = None,
             uncertainty: dict | None = None, correlation: dict | None = None, iterations: int = DEFAULT_ITERATIONS,
             seed: int | None = None, target_output: str | None = None, target_value: float | None = None,
             target_direction: str = "at_least", verify: bool = True) -> dict:
    """Run a Monte Carlo simulation of one scenario (the baseline plus `adjustments`)."""
    started = time.perf_counter()
    ctx = _prepare(model, baseline, adjustments, assumptions, uncertainty, correlation, iterations, seed,
                   target_output, target_value, target_direction)
    raw = _raw(ctx)
    spec, tgt = ctx["spec"], ctx["target_output"]
    model_name = ctx["model"]
    base_out = scenarios._outputs(model, ctx["baseline"], ctx["assumptions"])        # noqa: SLF001
    center_out = scenarios._outputs(model, ctx["center"], ctx["assumptions"])        # noqa: SLF001
    y = raw["outputs"][tgt]
    summary = summarize(y)
    target = {"output": tgt, "label": spec["outputs"][tgt], "baseline_value": base_out.get(tgt),
              "deterministic_scenario_value": center_out.get(tgt),
              "low": summary["p5"], "expected": summary["mean"], "median": summary["median"], "high": summary["p95"],
              "definition": "low and high are the 5th and 95th percentiles of the simulated outcomes; expected is "
                            "their mean. They describe this simulation, not guaranteed bounds.",
              "probabilities": probabilities(y, base_out.get(tgt), ctx["target_value"], ctx["target_direction"]),
              "histogram": _histogram(y)}
    fixed = [k for k in spec["inputs"] if k not in ctx["dists"]]
    result = {
        "kind": "simulation", "disclaimer": DISCLAIMER, "engine_version": SIM_VERSION, "model": model_name,
        "title": spec["title"],
        "config": {"iterations": ctx["iterations"], "seed": ctx["seed"], "adjustments": ctx["adjustments"],
                   "assumptions": ctx["assumptions"], "target_output": tgt, "target_value": ctx["target_value"],
                   "target_direction": ctx["target_direction"], "distributions": ctx["dists"],
                   "correlation": ({"inputs": ctx["corr"][0], "matrix": ctx["corr"][1].tolist()}
                                   if ctx["corr"] else None)},
        "baseline_inputs": ctx["baseline"], "scenario_inputs": ctx["center"],
        "uncertain_inputs": list(ctx["dists"]), "fixed_inputs": fixed,
        "outputs": {k: summarize(v) for k, v in raw["outputs"].items()},
        "output_labels": spec["outputs"], "target": target,
        "truncated_probability": {k: round(v, 6) for k, v in raw["cut"].items() if v > 1e-9},
        "disclosures": _disclosures(ctx, fixed, raw["cut"], summary["n_valid"], raw["constraint_rejected"]),
        "constraint_rejected_fraction": raw["constraint_rejected"],
    }
    if verify:
        from app import scenario_verify
        result["verification"] = scenario_verify.safe(scenario_verify.verify_simulation, result, raw, lambda: _raw(ctx), spec)
    result["runtime_ms"] = int((time.perf_counter() - started) * 1000)
    return result


def _disclosures(ctx: dict, fixed: list[str], cut: dict[str, float], n_valid: int, rejected: float = 0.0) -> list[str]:
    spec, notes = ctx["spec"], []
    labels = lambda names: ", ".join(spec["inputs"][k]["label"] for k in names)  # noqa: E731
    notes.append("The input distributions are assumptions you supplied. They were not estimated from your data.")
    if fixed:
        notes.append(f"Held fixed at their scenario values because no uncertainty was given: {labels(fixed)}.")
    names = list(ctx["dists"])
    in_corr = set(ctx["corr"][0]) if ctx["corr"] else set()
    if len(names) > 1 and len(in_corr) < len(names):
        independent = [n for n in names if n not in in_corr] if in_corr else names
        notes.append("Sampled independently of each other (no correlation was given): " + labels(independent) + ".")
    if ctx["corr"]:
        notes.append("Correlation is applied through a Gaussian copula: the matrix is the correlation of the "
                     "underlying normal variables, so the correlation of the sampled inputs is close to it but not "
                     "identical, especially for skewed or bounded distributions.")
    if _elasticity_applies(ctx):
        notes.append(f"Volume follows each sampled price through the stated price elasticity of "
                     f"{ctx['assumptions']['price_elasticity']:g}.")
    elif ctx["model"] == "unit_economics" and ctx["assumptions"].get("price_elasticity") and "price" in ctx["dists"]:
        notes.append("The price elasticity is not applied to sampled prices because volume was adjusted or given its "
                     "own distribution.")
    for name, mass in cut.items():
        if mass > 1e-9:
            notes.append(f"{spec['inputs'][name]['label']}: {mass * 100:.2f}% of the stated distribution lay outside "
                         "its allowed range; the distribution was truncated to the allowed range.")
    if rejected > 0:
        notes.append(f"{rejected * 100:.2f}% of the draws broke one of the model's constraints and are left out of the summaries.")
    if n_valid < ctx["iterations"] - int(round(rejected * ctx["iterations"])):
        notes.append(f"{ctx['iterations'] - n_valid - int(round(rejected * ctx['iterations'])):,} of {ctx['iterations']:,} draws had no defined "
                     f"{spec['outputs'][ctx['target_output']].lower()} (a division by zero) and are left out of the "
                     "summaries.")
    return notes
