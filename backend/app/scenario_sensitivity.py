"""Two-way sensitivity: one output over a grid of two inputs (price x volume, conversion x order value, any two inputs).

The grid is the model's own array formulas evaluated at every combination, so it is the same arithmetic as a scenario,
including a custom model's formulas and constraints and the built-in price-elasticity link. Alongside the values it
returns what a decision-maker looks for on such a chart:

* the **current position** (the scenario's own values of the two inputs) and its output, computed exactly, not read off
  the nearest cell;
* the **best** and **worst** cell on the grid (the largest and smallest output). "Best" is the best *on this grid*,
  inside the ranges you chose; it is not claimed to be a global optimum;
* the **boundary** where the output crosses a threshold (break-even for profit when the threshold is 0), as the points
  where it crosses between neighbouring cells, found by linear interpolation along the grid's edges.

The axes replace any adjustment to those two inputs; every other adjustment stays. Each axis is a range of percent change
around the scenario value, or absolute limits, and must stay inside the input's allowed range: it is refused, never
clipped. At most 61 x 61 cells.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from app import scenarios
from app.scenarios import ScenarioError

SENSITIVITY_VERSION = "scenario-sensitivity.1"
MIN_RESOLUTION = 3
MAX_RESOLUTION = 61
DEFAULT_RESOLUTION = 21


class SensitivityError(ScenarioError):
    """The grid request is invalid; the message says why."""


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise SensitivityError(f"{what} must be a finite number")
    return float(v)


def _axis(name: str, rule: dict, center: float, spec: Any, resolution: int) -> np.ndarray:
    if not isinstance(spec, dict) or set(spec) - {"mode", "low", "high"} or "low" not in spec or "high" not in spec:
        raise SensitivityError(f"The range for {rule['label']} needs {{mode: pct|absolute, low, high}}")
    mode = spec.get("mode", "pct")
    low, high = _num(spec["low"], f"The low end for {rule['label']}"), _num(spec["high"], f"The high end for {rule['label']}")
    if mode == "pct":
        if center <= 0:
            raise SensitivityError(f"{rule['label']} is {center:g} in this scenario, so a percentage range around it is empty; use absolute limits")
        if low <= -100:
            raise SensitivityError(f"The low end for {rule['label']} cannot be -100% or lower")
        lo, hi = center * (1 + low / 100), center * (1 + high / 100)
    elif mode == "absolute":
        lo, hi = low, high
    else:
        raise SensitivityError("mode must be 'pct' or 'absolute'")
    if not lo < hi:
        raise SensitivityError(f"For {rule['label']}, the low end must be below the high end")
    rmin, rmax = rule.get("min", -math.inf), rule.get("max", math.inf)
    if lo < rmin - 1e-12 or hi > rmax + 1e-12:
        raise SensitivityError(f"The range for {rule['label']} ({lo:,.6g} to {hi:,.6g}) goes outside its allowed range "
                               f"({rmin:g} to {rmax:g}); narrow it")
    return np.linspace(lo, hi, resolution)


def _cell(grid: np.ndarray, i: int, j: int) -> dict:
    return {"i": int(i), "j": int(j), "value": float(grid[i, j])}


def two_way(model: "str | dict", baseline: dict, x_input: str, y_input: str, output: str | None = None, *,
            adjustments: dict | None = None, assumptions: dict | None = None, x_range: dict | None = None,
            y_range: dict | None = None, resolution: int = DEFAULT_RESOLUTION, threshold: float | None = 0.0,
            verify: bool = True) -> dict:
    spec = scenarios._model(model)  # noqa: SLF001
    key = scenarios.model_key(model)
    baseline = scenarios.validate_inputs(model, baseline)
    assumptions = scenarios.validate_assumptions(model, assumptions)
    adjustments = dict(adjustments or {})
    center = scenarios.apply_adjustments(model, baseline, adjustments, assumptions)
    for what, name in (("x_input", x_input), ("y_input", y_input)):
        if name not in spec["inputs"]:
            raise SensitivityError(f"Unknown {what} '{name}'. Available: {', '.join(spec['inputs'])}")
    if x_input == y_input:
        raise SensitivityError("Choose two different inputs for the two axes")
    output = output or spec["primary_output"]
    if output not in spec["outputs"]:
        raise SensitivityError(f"Unknown output '{output}'. Available: {', '.join(spec['outputs'])}")
    if isinstance(resolution, bool) or not isinstance(resolution, int) or not MIN_RESOLUTION <= resolution <= MAX_RESOLUTION:
        raise SensitivityError(f"resolution must be a whole number from {MIN_RESOLUTION} to {MAX_RESOLUTION}")
    if threshold is not None:
        threshold = _num(threshold, "The threshold")
    default = {"mode": "pct", "low": -20.0, "high": 20.0}
    xs = _axis(x_input, spec["inputs"][x_input], center[x_input], x_range or default, resolution)
    ys = _axis(y_input, spec["inputs"][y_input], center[y_input], y_range or default, resolution)

    X, Y = np.meshgrid(xs, ys)                       # rows follow y, columns follow x
    arrays = {k: float(v) for k, v in center.items()}
    arrays[x_input], arrays[y_input] = X.ravel(), Y.ravel()
    linked = (key == "unit_economics" and bool(assumptions.get("price_elasticity")) and bool(baseline["price"])
              and "volume" not in adjustments and "volume" not in (x_input, y_input) and "price" in (x_input, y_input))
    if linked:
        price = arrays["price"]
        arrays["volume"] = np.maximum(0.0, baseline["volume"] * (1 + assumptions["price_elasticity"] * (price / baseline["price"] - 1)))
    outs = {k: np.asarray(v, dtype=float).copy() for k, v in scenarios.outputs_vec(model, arrays, assumptions).items()}
    rejected = 0
    mask_fn = spec.get("constraint_mask")
    if mask_fn is not None:
        ok = mask_fn({**arrays, **assumptions, **outs})
        rejected = int((~ok).sum())
        for arr in outs.values():
            arr[~ok] = np.nan
    z = outs[output].reshape(resolution, resolution)

    center_out = scenarios._outputs(model, center, assumptions)          # noqa: SLF001
    finite = np.isfinite(z)
    best = worst = None
    if finite.any():
        zz = np.where(finite, z, -np.inf)
        bi, bj = np.unravel_index(int(np.argmax(zz)), z.shape)
        zz = np.where(finite, z, np.inf)
        wi, wj = np.unravel_index(int(np.argmin(zz)), z.shape)
        best = {**_cell(z, bi, bj), "x": float(xs[bj]), "y": float(ys[bi])}
        worst = {**_cell(z, wi, wj), "x": float(xs[wj]), "y": float(ys[wi])}
    boundary = _boundary(xs, ys, z, threshold) if threshold is not None else None
    labels = {k: spec["inputs"][k]["label"] for k in (x_input, y_input)}
    disclosures = [f"Each cell is {spec['outputs'][output]} for that pair of values; all other inputs stay at their scenario values.",
                   f"The axes replace any adjustment to {labels[x_input]} and {labels[y_input]}.",
                   "Best and worst are the largest and smallest cells on this grid, inside the ranges chosen. They are not "
                   "claimed to be a global optimum."]
    if linked:
        disclosures.append(f"Volume follows each price on the grid through the stated price elasticity of {assumptions['price_elasticity']:g}.")
    if rejected:
        disclosures.append(f"{rejected} of {resolution * resolution} cells break one of the model's constraints and are left empty.")
    if boundary is not None and not boundary["points"] and finite.any():
        disclosures.append(f"The output never crosses {threshold:g} inside these ranges, so there is no boundary to draw.")
    result = {
        "kind": "sensitivity_grid", "data_kind": "scenario", "engine_version": SENSITIVITY_VERSION, "model": key, "title": spec["title"],
        "x": {"input": x_input, "label": labels[x_input], "unit": spec["inputs"][x_input]["unit"], "values": xs.tolist(), "center": center[x_input]},
        "y": {"input": y_input, "label": labels[y_input], "unit": spec["inputs"][y_input]["unit"], "values": ys.tolist(), "center": center[y_input]},
        "output": {"key": output, "label": spec["outputs"][output]},
        "z": [[None if not math.isfinite(v) else float(v) for v in row] for row in z],
        "resolution": resolution, "cells": resolution * resolution, "empty_cells": int((~finite).sum()),
        "current": {"x": center[x_input], "y": center[y_input], "value": center_out.get(output)},
        "baseline_value": scenarios._outputs(model, baseline, assumptions).get(output),          # noqa: SLF001
        "best": best, "worst": worst, "threshold": threshold, "boundary": boundary,
        "baseline_inputs": baseline, "adjustments": adjustments, "assumptions": assumptions, "disclosures": disclosures,
        "constraint_rejected_cells": rejected,
    }
    if verify:
        from app import scenario_verify
        result["verification"] = scenario_verify.safe(scenario_verify.verify_two_way, result, spec)
    return result


def _boundary(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, t: float) -> dict:
    """Where the output crosses `t` between neighbouring cells, by linear interpolation along each grid edge."""
    d = z - t
    points: list[dict] = []
    m, n = z.shape
    for i in range(m):
        for j in range(n - 1):
            a, b = d[i, j], d[i, j + 1]
            if np.isfinite(a) and np.isfinite(b) and a * b < 0:
                points.append({"x": float(xs[j] + (xs[j + 1] - xs[j]) * a / (a - b)), "y": float(ys[i]), "edge": "row"})
    for i in range(m - 1):
        for j in range(n):
            a, b = d[i, j], d[i + 1, j]
            if np.isfinite(a) and np.isfinite(b) and a * b < 0:
                points.append({"x": float(xs[j]), "y": float(ys[i] + (ys[i + 1] - ys[i]) * a / (a - b)), "edge": "column"})
    above = int((d > 0).sum())
    below = int((d < 0).sum())
    return {"threshold": t, "points": points, "cells_above": above, "cells_below": below, "cells_on": int((d == 0).sum()),
            "method": "linear interpolation of the output along each grid edge whose two ends lie on opposite sides"}
