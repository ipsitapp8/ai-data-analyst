"""What-if simulator: explicit, deterministic scenario models.

A scenario is arithmetic on stated inputs under stated assumptions. It is not a
forecast: nothing here estimates how customers, costs or competitors would
actually respond. Every result is labelled `kind: "scenario"`, carries the
formulas that produced it, and lists the assumptions the dataset does not
support (for example, any price elasticity is a number the user typed, not one
estimated from data).

Two models, both fully specified below:

    unit_economics   revenue = price x volume
                     variable_cost = unit_cost x volume
                     profit = revenue - variable_cost - fixed_cost
                     volume responds to a price change only through the
                     `price_elasticity` assumption (default 0 = no response)

    funnel           orders = visitors x conversion_rate
                     revenue = orders x average_order_value
                     profit = revenue - visitors x cost_per_visitor

Inputs are validated against physical constraints (no negative prices, rates
within 0..1, no change below -100%) and impossible inputs are rejected with the
reason, not clamped silently.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Any, Callable

import numpy as np
import pandas as pd

from app import metric_engine as me

ENGINE_VERSION = "scenarios.2"

DISCLAIMER = ("Scenario output: arithmetic on the stated inputs and assumptions. It is not an observed "
              "fact and not a validated forecast.")


class ScenarioError(ValueError):
    pass


def _unit_economics(i: dict, a: dict) -> dict:
    revenue = i["price"] * i["volume"]
    variable_cost = i["unit_cost"] * i["volume"]
    gross_profit = revenue - variable_cost
    profit = gross_profit - i["fixed_cost"]
    return {"revenue": revenue, "variable_cost": variable_cost, "gross_profit": gross_profit, "profit": profit,
            "margin_pct": profit / revenue * 100 if revenue else None}


def _funnel(i: dict, a: dict) -> dict:
    orders = i["visitors"] * i["conversion_rate"]
    revenue = orders * i["average_order_value"]
    acquisition_cost = i["visitors"] * i["cost_per_visitor"]
    return {"orders": orders, "revenue": revenue, "acquisition_cost": acquisition_cost,
            "profit": revenue - acquisition_cost,
            "cost_per_order": acquisition_cost / orders if orders else None}


def _unit_economics_vec(i: dict, a: dict) -> dict:
    """The same formulas as `_unit_economics`, on arrays (Monte Carlo). A margin on zero revenue is NaN, not 0."""
    revenue = i["price"] * i["volume"]
    variable_cost = i["unit_cost"] * i["volume"]
    gross_profit = revenue - variable_cost
    profit = gross_profit - i["fixed_cost"]
    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(revenue != 0, profit / revenue * 100, np.nan)
    return {"revenue": revenue, "variable_cost": variable_cost, "gross_profit": gross_profit, "profit": profit,
            "margin_pct": margin}


def _funnel_vec(i: dict, a: dict) -> dict:
    orders = i["visitors"] * i["conversion_rate"]
    revenue = orders * i["average_order_value"]
    acquisition_cost = i["visitors"] * i["cost_per_visitor"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cost_per_order = np.where(orders != 0, acquisition_cost / orders, np.nan)
    return {"orders": orders, "revenue": revenue, "acquisition_cost": acquisition_cost,
            "profit": revenue - acquisition_cost, "cost_per_order": cost_per_order}


MODELS: dict[str, dict[str, Any]] = {
    "unit_economics": {
        "title": "Unit economics (price, volume, cost)",
        "compute": _unit_economics,
        "compute_vec": _unit_economics_vec,
        "inputs": {
            "price": {"label": "Price per unit", "unit": "currency/unit", "min": 0.0, "good": "up"},
            "volume": {"label": "Units sold", "unit": "units", "min": 0.0, "good": "up"},
            "unit_cost": {"label": "Variable cost per unit", "unit": "currency/unit", "min": 0.0, "good": "down"},
            "fixed_cost": {"label": "Fixed cost", "unit": "currency", "min": 0.0, "good": "down"},
        },
        "assumptions": {
            "price_elasticity": {"label": "Price elasticity of volume", "default": 0.0, "min": -10.0, "max": 10.0,
                                 "help": "% change in volume per 1% change in price. 0 = volume does not react."},
        },
        "outputs": {"revenue": "Revenue", "variable_cost": "Variable cost", "gross_profit": "Gross profit",
                    "profit": "Profit", "margin_pct": "Profit margin %"},
        "primary_output": "profit",
        "additive_outputs": ["revenue", "variable_cost", "gross_profit", "profit"],
        "ratio_outputs": {"margin_pct": {"numerator": "profit", "denominator": "revenue", "scale": 100.0}},
        "formulas": ["revenue = price × volume", "variable_cost = unit_cost × volume",
                     "gross_profit = revenue − variable_cost", "profit = gross_profit − fixed_cost",
                     "margin_pct = profit ÷ revenue × 100",
                     "volume after a price change = volume × (1 + price_elasticity × price change %)"],
    },
    "funnel": {
        "title": "Acquisition funnel (visitors, conversion, order value)",
        "compute": _funnel,
        "compute_vec": _funnel_vec,
        "inputs": {
            "visitors": {"label": "Visitors", "unit": "visitors", "min": 0.0, "good": "up"},
            "conversion_rate": {"label": "Conversion rate", "unit": "fraction 0–1", "min": 0.0, "max": 1.0, "good": "up"},
            "average_order_value": {"label": "Average order value", "unit": "currency/order", "min": 0.0, "good": "up"},
            "cost_per_visitor": {"label": "Cost per visitor", "unit": "currency/visitor", "min": 0.0, "good": "down"},
        },
        "assumptions": {},
        "outputs": {"orders": "Orders", "revenue": "Revenue", "acquisition_cost": "Acquisition cost",
                    "profit": "Profit", "cost_per_order": "Cost per order"},
        "primary_output": "profit",
        "additive_outputs": ["orders", "revenue", "acquisition_cost", "profit"],
        "ratio_outputs": {"cost_per_order": {"numerator": "acquisition_cost", "denominator": "orders", "scale": 1.0}},
        "formulas": ["orders = visitors × conversion_rate", "revenue = orders × average_order_value",
                     "acquisition_cost = visitors × cost_per_visitor", "profit = revenue − acquisition_cost",
                     "cost_per_order = acquisition_cost ÷ orders"],
    },
}


def describe_models() -> list[dict]:
    return [{"model": key, "title": m["title"], "inputs": m["inputs"], "assumptions": m["assumptions"],
             "outputs": m["outputs"], "primary_output": m["primary_output"], "formulas": m["formulas"]}
            for key, m in MODELS.items()]


def _model(name: "str | dict") -> dict:
    """A model spec from a built-in name, or a compiled custom spec passed through unchanged (app/scenario_models.py)."""
    if isinstance(name, dict):
        return name
    if name not in MODELS:
        raise ScenarioError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return MODELS[name]


def model_key(model: "str | dict") -> str:
    """The identifier of a model: the built-in name, or custom:<id>@<version> for a custom model."""
    return model["key"] if isinstance(model, dict) else model


for _name, _spec in MODELS.items():
    _spec["key"] = _name


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ScenarioError(f"{what} must be a finite number")
    return float(value)


def validate_inputs(model: str, inputs: dict) -> dict:
    spec = _model(model)["inputs"]
    unknown = set(inputs) - set(spec)
    if unknown:
        raise ScenarioError(f"Unknown input(s): {', '.join(sorted(unknown))}")
    clean = {}
    for name, rule in spec.items():
        if name not in inputs:
            raise ScenarioError(f"Missing input '{name}' ({rule['label']})")
        v = _number(inputs[name], rule["label"])
        if v < rule.get("min", -math.inf):
            raise ScenarioError(f"{rule['label']} cannot be below {rule['min']:g} (got {v:g})")
        if v > rule.get("max", math.inf):
            raise ScenarioError(f"{rule['label']} cannot be above {rule['max']:g} (got {v:g})")
        clean[name] = v
    return clean


def validate_assumptions(model: str, assumptions: dict | None) -> dict:
    spec = _model(model)["assumptions"]
    unknown = set(assumptions or {}) - set(spec)
    if unknown:
        raise ScenarioError(f"Unknown assumption(s): {', '.join(sorted(unknown))}")
    clean = {}
    for name, rule in spec.items():
        v = _number((assumptions or {}).get(name, rule["default"]), rule["label"])
        if not rule["min"] <= v <= rule["max"]:
            raise ScenarioError(f"{rule['label']} must be between {rule['min']:g} and {rule['max']:g}")
        clean[name] = v
    return clean


def apply_adjustments(model: str, baseline: dict, adjustments: dict | None, assumptions: dict) -> dict:
    """Adjusted inputs. An adjustment is {"type": "pct"|"abs"|"set", "value": x}."""
    spec = _model(model)["inputs"]
    out = dict(baseline)
    unknown = set(adjustments or {}) - set(spec)
    if unknown:
        raise ScenarioError(f"Cannot adjust unknown input(s): {', '.join(sorted(unknown))}")
    for name, adj in (adjustments or {}).items():
        if not isinstance(adj, dict) or adj.get("type") not in ("pct", "abs", "set"):
            raise ScenarioError(f"Adjustment for '{name}' needs type pct, abs or set")
        value = _number(adj.get("value"), f"Adjustment for {spec[name]['label']}")
        if adj["type"] == "pct":
            if value < -100:
                raise ScenarioError(f"{spec[name]['label']} cannot fall by more than 100%")
            out[name] = baseline[name] * (1 + value / 100)
        elif adj["type"] == "abs":
            out[name] = baseline[name] + value
        else:
            out[name] = value
    # The one behavioural link in the model, and it is an explicit assumption.
    elasticity = assumptions.get("price_elasticity", 0.0)
    if model_key(model) == "unit_economics" and elasticity and baseline["price"] and "volume" not in (adjustments or {}):
        price_change = out["price"] / baseline["price"] - 1
        out["volume"] = max(0.0, baseline["volume"] * (1 + elasticity * price_change))
    return validate_inputs(model, out)


def _outputs(model: str, inputs: dict, assumptions: dict) -> dict:
    return _model(model)["compute"](inputs, assumptions)


def _delta(base: dict, new: dict) -> dict:
    out = {}
    for k, v in new.items():
        b = base.get(k)
        out[k] = None if v is None or b is None else {"abs": v - b, "pct": (v - b) / abs(b) * 100 if b else None}
    return out


def outputs_vec(model: str, inputs: dict, assumptions: dict) -> dict:
    """Outputs for array-valued inputs (every value may be a scalar or an array)."""
    return _model(model)["compute_vec"](inputs, assumptions)


# ------------------------------------------------------------- break-even --

def break_even(model: str, inputs: dict, curve_points: int = 30) -> dict:
    """Where profit crosses zero, from the model's own formulas.

    unit_economics: profit(q) = (price - unit_cost) x q - fixed_cost, so the
    break-even quantity is fixed_cost / (price - unit_cost) when a unit earns
    a positive contribution, and does not exist when it does not.

    funnel: the model has no fixed cost, so profit = visitors x (conversion x order value - cost per
    visitor) is proportional to visitors. It is profitable at every scale or at none, and the break-even
    that exists is a threshold on conversion rate, order value or cost per visitor.
    """
    _model(model)
    if isinstance(model, dict):
        return None        # a custom model has no built-in break-even; use goal seek on its profit-like output
    if model == "unit_economics":
        out = _break_even_units(inputs)
    else:
        out = _break_even_funnel(inputs)
    out["model"] = model
    out["curve"] = _break_even_curve(model, inputs, out, curve_points)
    return out


def _break_even_units(i: dict) -> dict:
    price, volume, unit_cost, fixed = i["price"], i["volume"], i["unit_cost"], i["fixed_cost"]
    contribution = price - unit_cost
    out: dict[str, Any] = {
        "formulas": ["contribution per unit = price − unit_cost",
                     "break-even quantity = fixed_cost ÷ contribution per unit",
                     "break-even revenue = break-even quantity × price",
                     "safety margin % = (units sold − break-even quantity) ÷ units sold × 100"],
        "contribution_per_unit": contribution, "current_units": volume,
        "current_profit": contribution * volume - fixed,
    }
    if contribution > 0:
        q = fixed / contribution
        out.update(status="reachable", break_even_quantity=q, break_even_revenue=q * price,
                   distance_units=volume - q, additional_units_needed=max(0.0, q - volume),
                   above_break_even=volume >= q,
                   safety_margin_pct=(volume - q) / volume * 100 if volume > 0 else None,
                   explanation=(f"Each unit contributes {contribution:,.2f} towards fixed costs of {fixed:,.2f}, so "
                                f"{q:,.2f} units are needed to break even; {volume:,.2f} are sold."))
    elif contribution == 0 and fixed == 0:
        out.update(status="flat", explanation="Price equals unit cost and there is no fixed cost: profit is zero at "
                                              "every volume, so there is no single break-even point.")
    elif contribution == 0:
        out.update(status="never", explanation="Price equals unit cost, so no unit contributes towards the fixed "
                                               f"cost of {fixed:,.2f}: the business cannot break even at any volume.")
    else:
        out.update(status="never", explanation=f"Each unit sells for {-contribution:,.2f} less than it costs, so selling "
                                               "more increases the loss: there is no break-even volume at this price "
                                               "and unit cost.")
    return out


def _break_even_funnel(i: dict) -> dict:
    visitors, cr, aov, cpv = i["visitors"], i["conversion_rate"], i["average_order_value"], i["cost_per_visitor"]
    per_visitor = cr * aov - cpv
    out: dict[str, Any] = {
        "formulas": ["profit per visitor = conversion_rate × average_order_value − cost_per_visitor",
                     "break-even conversion rate = cost_per_visitor ÷ average_order_value",
                     "break-even order value = cost_per_visitor ÷ conversion_rate",
                     "break-even cost per visitor = conversion_rate × average_order_value"],
        "profit_per_visitor": per_visitor, "current_visitors": visitors, "current_profit": per_visitor * visitors,
        "break_even_conversion_rate": cpv / aov if aov > 0 else None,
        "break_even_average_order_value": cpv / cr if cr > 0 else None,
        "break_even_cost_per_visitor": cr * aov,
    }
    be_cr, be_aov, be_cpv = (out["break_even_conversion_rate"], out["break_even_average_order_value"],
                             out["break_even_cost_per_visitor"])
    out["safety_margin_pct"] = {
        "conversion_rate": (cr - be_cr) / cr * 100 if be_cr is not None and cr > 0 else None,
        "average_order_value": (aov - be_aov) / aov * 100 if be_aov is not None and aov > 0 else None,
        "cost_per_visitor": (be_cpv - cpv) / be_cpv * 100 if be_cpv > 0 else None,
    }
    note = " This model has no fixed cost, so the break-even is a threshold on the rates, not a visitor count."
    if per_visitor > 0:
        out.update(status="profitable_at_any_scale", break_even_visitors=0.0, above_break_even=True,
                   explanation=f"Each visitor earns {per_visitor:,.4f} after acquisition cost, so profit is positive at "
                               "any number of visitors." + note)
    elif per_visitor == 0:
        out.update(status="flat", break_even_visitors=None, above_break_even=False,
                   explanation="Each visitor earns exactly what it costs: profit is zero at any number of visitors." + note)
    else:
        out.update(status="never", break_even_visitors=None, above_break_even=False,
                   explanation=f"Each visitor loses {-per_visitor:,.4f}, so more visitors increase the loss." + note)
    return out


def _break_even_curve(model: str, i: dict, be: dict, points: int) -> dict:
    """Profit along the break-even variable, from the model's own array formulas, for the chart."""
    points = max(5, min(int(points), 200))
    if model == "unit_economics":
        top = max(i["volume"], be.get("break_even_quantity") or 0.0) * 1.5 or 1.0
        x = np.linspace(0.0, top, points)
        profit = outputs_vec(model, {**i, "volume": x}, {})["profit"]
        return {"x_label": "Units sold", "x": x.tolist(), "profit": profit.tolist(), "current_x": i["volume"],
                "break_even_x": be.get("break_even_quantity")}
    ref = max(i["conversion_rate"], be.get("break_even_conversion_rate") or 0.0)
    top = min(1.0, ref * 1.5) if ref > 0 else 1.0
    x = np.linspace(0.0, top, points)
    profit = outputs_vec(model, {**i, "conversion_rate": x}, {})["profit"]
    return {"x_label": "Conversion rate", "x": x.tolist(), "profit": profit.tolist(),
            "current_x": i["conversion_rate"], "break_even_x": be.get("break_even_conversion_rate")}


def shapley(fn: Callable[[dict], float], base: dict, scenario: dict, *, max_exact: int = 10, samples: int = 2000,
            seed: int = 0) -> dict:
    """How much of the change in an output each changed input accounts for, with how it was computed.

    Inputs interact (price x volume), so "the effect of price" depends on the order inputs are changed in. The Shapley
    value averages over every order. With up to `max_exact` changed inputs it is computed exactly from the subset
    formula, evaluating the output once per subset (2^n evaluations, not n! orders). Beyond that it is estimated by
    sampling random orders (each also taken reversed), with a standard error per input. Either way the contributions
    sum exactly to the total change, because every order's contributions do."""
    changed = [k for k in base if not math.isclose(base[k], scenario[k], rel_tol=1e-12, abs_tol=1e-12)]
    n = len(changed)
    if n == 0:
        return {"values": {}, "method": "exact", "changed_inputs": 0, "samples": None, "standard_errors": None}
    if n <= max_exact:
        value: dict[int, float] = {}

        def f(mask: int) -> float:
            if mask not in value:
                value[mask] = fn({**base, **{changed[i]: scenario[changed[i]] for i in range(n) if mask >> i & 1}})
            return value[mask]

        weight = [math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) for k in range(n)]
        values = {}
        for i, name in enumerate(changed):
            total = 0.0
            for mask in range(1 << n):
                if not mask >> i & 1:
                    total += weight[bin(mask).count("1")] * (f(mask | 1 << i) - f(mask))
            values[name] = total
        return {"values": values, "method": "exact", "changed_inputs": n, "samples": None, "standard_errors": None}
    rng = random.Random(seed)
    pairs = max(1, samples // 2)
    start = fn(dict(base))
    sums = {k: 0.0 for k in changed}
    sq = {k: 0.0 for k in changed}
    for _ in range(pairs):
        order = changed[:]
        rng.shuffle(order)
        avg = {k: 0.0 for k in changed}
        for seq in (order, order[::-1]):
            current, previous = dict(base), start
            for k in seq:
                current[k] = scenario[k]
                val = fn(current)
                avg[k] += (val - previous) / 2
                previous = val
        for k in changed:
            sums[k] += avg[k]
            sq[k] += avg[k] ** 2
    values = {k: sums[k] / pairs for k in changed}
    se = {k: math.sqrt(max(0.0, sq[k] / pairs - values[k] ** 2) / pairs) for k in changed}
    return {"values": values, "method": "sampled", "changed_inputs": n, "samples": pairs * 2, "standard_errors": se}


def shapley_contributions(fn: Callable[[dict], float], base: dict, scenario: dict, **kw) -> dict[str, float]:
    return shapley(fn, base, scenario, **kw)["values"]


def _enforce(spec: dict, label: str, inputs: dict, assumptions: dict, outputs: dict, notes: dict, key: str) -> None:
    """A custom model's constraints. A broken error-level constraint rejects the scenario like an impossible input;
    a warning is returned beside the result."""
    check = spec.get("constraint_violations")
    if check is None:
        return
    found = check({**inputs, **assumptions, **{k: (math.nan if v is None else v) for k, v in outputs.items()}})
    errors = [v for v in found if v["severity"] == "error"]
    if errors:
        raise ScenarioError(f"{label} breaks the model's constraint: {errors[0]['message']}")
    if found:
        notes[key] = found


def evaluate(model: "str | dict", baseline: dict, scenarios: dict[str, dict] | None = None,
             assumptions: dict | None = None, sensitivity_pct: float = 10.0) -> dict:
    """Evaluate the baseline and each named scenario, with a sensitivity table
    and a contribution breakdown for the model's primary output."""
    spec = _model(model)
    baseline = validate_inputs(model, baseline)
    assumptions = validate_assumptions(model, assumptions)
    if not 0 < sensitivity_pct <= 100:
        raise ScenarioError("sensitivity_pct must be between 0 and 100")
    if scenarios is not None and len(scenarios) > 12:
        raise ScenarioError("At most 12 scenarios per evaluation")
    primary = spec["primary_output"]
    base_out = _outputs(model, baseline, assumptions)
    constraint_notes: dict[str, list] = {}
    _enforce(spec, "The baseline", baseline, assumptions, base_out, constraint_notes, "baseline")

    def primary_of(inputs: dict) -> float:
        return _outputs(model, inputs, assumptions)[primary] or 0.0

    results = {}
    for name, adjustments in (scenarios or {}).items():
        inputs = apply_adjustments(model, baseline, adjustments, assumptions)
        outputs = _outputs(model, inputs, assumptions)
        _enforce(spec, f"Scenario '{name}'", inputs, assumptions, outputs, constraint_notes, str(name)[:60])
        sh = shapley(primary_of, baseline, inputs)
        results[str(name)[:60]] = {
            "adjustments": adjustments, "inputs": inputs, "outputs": outputs,
            "delta_vs_baseline": _delta(base_out, outputs),
            "contributions": {"output": primary, "by_input": sh["values"],
                              "method": ("Shapley average over the order in which inputs change" if sh["method"] == "exact" else
                                         f"Shapley average estimated from {sh['samples']:,} sampled orders (approximate)"),
                              "computation": {k: sh[k] for k in ("method", "changed_inputs", "samples", "standard_errors")}},
        }

    sensitivity = []
    for name, rule in spec["inputs"].items():
        low_in, high_in = dict(baseline), dict(baseline)
        low_in[name] = max(rule.get("min", -math.inf), baseline[name] * (1 - sensitivity_pct / 100))
        high_in[name] = min(rule.get("max", math.inf), baseline[name] * (1 + sensitivity_pct / 100))
        low, high = primary_of(low_in), primary_of(high_in)
        sensitivity.append({"input": name, "label": rule["label"], "low_input": low_in[name],
                            "high_input": high_in[name], "low": low, "high": high, "range": abs(high - low)})
    sensitivity.sort(key=lambda s: -s["range"])

    unsupported = ["Inputs change independently of each other, except where an assumption below links them.",
                   "Costs and prices scale linearly; no volume discounts, capacity limits or step costs."]
    if isinstance(model, dict):
        unsupported = ["Inputs change independently of each other, except where the model's own formulas link them.",
                       "This is a custom model: every relationship is a formula in its definition, and the engine adds no "
                       "behavioural link of its own (no price elasticity, no scaling)."]
    elif "price_elasticity" in assumptions:
        e = assumptions["price_elasticity"]
        unsupported.append(
            "Price elasticity is 0: volume is assumed not to react to price at all." if e == 0 else
            f"Price elasticity of {e:g} is a stated assumption. It was not estimated from the dataset.")
    return {
        "kind": "scenario", "data_kind": "scenario", "disclaimer": DISCLAIMER, "model": model_key(model), "title": spec["title"],
        "engine_version": ENGINE_VERSION, "constraint_notes": constraint_notes,
        "formulas": spec["formulas"], "assumptions": assumptions, "unsupported_assumptions": unsupported,
        "baseline": {"inputs": baseline, "outputs": base_out},
        "scenarios": results, "primary_output": primary,
        "sensitivity": {"pct": sensitivity_pct, "output": primary, "rows": sensitivity},
        "break_even": ({"baseline": break_even(model, baseline),
                        "scenarios": {name: break_even(model, r["inputs"]) for name, r in results.items()}}
                       if not isinstance(model, dict) else None),
    }


def standard_scenarios(model: str, spread_pct: float) -> dict[str, dict]:
    """Optimistic and pessimistic variants: every input moved `spread_pct` in
    its favourable / unfavourable direction. A convenience starting point."""
    if not 0 < spread_pct <= 100:
        raise ScenarioError("spread_pct must be between 0 and 100")
    spec = _model(model)["inputs"]
    sign = {"up": 1, "down": -1}
    return {
        "optimistic": {k: {"type": "pct", "value": sign[r["good"]] * spread_pct} for k, r in spec.items()},
        "pessimistic": {k: {"type": "pct", "value": -sign[r["good"]] * spread_pct} for k, r in spec.items()},
    }


# ------------------------------------------------------- baseline from data --

def baseline_from_dataset(df: pd.DataFrame, profile: dict, model: str, mapping: dict,
                          filters: list | None = None) -> dict:
    """Derive baseline inputs from dataset columns. Each input comes back with
    its source: the formula it was computed with, or "assumed" when the dataset
    has nothing for it and the caller must supply it."""
    spec = _model(model)["inputs"]
    numeric = {c["name"] for c in profile.get("columns", []) if c.get("kind") == "numeric"}

    def col(key: str) -> str | None:
        name = mapping.get(key)
        if name is None:
            return None
        if name not in numeric:
            raise ScenarioError(f"'{name}' is not a numeric column of this dataset")
        return name

    def value(formula: str) -> float | None:
        try:
            return me.evaluate(df, formula, filters)["value"]
        except me.FormulaError as e:
            raise ScenarioError(str(e)) from e

    inputs: dict[str, float | None] = {k: None for k in spec}
    sources: dict[str, dict] = {}

    def setv(name: str, formula: str) -> None:
        inputs[name] = value(formula)
        sources[name] = {"source": "dataset", "formula": formula}

    if model == "unit_economics":
        qty, amount, price = col("quantity_column"), col("amount_column"), col("price_column")
        unit_cost, total_cost = col("unit_cost_column"), col("total_cost_column")
        if qty is None:
            raise ScenarioError("unit_economics needs a quantity column")
        setv("volume", f'sum("{qty}")')
        if amount is not None:
            setv("price", f'sum("{amount}") / sum("{qty}")')
        elif price is not None:
            setv("price", f'avg("{price}")')
            sources["price"]["note"] = "simple average of the price column; not weighted by quantity"
        if total_cost is not None:
            setv("unit_cost", f'sum("{total_cost}") / sum("{qty}")')
        elif unit_cost is not None:
            setv("unit_cost", f'avg("{unit_cost}")')
    else:
        visitors, orders, revenue, spend = (col("visitors_column"), col("orders_column"),
                                            col("revenue_column"), col("spend_column"))
        if visitors is None or orders is None:
            raise ScenarioError("funnel needs a visitors column and an orders column")
        setv("visitors", f'sum("{visitors}")')
        setv("conversion_rate", f'sum("{orders}") / sum("{visitors}")')
        if revenue is not None:
            setv("average_order_value", f'sum("{revenue}") / sum("{orders}")')
        if spend is not None:
            setv("cost_per_visitor", f'sum("{spend}") / sum("{visitors}")')
    missing = [k for k, v in inputs.items() if v is None]
    for k in missing:
        sources[k] = {"source": "assumed", "formula": None,
                      "note": "The dataset has no column mapped for this input; enter a value."}
    return {"model": model, "inputs": inputs, "sources": sources, "missing": missing,
            "rows_used": int(len(me.apply_filters(df, filters)))}
