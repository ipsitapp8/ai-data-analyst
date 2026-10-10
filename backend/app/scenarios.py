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
from typing import Any, Callable

import pandas as pd

from app import metric_engine as me

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


MODELS: dict[str, dict[str, Any]] = {
    "unit_economics": {
        "title": "Unit economics (price, volume, cost)",
        "compute": _unit_economics,
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
        "formulas": ["revenue = price × volume", "variable_cost = unit_cost × volume",
                     "gross_profit = revenue − variable_cost", "profit = gross_profit − fixed_cost",
                     "margin_pct = profit ÷ revenue × 100",
                     "volume after a price change = volume × (1 + price_elasticity × price change %)"],
    },
    "funnel": {
        "title": "Acquisition funnel (visitors, conversion, order value)",
        "compute": _funnel,
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
        "formulas": ["orders = visitors × conversion_rate", "revenue = orders × average_order_value",
                     "acquisition_cost = visitors × cost_per_visitor", "profit = revenue − acquisition_cost",
                     "cost_per_order = acquisition_cost ÷ orders"],
    },
}


def describe_models() -> list[dict]:
    return [{"model": key, "title": m["title"], "inputs": m["inputs"], "assumptions": m["assumptions"],
             "outputs": m["outputs"], "primary_output": m["primary_output"], "formulas": m["formulas"]}
            for key, m in MODELS.items()]


def _model(name: str) -> dict:
    if name not in MODELS:
        raise ScenarioError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
    return MODELS[name]


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
    if model == "unit_economics" and elasticity and baseline["price"] and "volume" not in (adjustments or {}):
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


def shapley_contributions(fn: Callable[[dict], float], base: dict, scenario: dict) -> dict[str, float]:
    """How much of the change in an output each changed input accounts for.

    Inputs interact (price x volume), so "the effect of price" depends on the
    order inputs are changed in. The Shapley value averages over every order;
    the contributions always sum exactly to the total change."""
    changed = [k for k in base if not math.isclose(base[k], scenario[k], rel_tol=1e-12, abs_tol=1e-12)]
    contributions = {k: 0.0 for k in changed}
    if not changed:
        return contributions
    orders = list(itertools.permutations(changed))
    for order in orders:
        current = dict(base)
        previous = fn(current)
        for k in order:
            current[k] = scenario[k]
            value = fn(current)
            contributions[k] += value - previous
            previous = value
    return {k: v / len(orders) for k, v in contributions.items()}


def evaluate(model: str, baseline: dict, scenarios: dict[str, dict] | None = None,
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

    def primary_of(inputs: dict) -> float:
        return _outputs(model, inputs, assumptions)[primary] or 0.0

    results = {}
    for name, adjustments in (scenarios or {}).items():
        inputs = apply_adjustments(model, baseline, adjustments, assumptions)
        outputs = _outputs(model, inputs, assumptions)
        results[str(name)[:60]] = {
            "adjustments": adjustments, "inputs": inputs, "outputs": outputs,
            "delta_vs_baseline": _delta(base_out, outputs),
            "contributions": {"output": primary, "by_input": shapley_contributions(primary_of, baseline, inputs),
                              "method": "Shapley average over the order in which inputs change"},
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
    if "price_elasticity" in assumptions:
        e = assumptions["price_elasticity"]
        unsupported.append(
            "Price elasticity is 0: volume is assumed not to react to price at all." if e == 0 else
            f"Price elasticity of {e:g} is a stated assumption. It was not estimated from the dataset.")
    return {
        "kind": "scenario", "disclaimer": DISCLAIMER, "model": model, "title": spec["title"],
        "formulas": spec["formulas"], "assumptions": assumptions, "unsupported_assumptions": unsupported,
        "baseline": {"inputs": baseline, "outputs": base_out},
        "scenarios": results, "primary_output": primary,
        "sensitivity": {"pct": sensitivity_pct, "output": primary, "rows": sensitivity},
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
