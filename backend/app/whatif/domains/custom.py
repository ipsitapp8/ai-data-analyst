"""Custom: no sector assumptions. The user names the variables, baselines, formulas and constraints, or points at any numeric
column of any table. The generic templates below are also offered, bound to real columns, for datasets of every domain."""
from __future__ import annotations

from .base import ESTIMATE, FORECAST, Concept, Domain, InputSpec, Template

GENERIC_SHIFT = Template(
    id="generic_shift", title="Change a column", question="What if {metric} changes by {change}%?", method="deterministic", operation="shift_metric",
    requires=("metric",), inputs=(InputSpec("change", "Change", "%", None, -100, 1000),),
    params={"concept": "metric", "mode": "pct", "noun": "records"},
    reason="A plain what-if on one of your columns: the new average and total, with nothing else allowed to react.",
    tags=frozenset({"change", "increase", "decrease"}), limitation="Applies the same percentage to every record; shows arithmetic, not consequences.")

GENERIC_ASSOCIATION = Template(
    id="generic_association", title="Change a driver", question="What if {x} changes by {change}%: what goes with it in {y}?", method="regression", operation="association",
    requires=("x", "y"), evidence=ESTIMATE, inputs=(InputSpec("change", "Change in the driver", "%", None, -100, 1000),),
    params={"x": "x", "y": "y", "mode": "pct"},
    reason="These two columns move together in your data, so what a change might go with can be estimated, if the evidence is reliable.",
    tags=frozenset({"relationship", "driver", "effect"}), limitation="An association in past data, not proof of cause.")

GENERIC_THRESHOLD = Template(
    id="generic_threshold", title="Move a cut-off", question="What if the cut-off on {metric} moves from {current_threshold} to {new_threshold}?",
    method="deterministic", operation="threshold", requires=("metric",),
    inputs=(InputSpec("current_threshold", "Current cut-off", help="Records at or above this value meet the cut-off today"),
            InputSpec("new_threshold", "New cut-off")),
    params={"concept": "metric", "noun": "records", "rule": "the cut-off"},
    reason="How many records are at or above a value you choose, before and after it moves.",
    tags=frozenset({"threshold", "cutoff", "minimum", "limit"}), limitation="Counts existing records against the new cut-off; it does not predict how values would change in response.")

GENERIC_FORECAST = Template(
    id="generic_forecast", title="Continue a trend", question="What if the monthly trend in records by {date} continues for {periods_ahead} more period(s)?",
    method="forecast", operation="forecast_trend", requires=("date",), evidence=FORECAST,
    inputs=(InputSpec("periods_ahead", "Periods ahead", "months", 1, 1, 24),), params={"date": "date", "noun": "Records per month", "freq": "M"},
    reason="Records per month, counted from a date column, projected forward.",
    tags=frozenset({"forecast", "trend", "future"}), limitation="A straight-line trend; it ignores seasons, policy changes and capacity limits.")

CUSTOM_MODEL = Template(
    id="custom_model", title="Define your own model", question="What if my own variables change?", method="deterministic", operation="custom_model", manual=True,
    reason="Name your variables, baseline values, formulas, assumptions and constraints; Silt runs them on the same engine as the built-in models.",
    tags=frozenset({"custom", "own", "define", "model"}),
    limitation="Silt checks your formulas but cannot check that they describe reality.")

GENERIC_TEMPLATES = {t.id: t for t in (GENERIC_SHIFT, GENERIC_ASSOCIATION, GENERIC_THRESHOLD, GENERIC_FORECAST)}

CUSTOM = Domain(key="custom", title="Custom", description="Your own variables, formulas and constraints, or any column of any table.", entities=(), concepts=(),
                templates=(CUSTOM_MODEL,), metrics=(), notes=("Formulas are yours; Silt validates them and runs them but does not vouch for them.",))
