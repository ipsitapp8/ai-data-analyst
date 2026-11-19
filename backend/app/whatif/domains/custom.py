"""Custom: no sector assumptions. The user names the variables, baselines, formulas and constraints, or points at any numeric
column of any table. The generic templates below are also offered, bound to real columns, for datasets of every domain."""
from __future__ import annotations

from .base import ESTIMATE, Concept, Domain, InputSpec, Template

GENERIC_SHIFT = Template(
    id="generic_shift", title="Change a column", question="What if {column} changes by {change}%?", method="deterministic", operation="shift_metric",
    requires=("metric",), inputs=(InputSpec("change", "Change", "%", None, -100, 1000),),
    params={"concept": "metric", "mode": "pct", "noun": "records"},
    reason="A plain what-if on one of your columns: the new average and total, with nothing else allowed to react.",
    tags=frozenset({"change", "increase", "decrease"}), limitation="Applies the same percentage to every record; shows arithmetic, not consequences.")

GENERIC_ASSOCIATION = Template(
    id="generic_association", title="Change a driver", question="What if {x} changes by {change}%?", method="regression", operation="association",
    requires=("x", "y"), evidence=ESTIMATE, inputs=(InputSpec("change", "Change in the driver", "%", None, -100, 1000),),
    params={"x": "x", "y": "y", "mode": "pct"},
    reason="These two columns move together in your data, so what a change might go with can be estimated, if the evidence is reliable.",
    tags=frozenset({"relationship", "driver", "effect"}), limitation="An association in past data, not proof of cause.")

CUSTOM_MODEL = Template(
    id="custom_model", title="Define your own model", question="What if my own variables change?", method="deterministic", operation="custom_model", manual=True,
    reason="Name your variables, baseline values, formulas, assumptions and constraints; Silt runs them on the same engine as the built-in models.",
    tags=frozenset({"custom", "own", "define", "model"}),
    limitation="Silt checks your formulas but cannot check that they describe reality.")

GENERIC_TEMPLATES = {"generic_shift": GENERIC_SHIFT, "generic_association": GENERIC_ASSOCIATION}

CUSTOM = Domain(key="custom", title="Custom", description="Your own variables, formulas and constraints, or any column of any table.", entities=(), concepts=(),
                templates=(CUSTOM_MODEL,), metrics=(), notes=("Formulas are yours; Silt validates them and runs them but does not vouch for them.",))
