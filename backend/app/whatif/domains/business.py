"""Business: price, cost, volume, marketing, revenue, headcount. The price/cost/break-even templates run on the built-in
`unit_economics` model that the existing What-If page uses; nothing here re-implements it."""
from __future__ import annotations

from .base import ESTIMATE, HYPOTHETICAL, Concept, Domain, InputSpec, Template

CONCEPTS = (
    Concept("revenue", "Revenue", frozenset({"revenue", "sales", "turnover", "income", "gmv"}), "numeric", 1.0),
    Concept("marketing", "Marketing spend", frozenset({"marketing", "advertising", "ads", "ad", "promotion", "campaign"}), "numeric", 1.5),
    Concept("cost", "Cost", frozenset({"cost", "costs", "expense", "expenses", "cogs", "spend"}), "numeric", 1.0),
    Concept("units", "Units sold", frozenset({"units", "quantity", "qty", "volume", "orders"}), "numeric", 1.0),
    Concept("price", "Price", frozenset({"price", "unit_price"}), "numeric", 1.0),
    Concept("profit", "Profit", frozenset({"profit", "margin", "earnings"}), "numeric", 1.0),
    Concept("customer", "Customer", frozenset({"customer", "customers", "client", "clients"}), "id", 0.5),
    Concept("date", "Date", frozenset({"date", "month", "day", "week"}), "datetime", 0.3),
)

UNIT_INPUTS = (
    InputSpec("price", "Price per unit", "currency/unit", None, 0, None, role="baseline"),
    InputSpec("volume", "Units sold", "units", None, 0, None, role="baseline"),
    InputSpec("unit_cost", "Variable cost per unit", "currency/unit", None, 0, None, role="baseline"),
    InputSpec("fixed_cost", "Fixed cost", "currency", None, 0, None, role="baseline", help="Rarely in a sales file; you supply it"))

MARKETING = Template(
    id="biz_marketing_spend", title="Change marketing spend", question="What if marketing spend changes by {change}%?",
    method="regression", operation="association", requires=("marketing", "revenue"), evidence=ESTIMATE,
    inputs=(InputSpec("change", "Change in marketing spend", "%", None, -100, 1000),),
    params={"x": "marketing", "y": "revenue", "mode": "pct", "x_unit": "currency", "y_unit": "currency"},
    reason="Your data holds marketing spend and revenue, so their historical relationship can be checked before any effect is claimed.",
    tags=frozenset({"marketing", "advertising", "spend", "revenue", "campaign"}),
    patterns=(r"marketing\s+(?:spend\w*|expenditure|budget)", r"advertis\w+\s+(?:spend\w*|budget)", r"(?:spend|spending)\s+on\s+(?:marketing|ads|advertising)"),
    limitation="Shows what past marketing spend went with; it does not prove extra spend would produce extra revenue.")

PRICE = Template(
    id="biz_price_change", title="Change the price", question="What if the price changes by {price_change}%?",
    method="deterministic", operation="formula_model", model="unit_economics", manual=True, requires=("units", "revenue", "cost"),
    inputs=(*UNIT_INPUTS, InputSpec("price_elasticity", "Price elasticity of volume (0 = volume does not react)", "", 0.0, -10, 10, role="assumption"),
            InputSpec("price_change", "Price change", "%", None, -100, 1000, role="change", target="price", change_type="pct")),
    reason="Price, volume and cost can be read from your data; the unit-economics model then gives the new profit.",
    tags=frozenset({"price", "pricing", "profit", "revenue", "margin"}),
    patterns=(r"\bprices?\b", r"\bpricing\b"),
    limitation="By default volume does not react to price. Set an elasticity if you believe it would, and it is then a stated assumption.")

COST = Template(
    id="biz_cost_change", title="Change production cost", question="What if production costs change by {cost_change}%?",
    method="deterministic", operation="formula_model", model="unit_economics", manual=True, requires=("units", "revenue", "cost"),
    inputs=(*UNIT_INPUTS, InputSpec("cost_change", "Change in cost per unit", "%", None, -100, 1000, role="change", target="unit_cost", change_type="pct")),
    reason="Unit cost, volume and price give the effect on profit directly.",
    tags=frozenset({"cost", "costs", "production", "expense", "profit", "inflation"}),
    patterns=(r"(?:production|manufacturing|unit|variable|input|material)\s+costs?",),
    limitation="Assumes price and volume are unchanged.")

BREAK_EVEN = Template(
    id="biz_break_even", title="Break-even volume", question="How many units must we sell to break even?",
    method="break_even", operation="formula_model", model="unit_economics", manual=True, requires=("units", "revenue", "cost"),
    inputs=UNIT_INPUTS, reason="With price, unit cost and fixed cost, the break-even volume is exact.",
    tags=frozenset({"break", "even", "breakeven", "loss", "units"}),
    patterns=(r"break[\s-]?even",), limitation="Assumes price and unit cost stay constant at every volume.")

HIRE = Template(
    id="biz_hire", title="Hire additional employees", question="What if we hire {extra_employees} more employees?",
    method="deterministic", operation="formula_model", manual=True, evidence=HYPOTHETICAL,
    model={"inputs": {"employees": {"label": "Employees", "unit": "employees", "min": 0}, "cost_per_employee": {"label": "Annual cost per employee", "unit": "currency/employees", "min": 0, "good": "down"},
                      "revenue_per_employee": {"label": "Annual revenue per employee", "unit": "currency/employees", "min": 0, "good": "up"}},
           "outputs": {"payroll": {"label": "Payroll", "unit": "currency", "formula": "employees * cost_per_employee", "additive": True},
                       "revenue": {"label": "Revenue attributed to staff", "unit": "currency", "formula": "employees * revenue_per_employee", "additive": True},
                       "net": {"label": "Net contribution", "unit": "currency", "formula": "revenue - payroll", "additive": True}},
           "primary_output": "net"},
    inputs=(InputSpec("employees", "Current employees", "employees", None, 0, None, role="baseline"),
            InputSpec("cost_per_employee", "Annual cost per employee", "currency", None, 0, None, role="baseline"),
            InputSpec("revenue_per_employee", "Annual revenue per employee", "currency", None, 0, None, role="baseline",
                      help="An assumption: a new hire may not produce the average"),
            InputSpec("extra_employees", "Additional employees", "employees", None, 0, None, role="change", target="employees", change_type="abs")),
    reason="Headcount, cost and productivity are rarely in a sales file, so you supply them.",
    tags=frozenset({"hire", "hiring", "employees", "headcount", "staff", "payroll"}),
    patterns=(r"\bhire\b", r"additional\s+employees", r"more\s+(?:employees|staff)"),
    limitation="Assumes a new hire is as productive as the average, which is a hypothesis.")

SENSITIVITY = Template(
    id="biz_sensitivity", title="Which inputs matter most for profit?", question="Which inputs move profit the most?",
    method="sensitivity", operation="formula_model", model="unit_economics", manual=True, requires=("units", "revenue", "cost"),
    inputs=UNIT_INPUTS, reason="Moves each input by the same percentage to rank their effect on profit.",
    tags=frozenset({"sensitivity", "drivers", "matter", "important", "inputs"}),
    patterns=(r"which\s+(?:inputs?|factors?|drivers?)", r"sensitiv"), limitation="One input at a time, from the same baseline.")

BUSINESS = Domain(
    key="business", title="Business", description="Pricing, cost, volume, marketing and headcount decisions.", entities=("customer", "product", "order"),
    concepts=CONCEPTS, templates=(MARKETING, PRICE, COST, BREAK_EVEN, SENSITIVITY, HIRE),
    metrics=("revenue", "profit", "margin", "break-even volume"), keywords=frozenset({"sku", "invoice", "order"}),
    notes=("Scenarios are arithmetic on the inputs you set; they are not forecasts.",))
