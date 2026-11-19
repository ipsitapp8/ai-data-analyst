"""Government and public services: population, service capacity, budgets."""
from __future__ import annotations

from .base import HYPOTHETICAL, Concept, Domain, InputSpec, Template

CONCEPTS = (
    Concept("population", "Population", frozenset({"population", "residents", "inhabitants", "households"}), "numeric", 1.5),
    Concept("capacity", "Service capacity", frozenset({"capacity", "seats", "routes", "buses", "vehicles"}), "numeric", 1.0),
    Concept("budget", "Budget", frozenset({"budget", "allocation", "grant", "funding", "expenditure"}), "numeric", 1.5),
    Concept("district", "District / area", frozenset({"district", "ward", "zone", "region", "area", "municipality", "county", "state"}), "category", 1.0),
    Concept("ridership", "Service use", frozenset({"ridership", "passengers", "riders", "usage", "users", "beneficiaries"}), "numeric", 1.0),
    Concept("date", "Date", frozenset({"date", "year", "month"}), "datetime", 0.3),
)

DEMAND = {
    "inputs": {"population": {"label": "Population", "unit": "people", "min": 0},
               "usage_rate": {"label": "Share of people using the service", "unit": "fraction", "min": 0, "max": 1},
               "capacity": {"label": "Service capacity", "unit": "people", "min": 0, "good": "up"}},
    "outputs": {"demand": {"label": "People needing the service", "unit": "people", "formula": "population * usage_rate"},
                "utilisation_pct": {"label": "Capacity used %", "unit": "percent", "formula": "population * usage_rate / capacity * 100"},
                "shortfall": {"label": "People beyond capacity", "unit": "people", "formula": "max(population * usage_rate - capacity, 0)"}},
    "primary_output": "utilisation_pct"}
DEMAND_BASE = (InputSpec("population", "Population", "people", None, 0, None, from_data="sum:population", role="baseline"),
               InputSpec("usage_rate", "Share of people using the service (0-1)", "fraction", None, 0, 1, role="baseline", help="An assumption unless you have usage data"),
               InputSpec("capacity", "Service capacity (people served)", "people", None, 0, None, from_data="sum:capacity", role="baseline"))

POP = Template(id="gov_population", title="Population changes", question="What if the population changes by {population_change}%?", method="deterministic",
               operation="formula_model", model=DEMAND, manual=True, evidence=HYPOTHETICAL,
               inputs=(*DEMAND_BASE, InputSpec("population_change", "Change in population", "%", None, -100, 1000, role="change", target="population", change_type="pct")),
               reason="Demand is population × the share who use the service; compared with capacity.", tags=frozenset({"population", "growth", "demand", "residents"}),
               patterns=(r"population\s+(?:of\s+\w+\s+)?(?:increase|decrease|grow|change|rise|fall)", r"population\s+growth"),
               limitation="Assumes the share of people using the service stays the same.")
CAP = Template(id="gov_capacity", title="Service capacity reduced", question="What if public transport capacity changes by {capacity_change}%?", method="deterministic",
               operation="formula_model", model=DEMAND, manual=True, evidence=HYPOTHETICAL,
               inputs=(*DEMAND_BASE, InputSpec("capacity_change", "Change in capacity", "%", None, -100, 1000, role="change", target="capacity", change_type="pct")),
               reason="A capacity change shows up directly in utilisation and shortfall.", tags=frozenset({"capacity", "transport", "reduced", "service"}),
               patterns=(r"(?:transport|service)\s+capacity", r"capacity\s+(?:is\s+)?(?:reduced|cut|increased)"),
               limitation="Ignores people switching to other services.")
BUDGET_MODEL = {"inputs": {"budget": {"label": "Project budget", "unit": "currency", "min": 0, "good": "up"},
                           "cost_per_unit": {"label": "Cost per unit delivered", "unit": "currency/units", "min": 0, "good": "down"}},
                "outputs": {"units_funded": {"label": "Units that can be funded", "unit": "units", "formula": "budget / cost_per_unit"}},
                "primary_output": "units_funded"}
BUDGET = Template(id="gov_budget", title="Project budget changes", question="What if the project budget changes by {budget_change}%?", method="deterministic",
                  operation="formula_model", model=BUDGET_MODEL, manual=True, evidence=HYPOTHETICAL,
                  inputs=(InputSpec("budget", "Project budget", "currency", None, 0, None, from_data="sum:budget", role="baseline"),
                          InputSpec("cost_per_unit", "Cost per unit delivered", "currency/unit", None, 0, None, role="baseline"),
                          InputSpec("budget_change", "Change in budget", "%", None, -100, 1000, role="change", target="budget", change_type="pct")),
                  reason="Units funded = budget ÷ cost per unit; the unit cost is yours to supply.", tags=frozenset({"budget", "cut", "project", "funding"}),
                  patterns=(r"budget\s+(?:for\s+\w+\s+(?:\w+\s+)?)?(?:decrease|increase|cut|reduc|change)", r"(?:project|public)\s+budget"),
                  limitation="Assumes cost per unit does not change with scale.")
NEWLINE = Template(id="gov_new_line", title="A new service opens", question="What if a new metro line opens?", method="deterministic", operation="none",
                   external=("expected ridership of the new service", "its capacity"), reason="Needs forecasts from outside the dataset.",
                   tags=frozenset({"new", "metro", "line", "open"}), patterns=(r"new\s+(?:metro|line|route|hospital|school)",), limitation="No dataset can say how many people would use a service that does not exist yet.")

GOVERNMENT = Domain(key="government", title="Government & public services", description="Population, service capacity and project budgets.",
                    entities=("district", "service"), concepts=CONCEPTS, templates=(POP, CAP, BUDGET, NEWLINE), metrics=("service utilisation", "units funded"),
                    keywords=frozenset({"municipal", "public", "census", "civic"}), notes=("Scenarios describe arithmetic on stated values; they are not policy recommendations.",))
