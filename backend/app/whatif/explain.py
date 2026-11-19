"""The eight-part explanation every result carries, written from the scenario and its result alone (no language model).

    changed · held constant · data used · method · assumptions · result · reliability · what cannot be concluded

It also names what kind of statement the result is, so an exact count is never mistaken for a forecast or a hypothesis.
"""
from __future__ import annotations

from app.whatif.domains.base import ESTIMATE, EXACT, FORECAST, HYPOTHETICAL

KIND_LABEL = {EXACT: "Exact calculation", ESTIMATE: "Statistical estimate", FORECAST: "Forecast", HYPOTHETICAL: "Hypothetical assumption",
              "unsupported_conclusion": "Unsupported conclusion"}
METHOD_TEXT = {
    "deterministic": "A deterministic calculation: the same inputs always give the same answer.",
    "regression": "A least-squares fit of the relationship in your historical data, with a 95% confidence interval on its slope.",
    "forecast": "A straight-line trend over past periods, projected forward with a 95% prediction interval.",
    "monte_carlo": "A Monte Carlo simulation: many random draws from the input distributions you stated, with a fixed seed so the run can be repeated.",
    "sensitivity": "A sensitivity analysis: each input is moved in turn from the same baseline to rank its effect.",
    "break_even": "A break-even calculation on the model's own formulas.",
    "goal_seek": "A goal seek: the model's formulas are solved for the input value that reaches your target.",
}
RELIABILITY = {
    EXACT: "High for what it states: it is an exact count or sum over the records in your table.",
    ESTIMATE: "Moderate and conditional: it is an estimate from past data with the interval shown; it does not establish cause and effect.",
    FORECAST: "Low to moderate: it extends a past trend and cannot anticipate changes in the underlying situation.",
    HYPOTHETICAL: "As reliable as the values you entered: the arithmetic is exact, but the inputs are your assumptions, not verified facts.",
    "unsupported_conclusion": "No estimate is given because the data does not support one.",
}


def _n(x) -> str:
    if x is None:
        return "—"
    return f"{x:,.0f}" if abs(x) >= 1000 or float(x).is_integer() else f"{x:,.2f}"


def explain(s: dict, template, domain) -> dict:
    r = s["results"]
    kind = r["evidence_kind"]
    metrics = {m["key"]: m for m in r["metrics"]}
    head = metrics.get(r.get("primary"))
    if r["status"] != "ok":
        headline = r.get("reason") or "The data does not support an answer to this question."
    elif head:
        pct = f" ({head['pct_change']:+.1f}%)" if head.get("pct_change") is not None else ""
        headline = f"{head['label']}: {_n(head['baseline'])} → {_n(head['scenario'])}{pct}."
    elif r["details"].get("goal_seek"):
        headline = r.get("reason") or "Goal seek finished."
    else:
        headline = "Scenario evaluated."
    changed = []
    if s["dataset"] and r["engine"] == "dataset-operation":
        changed = [f"{sp.label}: {s['inputs'][sp.name]:g}" for sp in template.inputs if sp.name in s["inputs"] and sp.role in ("value", "change")]
    changed = r["details"].get("changed") or changed or ["Nothing (this is the baseline)"]
    data = []
    if s.get("dataset"):
        d = s["dataset"]
        data.append(f"Dataset: {d.get('filename') or d.get('dataset_id')}" + (f", {r['rows_used']:,} records used" if r.get("rows_used") else ""))
        if s.get("columns"):
            data.append("Columns used: " + "; ".join(f"{k} → {', '.join(v)}" for k, v in s["columns"].items() if v))
    src = {k: v for k, v in (s.get("baseline_sources") or {}).items()}
    if src:
        data.append("Starting values: " + "; ".join(f"{k} ({v})" for k, v in src.items()))
    if not data:
        data.append("No dataset: every value is one you entered.")
    u = r.get("uncertainty") or {}
    unc = u.get("note", "")
    if u.get("kind") == "confidence_interval":
        unc = f"Estimated change in {u['target'].replace('_', ' ')}: between {_n(u['change_low'])} and {_n(u['change_high'])} (95% confidence interval). " + unc
    elif u.get("kind") in ("prediction_interval", "simulation_percentiles"):
        unc = f"Likely range {_n(u['low'])} to {_n(u['high'])} ({int(u['level'] * 100)}% interval). " + unc
    cannot = [template.limitation] if template.limitation else []
    cannot += ["It does not show that the change would be achievable, or what else would change with it."]
    if kind == ESTIMATE:
        cannot.append("It does not prove that changing the driver causes the outcome to change.")
    cannot += list(domain.notes)
    result = []
    for m in r["metrics"]:
        pct = f", {m['pct_change']:+.1f}%" if m.get("pct_change") is not None else ""
        d = f" ({m['delta']:+,.2f}{pct})" if m.get("delta") is not None else ""
        result.append(f"{m['label']}: {_n(m['baseline'])} → {_n(m['scenario'])}{d}")
    if r["status"] != "ok":
        result = [r.get("reason") or "No result."]
    return {"headline": headline, "evidence_kind": kind, "evidence_label": KIND_LABEL.get(kind, kind),
            "sections": {"changed": changed, "held_constant": r.get("held_constant") or [], "data_used": data, "method": [METHOD_TEXT.get(s["method"], s["method"])],
                         "assumptions": r.get("assumptions") or [], "result": result, "reliability": [RELIABILITY.get(kind, ""), *( [unc] if unc else [])],
                         "cannot_conclude": cannot},
            "warnings": r.get("warnings") or []}


def as_text(e: dict) -> str:
    """The same explanation as plain text, for the Copilot's reply."""
    lines = [f"**{e['evidence_label']}.** {e['headline']}"]
    titles = {"changed": "What changed", "held_constant": "Kept constant", "data_used": "Data used", "method": "Method", "assumptions": "Assumptions",
              "result": "Result", "reliability": "How reliable", "cannot_conclude": "What this cannot tell you"}
    for k, title in titles.items():
        items = e["sections"].get(k) or []
        if items:
            lines.append(f"\n**{title}**\n" + "\n".join(f"- {i}" for i in items))
    for w in e.get("warnings") or []:
        lines.append(f"\n_Note: {w}_")
    return "\n".join(lines)
