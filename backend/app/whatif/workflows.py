"""The two rules-based starting points of the What-if page. Both end in the same suggestion record (app.whatif.discovery) and the
same engine; they differ in what they start from.

    explore            starts from a domain: what can be asked, grouped by the kind of calculation, and what each question needs
    dataset_scenarios  starts from a table: only questions its real columns support, grouped by how far the answer can be trusted

No model is used and no column is ever named that the table does not have.
"""
from __future__ import annotations

import pandas as pd

from app.whatif.discovery import RANK, _suggestion, assess, generic_suggestions, profile_col
from app.whatif.domains import GENERIC_TEMPLATES, get_domain
from app.whatif.domains.base import ESTIMATE, FORECAST, UNSUPPORTED, Template
from app.whatif.profile import map_concepts

MAX_METRIC_COLUMNS = 8
MAX_RELATIONSHIPS = 6

# what Silt can calculate, in the words a person choosing a question would use: operation -> (key, title, what it does)
CAPABILITIES = {
    "threshold": ("cutoff", "Move a rule or cut-off", "Counts who meets a requirement before and after it moves."),
    "shift_metric": ("shift", "Raise or lower a measure", "Applies one change to every record and recomputes the averages and totals."),
    "association": ("estimate", "Estimate a knock-on effect", "Uses the relationship in past records, only when it is strong enough to trust."),
    "forecast_trend": ("trend", "Project a trend", "Carries a past monthly trend forward, with a likely range."),
    "formula_model": ("model", "Plan from your own figures", "Applies stated formulas to starting values you enter or Silt reads from your data."),
    "none": ("outside", "Needs outside evidence", "Worth asking, but no dataset of past records can answer it."),
}
GROUPS = {"exact": "Exact calculations from your columns", "estimate": "Estimates from your data",
          "assumptions": "Calculations that also need your assumptions", "cannot": "Cannot be answered from this data"}


def _label(domain_key: str, concept: str) -> str:
    c = get_domain(domain_key).concept(concept)
    return c.label if c else concept


def data_needed(template: Template, domain_key: str, profile: dict | None, columns: dict) -> list[str]:
    """What it takes to run this question, in plain words: the columns it reads, what you type, and what no dataset holds."""
    out = []
    if template.requires:
        parts = []
        for c in template.requires:
            have = columns.get(c) or (map_concepts(profile, domain_key).get(c, {}).get("candidates") if profile and domain_key != "custom" else None)
            parts.append(", ".join(have) if have else f"a column for {_label(domain_key, c).lower()}")
        out.append("From a dataset: " + "; ".join(parts))
    typed = [s.label.lower() for s in template.inputs if not s.optional and s.default is None
             and not (profile is not None and s.from_data and s.from_data.startswith(("row_count", "entity_count")))]
    if typed:
        out.append("You enter: " + ", ".join(typed))
    if template.external:
        out.append("From outside: " + ", ".join(template.external))
    return out or ["Nothing beyond the defaults shown"]


def describe(item: dict, template: Template, profile: dict | None) -> dict:
    """Add what a card needs to explain itself: title, capability and the data required."""
    key, title, what = CAPABILITIES.get(template.operation, CAPABILITIES["none"])
    return {**item, "title": template.title, "capability": key, "capability_title": title, "capability_what": what,
            "needs": data_needed(template, item["domain"], profile, item["columns"])}


def explore(profile: dict | None, domain_key: str, df: pd.DataFrame | None) -> dict:
    """The domain's questions, grouped by capability. A domain without its own questions (Custom, or data of an unfamiliar
    kind) is explored through the calculations that work on any table."""
    dom = get_domain(domain_key)
    items = [describe(_suggestion(t, domain_key, assess(t, domain_key, profile, df), "domain"), t, profile)
             for t in dom.templates if t.operation != "custom_model"]
    if not items and profile is not None:
        items = [describe(s, GENERIC_TEMPLATES[s["template_id"]], profile) for s in generic_suggestions(profile, df, domain_key, set())]
    order = [c[0] for c in CAPABILITIES.values()]
    items.sort(key=lambda i: (order.index(i["capability"]), RANK[i["feasibility"]]))
    found = [{"key": k, "title": t, "what": w, "count": sum(1 for i in items if i["capability"] == k)} for k, t, w in CAPABILITIES.values()]
    return {"suggestions": items, "capabilities": [c for c in found if c["count"]],
            "meta": {"workflow": "explore", "ranked_by": "rules", "llm": None, "with_data": profile is not None}}


def _group(item: dict, template: Template) -> str:
    if item["feasibility"] == UNSUPPORTED:
        return "cannot"
    if template.evidence in (ESTIMATE, FORECAST):
        return "estimate"
    return "assumptions" if template.operation == "formula_model" else "exact"


def _domain_template(t: Template, domain_key: str, profile: dict, df: pd.DataFrame) -> dict | None:
    """A domain question, only if this table's own columns feed it. None otherwise."""
    if t.operation in ("none", "custom_model"):
        return None
    mapped = map_concepts(profile, domain_key) if domain_key != "custom" else {}
    found = {k: v["candidates"] for k, v in mapped.items() if v["candidates"]}
    if any(c not in found for c in t.requires):
        return None
    used = [c for concept in t.requires for c in found[concept]]
    for s in t.inputs:                                     # a starting value read from the table
        kind, _, arg = (s.from_data or "").partition(":")
        if kind == "entity_count" and profile["entities"]:
            used.append(profile["entities"][0]["column"])
        elif kind in ("mean", "sum", "rows_per_day"):
            used += found.get(arg, [])
    used = list(dict.fromkeys(used))
    if not used:
        return None
    item = describe(_suggestion(t, domain_key, assess(t, domain_key, profile, df), "dataset"), t, profile)
    return {**item, "used_columns": used, "group": _group(item, t)}


def _column_item(template_id: str, domain_key: str, profile: dict, df: pd.DataFrame, columns: dict, reason: str) -> dict:
    t = GENERIC_TEMPLATES[template_id]
    item = describe(_suggestion(t, domain_key, assess(t, domain_key, profile, df, columns), "column", reason, columns), t, profile)
    item["id"] = f"{domain_key}:{t.id}:" + "+".join(c for v in columns.values() for c in v)
    return {**item, "used_columns": list(dict.fromkeys(c for v in columns.values() for c in v)), "group": _group(item, t)}


def _column_report(profile: dict) -> list[dict]:
    """Every column, what was read from it, and whether a scenario can use it (and if not, why)."""
    out = []
    for c in profile["columns"]:
        name = c["name"]
        if c["sensitive"]:
            seen = "not shown"
        elif "min" in c:
            seen = f"{c['min']:,.6g} to {c['max']:,.6g} (average {c['mean']:,.6g})"
        elif "start" in c:
            seen = f"{c['start'][:10]} to {c['end'][:10]}"
        elif c.get("values"):
            seen = ", ".join(c["values"][:5]) + ("…" if c["unique"] > 5 else "")
        else:
            seen = f"{c['unique']:,} different values"
        if c["sensitive"]:
            use = "Not used: looks personal"
        elif name in profile["metrics"]:
            use = "Can be changed or compared with a cut-off"
        elif c["role"] == "datetime":
            use = "Can drive a trend forecast"
        elif c["role"] in ("identifier", "person_name"):
            use = "Identifies records; counted, never changed"
        else:
            use = "Read only: no calculation changes a text or category column"
        out.append({"column": name, "type": c["role"], "missing": c["missing"], "observed": seen, "use": use})
    return out


def dataset_scenarios(profile: dict, domain_key: str, df: pd.DataFrame) -> dict:
    """Scenarios this table can support: the domain's questions whose columns exist, then questions built column by column."""
    items = [i for t in get_domain(domain_key).templates if (i := _domain_template(t, domain_key, profile, df))]
    metrics = [m for m in profile["metrics"] if not profile_col(profile, m)["sensitive"] and "min" in profile_col(profile, m)]
    for name in metrics[:MAX_METRIC_COLUMNS]:
        c = profile_col(profile, name)
        seen = f"'{name}' runs from {c['min']:,.6g} to {c['max']:,.6g}, average {c['mean']:,.6g}."
        items.append(_column_item("generic_shift", domain_key, profile, df, {"metric": [name]}, seen))
        if c["min"] != c["max"]:
            items.append(_column_item("generic_threshold", domain_key, profile, df, {"metric": [name]}, f"{seen} The middle value is {c['median']:,.6g}."))
    pairs = [r for r in profile["relationships"] if r["a"] in metrics and r["b"] in metrics]
    for r in pairs[:MAX_RELATIONSHIPS]:
        items.append(_column_item("generic_association", domain_key, profile, df, {"x": [r["a"]], "y": [r["b"]]},
                                  f"{r['a']} and {r['b']} move together in your records (correlation {r['pearson']:+.2f} over {r['n']} rows)."))
    dates = [c for c in profile["datetime"] if not profile_col(profile, c)["sensitive"]]
    for name in dates[:2]:
        items.append(_column_item("generic_forecast", domain_key, profile, df, {"date": [name]}, f"'{name}' dates each record."))
    notes = []
    if len(metrics) > MAX_METRIC_COLUMNS:
        notes.append(f"Column-by-column scenarios are shown for the first {MAX_METRIC_COLUMNS} of {len(metrics)} numeric columns.")
    if not pairs:
        notes.append("No two numeric columns move together clearly enough (correlation of 0.3 or more), so no estimate between columns is offered.")
    if not dates:
        notes.append("No column was read as a date, so no trend forecast is offered.")
    order = list(GROUPS)
    items.sort(key=lambda i: (order.index(i["group"]), RANK[i["feasibility"]], 0 if i["source"] == "dataset" else 1))
    return {"suggestions": items, "groups": [{"key": k, "title": v, "count": sum(1 for i in items if i["group"] == k)} for k, v in GROUPS.items()],
            "columns": _column_report(profile),
            "meta": {"workflow": "dataset", "ranked_by": "rules", "llm": None, "rows": profile["rows"], "filename": profile["filename"],
                     "relationships": profile["relationships"][:MAX_RELATIONSHIPS], "notes": notes}}
