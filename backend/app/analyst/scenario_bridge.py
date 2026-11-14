"""The Silt Copilot's route into the shared what-if service.

The Copilot answers a what-if or goal sentence about the dataset it holds ("what if the price goes up 10%?", "what price gives
me a 25% margin?", "how many units to break even?") by calling the same service the What-If page calls
(app/scenario_service.py), so the formulas, validation, assumptions, independent checks and explanation are identical in both.
It has no calculations of its own.

It only acts when it is sure:

* the sentence reads as a what-if or goal request (scenario_nl does the reading);
* the dataset's columns map cleanly onto a built-in model (units, revenue and cost for unit economics; visitors or leads,
  conversions, revenue and spend for the funnel). Otherwise it steps aside and the Copilot's other paths answer as before;
* every input the model needs has a value. A fixed cost the data does not hold is asked for, never assumed.

Anything it cannot handle returns None, so adding this changes no answer the Copilot gave before.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app import scenario_nl as nl
from app import scenario_service as svc
from app import scenarios
from app.analyst.data import business_roles, kind_of
from app.scenarios import ScenarioError

TRIGGER = re.compile(r"\bwhat if\b|\bwhat (?:happens|would happen|will happen)\b|\b(?:increas|rais|reduc|decreas|cut|lower|grow|boost|improv|drop)\w*\b|"
                     r"\bbreak[\s-]?even\b|\bwhat\s+\w+(?:\s+\w+)?\s+(?:would|should)\b|\bhow (?:many|much)\b.*\b(?:need|should|must)\b|\bcompare\b", re.I)
FOLLOW_UP = re.compile(r"^\s*why\b|\bwhich\b.*\b(?:variable|input|factor|driver|lever)\b|\bbiggest (?:impact|driver|effect)\b|\bcompare (?:that|this|it)\b", re.I)
FIXED_STATEMENT = re.compile(r"\bfixed\s+costs?\b[^0-9]{0,25}(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*(?:of\s+)?fixed\s+costs?", re.I)
QTY = re.compile(r"^(?:units?|quantit\w*|qty|volume|units_sold|items?|pieces)$", re.I)
AMT = re.compile(r"^(?:revenue|sales|amount|total_sales|turnover|total_revenue)$", re.I)
COST = re.compile(r"^(?:cost|total_cost|cogs|expenses?|total_expenses?|variable_cost)$", re.I)
UNIT_COST = re.compile(r"^(?:unit_cost|cost_per_unit)$", re.I)
PRICE = re.compile(r"^(?:unit_price|price|selling_price)$", re.I)


def infer_model(df: pd.DataFrame) -> tuple[str, dict, list[str]] | None:
    """The built-in model this dataset fits, how its columns map, and the columns used. None if it fits neither cleanly."""
    numeric = [c for c in df.columns if kind_of(df[c]) == "numeric"]

    def one(rx: re.Pattern) -> str | None:
        hits = [c for c in numeric if rx.match(re.sub(r"[\s-]+", "_", str(c).strip()))]
        return hits[0] if len(hits) == 1 else None

    qty, amt, cost, unit_cost = one(QTY), one(AMT), one(COST), one(UNIT_COST)
    if qty and amt and (cost or unit_cost):
        mapping = {"quantity_column": qty, "amount_column": amt}
        mapping["total_cost_column" if cost else "unit_cost_column"] = cost or unit_cost
        return "unit_economics", mapping, [qty, amt, cost or unit_cost]
    roles = business_roles(df)["roles"]
    if all(k in roles for k in ("leads", "conversions", "revenue", "spend")) and not business_roles(df)["ambiguous"]:
        mapping = {"visitors_column": roles["leads"], "orders_column": roles["conversions"], "revenue_column": roles["revenue"],
                   "spend_column": roles["spend"]}
        return "funnel", mapping, [roles["leads"], roles["conversions"], roles["revenue"], roles["spend"]]
    return None


def _profile(df: pd.DataFrame) -> dict:
    return {"columns": [{"name": c, "kind": kind_of(df[c])} for c in df.columns]}


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def _stated_fixed_cost(message: str) -> float | None:
    m = FIXED_STATEMENT.search(nl._normalize(message))  # noqa: SLF001
    return None if not m else _number(next(g for g in m.groups() if g))


def _table(result: dict, name: str | None) -> str:
    """A small plain-text comparison of the headline outputs, from the result itself."""
    p = result["primary_output"]
    lines = []
    for k in result["output_labels"]:
        b = result["baseline"]["outputs"][k]
        s = result["scenarios"][name]["outputs"][k] if name else None
        if b is None and s is None:
            continue
        lines.append(f"- {result['output_labels'][k]}: " + (f"{svc_fmt(b)}" if s is None else f"{svc_fmt(b)} → {svc_fmt(s)}") + (" (headline)" if k == p else ""))
    return "\n".join(lines)


def svc_fmt(x: float | None) -> str:
    from app import scenario_explain as ex
    return ex.n(x)


def try_answer(message: str, df: pd.DataFrame, memory: dict, filename: str | None = None, scope_note: str | None = None) -> dict | None:
    """A reply for a what-if or goal sentence, or None to let the Copilot's other paths answer."""
    pending = memory.get("pending_scenario")
    fixed_now = _stated_fixed_cost(message)
    if pending and fixed_now is not None and not re.search(r"\b(?:what if|increas|reduc|cut|raise)\w*", message, re.I):
        memory.setdefault("scenario_inputs", {})["fixed_cost"] = fixed_now
        message = pending["message"]
        memory.pop("pending_scenario", None)
    elif FOLLOW_UP.search(message):
        if not memory.get("scenario_context"):
            return None                              # a "why" or "which input" question with no scenario to explain is not ours
    elif not TRIGGER.search(message):
        return None
    fit = infer_model(df)
    if fit is None:
        return None
    model, mapping, used = fit
    spec = scenarios._model(model)  # noqa: SLF001
    try:
        base = scenarios.baseline_from_dataset(df, _profile(df), model, mapping)
    except ScenarioError:
        return None
    inputs = dict(base["inputs"])
    given = dict(memory.get("scenario_inputs") or {})
    if fixed_now is not None:
        given["fixed_cost"] = fixed_now
        memory["scenario_inputs"] = given
        # the statement is a value being given, not a change being asked for: take it out before the sentence is read
        message = re.sub(r"\s+", " ", FIXED_STATEMENT.sub(" ", nl._normalize(message))).strip(" ,;")  # noqa: SLF001
    for name in base["missing"]:
        if name in given:
            inputs[name] = float(given[name])
    still = [n for n in inputs if inputs[n] is None]
    if still:
        if nl.parse(message, spec)["status"] != "ok":
            return None                              # unclear or not a what-if: leave it to the Copilot's other paths
        memory["pending_scenario"] = {"message": message}
        labels = ", ".join(spec["inputs"][n]["label"].lower() for n in still)
        return {"answer": (f"I can work that out from this dataset (columns {', '.join(f'`{c}`' for c in used)}), but the data has no value for "
                           f"**{labels}**, and I will not assume one. Tell me what it is, for example *fixed cost is 3000* (use 0 if there is none)."),
                "plan": {"type": "clarify", "scenario_model": model}, "calls": [],
                "notes": ["The what-if model needs every input to have a value; a missing one is asked for, never filled in."]}
    inputs = scenarios.validate_inputs(model, inputs)
    ctx = memory.get("scenario_context")
    try:
        out = svc.ask(model, inputs, None, message, ctx)
    except ScenarioError as e:
        return {"answer": f"I could not run that what-if: {e}", "plan": {"type": "clarify", "scenario_model": model}, "calls": [], "notes": []}
    if out["status"] in ("unsupported", "clarify"):
        return None
    if out.get("context"):
        memory["scenario_context"] = out["context"]
    notes = ([f"Computed over the records in scope only: {scope_note}."] if scope_note else []) + [
        f"Answered by the same what-if service as the What-If page ({svc.SERVICE_VERSION}); no number came from a model.",
             f"Built-in model: {spec['title']}; columns used: {', '.join(used)}."]
    notes += [f"Interpreted as: {i}" for i in out.get("interpretation", [])] + out.get("notes", [])
    for k, v in (base["sources"] or {}).items():
        if v.get("source") == "dataset":
            notes.append(f"{spec['inputs'][k]['label']} = {v['formula']} over {base['rows_used']:,} rows.")
    for n, v in (memory.get("scenario_inputs") or {}).items():
        if n in spec["inputs"]:
            notes.append(f"{spec['inputs'][n]['label']} = {v:,.10g}, as you told me.")
    result = out.get("result")
    if out["status"] == "clarify":
        opts = "; ".join(o["label"] for o in out["clarification"]["options"])
        return {"answer": out["answer"] + (f" ({opts})" if opts else ""), "plan": {"type": "clarify", "scenario_model": model},
                "calls": [], "notes": notes}
    answer = out["answer"]
    name = out.get("last_scenario")
    details = ""
    if result and result.get("kind") == "scenario" and name:
        answer += "\n\n" + _table(result, name)
        ex = result["explanation"]["sections"]
        details = "\n".join(f"- {x}" for k in ("why", "assumptions", "risks", "cannot_conclude") for x in ex.get(k, []))
    ver = (result or {}).get("verification", {}).get("status")
    if ver:
        notes.append(f"Independent check of the arithmetic: {ver}.")
    compact = None
    if result:
        compact = {k: result[k] for k in ("kind", "status", "model", "primary_output", "message") if k in result}
    return {"answer": answer, "plan": {"type": "scenario", "model": model, "action": (out.get("intent") or {}).get("action"), "mapping": mapping},
            "calls": [{"tool": "scenario_service", "status": "ok", "result": compact or {}}], "notes": notes, "details": details}
