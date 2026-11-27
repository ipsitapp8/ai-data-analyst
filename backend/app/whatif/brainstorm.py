"""Brainstorm: a language model proposes what-if scenarios for a stated objective; Silt checks every one before offering it.

The model sees the objective, the column names (with types, and ranges of non-personal numeric columns; never rows) and the
calculations Silt can run. It may choose among those calculations, name real columns, or sketch a small formula model. It supplies
no numbers. Each idea is then grounded here: an unknown calculation, a column the table lacks or a formula that does not validate
drops the idea. How feasible an idea is, and what data it needs, are worked out by Silt; only the wording and the usefulness
rating are the model's. There is no rules fallback in this module: if the model cannot answer, an exception says so.
"""
from __future__ import annotations

import pandas as pd

from app import runtime, scenario_formula as sf, scenario_models
from app.whatif.discovery import RANK, _suggestion, assess, profile_col
from app.whatif.domains import GENERIC_TEMPLATES, get_domain, list_domains
from app.whatif.domains.base import NEEDS_ASSUMPTIONS, NEEDS_EXTERNAL_DATA, Template
from app.whatif.workflows import describe

TOOL = "brainstorm_scenarios"
MAX_IDEAS = 8
MAX_COLUMNS = 60
NO_LOWER_LIMIT = -1e15
PLAIN_NUMBERS, PLAIN_POWERS = (0.0, 1.0, 100.0), (0.5, 1.0, 2.0)
KINDS = ("calculation", "column_change", "column_cutoff", "column_relationship", "own_model", "outside_evidence")
_TEXT = {"type": "string"}
_NAMED = {"name": _TEXT, "label": _TEXT, "unit": _TEXT}
SCHEMA = {"type": "object", "required": ["ideas"], "properties": {"ideas": {"type": "array", "items": {
    "type": "object", "required": ["title", "question", "why", "usefulness", "kind"],
    "properties": {
        "title": _TEXT, "question": _TEXT, "why": _TEXT, "usefulness": {"type": "integer"}, "kind": {"type": "string", "enum": list(KINDS)},
        "calculation_id": _TEXT, "column": _TEXT, "x_column": _TEXT, "y_column": _TEXT,
        "variables": {"type": "array", "items": {"type": "object", "properties": _NAMED, "required": ["name", "label"]}},
        "outcomes": {"type": "array", "items": {"type": "object", "properties": {**_NAMED, "formula": _TEXT}, "required": ["name", "label", "formula"]}},
        "data_needed": {"type": "array", "items": _TEXT},
    }}}}}
SYSTEM = f"""You are Silt AI. You turn a person's objective into what-if scenarios worth testing.
Propose 5 to {MAX_IDEAS} different ideas that directly serve the objective, most useful first. Each idea has one kind:
- calculation: one of the listed calculations; give its id in calculation_id.
- column_change: raise or lower one listed numeric column; give it in column.
- column_cutoff: move a minimum or cut-off on one listed numeric column; give it in column.
- column_relationship: estimate what a change in x_column goes with in y_column (both listed numeric columns).
- own_model: a small formula model when nothing above fits. variables: names in lower_case_with_underscores. outcomes: each with a
  formula using only those names, + - * / **, min, max, abs, sqrt, log, exp. A formula may contain no number except 0, 1 and
  100: every weight, rate or coefficient must be a variable. The person will type every value.
- outside_evidence: an important question the data and these calculations cannot answer; list what is missing in data_needed.
Rules: copy column names and calculation ids exactly. Never invent a column. Never state, assume or estimate a number: Silt
calculates and the person supplies values. title is short; question is one what-if question; why says in one or two sentences how it
serves the objective; usefulness is 1 (marginal) to 5 (central to the objective). Plain language."""


class AIUnavailable(RuntimeError):
    """No model could be reached; the message says why in plain words."""


class AIUnusable(RuntimeError):
    """The model answered, but nothing in the answer could be used."""


def _catalogue(domain_key: str | None) -> list[tuple[str, Template]]:
    domains = [get_domain(domain_key)] if domain_key else list_domains()
    return [(d.key, t) for d in domains for t in d.templates if t.operation not in ("none", "custom_model")]


def _prompt(objective: str, profile: dict | None, domain_key: str | None) -> str:
    lines = [f"Objective: {objective.strip()}", f"Domain: {get_domain(domain_key).title if domain_key else 'not chosen'}", ""]
    if profile is None:
        lines.append("Dataset: none. Do not use the column_* kinds.")
    else:
        lines.append(f"Dataset columns ({profile['rows']} records):")
        for c in profile["columns"][:MAX_COLUMNS]:
            use = "numeric" if c["name"] in profile["metrics"] and not c["sensitive"] else c["role"]
            seen = f", {c['min']:g} to {c['max']:g}, average {c['mean']:.4g}" if "min" in c and not c["sensitive"] else ""
            lines.append(f"- {c['name']} ({use}{seen}{'; personal, do not use' if c['sensitive'] else ''})")
        for r in profile["relationships"][:5]:
            lines.append(f"- relationship: {r['a']} and {r['b']}, correlation {r['pearson']:+.2f}")
    lines += ["", "Calculations Silt can run (id: question; needs):"]
    for _, t in _catalogue(domain_key):
        lines.append(f"- {t.id}: {t.question}; {', '.join(t.requires) or 'values typed by the person'}")
    return "\n".join(lines)


def _ask(objective: str, profile: dict | None, domain_key: str | None) -> tuple[list, str]:
    from app.agents import llm_client
    if not llm_client.provider_available(TOOL):
        raise AIUnavailable("Silt AI is not set up on this server (no AI model key is configured), so it cannot brainstorm.")
    try:
        raw = llm_client.call_tool(SYSTEM, _prompt(objective, profile, domain_key), TOOL, SCHEMA, "Return the scenario ideas.", 4000)
    except runtime.ProviderUnavailable as e:
        why = "; ".join(f"{str(f.get('provider', '')).title()}: {e.WHY.get(f.get('kind'), 'the request failed')}" for f in e.failures)
        raise AIUnavailable("Silt AI could not be reached, so nothing was brainstormed." + (f" {why}." if why else "")) from e
    except Exception as e:  # noqa: BLE001 - any failure to get an answer is reported, never replaced with made-up ideas
        raise AIUnavailable(f"Silt AI could not answer ({type(e).__name__}), so nothing was brainstormed.") from e
    ideas = raw.get("ideas") if isinstance(raw, dict) else None
    if not isinstance(ideas, list) or not ideas:
        raise AIUnusable("Silt AI replied without any scenario ideas. Try again, or describe the objective differently.")
    return ideas, llm_client.LAST_PROVIDER


def _text(value, limit: int) -> str:
    return " ".join(value.split())[:limit] if isinstance(value, str) else ""


def _items(value) -> list:
    return list(value) if isinstance(value, (list, tuple)) else []


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _own_model(idea: dict) -> dict | None:
    """A runnable definition from the model's sketch, or None if it does not validate. It carries names and formulas only."""
    variables = [v for v in map(_as_dict, _items(idea.get("variables"))) if v.get("name")]
    outcomes = [o for o in map(_as_dict, _items(idea.get("outcomes"))) if o.get("name") and o.get("formula")]
    if not variables or not outcomes:
        return None
    defn = {"inputs": {str(v["name"]): {"label": _text(v.get("label"), 80) or str(v["name"]), "unit": _text(v.get("unit"), 40), "min": NO_LOWER_LIMIT}
                       for v in variables},
            "outputs": {str(o["name"]): {"label": _text(o.get("label"), 80) or str(o["name"]), "unit": _text(o.get("unit"), 40), "formula": str(o["formula"])}
                        for o in outcomes},
            "primary_output": str(outcomes[0]["name"]), "constraints": []}
    if len(defn["inputs"]) != len(variables) or not scenario_models.analyze(defn)["valid"]:
        return None
    return None if any(_invents_numbers(sf.parse(o["formula"]).tree) for o in defn["outputs"].values()) else defn


def _invents_numbers(node) -> bool:
    """Whether a formula carries a weight, rate or coefficient of the model's own. Only 0, 1, 100 (and squares and roots) pass."""
    if isinstance(node, list):
        return any(_invents_numbers(x) for x in node)
    if not isinstance(node, tuple):
        return False
    if node[0] == "num":
        return node[1] not in PLAIN_NUMBERS
    if node[0] == "pow" and node[2] not in PLAIN_POWERS:
        return True
    return any(_invents_numbers(x) for x in node[1:])


def _ground(idea: dict, profile: dict | None, domain_key: str | None, df: pd.DataFrame | None) -> dict | None:
    """The idea as a runnable (or honestly unrunnable) suggestion, or None when it refers to something that does not exist."""
    kind = idea.get("kind")
    home = domain_key or "custom"
    if kind == "calculation":
        hit = next(((dk, t) for dk, t in _catalogue(domain_key) if t.id == idea.get("calculation_id")), None)
        if hit is None:
            return None
        dk, t = hit
        own = profile is not None and domain_key in (None, dk)
        a = assess(t, dk, profile if own else None, df if own else None)
        return describe(_suggestion(t, dk, a, "ai"), t, profile if own else None)
    if kind in ("column_change", "column_cutoff", "column_relationship"):
        if profile is None:
            return None
        usable = {m for m in profile["metrics"] if not profile_col(profile, m)["sensitive"]}
        if kind == "column_relationship":
            if idea.get("x_column") == idea.get("y_column"):
                return None
            t, cols = GENERIC_TEMPLATES["generic_association"], {"x": [idea.get("x_column")], "y": [idea.get("y_column")]}
        else:
            t = GENERIC_TEMPLATES["generic_shift" if kind == "column_change" else "generic_threshold"]
            cols = {"metric": [idea.get("column")]}
        if any(not isinstance(v[0], str) or v[0] not in usable for v in cols.values()):
            return None
        return describe(_suggestion(t, home, assess(t, home, profile, df, cols), "ai", columns=cols), t, profile)
    if kind == "own_model":
        defn = _own_model(idea)
        if defn is None:
            return None
        t = get_domain("custom").template("custom_model")
        a = {"feasibility": NEEDS_ASSUMPTIONS, "available": {}, "missing": ["a starting and a changed value for your variables"], "limitations": [t.limitation]}
        item = describe(_suggestion(t, "custom", a, "ai"), t, profile)
        return {**item, "question": _text(idea.get("question"), 200), "custom_model": defn, "capability_title": "Your own formula model",
                "needs": ["You enter: " + ", ".join(v["label"].lower() for v in defn["inputs"].values()),
                          "Formulas to check: " + "; ".join(f"{k} = {o['formula']}" for k, o in defn["outputs"].items())]}
    if kind == "outside_evidence":
        missing = [_text(x, 160) for x in _items(idea.get("data_needed")) if _text(x, 160)] or ["information that is not in this data"]
        a = {"feasibility": NEEDS_EXTERNAL_DATA, "available": {}, "missing": missing, "limitations": []}
        t = Template(id="ai_outside", title="Needs outside evidence", question=_text(idea.get("question"), 200).replace("{", "(").replace("}", ")"),
                     method="deterministic", operation="none", external=tuple(missing), reason="Needs information outside this data.")
        return describe(_suggestion(t, home, a, "ai"), t, profile)
    return None


def brainstorm(objective: str, profile: dict | None, domain_key: str | None, df: pd.DataFrame | None = None) -> dict:
    """{"suggestions": [...ranked...], "meta": {...}}. Ranked by feasibility (Silt's check), then usefulness (the model's rating)."""
    raw, provider = _ask(objective, profile, domain_key)
    kept, seen, dropped = [], set(), 0
    for idea in map(_as_dict, raw):
        item = _ground(idea, profile, domain_key, df)
        title, question, why = _text(idea.get("title"), 100), _text(idea.get("question"), 200), _text(idea.get("why"), 400)
        if item is None or not title or not question:
            dropped += 1
            continue
        key = (item["template_id"], str(item.get("columns")), str(item.get("custom_model")), question if item["template_id"] == "ai_outside" else "")
        if key in seen:
            continue
        seen.add(key)
        score = idea.get("usefulness")
        score = min(5, max(1, int(score))) if isinstance(score, (int, float)) and not isinstance(score, bool) else 1
        kept.append({**item, "ai": {"title": title, "question": question, "why": why, "usefulness": score}})
    if not kept:
        raise AIUnusable("Silt AI replied, but none of its ideas matched your columns or a calculation Silt can run. "
                         "Try again, or describe the objective differently.")
    kept.sort(key=lambda i: (RANK[i["feasibility"]], -i["ai"]["usefulness"]))
    kept = kept[:MAX_IDEAS]
    for n, item in enumerate(kept, start=1):
        item["id"], item["rank"] = f"ai{n}:{item['id']}", n
    return {"suggestions": kept, "meta": {"workflow": "brainstorm", "llm": "used", "generated_by": provider, "objective": objective, "dropped": dropped,
                                          "domain": domain_key, "ranked_by": "feasibility, then the AI's usefulness rating"}}
