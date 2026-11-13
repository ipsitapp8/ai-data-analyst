"""Checking a model's decline or clarification against the dataset it was
given. A plan that says the data lacks something the schema plainly has is a
contradiction: it is reported, and the model may reconsider once (see
orchestrator.ask). Nothing here writes a plan; it only finds the conflict."""
from __future__ import annotations

import re

import pandas as pd

from app.analyst.data import business_roles, kind_of

_ADS = re.compile(r"advertis|campaign|marketing|budget|\bad\b|\bads\b|spend|roi\b|\bcac\b|acquisition", re.I)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contradiction(plan: dict, df: pd.DataFrame, message: str) -> str | None:
    """Why the plan contradicts the dataset, or None."""
    if plan.get("type") not in ("decline", "clarify"):
        return None
    said = " ".join(str(plan.get(k) or "") for k in ("reason", "question"))
    # 1. "no advertising history" when spend, revenue/conversions and an area or campaign column exist
    if plan["type"] == "decline" and (_ADS.search(said) or _ADS.search(message)):
        roles = business_roles(df)["roles"]
        group = roles.get("area") or roles.get("campaign")
        outcome = roles.get("revenue") or roles.get("conversions")
        if roles.get("spend") and outcome and group:
            return (f"the dataset does contain advertising history: spend column '{roles['spend']}', "
                    f"outcome column '{outcome}', grouped by '{group}'")
    # 2. "there is no <X> column" when a column named X exists
    for m in re.finditer(r"\bno\s+([a-z][a-z0-9_ ]{1,30}?)\s+column", said, re.I):
        wanted = _norm(m.group(1))
        hits = [c for c in df.columns if wanted and (wanted == _norm(c) or wanted in _norm(c).split())]
        if hits:
            return f"it says there is no '{m.group(1)}' column, but the dataset has {', '.join(repr(h) for h in hits)}"
    # 3. "which column should I predict?" when the question names exactly one column to predict
    if plan["type"] == "clarify" and re.search(r"predict|target", said, re.I):
        m = re.search(r"\b(predict|forecast|model)\w*\s+(?:the\s+)?([a-z][\w ]{1,40}?)(?:\s+(?:with|using|for|from|based)\b|[.?!]|$)",
                      message, re.I)
        if m:
            named = [c for c in df.columns if _norm(c) == _norm(m.group(2)) or _norm(c) == _norm(m.group(2)).rstrip("s")]
            if len(named) == 1:
                return f"the question names the column to predict: '{named[0]}'"
    # 4. "no suitable prediction target" when suitable targets exist
    if plan["type"] == "decline" and re.search(r"target", said, re.I):
        targets = [c for c in df.columns if (kind_of(df[c]) == "categorical" and 2 <= df[c].nunique(dropna=True) <= 10)
                   or (kind_of(df[c]) == "numeric" and df[c].nunique(dropna=True) > 10)]
        if targets:
            return f"it says no column can be predicted, but {', '.join(repr(t) for t in targets[:6])} could be"
    return None



_RATE_Q = re.compile(r"\b(rate|share|percent\w*|proportion|fraction|how likely)\b", re.I)
_AVG_Q = re.compile(r"\b(average|mean|typical|per (record|row|employee|person|customer|store))\b", re.I)
_TOTAL_Q = re.compile(r"\b(total|sum|overall amount|in all)\b", re.I)


def metric_mismatch(message: str, calls: list[dict], df: pd.DataFrame) -> str | None:
    """Why a computed metric does not answer the question asked (a count for a rate, a sum for an average, a different
    column than the one named), or None. Only the trained planner's compute_metric results are judged here; the
    calculator's plans are checked by its own validator."""
    from app.analyst.data import find_column
    for c in calls:
        if c.get("tool") != "compute_metric" or c.get("status") != "ok":
            continue
        a = c.get("args") or {}
        agg, col = str(a.get("aggregation") or ""), a.get("column")
        named = [x for x in find_column(df, message) if kind_of(df[x]) == "numeric"]
        if _RATE_Q.search(message) and agg in ("count", "sum") and not (col and re.search(r"rate|pct|percent|share", col)):
            return f"the question asks for a rate or share, but the plan computed a {agg} of {col or 'records'}"
        if _AVG_Q.search(message) and not _TOTAL_Q.search(message) and agg in ("sum", "count"):
            return f"the question asks for an average, but the plan computed a {agg}"
        if _TOTAL_Q.search(message) and not _AVG_Q.search(message) and agg in ("avg", "mean", "median"):
            return f"the question asks for a total, but the plan computed an {agg}"
        if named and col and col not in named and agg != "count":
            return f"the question names {', '.join(named)}, but the plan computed {col}"
    return None
