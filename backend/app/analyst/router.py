"""Requests the fine-tuned planner has no tool for, routed to deterministic
tools before the model is asked.

The model was trained and evaluated on a fixed tool catalogue (tools.TOOLS);
adding tools to its prompt would change what it was evaluated on. So dataset
understanding, data quality, outliers, trends, unusual patterns and general
what-if scenarios are recognised here by their wording and answered by the
tools below. Every such answer is labelled "Silt tool router" with the reason;
it is never presented as the model's plan. Anything not clearly one of these
goes to the model as before.

Also here: the train/test split and seed a user asks for in words
("use a 70/30 split, seed 7"), applied to Random Forest training.
"""
from __future__ import annotations

import re

import pandas as pd

from app.analyst import overview, scenario
from app.analyst.business import Insufficient
from app.analyst.data import business_roles, find_column, kind_of

_PCT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b|percentage points?\b|pp\b|points?\b)")
_SPLIT = re.compile(r"\b(80|75|70)\s*/\s*(20|25|30)\b")


def ml_options(message: str) -> dict:
    low = message.lower()
    out = {}
    m = _SPLIT.search(low)
    if m and int(m.group(1)) + int(m.group(2)) == 100:
        out["test_size"] = int(m.group(2)) / 100
    else:
        t = re.search(r"(20|25|30)\s*%?\s*(?:test|for testing|held[- ]out)", low) or \
            re.search(r"test(?:ing)?\s*(?:set|share|size|split)?\s*(?:of|=|:)?\s*(20|25|30)\s*%", low)
        if t:
            out["test_size"] = int(t.group(1)) / 100
    s = re.search(r"\b(?:seed|random state|random_state)\s*(?:=|:|of)?\s*(\d{1,9})\b", low)
    if s:
        out["seed"] = int(s.group(1))
    return out


def _numeric_named(df: pd.DataFrame, low: str) -> list[str]:
    cols = [c for c in find_column(df, low) if kind_of(df[c]) == "numeric"]
    if not cols and re.search(r"\b(salar(y|ies)|pay|wages?|earnings)\b", low):
        cols = [c for c in df.columns if kind_of(df[c]) == "numeric" and re.search(r"salar|income|pay|wage|earning", c.lower())]
        if not cols:      # no salary column: pay in an hourly dataset is its rate
            cols = [c for c in df.columns if kind_of(df[c]) == "numeric" and re.search(r"(hourly|pay|wage)_?rate|^rate$",
                                                                                      c.lower())]
    if not cols:
        # "salaries", "sales", "revenue": find_column handles plurals; try word stems as a fallback
        for c in df.columns:
            stem = re.sub(r"[^a-z]", "", c.lower())[:5]
            if kind_of(df[c]) == "numeric" and len(stem) >= 4 and re.search(rf"\b{stem}", low):
                cols.append(c)
    return cols


def direction_of(low: str) -> int:
    return -1 if re.search(r"decreas|reduc|cut|drop|fall|lower|declin|shrink", low) else 1


def _clarify(question: str, options: list[str], why: str) -> dict:
    return {"type": "clarify", "question": question, "options": options[:8], "router": why}


def route(message: str, df: pd.DataFrame, memory: dict | None = None) -> dict | None:
    """A plan for a request the planner's tools do not cover, or None. `memory` holds the group
    the previous answer ranked first, so "why is that region better?" knows which one."""
    low = message.lower().strip()
    # Wording is matched with column names blanked out, so a column called "monthly_income" is not read as
    # "monthly"; columns are found in the original text.
    bare = low
    for c in sorted(df.columns, key=len, reverse=True):
        for form in {c.lower(), c.lower().replace("_", " ")}:
            bare = re.sub(rf"(?<![a-z0-9_]){re.escape(form)}(?![a-z0-9_])", " ", bare)
    roles = business_roles(df)["roles"]
    numeric = [c for c in df.columns if kind_of(df[c]) == "numeric" and not overview._id_like(c, df[c])]  # noqa: SLF001

    # --- why does a group do better: compare it with the rest --------------------
    if re.search(r"\bwhy\b.*\b(better|higher|best|more|outperform|lead|ahead|top|stronger|worse|lower|weaker)\b", bare) \
            and not re.search(r"recommend|suggest|predict|model", bare):
        last = (memory or {}).get("last_focus")
        named = None
        for c in df.columns:
            if kind_of(df[c]) == "categorical" and df[c].nunique() <= 50:
                hit = [v for v in df[c].dropna().astype(str).unique() if re.search(rf"\b{re.escape(v.lower())}\b", low)]
                if hit:
                    named = {"group_column": c, "group": hit[0], "outcome": (last or {}).get("outcome")}
                    break
        focus = named or last
        if focus:
            return {"type": "router", "tool": "drivers", "args": focus, "reason": "comparing a group with the rest"}
        return _clarify("Which group should I compare with the rest? Ask a ranking question first (for example "
                        "'Which region has the highest sales?') or name the group.", [], "group to explain")

    # --- dataset understanding -------------------------------------------------
    if re.search(r"\bunderstand (it|this|the (data|dataset|file))|summari[sz]e (it|the|this|my)\b.*\b(data|dataset|file)?|"
                 r"what('s| is) in (this|the|my) (data|dataset|file)|describe (the|this|my) (data|dataset|file)|"
                 r"tell me about (this|the|my) (data|dataset|file)|overview of (the|this|my) (data|dataset)|"
                 r"explore (the|this|my) (data|dataset)|explain (the|this|my) (data|dataset|file)|^(summary|overview)\W*$|what (does|is) (this|the) dataset", low):
        return {"type": "router", "tool": "understand", "args": {}, "reason": "dataset overview"}
    if re.search(r"what should (i|we) (investigate|look at|analy[sz]e|ask)|what (else )?can (i|we) (ask|analy[sz]e|do)|"
                 r"suggest (some )?(analys|questions)|where (should|do) (i|we) start", low):
        return {"type": "router", "tool": "understand", "args": {"focus": "suggestions"}, "reason": "suggested next analyses"}
    if re.search(r"missing (values|data)|null values|\bnulls?\b|data quality|duplicate|how clean|incomplete", bare):
        return {"type": "router", "tool": "missing_values", "args": {}, "reason": "data quality"}
    if re.search(r"outlier|unusual (values?|records?|rows?|entries|observations?)|anomal|extreme values?", bare):
        named = _numeric_named(df, low)
        return {"type": "router", "tool": "outliers", "args": {"column": named[0]} if named else {}, "reason": "outliers"}
    if re.search(r"unusual pattern|interesting pattern|find (any )?patterns|what stands out|anything (unusual|interesting)|"
                 r"conclusions?|insights?|key findings|takeaways?|what does (the|this) data (tell|say|show)", low):
        return {"type": "router", "tool": "patterns", "args": {}, "reason": "unusual patterns"}

    # --- trends over time --------------------------------------------------------
    if re.search(r"\btrends?\b|over time|monthly|by month|per month|each month|month by month|weekly|by week|"
                 r"quarterly|by quarter|time series", bare):
        dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
        named = _numeric_named(df, low)
        value = named[0] if named else (roles.get("revenue") or (numeric[0] if len(numeric) == 1 else None))
        if not dates:
            return {"type": "decline", "reason": "This dataset has no date column, so a trend over time cannot be "
                                                 "computed.", "router": "trend over time"}
        if value is None:
            return _clarify("Which value should I show over time?", numeric, "trend")
        from app.analyst.followup import RATE_LIKE, auto_freq
        freq = ("W" if re.search(r"week", low) else "Q" if re.search(r"quarter", low) else
                "M" if re.search(r"month", low) else "D" if re.search(r"\b(day|daily)\b", low) else auto_freq(df[dates[0]]))
        agg = ("avg" if re.search(r"average|mean", low) or (RATE_LIKE.search(value) and not re.search(r"total|sum", low))
               else "sum")
        return {"type": "router", "tool": "trend", "args": {"value": value, "freq": freq, "agg": agg}, "reason": "trend"}

    # --- hypothetical scenarios -------------------------------------------------
    pcts = [float(x) for x in _PCT.findall(low)]
    hypothetical = re.search(r"what if|what happens|what would happen|suppose|assum|imagine|if we|scenario|"
                             r"can we (increase|grow|raise|reduce|cut|improve)|could we", low)
    if not hypothetical:
        return None
    if re.search(r"\b(budget|ad spend|advertis|marketing spend|spend)\b", low) and "conversion" not in low:
        rank = re.search(r"\b(lowest|worst|weakest|bottom|least)[- ]?(performing)?\b|\b(highest|best|strongest|top)[- ]"
                         r"performing\b", low)
        if rank and pcts:
            return {"type": "router", "tool": "budget_ranked",
                    "args": {"change_pct": direction_of(low) * abs(pcts[0]), "worst": bool(rank.group(1))},
                    "reason": "budget scenario for a ranked group"}
        return None                     # budget scenarios: the planner's own budget_scenario tool
    if re.search(r"open (a |an )?(new |another )|new (store|branch|shop|location)|another (store|branch|shop)", low):
        return _clarify("To estimate a new location I need an assumption about how it would perform. Should I assume "
                        "it performs like the average existing location in that group, or will you give the expected "
                        "figures (for example monthly sales)?",
                        ["like the average existing location", "I will give the figures"], "new location")
    if "conversion" in low:
        if not pcts:
            return _clarify("What conversion rate should I assume (for example, 'from 3% to 5%')?", [], "conversion")
        if len(pcts) == 1 and re.search(r"\bby\s+-?\d", low) and not re.search(r"\bto\s+-?\d", low):
            return _clarify(f"Do you mean the conversion rate rises by {pcts[0]:g}% of its current value (relative), "
                            f"or by {pcts[0]:g} percentage points, or that it becomes {pcts[0]:g}%? For example: "
                            f"'conversion improves from 3% to {pcts[0]:g}%' or 'conversion improves by {pcts[0]:g} "
                            "percentage points'.",
                            [f"{pcts[0]:g}% relative", f"{pcts[0]:g} percentage points", f"becomes {pcts[0]:g}%"],
                            "conversion change size")
        to_rate = pcts[-1] / 100
        from_rate = pcts[0] / 100 if len(pcts) >= 2 else None
        return {"type": "router", "tool": "conversion_change", "args": {"to_rate": to_rate, "from_rate": from_rate},
                "reason": "conversion scenario"}
    direction = direction_of(low)
    cats = [c for c in find_column(df, low) if kind_of(df[c]) == "categorical"]
    if not cats and re.search(r"churn|attrition", low):
        cats = [c for c in df.columns if re.search(r"churn|attrition|left", c.lower()) and kind_of(df[c]) == "categorical"]
    if cats and pcts and not _numeric_named(df, low):
        col = cats[0]
        values = df[col].dropna().astype(str).value_counts()
        positive = next((v for v in values.index if v.lower() in ("yes", "true", "1", "churned", "left")), values.index[-1])
        unit = "points" if re.search(r"percentage points?|\bpp\b|points?\b", low) else \
            "relative" if re.search(r"relative|of the current|of current|proportionally", low) else None
        if unit is None:
            return _clarify(f"Do you mean the {col} rate falls by {pcts[0]:g}% of its current value (relative), or by "
                            f"{pcts[0]:g} percentage points?",
                            [f"{pcts[0]:g}% relative", f"{pcts[0]:g} percentage points"], "rate unit")
        return {"type": "router", "tool": "rate_change",
                "args": {"column": col, "value": positive, "change": direction * pcts[0], "unit": unit},
                "reason": "rate scenario"}
    named = _numeric_named(df, low)
    if pcts and named:
        group = roles.get("area") or roles.get("campaign")
        return {"type": "router", "tool": "column_change",
                "args": {"column": named[0], "change_pct": direction * abs(pcts[0]), "group": group},
                "reason": "percentage-change scenario"}
    if pcts:
        return _clarify("Which value should change by that percentage?", numeric, "scenario column")
    if named:
        return _clarify(f"By how much should {named[0]} change? Give a percentage, for example 10%.", [], "scenario size")
    return None


def run(plan: dict, df: pd.DataFrame, memory: dict, filename: str | None = None) -> dict:
    t, a = plan["tool"], plan["args"]
    if t == "understand":
        return overview.understand(df, filename)
    if t == "missing_values":
        return overview.missing_values(df)
    if t == "outliers":
        return overview.outliers(df, a.get("column"))
    if t == "patterns":
        return overview.patterns(df)
    if t == "trend":
        return overview.trend(df, a["value"], freq=a.get("freq", "M"), agg=a.get("agg", "sum"))
    if t == "drivers":
        return overview.drivers(df, a["group_column"], a["group"], a.get("outcome"))
    if t == "column_change":
        return scenario.column_change(df, a["column"], a["change_pct"], memory, a.get("group"))
    if t == "budget_ranked":
        from app.analyst import business
        perf = business.performance(df)
        ranked = perf["ranking"]
        if not ranked:
            raise Insufficient("No group has enough rows to rank performance.")
        focus = ranked[-1] if a["worst"] else ranked[0]
        r = business.budget_scenario(df, a["change_pct"], perf["group_column"], focus)
        r["chosen_by"] = (f"{focus} is the {'lowest' if a['worst'] else 'highest'}-performing {perf['group_column']} by "
                          f"{perf['ranking_metric']} (ranking: {', '.join(ranked)})")
        return r
    if t == "conversion_change":
        return scenario.conversion_change(df, a["to_rate"], a.get("from_rate"))
    if t == "rate_change":
        return scenario.rate_change(df, a["column"], a["value"], a["change"], a["unit"])
    raise Insufficient(f"Unknown routed tool '{t}'.")
