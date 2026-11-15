"""What each column of an unfamiliar dataset is, judged from its values as well as its name.

`profile(df)` gives every column a role with the evidence and a confidence:

  identifier     (nearly) unique per row, or named id/code/ref with many values
  person_name    text of two or more capitalised words, many distinct values
  category       a few to a few dozen repeated labels
  flag           exactly two values forming a yes/no-like pair
  datetime       parsed dates                time_of_day   "09:30"-style clock times
  year           integers that are all plausible calendar years
  amount         money-like numbers (currency in the source, or named cost/price/revenue/...)
  count          non-negative integers
  rate           proportions in 0..1, or percentages, or named rate/pct/share
  measurement    any other numeric column
  free_text      long or mostly unique text        constant   one value only

Names alone never decide a role when the values contradict them. A concept the question uses but the data does not
name (`concept_candidates`, e.g. "churn" for a yes/no column called attrition) is only ever offered as a question
to the user -- never mapped silently.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.analyst.data import _words, kind_of

_AMOUNT_WORDS = {"cost", "costs", "price", "revenue", "sales", "amount", "amt", "spend", "fee", "fees", "income",
                 "salary", "wage", "pay", "bill", "budget", "profit", "value", "charge", "fare", "premium", "gmv"}
_RATE_WORDS = {"rate", "pct", "percent", "percentage", "share", "ratio", "proportion", "margin", "score"}
_OUTCOME_WORDS = {"churn", "churned", "attrition", "left", "default", "defaulted", "failure", "failed", "fault",
                  "converted", "conversion", "outcome", "target", "label", "status", "result", "returned", "cancelled",
                  "fraud", "readmitted", "passed", "survived", "won", "approved", "delayed", "late", "escalated",
                  "diagnosis", "class", "response", "grade", "severity", "progression", "y"}
# words people use for the same idea; used only to OFFER a column, never to map one silently
CONCEPTS = [
    {"churn", "attrition", "left", "leave", "leaver", "leavers", "quit", "resign", "resignation", "turnover", "retention",
     "retained", "cancelled", "cancel"},
    {"failure", "failures", "fault", "faults", "breakdown", "defect", "defects", "failed", "malfunction"},
    {"revenue", "sales", "income", "turnover", "takings", "gmv"},
    {"cost", "costs", "expense", "expenses", "spend", "spending", "outlay"},
    {"customer", "customers", "client", "clients", "buyer", "buyers", "account", "accounts"},
    {"employee", "employees", "staff", "worker", "workers", "personnel", "people"},
    {"profit", "margin", "earnings"},
    {"late", "delay", "delayed", "lateness", "overdue"},
]


# outcome words include every word of the outcome-like concept groups, so a renamed outcome ("breakdown" for
# "failure", "leaver" for "churn") is still recognised
_OUTCOME_WORDS = _OUTCOME_WORDS | CONCEPTS[0] | CONCEPTS[1] | CONCEPTS[7]


def _tokens(c) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", _words(str(c))) if t}


def column_role(df: pd.DataFrame, c) -> dict:
    s = df[c]
    v = s.dropna()
    n, nu = len(v), v.nunique()
    t = _tokens(c)
    ev = []
    if n == 0 or nu <= 1:
        return {"column": c, "role": "constant", "confidence": 1.0, "evidence": ["one value (or none)"]}
    k = kind_of(s)
    if k == "datetime":
        return {"column": c, "role": "datetime", "confidence": 1.0, "evidence": ["parsed as dates"]}
    if k == "numeric":
        x = pd.to_numeric(v, errors="coerce").dropna()
        integral = bool((x == x.round()).all())
        if integral and x.between(1800, 2100).all() and (t & {"year", "yr", "fy", "season"} or nu <= 60):
            return {"column": c, "role": "year", "confidence": 0.9 if t & {"year", "yr"} else 0.6,
                    "evidence": ["integers within 1800–2100"]}
        countish = bool(t & {"n", "num", "count", "qty", "quantity", "items", "units", "total", "visits", "calls"})
        if integral and nu > 0.95 * n and not countish and (t & {"id", "code", "no", "number", "ref", "key"}
                                                            or x.is_monotonic_increasing):
            return {"column": c, "role": "identifier", "confidence": 0.85, "evidence": ["unique per row"]}
        if nu == 2 and integral:         # two integer codes (0/1, 1/2) are a flag, not a quantity
            vals = sorted(x.unique())
            pos = 1 if set(vals) <= {0, 1} else None
            if pos is None:
                vc = x.value_counts()
                pos = vc.index[-1]
            return {"column": c, "role": "flag", "confidence": 0.9 if set(vals) <= {0, 1} else 0.6,
                    "evidence": [f"two numeric codes ({vals[0]:g}, {vals[1]:g})"], "positive": f"{pos:g}",
                    "positive_is_minority": not set(vals) <= {0, 1}}
        if t & _RATE_WORDS and t & (_AMOUNT_WORDS | {"hourly", "daily", "nightly", "weekly", "monthly", "pay", "wage"}) \
                and not x.between(0, 1).all():
            # "hourly_rate", "daily_rate", "pay_rate": money per unit of time, not a share
            return {"column": c, "role": "amount", "confidence": 0.8, "evidence": ["named like a pay or price rate"]}
        if t & _RATE_WORDS or (x.between(0, 1).all() and not integral):
            return {"column": c, "role": "rate", "confidence": 0.8 if t & _RATE_WORDS else 0.6,
                    "evidence": ["named like a rate" if t & _RATE_WORDS else "values between 0 and 1"],
                    "scale": "0-1" if x.max() <= 1 else "0-100"}
        if t & _AMOUNT_WORDS:
            return {"column": c, "role": "amount", "confidence": 0.8, "evidence": ["named like money"]}
        if integral and (x >= 0).all():
            return {"column": c, "role": "count", "confidence": 0.6, "evidence": ["non-negative integers"]}
        return {"column": c, "role": "measurement", "confidence": 0.6, "evidence": ["numeric"]}
    sv = v.astype(str).str.strip()
    from app.analyst.metrics import yes_no
    if sv.str.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?").mean() > 0.9:
        return {"column": c, "role": "time_of_day", "confidence": 0.95, "evidence": ["clock times"]}
    if nu == 2:
        pair = yes_no(df, c)
        minority = str(sv.value_counts().index[-1])
        return {"column": c, "role": "flag", "confidence": 0.95 if pair else 0.6,
                "evidence": ["yes/no-like values" if pair else "two values"], "positive": pair[0] if pair else None,
                "event": pair[0] if pair else minority, "positive_is_minority": not pair}
    if sv.str.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?").mean() > 0.9:
        return {"column": c, "role": "time_of_day", "confidence": 0.95, "evidence": ["clock times"]}
    if nu > 0.9 * n and n > 20:
        if sv.str.fullmatch(r"[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)+").mean() > 0.8:
            return {"column": c, "role": "person_name", "confidence": 0.7, "evidence": ["capitalised multi-word names"]}
        if sv.str.len().mean() > 40:
            return {"column": c, "role": "free_text", "confidence": 0.8, "evidence": ["long, mostly unique text"]}
        return {"column": c, "role": "identifier", "confidence": 0.85, "evidence": ["unique per row"]}
    if sv.str.fullmatch(r"[A-Z][a-z'’-]+(?:\s+[A-Z][a-z'’-]+)+").mean() > 0.8 and nu >= 3:
        return {"column": c, "role": "person_name", "confidence": 0.6, "evidence": ["capitalised multi-word names"]}
    if t & {"id", "code", "ref", "key"} and nu > 0.5 * n:
        return {"column": c, "role": "identifier", "confidence": 0.7, "evidence": ["named like an id"]}
    if sv.str.len().mean() > 60:
        return {"column": c, "role": "free_text", "confidence": 0.7, "evidence": ["long text"]}
    return {"column": c, "role": "category", "confidence": 0.8 if nu <= 30 else 0.5,
            "evidence": [f"{nu} repeated labels"]}


def profile(df: pd.DataFrame) -> dict:
    cols = [column_role(df, c) for c in df.columns]
    roles = {r["column"]: r for r in cols}
    measures = [r["column"] for r in cols if r["role"] in ("amount", "count", "measurement", "rate")]
    flags = [r["column"] for r in cols if r["role"] == "flag"]
    # an outcome is a flag named like one (failure, churn ...) or the only yes/no-like flag -- never just any
    # two-valued label column
    yesno = [c for c in flags if roles[c].get("positive") and not roles[c].get("positive_is_minority")]
    key_amounts = [c for c in measures if roles[c]["role"] in ("amount", "count")]
    outcomes = [c for c in flags if _tokens(c) & _OUTCOME_WORDS] or (yesno if len(yesno) == 1 and not key_amounts else [])
    groups = [r["column"] for r in cols if r["role"] in ("category", "flag") and 2 <= df[r["column"]].nunique() <= 30]
    dates = [r["column"] for r in cols if r["role"] == "datetime"]
    years = [r["column"] for r in cols if r["role"] == "year"]
    entity = next((r["column"] for r in cols if r["role"] == "person_name"), None)
    # a numeric outcome named like one (progression, score, response ...) leads the measures
    num_out = [c for c in measures if _tokens(c) & _OUTCOME_WORDS]
    if num_out and not outcomes:
        measures = num_out + [m for m in measures if m not in num_out]
    nonfactors = [r["column"] for r in cols if r["role"] in ("year", "datetime", "identifier", "free_text", "constant",
                                                              "time_of_day")]
    return {"columns": cols, "roles": roles, "measures": measures, "flags": flags, "outcomes": outcomes,
            "nonfactors": nonfactors, "numeric_outcomes": num_out,
            "groups": groups, "dates": dates, "years": years, "entity": entity, "rows": int(len(df))}


def _forms(w: str) -> set[str]:
    """quitting -> quit, leaving -> leave, failed -> fail, faults -> fault: enough to match a concept word."""
    out = {w, w.rstrip("s")}
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            b = w[: -len(suf)]
            out |= {b, b + "e"}
            if len(b) > 3 and b[-1] == b[-2]:
                out.add(b[:-1])
    return out


def concept_candidates(word: str, df: pd.DataFrame) -> list[str]:
    """Columns that may mean `word` although they are named differently -- to OFFER, never to apply silently.
    Identifiers, free text and constants are never offered: no analysis question means them."""
    forms = _forms(word.lower())
    group = next((g for g in CONCEPTS if forms & g), None)
    if not group:
        return []
    out = []
    for c in df.columns:
        toks = _tokens(c)
        vals = {str(x).lower() for x in df[c].dropna().unique()[:30]} if kind_of(df[c]) == "categorical" else set()
        if (toks & group or vals & group) and column_role(df, c)["role"] not in ("identifier", "free_text", "constant"):
            out.append(str(c))
    return out


def outcome_concept(word: str) -> bool:
    return bool(_forms(word.lower()) & (CONCEPTS[0] | CONCEPTS[1] | CONCEPTS[7]))


# Everyday words for a measure that columns name differently ("trip length" -> km_driven, "visitors" -> footfall,
# "spend" -> total_bill). Each pair: the words people use, and the name tokens columns use for that measure.
MEASURE_WORDS = [
    ({"length", "distance", "far", "mileage", "kilometres", "kilometers", "miles", "travelled", "traveled"},
     {"km", "kms", "kilometre", "kilometres", "kilometer", "kilometers", "mi", "mile", "miles", "distance", "mileage",
      "length", "odometer"}),
    ({"visitors", "visitor", "visits", "footfall", "traffic", "shoppers", "walkins"},
     {"footfall", "visitors", "visitor", "visits", "traffic", "shoppers"}),
    ({"spend", "spending", "spent", "bill", "bills", "expenditure", "outlay"},
     {"bill", "spend", "spending", "spent", "expenditure", "payment", "paid"}),
]


# Everyday phrases for a yes/no outcome that columns name with one word ("behind schedule" -> late). Each pair: the
# phrases people use, and the name tokens of a yes/no column recording it. Only a flag column is a candidate.
FLAG_WORDS = [
    (("behind schedule", "running behind", "ran behind", "run behind", "runs behind", "not on time", "delayed",
      "overdue", "past due", "past the deadline", "missed the deadline", "missed deadline", "behind time", "tardy"),
     {"late", "lateness", "delay", "delayed", "overdue", "tardy"}),
]


def _flag_words(message: str, df: pd.DataFrame, named: set, out: dict) -> None:
    low = message.lower()
    for phrases, tokens in FLAG_WORDS:
        hit = next((ph for ph in sorted(phrases, key=len, reverse=True) if re.search(rf"\b{re.escape(ph)}\b", low)), None)
        if hit is None:
            continue
        cands = [c for c in df.columns if (_tokens(c) & tokens) and column_role(df, c)["role"] == "flag"]
        if not cands or set(cands) & named or any(re.search(rf"\b{t}\b", low) for t in tokens if t != hit):
            continue                                    # the column (or its own word) is already named
        (out["rewrite"].append((hit, cands[0])) if len(cands) == 1 else out["ambiguous"].append((hit, cands)))


def measure_words(message: str, df: pd.DataFrame) -> dict:
    """Measure words in a message that no column is named after, mapped through MEASURE_WORDS to the numeric columns
    that measure it: {"rewrite": [(word, column)], "ambiguous": [(word, [columns])]}. One candidate is used (and the
    answer says how the word was read); two or more are asked about. Columns the message already names win."""
    from app.analyst.data import find_column
    low = message.lower()
    named = {c for c in df.columns if re.search(rf"\b{re.escape(_words(str(c)).lower())}\b", low)}
    out = {"rewrite": [], "ambiguous": []}
    for words, tokens in MEASURE_WORDS:
        for w in sorted(words, key=len, reverse=True):
            if not re.search(rf"\b{w}\b", low) or find_column(df, w) or any(
                    _forms(w) & _tokens(c) for c in df.columns):
                continue                                # a column is named after the word: it is not a synonym here
            cands = [c for c in df.columns if kind_of(df[c]) == "numeric" and _tokens(c) & tokens
                     and column_role(df, c)["role"] not in ("identifier", "year", "constant")]
            if not cands or set(cands) & named:
                break
            (out["rewrite"].append((w, cands[0])) if len(cands) == 1 else out["ambiguous"].append((w, cands)))
            break
    _flag_words(message, df, named, out)
    return out
