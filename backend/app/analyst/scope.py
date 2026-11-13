"""The analytical scope of a question, established once and used by every path.

Before anything runs, the question is read into a structured intent:
  entities  {column: [values]}   people / categories / groups the question names (found in the data, not guessed)
  time      {column, year, month} a period the question names ("September 2026", "in 2025")
  inherited bool                 the follow-up named nothing new, so the previous question's scope carries over
  clarify   str | None           the scope cannot be established without asking (e.g. "Yes" names a value of two
                                  columns and the question does not say which)

Every tool then works on the records inside that scope, and the answer is verified against it afterwards (`verify`).
Nothing here knows any dataset: entities are values that occur in the data and that the question names; time filters
need a real date column; record validity is judged from the meaning of the columns used (see `eligibility`).
"""
from __future__ import annotations

import datetime as _dt
import re

import numpy as np
import pandas as pd

from app.analyst.data import kind_of

MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
_MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS + [m[:3] for m in MONTHS if m != "may"]) + r")\b\.?(?:\s+(\d{4}))?", re.I)
_YEAR_RE = re.compile(r"\b(?:in|during|for|of)\s+(\d{4})\b", re.I)


_PREDICTIVE = re.compile(r"\b(predict\w*|forecast\w*|project(ed|ion)?|estimate\w*|will be|expect\w*)\b", re.I)
_RATE_WORDS = re.compile(r"\b(rate|share|percent\w*|proportion|fraction|ratio|how many|how often|likel\w*)\b", re.I)


def today() -> pd.Timestamp:
    return pd.Timestamp(_dt.date.today())


def _date_col(df: pd.DataFrame) -> str | None:
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    return dates[0] if len(dates) == 1 else None


def _named(df: pd.DataFrame, message: str) -> tuple[dict, list]:
    """Values the question names, by column; and values that name more than one column with nothing saying which."""
    from app.analyst.calc import named_values
    hits = named_values(df, message)
    by_value: dict[str, list[str]] = {}
    for col, val in hits:
        by_value.setdefault(val.lower(), []).append(col)
    entities: dict[str, list[str]] = {}
    ambiguous = []
    low = message.lower()
    for col, val in hits:
        cols = by_value[val.lower()]
        if len(cols) > 1:
            said = [c for c in cols if c.lower() in low or c.lower().replace("_", " ") in low]
            if len(said) != 1:
                if (val, tuple(cols)) not in ambiguous:
                    ambiguous.append((val, tuple(cols)))
                continue
            if said[0] != col:
                continue
        entities.setdefault(col, [])
        if val not in entities[col]:
            entities[col].append(val)
    return entities, ambiguous


def _time(df: pd.DataFrame, message: str) -> dict | None:
    col = _date_col(df)
    if col is None:
        return None
    m = _MONTH_RE.search(message)
    if m and not re.search(r"\b(by|per|each|every|monthly)\s+month", message, re.I):
        name = m.group(1).lower()
        month = next(i + 1 for i, x in enumerate(MONTHS) if x.startswith(name[:3]))
        year = int(m.group(2)) if m.group(2) else None
        return {"column": col, "year": year, "month": month}
    y = _YEAR_RE.search(message)
    if y:
        return {"column": col, "year": int(y.group(1)), "month": None}
    return None


def establish(message: str, df: pd.DataFrame, memory: dict | None = None) -> dict:
    from app.analyst.calc import _ALL_SCOPE, _ELLIPTIC  # noqa: PLC0415 - one definition of "follow-up" and "everyone"
    memory = memory or {}
    entities, ambiguous = _named(df, message)
    if _RATE_WORDS.search(message):
        # "what share of employees have attrition Yes": a value of a two-valued column is the event being measured,
        # not a filter on the records (filtering to it would make every share 100%)
        binary = {c for c in df.columns if df[c].nunique(dropna=True) == 2}
        entities = {c: v for c, v in entities.items() if c not in binary}
        ambiguous = [(v, cols) for v, cols in ambiguous if not set(cols) <= binary]
    time = _time(df, message)
    target_period = None
    if time and _PREDICTIVE.search(message):
        target_period, time = time, None     # "forecast X for September 4, 2026": the date is what is predicted
    scope = {"entities": entities, "time": time, "inherited": False, "clarify": None, "target_period": target_period}
    if ambiguous:
        val, cols = ambiguous[0]
        scope["clarify"] = (f"'{val}' is a value of more than one column ({', '.join(cols)}). Which one do you mean — "
                            f"for example '{cols[0].replace('_', ' ')} {val}'?")
        return scope
    prev = memory.get("scope") or {}
    if not entities and not time and (prev.get("entities") or prev.get("time")) and _ELLIPTIC.search(message) \
            and not _ALL_SCOPE.search(message):
        scope.update(entities={k: list(v) for k, v in (prev.get("entities") or {}).items() if k in df.columns},
                     time=prev.get("time"), inherited=True)
    return scope


def active(scope: dict | None) -> bool:
    return bool(scope and (scope.get("entities") or scope.get("time")))


def mask(df: pd.DataFrame, scope: dict | None) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    if not scope:
        return keep
    for col, vals in (scope.get("entities") or {}).items():
        if col in df.columns:
            keep &= df[col].astype(str).str.lower().isin([v.lower() for v in vals])
    t = scope.get("time")
    if t and t["column"] in df.columns:
        d = pd.to_datetime(df[t["column"]], errors="coerce")
        if t.get("year"):
            keep &= d.dt.year == t["year"]
        if t.get("month"):
            keep &= d.dt.month == t["month"]
    return keep


def describe(scope: dict | None, n_in: int | None = None, n_all: int | None = None) -> str:
    if not active(scope):
        return ""
    parts = [f"{c.replace('_', ' ')} = {' or '.join(v)}" for c, v in scope["entities"].items()]
    t = scope.get("time")
    if t:
        parts.append(f"{t['column'].replace('_', ' ')} in "
                     + (f"{MONTHS[t['month'] - 1].capitalize()} " if t.get("month") else "")
                     + (str(t["year"]) if t.get("year") else "(any year)"))
    out = "; ".join(parts)
    if n_in is not None and n_all is not None:
        out += f" — {n_in} of {n_all} records"
    return out + (" (carried over from the previous question)" if scope.get("inherited") else "")


def as_exclusions(scope: dict) -> list[dict]:
    """The scope as calculator exclusions, so a calculation can be held to it."""
    out = []
    for col, vals in (scope.get("entities") or {}).items():
        if len(vals) == 1:
            out.append({"reason": f"only {vals[0]}", "where": f"{col} != '{vals[0]}'"})
        else:
            out.append({"reason": "only " + " and ".join(vals),
                        "where": f"{col} not in [" + ", ".join(f"'{v}'" for v in vals) + "]"})
    t = scope.get("time")
    if t:
        cond = []
        if t.get("year"):
            cond.append(f"year({t['column']}) != {t['year']}")
        if t.get("month"):
            cond.append(f"month({t['column']}) != {t['month']}")
        if cond:
            out.append({"reason": "outside " + describe({"entities": {}, "time": t}).split(" in ", 1)[-1],
                        "where": " or ".join(cond)})
    return out


# ------------------------------------------------------------ record validity --
_AMOUNT = re.compile(r"rate|price|cost|amount|salary|pay|wage|hours?|qty|quantity|revenue|sales|spend|units|count|fee",
                     re.I)


def eligibility(df: pd.DataFrame, columns: list[str], historical: bool) -> tuple[pd.Series, list[dict]]:
    """Which records are valid observations for an analysis that uses `columns`, and why the others are not.
    Judged from the meaning of the columns used, not by generic cleaning:
      exact duplicate of an earlier record     -> the same observation twice (a repeated event differs in some field)
      negative value of an amount-like column  -> invalid (a cost, rate, count or duration cannot be negative)
      end time before start time (two time-of-day columns used) -> invalid as recorded
      missing value in a column used           -> unusable for that analysis only (zero is a value, not missing)
      dated after today, for historical analysis (trends, forecasts) -> a planned or future record, not history
    Unusual-but-possible values (outliers) are NOT excluded."""
    from app.analyst.calc import _looks_like_time, _to_hours  # noqa: PLC0415
    keep = pd.Series(True, index=df.index)
    reasons: list[dict] = []

    def drop(hit: pd.Series, why: str) -> None:
        nonlocal keep
        hit = hit.fillna(False).astype(bool) & keep
        if hit.any():
            reasons.append({"reason": why, "records": int(hit.sum()),
                            "lines": [int(i) + 2 for i in hit[hit].index][:20]})     # file lines (header is line 1)
            keep &= ~hit
    drop(df.duplicated(keep="first"), "exact duplicate of an earlier record")
    cols = [c for c in columns if c in df.columns]
    for c in cols:
        if kind_of(df[c]) == "numeric" and _AMOUNT.search(c):
            drop(pd.to_numeric(df[c], errors="coerce") < 0, f"negative {c.replace('_', ' ')}")
    times = [c for c in cols if kind_of(df[c]) == "categorical" and len(df[c].dropna()) and _looks_like_time(df[c].dropna())]
    if len(times) >= 2:
        drop(_to_hours(df[times[1]]) < _to_hours(df[times[0]]),
             f"{times[1].replace('_', ' ')} earlier than {times[0].replace('_', ' ')}")
    for c in cols:
        drop(df[c].isna(), f"missing {c.replace('_', ' ')}")
    date = _date_col(df)
    if historical and date:
        drop(pd.to_datetime(df[date], errors="coerce") > today(), f"dated after today ({today().date()}): a planned "
                                                                  "or future record, not history")
    return keep, reasons


# ------------------------------------------------------------- verification --
def verify_rows(used: pd.Series, df: pd.DataFrame, scope: dict) -> list[str]:
    """Problems if the records an answer used are not inside the requested scope."""
    if not active(scope):
        return []
    inside = mask(df, scope)
    outside = used & ~inside
    probs = []
    if outside.any():
        probs.append(f"{int(outside.sum())} record(s) outside the requested scope ({describe(scope)}) were used")
    if not (used & inside).any():
        probs.append(f"no record inside the requested scope ({describe(scope)}) was used")
    return probs


_SEL = re.compile(r"^\s*\(?\s*(\w+)\s*(?:!=|not\s+in\b)")


def replace_conflicting(exclusions: list[dict], sc: dict | None) -> list[dict]:
    """Earlier selections that the current scope replaces: "only Rahul Sharma" from a previous question must not stay
    when this question is about Aarav Mehta (same column), and an earlier period goes when a new one is named."""
    if not active(sc):
        return list(exclusions)
    cols = set((sc.get("entities") or {}).keys())
    tcol = (sc.get("time") or {}).get("column")
    out = []
    for e in exclusions:
        m = _SEL.match(e.get("where", ""))
        if m and m.group(1) in cols:
            continue
        if tcol and re.search(rf"\b(year|month)\(\s*{re.escape(tcol)}\s*\)", e.get("where", "")):
            continue
        out.append(e)
    return out
