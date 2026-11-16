"""Loading a dataset and describing its schema for the analyst."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

MAX_CATEGORIES_LISTED = 12


def load_table(path: str) -> pd.DataFrame:
    """CSV or Excel (first sheet), read by app/ingest.py: delimiter, encoding, decimal mark, header and numbers
    stored as text are detected, not assumed (files stored before uploads were normalised still read correctly).
    The read report is kept in df.attrs["ingestion"]. Columns that look like dates are parsed."""
    from app import ingest
    df, report = ingest.read_table(path)
    df.attrs["ingestion"] = report
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object and re.search(r"date|month|week|period|time", c, re.I):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[c] = parsed
    return df


def file_lines(frame: pd.DataFrame) -> dict:
    """{row label: its line in the uploaded file} (the header is line 1). A table cut from a file of several tables
    (app/analyst/tables.py) keeps its rows' labels, so its lines are label + 2; any other frame is the file itself."""
    if frame.attrs.get("file_lines"):
        return {i: int(i) + 2 for i in frame.index}
    return {i: n + 2 for n, i in enumerate(frame.index)}


def kind_of(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "categorical"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def schema(df: pd.DataFrame) -> dict:
    """Column names, kinds, missing counts and a few values: what the planner
    needs to read a question. Values listed are the data's own."""
    cols = []
    for c in df.columns:
        s = df[c]
        item = {"name": c, "kind": kind_of(s), "missing": int(s.isna().sum()), "unique": int(s.nunique(dropna=True))}
        if item["kind"] == "categorical":
            item["values"] = [str(v) for v in s.dropna().astype(str).value_counts().index[:MAX_CATEGORIES_LISTED]]
        elif item["kind"] == "numeric" and s.notna().any():
            item["min"], item["max"] = float(s.min()), float(s.max())
        cols.append(item)
    return {"rows": int(len(df)), "columns": cols}


def schema_text(sch: dict) -> str:
    """Compact schema description for a model prompt."""
    parts = []
    for c in sch["columns"]:
        extra = ""
        if c["kind"] == "categorical" and c.get("values"):
            extra = " values: " + ", ".join(c["values"][:8])
        parts.append(f"- {c['name']} ({c['kind']}{', ' + str(c['unique']) + ' unique' if c['kind'] == 'categorical' else ''}){extra}")
    return f"{sch['rows']} rows\n" + "\n".join(parts)


# ------------------------------------------------------- business column roles --

# Patterns are matched against the column name split into words; the first role
# pattern that matches exactly one numeric column decides it. Order matters:
# more specific names first.
ROLE_PATTERNS: dict[str, list[str]] = {
    "spend": [r"ad_?spend|spend", r"budget", r"cost", r"expense"],
    "revenue": [r"revenue", r"sales", r"income|gmv"],
    "profit": [r"profit"],
    "leads": [r"leads?", r"clicks?", r"visits?|visitors|sessions"],
    "conversions": [r"conversions?|converted", r"new_?customers|customers_?acquired|acquisitions?|acquired",
                    r"customers", r"signups?|orders?|purchases?"],
    "repeat": [r"repeat|returning|retained|retention"],
}
GROUP_PATTERNS: dict[str, list[str]] = {
    "area": [r"area", r"region", r"city|market|territory|zone", r"state|country|location"],
    "campaign": [r"campaign", r"channel", r"source|medium|ad_?group"],
}


def _words(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def business_roles(df: pd.DataFrame) -> dict:
    """Which columns play which business role, with the reason each was chosen.
    A role two columns match equally is reported as ambiguous, not guessed."""
    numeric = [c for c in df.columns if kind_of(df[c]) == "numeric"]
    categorical = [c for c in df.columns if kind_of(df[c]) == "categorical"]
    roles, evidence, ambiguous = {}, {}, {}

    def pick(role: str, patterns: list[str], pool: list[str]) -> None:
        for pat in patterns:
            hits = [c for c in pool if re.search(rf"(?:^|_)(?:{pat})(?:_|$)", _words(c)) and c not in roles.values()]
            if len(hits) == 1:
                roles[role] = hits[0]
                evidence[role] = f"column name '{hits[0]}' matches '{pat.split('|')[0]}'"
                return
            if len(hits) > 1:
                ambiguous[role] = hits
                return

    for role, pats in ROLE_PATTERNS.items():
        pick(role, pats, numeric)
    for role, pats in GROUP_PATTERNS.items():
        pick(role, pats, categorical)
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    if len(dates) == 1:
        roles["date"], evidence["date"] = dates[0], f"'{dates[0]}' is the only date column"
    elif len(dates) > 1:
        ambiguous["date"] = dates
    return {"roles": roles, "evidence": evidence, "ambiguous": ambiguous}


_STOP = {"the", "a", "an", "of", "and", "or", "in", "on", "by", "per", "for", "to", "is", "are", "what", "which", "how",
         "with", "each", "all", "me", "show", "average", "total", "mean", "sum", "number", "count", "value", "values"}


def _token_matches(t: str, w: str) -> bool:
    """Column-name token `t` written as question word `w`: identical, a prefix of it ('exp' / 'expectancy',
    'pop' / 'population'), or an in-order abbreviation with the same first letter ('dept' / 'department')."""
    if t == w or (t.endswith("s") and t[:-1] == w) or (w.endswith("s") and w[:-1] == t):
        return True
    if len(t) < 3 or t.isdigit():
        return False
    if w.startswith(t) and len(w) - len(t) <= 8:
        return True
    if t[0] == w[0] and len(t) < len(w):
        it = iter(w)
        return all(ch in it for ch in t) and len(t) >= 4
    return False


def find_column(df: pd.DataFrame, words: str) -> list[str]:
    """Columns the text names, best first: the column's name as words ('total bill' for total_bill, plurals too);
    otherwise every word of the column name abbreviated in the text ('life expectancy' for lifeExp, 'GDP per
    capita' for gdpPercap, 'population' for pop, 'department' for dept_name)."""
    text = " " + re.sub(r"[^a-z0-9]+", " ", words.lower()) + " "
    hits = []
    for c in df.columns:
        w = re.sub(r"[^a-z0-9]+", " ", _words(c)).strip()
        forms = {w, w + "s", w + "es", w[:-1] if w.endswith("s") else w, w[:-1] + "ies" if w.endswith("y") else w}
        if any(f and f" {f} " in text for f in forms):
            hits.append(c)
    exact = sorted(hits, key=lambda c: -len(c))
    qwords = [x for x in text.split() if x not in _STOP]
    allw = text.split()
    joined = [a + b for a, b in zip(allw, allw[1:])]            # "per capita" -> "percapita" (fillers kept here)
    fuzzy = []
    for c in df.columns:
        if c in hits:
            continue
        toks = [t for t in re.sub(r"[^a-z0-9]+", " ", _words(c)).split() if t]
        if not toks or all(len(t) <= 2 and not t.isalpha() for t in toks):
            continue
        used, ok = set(), True
        for t in toks:
            m = next((i for i, x in enumerate(qwords) if i not in used and _token_matches(t, x)), None)
            if m is None:
                j = next((k for k, x in enumerate(joined) if len(t) >= 4 and x.startswith(t)), None)
                if j is None:
                    ok = False
                    break
                used |= {j, j + 1}
            else:
                used.add(m)
        if ok:
            fuzzy.append(c)
    return exact + sorted(fuzzy, key=lambda c: -len(c))


def finite(values) -> np.ndarray:
    a = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return a[np.isfinite(a)]
