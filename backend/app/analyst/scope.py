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


# "expected", "expect" -- not "life expectancy", which is a measure, not a prediction
_PREDICTIVE = re.compile(r"\b(predict\w*|forecast\w*|project(ed|ion)?|estimate\w*|will be|expect(?:s|ed|ing)?)\b", re.I)
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
    as_cols = column_values(message, df, hits)
    hits = [h for h in hits if h not in as_cols]
    # an ordinary lower-case word that only matches a capitalised value ("a business overview" against the Segment value
    # "Business") is the word, not that value -- unless its column is named with it ("business segment",
    # "segment = business"), which makes it the value
    hits = [(c, v) for c, v in hits if not _lowercase_word_only(message, c, v)]
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


def column_named(message: str, col: str) -> bool:
    """The question names the column itself ("segment", "sales rep", "customer_id")."""
    name = re.sub(r"[_\s]+", " ", str(col)).strip().lower()
    return bool(name) and re.search(rf"\b{re.escape(name)}s?\b", re.sub(r"[_\s]+", " ", message.lower())) is not None


def _lowercase_word_only(message: str, col: str, val: str) -> bool:
    """`val` occurs in the question only in lower case while the data writes it with capitals, and the question does
    not name `col` -- an everyday word, not a selection of records."""
    v = str(val)
    if v == v.lower():
        return False
    # the value, or one of its words as written ("Priya" for "Priya Singh"), appears with its capitals
    if any(re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", message) for w in [v] + re.findall(r"[A-Za-z][\w'-]{2,}", v)):
        return False
    return not column_named(message, col)


def unpopulated_filters(df: pd.DataFrame, entities: dict) -> list[tuple[str, dict]]:
    """Filter columns that hold no value at all in the records the question's OTHER filters select ("Segment =
    Business" within Table = Sales Transactions, where Segment is empty): (column, {other column: [values where the
    column is filled in]}) for each. Such a filter can only select nothing, whatever value it names."""
    out = []
    for col in entities:
        others = {c: v for c, v in entities.items() if c != col}
        if not others:
            continue
        keep = pd.Series(True, index=df.index)
        for c, vals in others.items():
            keep &= df[c].astype(str).str.lower().isin([str(x).lower() for x in vals])
        if keep.any() and df.loc[keep, col].isna().all():
            filled = {c: sorted(df.loc[df[col].notna(), c].dropna().astype(str).unique().tolist())[:5] for c in others}
            out.append((col, filled))
    return out


def column_values(message: str, df: pd.DataFrame, hits) -> set:
    """Named values that are also the name of a numeric column ("votes for Joly" where Joly is a column of votes and
    also a value of `winner`): read as that column -- the measure -- unless the message names the value's own column
    next to it ("winner Joly", "where the winner is Joly"). Returns the (column, value) pairs that are columns, not filters."""
    low = message.lower()
    out = set()
    for col, val in hits:
        same = [c for c in df.columns if str(c).lower() == str(val).lower() and c != col and kind_of(df[c]) == "numeric"]
        if not same:
            continue
        words = [w for w in re.split(r"[^a-z0-9]+", str(col).lower()) if len(w) >= 3]
        v = re.escape(str(val).lower())
        if any(re.search(rf"\b{re.escape(w)}\w*\s+(?:is\s+|was\s+|were\s+|=\s*|of\s+|equals?\s+|being\s+)?{v}\b|"
                         rf"\b{v}\s+(?:is|was|as)\s+(?:the\s+)?{re.escape(w)}", low) for w in words):
            continue                                     # "leader is Ash", "leader Ash": the value's own column, a filter
        out.add((col, val))
    return out


_NEGATION = re.compile(r"\b(?:exclud\w*|except(?: for)?|other than|besides|apart from|without|not (?:counting|including)|"
                       r"ignor\w*|leav\w* out|minus|but not|aside from|everyone but|everybody but|all but|but)\b", re.I)


def _negated(message: str, value: str) -> bool:
    """The question names the value to leave it OUT ("everyone except Aarav", "excluding Rahul and Priya")."""
    for m in _NEGATION.finditer(message):
        tail = message[m.end():m.end() + 80]
        clause = re.split(r"[.;?!]|\b(?:by|per|for each|in|during|over)\b", tail, maxsplit=1)[0]
        if re.search(rf"\b{re.escape(value)}\b", clause, re.I):
            return True
    return False


def _time(df: pd.DataFrame, message: str) -> dict | None:
    col = _date_col(df)
    if col is None:
        return None
    from app.analyst.intent import _OPEN, _RANGE  # noqa: PLC0415
    if _RANGE.search(message) or _OPEN.search(message):
        return None                     # "between 1 and 15 August": a date range (intent.explicit_period), not August
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


_SUP = r"(highest|lowest|largest|biggest|smallest|top|best|worst|leading)"
_STOP = r"(?=\s*[,;:?.!]|\s+(?:what|which|who|show|list|give|are|is|was|were|how|and|by)\b|\s*$)"
_SUPERLATIVE_REF = (
    # "within the highest-revenue region", "for the lowest cost category"
    re.compile(rf"\b(?:within|in|inside|among|for|from|of)\s+(?:the|that)\s+{_SUP}[\s-]+([a-z][\w %-]*?)[\s-]+([a-z]+){_STOP}",
               re.I),
    # "in the region with the highest revenue"
    re.compile(rf"\b(?:within|in|inside|among|for|from|of)\s+(?:the|that)\s+([a-z]+)\s+with\s+the\s+{_SUP}\s+"
               rf"([a-z][\w %-]*?){_STOP}", re.I))


def superlative_group(message: str, df: pd.DataFrame) -> dict | None:
    """A group the question picks out by rank instead of by name ("within the highest-revenue region"): resolved
    from the data with the shared ranking rule (app/analyst/intent.ranking_aggregate -- revenue is compared by its
    total, "average revenue" by its mean; a measure that says neither is asked about). Returns {"column", "value",
    "text"}, {"clarify"}, or None when the question names no such group."""
    from app.analyst.data import find_column  # noqa: PLC0415
    from app.analyst.intent import ranking_aggregate  # noqa: PLC0415
    from app.analyst.metrics import group_column  # noqa: PLC0415
    for i, rx in enumerate(_SUPERLATIVE_REF):
        m = rx.search(message)
        if not m:
            continue
        if i == 0:
            sup, measure_txt, noun = m.group(1), m.group(2), m.group(3)
        else:
            noun, sup, measure_txt = m.group(1), m.group(2), m.group(3)
        col, _ = group_column(f"each {noun}", df)
        nums = [c for c in find_column(df, measure_txt) if kind_of(df[c]) == "numeric"]
        if col is None or len(nums) != 1:
            continue
        measure = nums[0]
        how = ranking_aggregate(m.group(0), measure)
        low = sup.lower() in ("lowest", "smallest", "worst")
        if how is None:
            return {"clarify": f"By “the {sup} {measure_txt} {noun}”, do you mean the {col} with the {sup} **total** "
                               f"{measure} or the {sup} **average** {measure}?"}
        vals = pd.to_numeric(df[measure], errors="coerce").groupby(df[col].astype(str)).agg(how).dropna()
        if vals.empty:
            continue
        pick = vals.idxmin() if low else vals.idxmax()
        word = {"sum": "total", "mean": "average", "median": "median", "max": "largest single",
                "min": "smallest single"}[how]
        return {"column": col, "value": str(pick), "span": m.span(), "phrase": m.group(0),
                "prep": m.group(0).split()[0],
                "text": f"the {col} with the {'lowest' if low else 'highest'} {word} {measure}: {vals[pick]:,.2f}"}
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
    # "for smokers", "among non-members": a yes/no column named as a noun selects its positive (or negative) value
    from app.analyst.metrics import _singular, _tokens, yes_no  # noqa: PLC0415
    both = re.search(r"\b(compar\w*|versus|vs\.?|between|difference|differ\w*|associat\w*|related|linked|explain\w*|"
                     r"drive\w*|factors?|predict\w*|influenc\w*|affect\w*)\b", message, re.I)
    for m in ([] if both else re.finditer(r"\b(?:for|among|of|from|with)\s+(?:the\s+)?(non[- ]?)?([a-z]{3,})\b", message, re.I)):
        neg, word = bool(m.group(1)), m.group(2).lower()
        cols = [c for c in df.columns if kind_of(df[c]) == "categorical" and (_singular(word) in _tokens(c)
                or word in _tokens(c)) and c not in entities and yes_no(df, c)]
        if len(cols) == 1:
            pos, negv = yes_no(df, cols[0])
            entities[cols[0]] = [negv if neg else pos]
    # The structured request (app/analyst/intent.py) is the one reading of the question; the scope carries its record
    # selection to every path: names found in long text columns (company names), thresholds ("csat below 3"), yes/no
    # filters ("only escalated", "not gift wrapped"), and -- taken OUT of the selection -- the event a rate is about.
    where: list[dict] = []
    try:
        from app.analyst import intent as _intent  # noqa: PLC0415
        req = _intent.read(message, df)
    except Exception:  # noqa: BLE001 - the scope never fails because the reader could not read a message
        req = None
    if req is not None:
        for c, v in req.entities:
            if c in df.columns and df[c].nunique() > 50 and c not in entities:
                entities.setdefault(c, [])
                if v not in entities[c]:
                    entities[c].append(v)
        if req.event and req.event[0] in entities:
            entities[req.event[0]] = [v for v in entities[req.event[0]] if str(v).lower() != str(req.event[1]).lower()]
            if not entities[req.event[0]]:
                del entities[req.event[0]]
        roles = None
        for c, yes in [(c, True) for c in req.restrict] + [(c, False) for c in req.restrict_not]:
            if c in entities:
                continue                                    # already selected by its value
            roles = roles or __import__("app.analyst.semantics", fromlist=["profile"]).profile(df)["roles"]
            pair = _intent.flag_values(df, c, roles[c])
            if pair is not None:
                where.append({"column": c, "op": "==", "value": pair[0] if yes else pair[1],
                              "text": f"{c.replace('_', ' ')} = {pair[0] if yes else pair[1]}"})
        for cd in req.conditions:
            where.append({"column": cd["column"], "op": cd["op"], "value": cd["value"],
                          "text": f"{cd['column'].replace('_', ' ')} {cd['op']} {cd['value']:g}"})
        dcol = _date_col(df)
        if req.period and not req.period.get("unreadable") and dcol is not None and not _PREDICTIVE.search(message) \
                and not re.search(r"\b(targets?|goals?|quotas?|on track|what if|reach)\b", message, re.I):
            # "the last 7 days", "between 1 and 15 August": the same dates for every path (a target or forecast
            # question uses its period as the horizon, not as a selection)
            period = {"column": dcol, "start": req.period.get("start"), "end": req.period.get("end"),
                      "label": req.period.get("label") or "the period named"}
    period = locals().get("period")
    excluded: dict[str, list[str]] = {}
    for col in list(entities):
        out = [v for v in entities[col] if _negated(message, v)]
        if out:
            excluded[col] = out
            entities[col] = [v for v in entities[col] if v not in out]
            if not entities[col]:
                del entities[col]
    # a filter on a column that is empty in the records the other filters select can only select nothing: named
    # outright ("Sales Transactions by Segment = Business") it is explained; met only as a value, it is not applied
    dropped, unsupported = [], None
    for col, filled in unpopulated_filters(df, entities):
        where = "; ".join(f"{c} = {' or '.join(v)}" for c, v in filled.items() if v)
        inside = "; ".join(f"{c} = {' or '.join(entities[c])}" for c in filled)
        text = (f"**{col}** has no values in the records where {inside}, so it can't be used to filter them"
                + (f" — it is only filled in where {where}." if where else "."))
        if column_named(message, col) and unsupported is None:
            unsupported = text
        else:
            dropped.append(f"“{' or '.join(entities[col])}” was not applied as a {col} filter: {col} has no values "
                           f"where {inside}")
        del entities[col]
    time = _time(df, message)
    numeric_eq: dict[str, list] = {}
    if time is None and not _PREDICTIVE.search(message):
        # "in 2007" with a numeric year column (no date column): a filter on that column, never silently dropped
        y = re.search(r"\b(?:in|for|during|of|from)\s+((?:18|19|20)\d{2})\b", message, re.I)
        yc = year_column(df)
        if y and yc and int(y.group(1)) in set(pd.to_numeric(df[yc], errors="coerce").dropna().astype(int)):
            numeric_eq[yc] = [int(y.group(1))]
    target_period = None
    if time and _PREDICTIVE.search(message):
        target_period, time = time, None     # "forecast X for September 4, 2026": the date is what is predicted
    scope = {"entities": entities, "time": time, "inherited": False, "clarify": None, "target_period": target_period}
    if excluded:
        scope["excluded_entities"] = excluded
    if numeric_eq:
        scope["numeric_eq"] = numeric_eq
    if where:
        scope["where"] = where
    if period:
        scope["period"] = period
    if dropped:
        scope["dropped"] = dropped
    if unsupported:
        scope["clarify"] = unsupported + " Nothing was filtered or calculated."
        scope["unsupported_filter"] = True
        return scope
    if ambiguous:
        val, cols = ambiguous[0]
        scope["clarify"] = (f"'{val}' is a value of more than one column ({', '.join(cols)}). Which one do you mean — "
                            f"for example '{cols[0].replace('_', ' ')} {val}'?")
        return scope
    prev = memory.get("scope") or {}
    if not entities and not time and not excluded and not numeric_eq and (prev.get("entities") or prev.get("time")) and _ELLIPTIC.search(message) \
            and not _ALL_SCOPE.search(message):
        scope.update(entities={k: list(v) for k, v in (prev.get("entities") or {}).items() if k in df.columns},
                     time=prev.get("time"), inherited=True)
    return scope


def from_request(req, df: pd.DataFrame) -> dict:
    """The record selection of a structured reading as a scope (what the answer to a follow-up is computed over):
    named values, exclusions, yes/no filters, thresholds and a date range -- the same shape establish() returns."""
    from app.analyst import intent as _intent  # noqa: PLC0415
    from app.analyst import semantics  # noqa: PLC0415
    entities: dict = {}
    for c, v in req.entities:
        if c != req.group or not req.compared:
            entities.setdefault(c, []).append(v)
    excluded: dict = {}
    for c, v in req.excluded:
        excluded.setdefault(c, []).append(v)
    where = []
    roles = semantics.profile(df)["roles"] if (req.restrict or req.restrict_not) else {}
    for c, yes in [(c, True) for c in req.restrict] + [(c, False) for c in req.restrict_not]:
        pair = _intent.flag_values(df, c, roles[c]) if c in roles else None
        if pair is not None:
            where.append({"column": c, "op": "==", "value": pair[0] if yes else pair[1],
                          "text": f"{c.replace('_', ' ')} = {pair[0] if yes else pair[1]}"})
    for cd in req.conditions:
        where.append({"column": cd["column"], "op": cd["op"], "value": cd["value"],
                      "text": f"{cd['column'].replace('_', ' ')} {cd['op']} {cd['value']:g}"})
    sc = {"entities": entities, "time": None, "inherited": False, "clarify": None, "target_period": None}
    if req.compared and req.group:
        named = [v for c, v in req.entities if c == req.group]
        if named:
            sc["entities"][req.group] = named                    # the values compared are the records used
    if excluded:
        sc["excluded_entities"] = excluded
    if where:
        sc["where"] = where
    dcol = _date_col(df)
    if req.period and not req.period.get("unreadable") and dcol:
        sc["period"] = {"column": dcol, "start": req.period.get("start"), "end": req.period.get("end"),
                        "label": req.period.get("label") or "the period named"}
    if req.calendar:
        sc["time"] = dict(req.calendar)
    return sc


def year_column(df: pd.DataFrame) -> str | None:
    """A numeric column holding calendar years (named year/yr, or every value a plausible year), when there is one."""
    cands = []
    for c in df.columns:
        if kind_of(df[c]) != "numeric":
            continue
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if not len(v) or not (v == v.round()).all() or not v.between(1800, 2100).all():
            continue
        if re.search(r"(^|_|\b)(year|yr|fy|season)s?($|_|\b)", str(c), re.I) or v.nunique() >= 2:
            cands.append(c)
    named = [c for c in cands if re.search(r"year|yr|fy|season", str(c), re.I)]
    pick = named or cands
    return pick[0] if len(pick) == 1 else None


def active(scope: dict | None) -> bool:
    return bool(scope and (scope.get("entities") or scope.get("time") or scope.get("excluded_entities")
                           or scope.get("numeric_eq") or scope.get("where") or scope.get("period")))


def where_mask(df: pd.DataFrame, where: list[dict]) -> pd.Series:
    """Records meeting every condition: a number compared ("csat < 3"; a missing value meets none), or a yes/no value."""
    keep = pd.Series(True, index=df.index)
    for w in where or []:
        c, op, v = w["column"], w["op"], w["value"]
        if c not in df.columns:
            continue
        if isinstance(v, str):
            x = df[c].astype(str).str.strip().str.lower()
            v = v.lower()
        else:
            x = pd.to_numeric(df[c], errors="coerce")
        keep &= {"==": x == v, "!=": x != v, ">": x > v, ">=": x >= v, "<": x < v, "<=": x <= v}[op].fillna(False)
    return keep


def mask(df: pd.DataFrame, scope: dict | None) -> pd.Series:
    keep = pd.Series(True, index=df.index)
    if not scope:
        return keep
    for col, vals in (scope.get("entities") or {}).items():
        if col in df.columns:
            keep &= df[col].astype(str).str.lower().isin([v.lower() for v in vals])
    for col, vals in (scope.get("excluded_entities") or {}).items():
        if col in df.columns:
            keep &= ~df[col].astype(str).str.lower().isin([v.lower() for v in vals])
    for col, vals in (scope.get("numeric_eq") or {}).items():
        if col in df.columns:
            keep &= pd.to_numeric(df[col], errors="coerce").isin(vals)
    if scope.get("where"):
        keep &= where_mask(df, scope["where"])
    p = scope.get("period")
    if p and p["column"] in df.columns:
        d = pd.to_datetime(df[p["column"]], errors="coerce").dt.normalize()
        if p.get("start"):
            keep &= (d >= pd.Timestamp(p["start"]).normalize()).fillna(False)
        if p.get("end"):
            keep &= (d <= pd.Timestamp(p["end"]).normalize()).fillna(False)
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
    parts = [f"{c.replace('_', ' ')} = {' or '.join(v)}" for c, v in (scope.get("entities") or {}).items()]
    parts += [f"{c.replace('_', ' ')} not {' or '.join(v)}" for c, v in (scope.get("excluded_entities") or {}).items()]
    parts += [f"{c.replace('_', ' ')} = {' or '.join(map(str, v))}" for c, v in (scope.get("numeric_eq") or {}).items()]
    parts += [w["text"] for w in scope.get("where") or []]
    if scope.get("period"):
        parts.append(f"{scope['period']['column'].replace('_', ' ')}: {scope['period']['label']}")
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
    for col, vals in (scope.get("numeric_eq") or {}).items():
        out.append({"reason": f"only {col.replace('_', ' ')} " + " or ".join(map(str, vals)),
                    "where": f"{col} != {vals[0]}" if len(vals) == 1 else f"{col} not in [{', '.join(map(str, vals))}]"})
    for col, vals in (scope.get("excluded_entities") or {}).items():
        out.append({"reason": "excluding " + " and ".join(vals),
                    "where": f"{col} == '{vals[0]}'" if len(vals) == 1 else f"{col} in [" + ", ".join(f"'{v}'" for v in vals) + "]"})
    _flip = {">": "<=", ">=": "<", "<": ">=", "<=": ">", "==": "!=", "!=": "=="}
    for w in scope.get("where") or []:
        v = f"'{w['value']}'" if isinstance(w["value"], str) else f"{w['value']:g}"
        out.append({"reason": f"only {w['text']}", "where": f"{w['column']} {_flip[w['op']]} {v} or isnull({w['column']})"})
    p = scope.get("period")
    if p:
        cond = [f"isnull({p['column']})"]
        if p.get("start"):
            cond.append(f"date_before({p['column']}, '{p['start']}')")
        if p.get("end"):
            cond.append(f"date_after({p['column']}, '{p['end']}')")
        out.append({"reason": f"outside {p['label']}", "where": " or ".join(cond)})
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
      negative value of a column that can never be negative (a rate, hours, a price, a quantity, a count) -> invalid;
        a negative revenue, cost, profit or other amount is a refund, credit or loss: kept (see signed_negatives)
      end time before start time (two time-of-day columns used) -> invalid as recorded
      missing value in a column used           -> unusable for that analysis only (zero is a value, not missing)
      dated after today, for historical analysis (trends, forecasts) -> a planned or future record, not history
    Unusual-but-possible values (outliers) are NOT excluded."""
    from app.analyst.calc import _looks_like_time, _to_hours  # noqa: PLC0415
    from app.analyst.metrics import never_negative, never_negative_reason  # noqa: PLC0415 - the one rule
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
        if kind_of(df[c]) == "numeric" and never_negative(c):
            drop(pd.to_numeric(df[c], errors="coerce") < 0,
                 f"negative {c.replace('_', ' ')} ({never_negative_reason(c)})")
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


def signed_negatives(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Negative values of amount columns that CAN be negative (revenue, cost, profit, salary ...) among `columns`:
    kept as recorded, and reported, as the calculator's data checks report them."""
    from app.analyst.metrics import never_negative  # noqa: PLC0415
    out = []
    for c in columns:
        if c in df.columns and kind_of(df[c]) == "numeric" and _AMOUNT.search(str(c)) and not never_negative(c):
            hit = pd.to_numeric(df[c], errors="coerce") < 0
            if hit.any():
                out.append({"reason": f"negative {str(c).replace('_', ' ')} (kept, as recorded)", "records": int(hit.sum()),
                            "lines": [int(i) + 2 for i in hit[hit].index][:20], "kept": True, "negative": str(c)})
    return out


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
    cols = set((sc.get("entities") or {}).keys()) | set((sc.get("excluded_entities") or {}).keys()) \
        | set((sc.get("numeric_eq") or {}).keys())
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


def valid_records(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """THE record-validity policy for a descriptive answer, the same on every path (the tool catalogue, rankings,
    column summaries, the calculator, analysis programs and the verifier):
      * a negative value of a column that can never be negative (a rate, price, duration, count ...) is impossible:
        the record is left out, and listed with the reason;
      * a negative amount that can be negative (revenue, cost, profit, salary: a refund, credit or loss) is kept as
        recorded, and flagged;
      * a missing value is unusable for that column only, and counted;
      * exact duplicates are kept as recorded and flagged (unless the question asks to exclude them);
      * unusual-but-possible values (outliers) are never removed.
    Returns (the records used, what was left out or flagged)."""
    cols = [c for c in columns if c in df.columns]
    base = df.drop(columns=[c for c in df.columns if str(c).startswith("__")], errors="ignore")
    ok, reasons = eligibility(base.assign(__row=range(len(base))), cols, historical=False)
    reasons = [r for r in reasons if not r["reason"].startswith("missing ")]  # counted by the aggregate itself
    kept = df[ok.values]
    flags = signed_negatives(kept, cols)
    dup = base.duplicated(keep="first")
    if dup[ok.values].any():
        flags.append({"reason": "exact duplicate of an earlier record (kept, as recorded)",
                      "records": int(dup[ok.values].sum()), "kept": True,
                      "lines": [int(i) + 2 for i in dup[ok.values][dup[ok.values]].index][:20]})
    return kept, reasons + flags


def describe_exclusions(items: list[dict]) -> str | None:
    """'Left out: 1 record — negative hourly rate (a rate cannot be negative), line 7.' and what was kept and flagged."""
    out_ = [x for x in items if not x.get("kept")]
    kept = [x for x in items if x.get("kept") and "duplicate" not in x["reason"]]
    parts = []
    if out_:
        parts.append("Left out: " + "; ".join(
            f"{x['records']} record{'s' if x['records'] != 1 else ''} — {x['reason']}"
            + (f", line{'s' if len(x.get('lines') or []) > 1 else ''} {', '.join(map(str, x['lines'][:10]))}"
               if x.get("lines") else "") for x in out_))
    if kept:
        parts.append("Kept and flagged: " + "; ".join(
            f"{x['records']} record{'s' if x['records'] != 1 else ''} — {x['reason']}" for x in kept))
    return ("_" + ". ".join(parts) + "._") if parts else None
