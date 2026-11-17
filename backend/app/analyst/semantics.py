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
        if integral and not countish and _code_named(c):
            # customer_id, zip_code, order_no: a label written in digits. It repeats (many orders per customer), so it
            # is not unique per row, but its values are still names -- never quantities to add or average
            return {"column": c, "role": "identifier", "confidence": 0.8,
                    "evidence": ["named like an identifier or code; whole numbers"], "repeats": bool(nu < 0.95 * n)}
        if integral and nu > 0.95 * n and not countish and (t & {"id", "code", "no", "number", "ref", "key"}
                                                            or (x.is_monotonic_increasing and not t & _AMOUNT_WORDS)):
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


_CODE_TOKENS = {"id", "ids", "code", "codes", "zip", "zipcode", "postcode", "postal", "pincode", "pin", "sku", "ref",
                "key", "uuid", "guid", "phone", "mobile", "ssn", "isbn", "barcode"}
_NUMBER_TOKENS = {"no", "number", "num", "nbr"}


def _code_named(c) -> bool:
    """customer_id, zip_code, sku -- or order_no / invoice_number (a number-word that ends a longer name; "number_of_
    orders" and a bare "number" are quantities)."""
    parts = [x for x in re.split(r"[^a-z0-9]+", _words(str(c)).lower()) if x]
    if set(parts) & _CODE_TOKENS:
        return True
    return len(parts) >= 2 and parts[-1] in _NUMBER_TOKENS


# Per-record levels: adding them up over records gives no quantity ("total unit price", "total rating"). Money and
# durations (salary, fee, hours) do add up, so they are not here.
_LEVEL_TOKENS = {"rate", "rates", "pct", "percent", "percentage", "ratio", "share", "margin", "score", "scores",
                 "rating", "ratings", "price", "avg", "average", "mean", "median", "age", "temperature", "temp", "csat",
                 "nps", "satisfaction", "grade", "rank", "ranking", "latitude", "longitude", "lat", "lon", "lng",
                 "index", "probability", "likelihood", "proportion"}
_FN_WORD = {"sum": "total", "mean": "average", "avg": "average", "median": "median", "min": "minimum", "max": "maximum",
            "std": "standard deviation", "var": "variance"}


def additive(df: pd.DataFrame, c, role: dict | None = None) -> bool:
    """Can the column's values be added up over records into a meaningful total?"""
    role = role or column_role(df, c)
    if role["role"] == "flag":
        return True                                     # a 0/1 flag adds up to a count of records
    if role["role"] == "year" or (role["role"] == "identifier" and _code_named(c)):
        return False
    # judged from the name: values alone (numbers between 0 and 1, a column unique per row) do not make a total
    # meaningless
    return not (_tokens(c) & _LEVEL_TOKENS)


def aggregation_problem(df: pd.DataFrame, c, fn: str, role: dict | None = None) -> str | None:
    """Why `fn` of column `c` is not a quantity -- or None when it is. One rule for every tool, program step and
    calculation: an identifier or code is counted, counted distinctly, filtered on or shown, never added or averaged;
    a per-record level (a price, a rate, a rating, a percentage) is averaged, never summed."""
    if c is None or c not in df.columns or fn in ("count", "count_distinct", "nunique"):
        return None
    role = role or column_role(df, c)
    word = _FN_WORD.get(fn, fn)
    # refused only on evidence from the name AND the values (customer_id, zip_code: whole numbers named as a code);
    # a column that merely happens to be unique per row in a small table is not refused
    if role["role"] == "identifier" and kind_of(df[c]) == "numeric" and _code_named(c):
        return (f"`{c}` identifies records (an identifier or code written in digits), so its {word} is not a quantity. "
                f"I can count its distinct values ({int(df[c].nunique()):,}), count records, or filter and group by it.")
    if fn == "sum" and role["role"] != "flag" and kind_of(df[c]) == "numeric" and not additive(df, c, role):
        return (f"`{c}` is a per-record level (a rate, price, score or percentage), so adding it up over records gives "
                f"no meaningful total. I can give its average, median, minimum or maximum instead"
                + (" — or a total of an amount built from it (for example price × quantity)."
                   if _tokens(c) & {"price", "rate"} else "."))
    return None


def measure_options(df: pd.DataFrame, limit: int = 8) -> list:
    """Columns to offer when Silt asks "which value?": numbers that measure something -- never an identifier or code,
    a year, a yes/no flag, a constant or a clock time. (Identifiers stay available for counting and grouping.)"""
    from app.analyst.ml import _is_identifier  # noqa: PLC0415
    out = []
    for c in df.columns:
        if kind_of(df[c]) != "numeric":
            continue
        r = column_role(df, c)
        if r["role"] in ("identifier", "year", "flag", "constant", "time_of_day") or _is_identifier(str(c), df[c]):
            continue
        out.append(c)
    return out[:limit]


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


# Everyday words for the money a business takes in. They name a column only when the data has exactly one money
# total that can carry that meaning -- a column named after one of these words, or else a generically named money
# total (amount, value, total, gross ...) whose name says nothing different. A column named for another kind of money
# (cost, price, discount, salary, tax ...) is never read as revenue.
MONEY_IN_WORDS = {"revenue", "revenues", "sales", "turnover", "takings", "income", "gmv", "money", "earnings"}
_MONEY_IN_TOKENS = {"revenue", "revenues", "sales", "sale", "turnover", "takings", "income", "gmv"}
_MONEY_GENERIC = {"amount", "amt", "value", "total", "gross", "net", "subtotal", "sum", "paid", "payment", "billed",
                  "invoiced", "bill", "spend", "spent", "price"}
_MONEY_OTHER = {"cost", "costs", "expense", "expenses", "discount", "discounts", "tax", "taxes", "fee", "fees", "salary",
                "salaries", "wage", "wages", "pay", "budget", "refund", "refunds", "profit", "margin", "commission",
                "shipping", "freight", "target", "quota", "forecast", "units", "unit", "qty", "quantity", "count",
                "rate", "pct", "percent", "tip", "tips", "deposit", "balance", "credit", "debt", "loan", "rent"}


def money_in_columns(df: pd.DataFrame) -> tuple[list, str]:
    """(candidate columns for "revenue / sales / money", how they were chosen). One candidate: the reading is used
    and stated. Several: Silt asks. None: the data has no such measure."""
    roles = {c: column_role(df, c) for c in df.columns if kind_of(df[c]) == "numeric"}
    usable = [c for c, r in roles.items() if r["role"] in ("amount", "measurement", "count")
              and not (_tokens(c) & _MONEY_OTHER) and not (_tokens(c) & {"id", "code", "no", "zip"})]
    named = [c for c in usable if _tokens(c) & _MONEY_IN_TOKENS]
    if named:
        return named, "named after it"
    generic = [c for c in usable if _tokens(c) & _MONEY_GENERIC and roles[c]["role"] in ("amount", "measurement")]
    return generic, "the only money total in this data whose name says nothing else"


def _is_value_word(df: pd.DataFrame, w: str) -> bool:
    """`w` is a value of a text column ("Sales" the department): a selection of records, not a measure."""
    for c in df.columns:
        if kind_of(df[c]) == "categorical" and df[c].nunique() <= 5000:
            if df[c].dropna().astype(str).str.strip().str.lower().isin({w, w.rstrip("s")}).any():
                return True
    return False


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
    _money_words(message, df, named, out)
    _flag_words(message, df, named, out)
    return out


def _money_words(message: str, df: pd.DataFrame, named: set, out: dict) -> None:
    from app.analyst import metrics
    from app.analyst.data import find_column
    low = message.lower()
    if metrics.recognise(message) is not None:
        return                          # "revenue per order", "profit": a defined metric reads its own inputs
    for w in sorted(MONEY_IN_WORDS, key=len, reverse=True):
        if not re.search(rf"\b{w}\b", low):
            continue
        if find_column(df, w) or any(_forms(w) & _tokens(c) for c in df.columns) or _is_value_word(df, w):
            return                      # a column is named after it, or it is a value ("the Sales department")
        cands, _ = money_in_columns(df)
        if set(cands) & named:
            return                      # the column is named in the question too
        if len(cands) == 1:
            out["rewrite"].append((w, cands[0]))
        elif len(cands) > 1:
            out["ambiguous"].append((w, cands))
        # none: no column is read as it -- the word is left for a defined metric (revenue = units x price) or a
        # clarification; it is never mapped to an unrelated measure
        return
