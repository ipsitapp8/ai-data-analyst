"""What a question asks for, item by item, and whether an answer contains each item.

A question such as "the mean, median, mode, standard deviation, minimum and maximum of the rating" asks for six
figures; an answer that shows four of them has not answered it. Every answer is checked against this list before it
is shown (app/analyst/orchestrator._finish): an item the answer does not contain is named under "Not answered here",
never silently dropped, and the answer is not presented as complete.

Only explicit statistical and profile words are items. Ranking words ("highest", "top") and measures are checked by
the planner and the verifier, not here.
"""
from __future__ import annotations

import re

# (item, how the question asks for it, how an answer shows it)
ITEMS = [
    ("number of records", r"\bhow many (?:rows|records|entries)\b|\bnumber of (?:rows|records|entries)\b|\brow count\b|"
                          r"\brecord count\b", r"\b\d[\d,]*\s*(?:rows|records)\b"),
    ("columns", r"\bwhat columns\b|\bwhich columns\b|\blist (?:the |all )?columns\b|\bcolumn names\b|\bnumber of columns\b|"
                r"\bhow many columns\b|\bcolumns does it have\b|\bcolumns (?:are there|it has)\b",
     r"\bcolumns?\b"),
    ("column types", r"\b(?:data )?types\b|\bdtypes?\b|\bkind of (?:columns|data)\b", r"\b(?:numeric|categorical|type)\b"),
    ("missing values", r"\bmissing\b|\bnull values?\b|\bblank values?\b|\bempty values?\b|\bnulls\b",
     r"\bmissing\b|\bno missing\b"),
    ("duplicates", r"\bduplicat\w*", r"\bduplicat\w*"),
    ("mean", r"\bmean\b|\baverage\b|\bavg\b", r"\bmean\b|\baverage\b"),
    ("median", r"\bmedian\b", r"\bmedian\b"),
    ("mode", r"\bmode\b|\bmost common (?:value|rating|score)\b", r"\bmode\b|\bno mode\b"),
    ("standard deviation", r"\bstandard deviation\b|\bstd\.?\b|\bstdev\b|\bsd\b", r"\bstandard deviation\b|\bstd\b"),
    ("variance", r"\bvariance\b", r"\bvariance\b"),
    ("minimum", r"\bminimum\b|\bmin\b", r"\bminimum\b|\bmin\b|\brange\b|\blowest\b"),
    ("maximum", r"\bmaximum\b|\bmax\b", r"\bmaximum\b|\bmax\b|\brange\b|\bhighest\b"),
    ("Pearson correlation", r"\bpearson\b", r"\bpearson\b"),
    ("Spearman correlation", r"\bspearman\b", r"\bspearman\b"),
    ("p-values", r"\bp[- ]?values?\b|\bsignifican\w*", r"\bp\s*[=<]|\bp-value"),
    ("sample size", r"\bsample size\b|\bhow many (?:pairs|observations)\b", r"\bn\s*=\s*\d"),
    ("z-scores", r"\bz[- ]?scores?\b", r"\bz[- ]?score|\babs\(z\)"),
    ("IQR rule", r"\biqr\b|\binterquartile\b", r"\biqr\b|\binterquartile\b"),
]
_Q = [(name, re.compile(q, re.I), re.compile(a, re.I)) for name, q, a in ITEMS]

_PROFILE = {"number of records", "columns", "column types", "missing values", "duplicates"}


def asked(message: str) -> list[str]:
    """The items the question names, in a fixed order."""
    return [name for name, q, _ in _Q if q.search(message)]


def missing_items(message: str, answer: str) -> list[str]:
    """Items the question asks for that the answer text does not contain."""
    out = []
    for name, q, a in _Q:
        if q.search(message) and not a.search(answer or ""):
            out.append(name)
    return out


def profile_request(message: str, df) -> bool:
    """The question asks for a description of the table itself -- at least two of: its records, columns, column
    types, missing values, duplicates -- and names no column to compute on."""
    items = set(asked(message)) & _PROFILE
    if len(items) < 2:
        return False
    from app.analyst.data import find_column
    return not find_column(df, message)


def note(missing: list[str]) -> str:
    return ("**Not answered here:** " + ", ".join(missing) + ". This answer does not include "
            + ("it" if len(missing) == 1 else "them") + "; ask for " + ("it" if len(missing) == 1 else "them")
            + " on its own so Silt can compute " + ("it" if len(missing) == 1 else "each one") + ".")
