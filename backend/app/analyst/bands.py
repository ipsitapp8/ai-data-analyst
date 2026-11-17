"""Counts of records in value bands the question states ("rated below 3, between 3 and 4, and 4 or above").

Two or more bands of one numeric column are needed; a single threshold is answered by the calculator. Where two
bands meet ("between 3 and 4" and "4 or above") the shared edge belongs to the higher band, and the answer says so.
Records with no value are counted separately; records in no band are counted too -- nothing is dropped silently.
"""
from __future__ import annotations

import math
import re

import pandas as pd

from app.analyst.data import find_column, kind_of

_N = r"(-?\d+(?:\.\d+)?)"
_PATTERNS = [
    ("lt", re.compile(rf"\b(?:below|under|less than|lower than|beneath)\s+{_N}", re.I)),
    ("ge", re.compile(rf"\b{_N}\s+(?:or|and)\s+(?:above|over|more|higher|up|greater)\b|\b(?:at least|no less than)\s+{_N}", re.I)),
    ("gt", re.compile(rf"\b(?:above|over|more than|greater than|higher than)\s+{_N}", re.I)),
    ("le", re.compile(rf"\b{_N}\s+or\s+(?:below|under|less|lower)\b|\b(?:at most|no more than|up to)\s+{_N}", re.I)),
    ("between", re.compile(rf"\bbetween\s+{_N}\s+and\s+{_N}|\b{_N}\s*(?:-|–|to)\s*{_N}\b", re.I)),
]


def parse(message: str, df: pd.DataFrame) -> dict | None:
    if df is None:
        return None
    nums = [c for c in find_column(df, message) if kind_of(df[c]) == "numeric"]
    if not nums:
        from app.analyst.followup import _stem_column
        nums = [c for c in (_stem_column(w, df) for w in re.findall(r"[a-z]+(?:ed|d)\b", message.lower())) if c]
    if len(nums) != 1:
        return None
    found = []
    taken = []
    for kind, rx in _PATTERNS:
        for m in rx.finditer(message):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            vals = [float(v) for v in m.groups() if v is not None]
            taken.append((m.start(), m.end()))
            if kind == "between":
                lo, hi = sorted(vals[:2])
                found.append({"lo": lo, "hi": hi, "lo_in": True, "hi_in": True, "at": m.start()})
            elif kind == "lt":
                found.append({"lo": -math.inf, "hi": vals[0], "lo_in": False, "hi_in": False, "at": m.start()})
            elif kind == "le":
                found.append({"lo": -math.inf, "hi": vals[0], "lo_in": False, "hi_in": True, "at": m.start()})
            elif kind == "gt":
                found.append({"lo": vals[0], "hi": math.inf, "lo_in": False, "hi_in": False, "at": m.start()})
            else:
                found.append({"lo": vals[0], "hi": math.inf, "lo_in": True, "hi_in": False, "at": m.start()})
    if len(found) < 2:
        return None
    found.sort(key=lambda b: (b["lo"], b["hi"]))
    for a, b in zip(found, found[1:]):            # a shared edge belongs to the higher band
        if a["hi"] == b["lo"] and a["hi_in"] and b["lo_in"]:
            a["hi_in"] = False
    noun = re.search(r"\bhow many\s+([a-z]+)|\bnumber of\s+([a-z]+)", message, re.I)
    from app.analyst.schema_profile import entity_column
    ent = entity_column(df, next(g for g in noun.groups() if g)) if noun else None
    return {"column": nums[0], "bands": found, "entity": ent}


def _label(b: dict) -> str:
    def f(x):
        return f"{x:g}"
    if b["lo"] == -math.inf:
        return f"{'≤' if b['hi_in'] else '<'} {f(b['hi'])}"
    if b["hi"] == math.inf:
        return f"{'≥' if b['lo_in'] else '>'} {f(b['lo'])}"
    return f"{f(b['lo'])} {'≤' if b['lo_in'] else '<'} x {'≤' if b['hi_in'] else '<'} {f(b['hi'])}"


def mask(x: pd.Series, b: dict) -> pd.Series:
    lo = (x >= b["lo"]) if b["lo_in"] else (x > b["lo"])
    hi = (x <= b["hi"]) if b["hi_in"] else (x < b["hi"])
    return lo & hi & x.notna()


def run(spec: dict, df: pd.DataFrame) -> dict:
    x = pd.to_numeric(df[spec["column"]], errors="coerce")
    rows, covered = [], pd.Series(False, index=df.index)
    for b in spec["bands"]:
        m = mask(x, b)
        covered |= m
        row = {"band": _label(b), "records": int(m.sum()), "share": float(m.sum() / x.notna().sum()) if x.notna().any() else None}
        if spec.get("entity"):
            row["distinct"] = int(df.loc[m, spec["entity"]].nunique())
        rows.append(row)
    return {"column": spec["column"], "entity": spec.get("entity"), "rows": rows, "records": int(len(df)),
            "missing": int(x.isna().sum()), "in_no_band": int((x.notna() & ~covered).sum()), "bands": spec["bands"]}


def text(r: dict) -> str:
    col, ent = r["column"].replace("_", " "), r.get("entity")
    lines = [f"**Records by {col} band** ({r['records']:,} records):", "",
             f"| {col} | records | share" + (f" | distinct {ent}" if ent else "") + " |",
             "|---|---:|---:|" + ("---:|" if ent else "")]
    for x in r["rows"]:
        lines.append(f"| {x['band']} | {x['records']:,} | {x['share'] * 100:.1f}% |"
                     + (f" {x['distinct']:,} |" if ent else ""))
    notes = ["Where two bands meet, the shared value belongs to the higher band."]
    if r["missing"]:
        notes.append(f"{r['missing']:,} record(s) have no {col} and are in no band.")
    if r["in_no_band"]:
        notes.append(f"{r['in_no_band']:,} record(s) fall outside every band asked for.")
    if ent:
        notes.append(f"Records are rows of the file; distinct {ent} counts a name listed more than once a single time.")
    return "\n".join(lines) + "\n\n_" + " ".join(notes) + "_"
