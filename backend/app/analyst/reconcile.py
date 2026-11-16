"""Does a stored profit column equal revenue minus cost? Computed record by record, with pandas.

When it does not, the records that differ are described from the data: how many, the total difference, and -- when
the difference falls on whole categories of one column (every Cancelled order, every Returned order) -- which, and
which simple rule the stored figure follows there (zero, revenue + cost, ...). That is a description of how the stored
numbers are built, not a claim about why; nothing is corrected or re-stated.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.analyst.data import business_roles, kind_of

ASK = re.compile(r"\bprofit\b[^.?!]{0,60}\b(difference|differences|gap|discrepanc\w*|mismatch\w*|reconcil\w*|"
                 r"(?:does not|doesn't|do not|don't|not) (?:match|equal|add up))\b|\breconcil\w*\b[^.?!]{0,60}\bprofit\b|"
                 r"\b(?:does|do|is|are|whether|if)\b[^.?!]{0,40}\bprofit\b[^.?!]{0,30}\b(?:equals?|match(?:es)?|adds? up)\b",
                 re.I)
_TOL = 0.005


def columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """(revenue, cost, profit) columns, when the data has all three as numbers."""
    if df is None:
        return None
    roles = business_roles(df)["roles"]
    rev, prof = roles.get("revenue"), roles.get("profit")
    cost = roles.get("cost") or next((c for c in df.columns if re.fullmatch(r"(total[ _])?costs?", str(c), re.I)), None) \
        or roles.get("spend")
    if not (rev and cost and prof) or len({rev, cost, prof}) < 3:
        return None
    if not all(kind_of(df[c]) == "numeric" for c in (rev, cost, prof)):
        return None
    return rev, cost, prof


_RULES = (("is 0", lambda r, c: r * 0), ("equals revenue + cost", lambda r, c: r + c),
          ("equals revenue", lambda r, c: r), ("equals − cost", lambda r, c: -c),
          ("equals cost − revenue", lambda r, c: c - r))


def check(df: pd.DataFrame) -> dict | None:
    cols = columns(df)
    if cols is None:
        return None
    rc, cc, pc = cols
    r, c, p = (pd.to_numeric(df[x], errors="coerce") for x in cols)
    ok = r.notna() & c.notna() & p.notna()
    gap = (p - (r - c)).where(ok)
    off = ok & (gap.abs() > _TOL)
    res = {"revenue": rc, "cost": cc, "profit": pc, "records": int(ok.sum()), "differ": int(off.sum()),
           "total_revenue": float(r[ok].sum()), "total_cost": float(c[ok].sum()), "total_profit": float(p[ok].sum()),
           "revenue_minus_cost": float((r[ok] - c[ok]).sum()), "difference": float(gap[ok].sum()), "by": None,
           "groups": []}
    if not off.any():
        return res
    # a category column whose values are either all-different or all-matching explains where the difference lies
    best = None
    for col in df.columns:
        if col in cols or kind_of(df[col]) != "categorical" or df[col].nunique() > 20:
            continue
        g = off[ok].groupby(df.loc[ok, col].astype(str)).mean()
        if ((g == 0) | (g == 1)).all() and 0 < (g == 1).sum() < len(g):
            if best is None or (g == 1).sum() < best[1]:
                best = (col, int((g == 1).sum()))
    if best:
        col = best[0]
        res["by"] = col
        for val, idx in df.loc[off].groupby(df.loc[off, col].astype(str)).groups.items():
            rr, cc_, pp = r[idx], c[idx], p[idx]
            rule = next((name for name, f in _RULES if bool(np.isclose(pp, f(rr, cc_), atol=_TOL).all())), None)
            res["groups"].append({"value": val, "records": int(len(idx)), "difference": float(gap[idx].sum()),
                                  "rule": rule})
        res["groups"].sort(key=lambda x: -abs(x["difference"]))
        res["matching"] = sorted(df.loc[ok & ~off, col].astype(str).unique().tolist())
    return res


def _n(x: float) -> str:
    return f"{x:,.2f}"


def text(res: dict) -> str:
    rv, cv, pv = res["revenue"], res["cost"], res["profit"]
    head = (f"**{rv} − {cv} vs the stored {pv}:** {rv} − {cv} = **{_n(res['revenue_minus_cost'])}**; the stored {pv} "
            f"totals **{_n(res['total_profit'])}**")
    if not res["differ"]:
        return head + f" — they agree in all {res['records']:,} records."
    out = [head + f" — a difference of **{_n(res['difference'])}**. The stored {pv} equals {rv} − {cv} in "
           f"{res['records'] - res['differ']:,} of {res['records']:,} records; {res['differ']:,} differ."]
    if res["by"]:
        out.append("")
        out.append(f"The records that differ are exactly the {res['by']} = "
                   + " and ".join(f"**{g['value']}**" for g in res["groups"]) + " records"
                   + (f" ({', '.join(res['matching'])} all agree)" if res.get("matching") else "") + ":")
        out.append("")
        out.append(f"| {res['by']} | records | difference | stored {pv} there |")
        out.append("|---|---:|---:|---|")
        for g in res["groups"]:
            out.append(f"| {g['value']} | {g['records']:,} | {_n(g['difference'])} | "
                       f"{(pv + ' ' + g['rule'].replace('revenue', rv).replace('cost', cv)) if g['rule'] else 'no single rule'} |")
    out.append("")
    out.append(f"_This describes how the stored {pv} figures are built in this file; the data does not say why. Silt "
               f"reports both totals and does not replace the stored {pv}._")
    return "\n".join(out)
