"""Understanding a dataset before analysing it: what it holds, how clean it is,
what stands out, how it moves over time, and what is worth asking next.

Everything here is computed from the data. Suggestions name real columns and
say which analysis would answer them; they are not findings.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from app.analyst import ml
from app.analyst.data import business_roles, finite, kind_of

MISSING_WARN_PCT = 5.0
IQR_K = 1.5
MAX_TREND_POINTS = 60


def _id_like(name: str, s: pd.Series) -> bool:
    return ml._is_identifier(name, s)  # noqa: SLF001 - one definition of "identifier" for the whole analyst


def column_summary(df: pd.DataFrame, c: str) -> dict:
    s = df[c]
    kind = kind_of(s)
    out = {"name": c, "kind": kind, "missing": int(s.isna().sum()),
           "missing_pct": float(s.isna().mean() * 100) if len(s) else 0.0, "unique": int(s.nunique(dropna=True))}
    if kind == "numeric":
        x = finite(s)
        if len(x):
            out.update(mean=float(x.mean()), median=float(np.median(x)), min=float(x.min()), max=float(x.max()),
                       sd=float(x.std(ddof=1)) if len(x) > 1 else None)
    elif kind == "datetime":
        d = s.dropna()
        if len(d):
            out.update(first=str(d.min().date()), last=str(d.max().date()))
    else:
        top = s.dropna().astype(str).value_counts()
        out["top_values"] = [{"value": str(k), "rows": int(v)} for k, v in top.head(5).items()]
        from app.analyst.calc import _looks_like_time, parse_time  # noqa: PLC0415 - one definition of a clock time
        if len(s.dropna()) and _looks_like_time(s.dropna()):
            hours = [h for h in (parse_time(v) for v in s.dropna()) if h is not None]
            out["time_of_day"] = True
            if hours:
                out.update(earliest=_hhmm(min(hours)), latest=_hhmm(max(hours)))
    out["identifier"] = _id_like(c, s)
    return out


def _hhmm(h: float) -> str:
    return f"{int(h) % 24:02d}:{int(round((h - int(h)) * 60)):02d}"


def _time_cols(df: pd.DataFrame) -> list[str]:
    from app.analyst.calc import _looks_like_time  # noqa: PLC0415
    return [c for c in df.columns if kind_of(df[c]) == "categorical" and len(df[c].dropna())
            and _looks_like_time(df[c].dropna())]


def _target_candidates(df: pd.DataFrame) -> list[dict]:
    out = []
    times = set(_time_cols(df))
    for c in df.columns:
        s, k = df[c], kind_of(df[c])
        if _id_like(c, s) or k == "datetime" or c in times or re.search(r"(^|_)(id|name|code)$", str(c).lower()):
            continue
        u = s.nunique(dropna=True)
        if k == "categorical" and 2 <= u <= 10:
            out.append({"column": c, "task": "classification", "classes": int(u)})
        elif k == "numeric" and u > 10:
            out.append({"column": c, "task": "regression"})
    # Names that usually denote an outcome first.
    out.sort(key=lambda t: (not re.search(r"churn|attrition|left|default|fraud|converted|outcome|target|label",
                                           t["column"].lower()),
                            not re.search(r"sales|revenue|profit|salary|income|score", t["column"].lower()),
                            not re.search(r"price", t["column"].lower())))
    return out[:6]


def quality_issues(df: pd.DataFrame, cols: list[dict]) -> list[str]:
    issues = []
    dup = int(df.duplicated().sum())
    if dup:
        issues.append(f"{dup} duplicated rows ({dup / len(df) * 100:.1f}% of the data).")
    for c in cols:
        if c["missing_pct"] >= MISSING_WARN_PCT:
            issues.append(f"'{c['name']}' is missing in {c['missing']} rows ({c['missing_pct']:.1f}%).")
        if c["unique"] == 1 and len(df) > 1:
            issues.append(f"'{c['name']}' has a single value, so it cannot explain any differences.")
        if c["identifier"]:
            issues.append(f"'{c['name']}' looks like an identifier (unique per row): useful for lookups, not for analysis.")
    for c in df.columns:
        s = df[c]
        if kind_of(s) == "categorical" and s.dtype == object:
            as_num = pd.to_numeric(s.dropna().astype(str).str.replace(",", "", regex=False), errors="coerce")
            if len(as_num) and 0.5 < as_num.notna().mean() < 1.0:
                bad = int(as_num.isna().sum())
                issues.append(f"'{c}' is mostly numbers but {bad} values are not, so it is treated as text.")
    for c in df.columns:
        if kind_of(df[c]) == "numeric" and re.search(r"rate|price|cost|amount|salary|pay|wage|hours|qty|quantity",
                                                       str(c).lower()):
            neg = pd.to_numeric(df[c], errors="coerce") < 0
            if neg.any():
                issues.append(f"'{c}' is negative in {int(neg.sum())} row{'s' if neg.sum() != 1 else ''} (line "
                              + ", ".join(str(i + 2) for i in np.flatnonzero(neg.to_numpy())[:10])
                              + "), which is probably a data error.")
    times = _time_cols(df)
    if len(times) >= 2:
        from app.analyst.calc import _to_hours  # noqa: PLC0415
        a, b = _to_hours(df[times[0]]), _to_hours(df[times[1]])
        back = (b < a).fillna(False)
        if back.any():
            issues.append(f"'{times[1]}' is earlier than '{times[0]}' in {int(back.sum())} row"
                          f"{'s' if back.sum() != 1 else ''} (line " + ", ".join(str(i + 2) for i in np.flatnonzero(back.to_numpy())[:10])
                          + "): an overnight shift or an entry error.")
    out = outliers(df)
    for o in out["columns"][:3]:
        issues.append(f"'{o['column']}' has {o['count']} unusual values outside {o['low']:,.2f} to {o['high']:,.2f} "
                      "(1.5 × IQR rule).")
    return issues


def suggestions(df: pd.DataFrame, targets: list[dict]) -> list[str]:
    roles = business_roles(df)["roles"]
    num = [c for c in df.columns if kind_of(df[c]) == "numeric" and not _id_like(c, df[c])]
    cat = [c for c in df.columns if kind_of(df[c]) == "categorical" and 2 <= df[c].nunique() <= 30
           and not _id_like(c, df[c])]
    group_role = roles.get("area") or roles.get("campaign")
    cat.sort(key=lambda c: (c != group_role, df[c].nunique()))    # the business group first, then fewest levels
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    out = []
    main = (roles.get("revenue") if roles.get("revenue") in num else None) or (targets[0]["column"] if targets
                                                                                and targets[0]["task"] == "regression" else None) \
        or (num[0] if num else None)
    if main and cat:
        out.append(f"Which {cat[0]} has the highest {'average ' if re.search(r'rate|price|percent|score', main.lower()) else ''}{main}?")
    if main and dates:
        out.append(f"Show the trend of {main} over time.")
    if len(num) >= 2:
        other = next((c for c in num if c != main), None)
        if main and other:
            out.append(f"Is {other} correlated with {main}?")
    if "spend" in roles and roles.get("area"):
        out.append("Which area should receive our next advertising campaign?")
        out.append("What if we increase the budget by 20%?")
    times = _time_cols(df)
    if len(times) >= 2:
        start, end = times[0], times[1]
        rate = next((c for c in num if re.search(r"rate|wage|pay|salary", c.lower())), None)
        by = f" by {cat[0]}" if cat else ""
        if rate:
            out.insert(0, f"What is the total labor cost (hours from {start} to {end} × {rate}){by}?")
        out.insert(1 if rate else 0, f"How many hours were worked (from {start} to {end}){by}?")
    if targets:
        out.append(f"Predict {targets[0]['column']}. ({targets[0]['task']} with a held-out test)")
    out.append("Find unusual values (outliers).")
    return out[:6]


def understand(df: pd.DataFrame, filename: str | None = None) -> dict:
    cols = [column_summary(df, c) for c in df.columns]
    targets = _target_candidates(df)
    roles = business_roles(df)
    return {
        "filename": filename, "rows": int(len(df)), "columns": len(df.columns),
        "kinds": {k: sum(c["kind"] == k for c in cols) for k in ("numeric", "categorical", "datetime")},
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "column_details": cols, "quality_issues": quality_issues(df, cols),
        "business_roles": roles["roles"], "possible_targets": targets, "suggested_questions": suggestions(df, targets),
    }


def missing_values(df: pd.DataFrame) -> dict:
    per = [{"column": c, "missing": int(df[c].isna().sum()), "missing_pct": float(df[c].isna().mean() * 100)}
           for c in df.columns]
    per.sort(key=lambda r: -r["missing"])
    rows_any = int(df.isna().any(axis=1).sum())
    return {"rows": int(len(df)), "missing_cells": int(df.isna().sum().sum()), "rows_with_missing": rows_any,
            "rows_with_missing_pct": float(rows_any / len(df) * 100) if len(df) else 0.0,
            "duplicates": int(df.duplicated().sum()), "columns": per}


def outliers(df: pd.DataFrame, column: str | None = None) -> dict:
    """Values outside [Q1 - 1.5 IQR, Q3 + 1.5 IQR] per numeric column."""
    cols = [column] if column else [c for c in df.columns if kind_of(df[c]) == "numeric" and not _id_like(c, df[c])]
    found = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        x = s.dropna()
        if len(x) < 8:
            continue
        q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
        mask = (s < lo) | (s > hi)
        n = int(mask.sum())
        if n:
            ex = s[mask].sort_values(key=lambda v: (v - x.median()).abs(), ascending=False).head(5)
            found.append({"column": c, "count": n, "pct": float(n / len(x) * 100), "low": lo, "high": hi,
                          "median": float(x.median()),
                          "examples": [{"row": int(i) + 1, "value": float(v)} for i, v in ex.items()]})
    found.sort(key=lambda o: -o["pct"])
    return {"method": "1.5 × interquartile range (IQR) rule", "checked": cols, "columns": found}


def trend(df: pd.DataFrame, value: str, date: str | None = None, agg: str = "sum", freq: str = "M") -> dict:
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    date = date or (dates[0] if dates else None)
    if date is None:
        raise ml.Unsuitable("This dataset has no date column, so a trend over time cannot be computed.")
    if kind_of(df[value]) != "numeric":
        raise ml.Unsuitable(f"'{value}' is not numeric, so its trend cannot be summed or averaged.")
    d = df[[date, value]].dropna()
    period = d[date].dt.to_period(freq)
    g = d.groupby(period)[value].agg("mean" if agg == "avg" else "sum")
    points = [{"period": str(k), "value": float(v)} for k, v in g.items()][-MAX_TREND_POINTS:]
    if len(points) < 2:
        raise ml.Unsuitable("Fewer than two time periods have data, so there is no trend to show.")
    first, last = points[0]["value"], points[-1]["value"]
    peak = max(points, key=lambda p: p["value"])
    low = min(points, key=lambda p: p["value"])
    x = np.arange(len(points), dtype=float)
    y = np.array([p["value"] for p in points])
    slope = float(np.polyfit(x, y, 1)[0])
    return {"value": value, "date": date, "aggregation": agg, "frequency": {"M": "month", "Q": "quarter", "W": "week", "D": "day"}[freq],
            "points": points, "first": points[0], "last": points[-1], "change": last - first,
            "change_pct": float((last - first) / abs(first) * 100) if first else None,
            "peak": peak, "low": low, "slope_per_period": slope,
            "note": "A trend describes what happened; it does not by itself predict what will happen next."}


def patterns(df: pd.DataFrame) -> dict:
    """Strongest pairwise correlations and the most unusual columns: leads, not findings."""
    num = [c for c in df.columns if kind_of(df[c]) == "numeric" and not _id_like(c, df[c])]
    pairs = []
    if len(num) >= 2:
        corr = df[num].apply(pd.to_numeric, errors="coerce").corr()
        for i, a in enumerate(num):
            for b in num[i + 1:]:
                r = corr.loc[a, b]
                if pd.notna(r):
                    pairs.append({"x": a, "y": b, "r": float(r)})
        pairs.sort(key=lambda p: -abs(p["r"]))
    return {"correlations": pairs[:5], "outliers": outliers(df)["columns"][:5],
            "note": "Correlation shows association, not cause; outliers may be errors or genuine extremes — check them."}


def drivers(df: pd.DataFrame, group_col: str, group: str, outcome: str | None = None) -> dict:
    """How one group differs from all others on every other numeric column: means, relative
    difference, Welch's t-test. Differences that go with better results are leads to check,
    not causes."""
    from app.stats import methods as sm
    if group_col not in df.columns:
        raise ml.Unsuitable(f"There is no column named '{group_col}'.")
    mask = df[group_col].astype(str) == str(group)
    if not mask.any():
        raise ml.Unsuitable(f"'{group}' does not occur in '{group_col}'.")
    rows = []
    for c in df.columns:
        if c in (group_col, outcome) or kind_of(df[c]) != "numeric" or _id_like(c, df[c]):
            continue
        a, b = finite(df.loc[mask, c]), finite(df.loc[~mask, c])
        if len(a) < 3 or len(b) < 3:
            continue
        w = sm.welch(b, a, 0.95)
        rel = (a.mean() - b.mean()) / abs(b.mean()) * 100 if b.mean() else None
        rows.append({"column": c, "group_mean": float(a.mean()), "others_mean": float(b.mean()),
                     "difference": float(a.mean() - b.mean()), "difference_pct": float(rel) if rel is not None else None,
                     "p_value": w["test"]["p_value"]})
    rows.sort(key=lambda r: (r["p_value"] >= 0.05, -abs(r["difference_pct"] or 0)))
    out_diff = None
    if outcome and outcome in df.columns and kind_of(df[outcome]) == "numeric":
        a, b = finite(df.loc[mask, outcome]), finite(df.loc[~mask, outcome])
        out_diff = {"column": outcome, "group_mean": float(a.mean()), "others_mean": float(b.mean())}
    cat = []
    for c in df.columns:
        if c in (group_col, outcome) or kind_of(df[c]) != "categorical" or _id_like(c, df[c]) or df[c].nunique() > 12:
            continue
        sa = df.loc[mask, c].astype(str).value_counts(normalize=True)
        sb = df.loc[~mask, c].astype(str).value_counts(normalize=True)
        gap = (sa.reindex(sa.index.union(sb.index), fill_value=0) - sb.reindex(sa.index.union(sb.index), fill_value=0))
        if len(gap) and gap.abs().max() >= 0.1:
            k = gap.abs().idxmax()
            cat.append({"column": c, "value": str(k), "group_share": float(sa.get(k, 0)), "others_share": float(sb.get(k, 0))})
    return {"group_column": group_col, "group": str(group), "outcome": out_diff, "rows_group": int(mask.sum()),
            "rows_others": int((~mask).sum()), "numeric": rows, "categorical": cat,
            "note": ("These are differences that go together with the result, found in observational data. They are "
                     "candidate explanations to check (for example with a controlled test), not proven causes.")}
