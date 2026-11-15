"""Analytical programs: typed, composable, validated steps executed deterministically.

A program is a list of steps. Each step has an id, an operation, its inputs (the dataset "data" or earlier step
ids) and arguments. Every step is validated before it runs -- columns exist, types fit, entity values exist, dates
parse, units agree -- and an invalid step rejects the whole program with the reason (ProgramError). Every result
carries its provenance: the operation, inputs, formula, records used and excluded (file lines), and unit.

Operations
  select     rows of a table: entities (only / except), a period, conditions; invalid records removed and listed
  derive     a per-row column from a calculator formula (app.analyst.calc's whitelisted expression language)
  aggregate  sum / mean / median / min / max / count / nunique of a column; optionally per group or per period
  project    a rate observed over a history window, carried over a horizon: expected total and a range
  compare    two results, or a result against a target: difference, ratio, and for targets a feasibility verdict
  scenario   a hypothetical change: remove an entity (for a number of days, or entirely), or scale a column
  ratio      one result divided by another: two values, or two per-group results group by group (hours per person)

Result kinds: table, scalar, series (per group / per period), projection, comparison. A scenario result is always
marked hypothetical; a projection always states its assumption. Nothing here knows a dataset or a question.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.analyst import calc
from app.analyst.data import kind_of

AGG = {"sum", "mean", "median", "min", "max", "count", "nunique"}
OPS = {"select", "derive", "aggregate", "project", "compare", "scenario", "rank", "anomalies", "associate", "profile",
       "ratio"}
CMP = {"==", "!=", ">", ">=", "<", "<="}


class ProgramError(ValueError):
    """The program cannot be run as written; the message says which step and why."""


@dataclass
class Result:
    kind: str                                   # table | scalar | series | projection | comparison
    value: Any                                  # DataFrame | float | dict | dict
    unit: str | None = None
    provenance: dict = field(default_factory=dict)
    hypothetical: bool = False

    def summary(self) -> dict:
        v = self.value
        if self.kind == "table":
            v = {"rows": int(len(v)), "columns": [str(c) for c in v.columns][:20]}
        return {"kind": self.kind, "value": _jsonable(v), "unit": self.unit, "hypothetical": self.hypothetical,
                "provenance": _jsonable(self.provenance)}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if (isinstance(x, float) or isinstance(x, np.floating)) and not np.isfinite(x) else float(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x.date())
    return x


def _date_col(df: pd.DataFrame) -> str | None:
    dates = [c for c in df.columns if kind_of(df[c]) == "datetime"]
    return dates[0] if len(dates) == 1 else None


# ------------------------------------------------------------------- steps --
def _table(results, ref, data, step) -> pd.DataFrame:
    src = data if ref == "data" else results.get(ref)
    if src is None:
        raise ProgramError(f"step '{step}': input '{ref}' does not exist (it must be 'data' or an earlier step)")
    if isinstance(src, Result):
        if src.kind != "table":
            raise ProgramError(f"step '{step}': input '{ref}' is a {src.kind}, a table is needed")
        return src.value
    return src


def _scalar(results, ref, step) -> Result:
    r = results.get(ref)
    if r is None:
        raise ProgramError(f"step '{step}': input '{ref}' does not exist")
    if r.kind not in ("scalar", "projection"):
        raise ProgramError(f"step '{step}': input '{ref}' is a {r.kind}, a single value is needed")
    return r


def _lines(df: pd.DataFrame, idx) -> list[int]:
    pos = {i: n + 2 for n, i in enumerate(df.index)}      # file lines: the header is line 1
    return [pos[i] for i in idx if i in pos][:20]


def op_select(s, results, data):
    df = _table(results, s.get("from", "data"), data, s["id"])
    keep = pd.Series(True, index=df.index)
    for col, vals in (s.get("entities") or {}).items():
        _need_col(df, col, s)
        present = set(df[col].astype(str).str.lower())
        missing = [v for v in vals if v.lower() not in present]
        if missing:
            raise ProgramError(f"step '{s['id']}': {', '.join(missing)} does not occur in '{col}'")
        keep &= df[col].astype(str).str.lower().isin([v.lower() for v in vals])
    for col, vals in (s.get("except") or {}).items():
        _need_col(df, col, s)
        keep &= ~df[col].astype(str).str.lower().isin([v.lower() for v in vals])
    per = s.get("period")
    if per:
        date = per.get("column") or _date_col(df)
        if not date:
            raise ProgramError(f"step '{s['id']}': a period needs exactly one date column")
        d = pd.to_datetime(df[date], errors="coerce")
        if per.get("start"):
            keep &= d >= pd.Timestamp(per["start"])
        if per.get("end"):
            keep &= d <= pd.Timestamp(per["end"])
    for w in s.get("where") or []:
        col, op, val = w["column"], w["op"], w["value"]
        _need_col(df, col, s)
        if op not in CMP:
            raise ProgramError(f"step '{s['id']}': '{op}' is not a comparison ({', '.join(sorted(CMP))})")
        x = pd.to_numeric(df[col], errors="coerce") if isinstance(val, (int, float)) else df[col].astype(str)
        keep &= {"==": x == val, "!=": x != val, ">": x > val, ">=": x >= val, "<": x < val, "<=": x <= val}[op].fillna(False)
    sel = df[keep]
    excluded = []
    if s.get("valid_for") is not None:
        from app.analyst.scope import eligibility
        base = sel.drop(columns=[c for c in sel.columns if c.startswith("__")], errors="ignore")
        keep_dupes = s.get("duplicates") == "keep"
        if keep_dupes:
            # descriptive totals follow the calculator: exact duplicates are kept (and flagged), only impossible
            # records are removed -- a row number makes every row distinct for the validity check
            dup = base.duplicated(keep="first")
            base = base.assign(__row=range(len(base)))
        ok, reasons = eligibility(base, [c for c in s["valid_for"] if c in sel.columns], historical=bool(s.get("historical")))
        excluded = reasons
        sel = sel[ok.values]
        if keep_dupes and dup[ok.values].any():
            excluded = excluded + [{"reason": "exact duplicate of an earlier record (kept, as recorded)",
                                    "records": int(dup[ok.values].sum()), "lines": _lines(df, dup[ok.values][dup[ok.values]].index),
                                    "kept": True}]
    return Result("table", sel, provenance={"op": "select", "rows_in": int(len(df)), "rows_out": int(len(sel)),
                                            "entities": s.get("entities"), "except": s.get("except"),
                                            "period": per, "where": s.get("where"), "excluded": excluded})


def op_derive(s, results, data):
    df = _table(results, s.get("from", "data"), data, s["id"]).copy()
    name, formula = s["name"], s["formula"]
    if not str(name).isidentifier():
        raise ProgramError(f"step '{s['id']}': '{name}' is not a valid column name")
    try:
        v = calc.evaluate(formula, df, df)
    except calc.CalcError as e:
        raise ProgramError(f"step '{s['id']}': {e}") from e
    df[name] = v if isinstance(v, pd.Series) else pd.Series(v, index=df.index)
    unit = s.get("unit") or ("hours" if "hours_between" in formula or "overnight_hours" in formula else None)
    df.attrs = {**df.attrs, "units": {**df.attrs.get("units", {}), name: unit}}
    return Result("table", df, unit=unit, provenance={"op": "derive", "column": name, "formula": formula,
                                                       "rows": int(len(df)), "missing": int(df[name].isna().sum())})


def op_aggregate(s, results, data):
    df = _table(results, s.get("from", "data"), data, s["id"])
    fn, col = s["fn"], s.get("column")
    if fn not in AGG:
        raise ProgramError(f"step '{s['id']}': '{fn}' is not an aggregation ({', '.join(sorted(AGG))})")
    if fn not in ("count",):
        _need_col(df, col, s)
        if pd.api.types.is_bool_dtype(df[col]) or (df[col].dropna().map(lambda x: isinstance(x, (bool,))).all()
                                                  and len(df[col].dropna())):
            df = df.assign(**{col: df[col].astype(float)})        # true/false counts as 1/0 (a share when averaged)
        if fn != "nunique" and kind_of(df[col]) != "numeric":
            raise ProgramError(f"step '{s['id']}': '{col}' is not numeric, so it cannot be {fn}med")
    unit = s.get("unit") or ((df.attrs.get("units") or {}).get(col) if col else "records")
    by, per = s.get("by"), s.get("per")

    def agg(x):
        if fn == "count":
            return float(len(x))
        if fn == "sum" and len(x) == 0:
            return 0.0                  # no records selected: nothing completed yet, which is 0 -- not unknown
        v = x[col] if fn == "nunique" else pd.to_numeric(x[col], errors="coerce").dropna()
        if fn == "nunique":
            return float(v.nunique())
        return float(getattr(v, fn)()) if len(v) else None
    if by or per:
        if by:
            _need_col(df, by, s)
            keys = df[by].astype(str)
        else:
            date = _date_col(df)
            if not date:
                raise ProgramError(f"step '{s['id']}': a per-period result needs exactly one date column")
            keys = pd.to_datetime(df[date]).dt.to_period(per).astype(str)
        out = {str(k): agg(g) for k, g in df.groupby(keys, sort=True)}
        return Result("series", out, unit=unit, provenance={"op": "aggregate", "fn": fn, "column": col, "by": by or per,
                                                             "rows": int(len(df))})
    return Result("scalar", agg(df), unit=unit, provenance={"op": "aggregate", "fn": fn, "column": col,
                                                             "rows": int(len(df))})


def op_project(s, results, data):
    """Daily totals of a column over a history window (days with no record count as 0), carried over a horizon.
    Expected = mean daily total x horizon days; range = +/- 1.96 x sd x sqrt(days), assuming days vary
    independently around the historical mean -- stated with every result."""
    df = _table(results, s.get("from", "data"), data, s["id"])
    col = s["column"]
    _need_col(df, col, s)
    date = s.get("date") or _date_col(df)
    if not date:
        raise ProgramError(f"step '{s['id']}': a projection needs a date column")
    days = s.get("days")
    if days is None or days < 0:
        raise ProgramError(f"step '{s['id']}': the horizon must be a number of days (0 or more)")
    d = pd.to_datetime(df[date], errors="coerce").dt.normalize()
    v = pd.to_numeric(df[col], errors="coerce")
    ok = d.notna() & v.notna()
    if not ok.any():
        raise ProgramError(f"step '{s['id']}': no dated values of '{col}' to learn a rate from")
    start = pd.Timestamp(s["history_start"]) if s.get("history_start") else d[ok].min()
    end = pd.Timestamp(s["history_end"]) if s.get("history_end") else d[ok].max()
    window = pd.date_range(start, end, freq="D")
    daily = v[ok].groupby(d[ok]).sum().reindex(window, fill_value=0.0)
    n = len(daily)
    mean = float(daily.mean())
    sd = float(daily.std(ddof=1)) if n > 1 else float("nan")
    expected = mean * days
    half = 1.96 * sd * math.sqrt(days) if n > 1 and days else float("nan")
    active = int((daily > 0).sum())
    unit = (df.attrs.get("units") or {}).get(col)
    return Result("projection", {"expected": expected, "low": max(0.0, expected - half) if half == half else None,
                                 "high": expected + half if half == half else None, "days": days, "daily_mean": mean,
                                 "daily_sd": sd if sd == sd else None, "history_days": n, "active_days": active,
                                 "history_start": str(start.date()), "history_end": str(end.date())},
                  unit=unit, provenance={"op": "project", "column": col, "date": date,
                                         "assumption": "each future day follows the historical daily total (days with "
                                                       "no record count as 0), independently",
                                         "rows": int(ok.sum())})


def op_compare(s, results, data):
    a = results.get(s["a"])
    if a is None:
        raise ProgramError(f"step '{s['id']}': input '{s['a']}' does not exist")
    av = _value(a)
    if "target" in s:
        target = float(s["target"])
        if s.get("unit") and a.unit and s["unit"] != a.unit:
            raise ProgramError(f"step '{s['id']}': the target is in {s['unit']} but the value is in {a.unit}")
        base = _value(results[s["base"]]) if s.get("base") else 0.0
        days = s.get("days")
        need = target - base
        required_daily = need / days if days else None
        proj = a.value if a.kind == "projection" else None
        expected_total = base + (proj["expected"] if proj else av)
        low = base + proj["low"] if proj and proj.get("low") is not None else None
        high = base + proj["high"] if proj and proj.get("high") is not None else None
        if need <= 0:
            verdict = "already reached"
        elif days is not None and days <= 0:
            verdict = "not reachable: no time left in the period"
        elif required_daily is not None and s.get("max_daily") and required_daily > s["max_daily"]:
            verdict = f"not physically possible: it needs more than {s['max_daily']:g} per day"
        elif low is not None and low >= target:
            verdict = "likely on the current pace"
        elif high is not None and high < target:
            verdict = "unlikely on the current pace"
        else:
            verdict = "uncertain: the target is within the range of likely outcomes"
        return Result("comparison", {"target": target, "completed": base, "remaining_needed": max(need, 0.0),
                                     "days_left": days, "required_daily": required_daily,
                                     "current_daily": proj["daily_mean"] if proj else None, "expected_total": expected_total,
                                     "expected_low": low, "expected_high": high, "gap": expected_total - target,
                                     "verdict": verdict}, unit=a.unit,
                      provenance={"op": "compare", "a": s["a"], "base": s.get("base"), "target": target})
    b = results.get(s["b"])
    if b is None:
        raise ProgramError(f"step '{s['id']}': input '{s['b']}' does not exist")
    if a.unit and b.unit and a.unit != b.unit:
        raise ProgramError(f"step '{s['id']}': cannot compare {a.unit} with {b.unit}")
    bv = _value(b)
    return Result("comparison", {"a": av, "b": bv, "difference": av - bv, "ratio": av / bv if bv else None,
                                 "a_label": s.get("a_label", s["a"]), "b_label": s.get("b_label", s["b"])},
                  unit=a.unit, hypothetical=a.hypothetical or b.hypothetical,
                  provenance={"op": "compare", "a": s["a"], "b": s["b"]})


def op_scenario(s, results, data):
    """A hypothetical table: an entity's records removed (entirely, or its share of a horizon via 'days'), or a
    column scaled. Marked hypothetical; nothing in the data changes."""
    df = _table(results, s.get("from", "data"), data, s["id"]).copy()
    notes = []
    for ch in s.get("changes") or []:
        if "remove" in ch:
            col, val = next(iter(ch["remove"].items()))
            _need_col(df, col, s)
            hit = df[col].astype(str).str.lower() == str(val).lower()
            if not hit.any():
                raise ProgramError(f"step '{s['id']}': '{val}' does not occur in '{col}'")
            df = df[~hit]
            notes.append(f"{val} removed")
        elif "scale" in ch:
            col, pct = ch["scale"]["column"], float(ch["scale"]["pct"])
            _need_col(df, col, s)
            if kind_of(df[col]) != "numeric":
                raise ProgramError(f"step '{s['id']}': '{col}' is not numeric, so it cannot be scaled")
            df[col] = pd.to_numeric(df[col], errors="coerce") * (1 + pct / 100)
            notes.append(f"{col} {pct:+g}%")
        else:
            raise ProgramError(f"step '{s['id']}': unsupported change {list(ch)} (use remove or scale)")
    return Result("table", df, hypothetical=True,
                  provenance={"op": "scenario", "name": s.get("name"), "changes": s.get("changes"), "notes": notes,
                              "rows": int(len(df))})


def _value(r: Result) -> float:
    if r.kind == "scalar":
        return float(r.value) if r.value is not None else float("nan")
    if r.kind == "projection":
        return float(r.value["expected"])
    if r.kind == "comparison" and "expected_total" in r.value:
        return float(r.value["expected_total"])
    raise ProgramError(f"a {r.kind} has no single value")


def _need_col(df, col, s):
    if col is None or col not in df.columns:
        raise ProgramError(f"step '{s['id']}': there is no column '{col}'")


def op_rank(s, results, data):
    """The top (or bottom) n of a per-group result."""
    r = results.get(s["from"])
    if r is None or r.kind != "series":
        raise ProgramError(f"step '{s['id']}': rank needs a per-group result, '{s.get('from')}' is not one")
    n = int(s.get("n", 5))
    if n < 1:
        raise ProgramError(f"step '{s['id']}': n must be at least 1")
    items = [(k, v) for k, v in r.value.items() if v is not None]
    items.sort(key=lambda kv: kv[1], reverse=s.get("order", "desc") == "desc")
    return Result("series", dict(items[:n]), unit=r.unit,
                  provenance={"op": "rank", "from": s["from"], "n": n, "order": s.get("order", "desc"),
                              "groups_ranked": len(items)})


def op_anomalies(s, results, data):
    """Values outside [Q1 - k IQR, Q3 + k IQR] of one numeric column (k = 1.5 by default), with their file lines."""
    df = _table(results, s.get("from", "data"), data, s["id"])
    col = s["column"]
    _need_col(df, col, s)
    if kind_of(df[col]) != "numeric":
        raise ProgramError(f"step '{s['id']}': '{col}' is not numeric, so it has no outliers by value")
    k = float(s.get("k", 1.5))
    x = pd.to_numeric(df[col], errors="coerce")
    v = x.dropna()
    if len(v) < 8:
        raise ProgramError(f"step '{s['id']}': '{col}' has fewer than 8 values, too few to call any unusual")
    q1, q3 = float(v.quantile(0.25)), float(v.quantile(0.75))
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    hit = (x < lo) | (x > hi)
    extreme = (x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)
    share = float(x[hit].abs().sum() / x.abs().sum()) if x.abs().sum() else None
    return Result("series", {"count": int(hit.sum()), "extreme": int(extreme.sum()), "checked": int(len(v)), "low": lo, "high": hi,
                             "median": float(v.median()), "share_of_total": share,
                             "examples": [{"line": ln, "value": float(x[i])} for ln, i in zip(_lines(df, x[hit].index), x[hit].index)][:5]},
                  unit=(df.attrs.get("units") or {}).get(col),
                  provenance={"op": "anomalies", "column": col, "rule": f"{k:g} x IQR", "rows": int(len(v))})


def op_associate(s, results, data):
    """Associations of every usable column with an outcome (app/analyst/associations.py): Spearman, eta squared or
    Cramer's V, with Holm-adjusted p-values."""
    from app.analyst import associations
    df = _table(results, s.get("from", "data"), data, s["id"])
    _need_col(df, s["target"], s)
    r = associations.compute(df, s["target"], exclude=s.get("exclude"))
    return Result("series", r, provenance={"op": "associate", "target": s["target"], "rows": int(len(df)),
                                           "tested": r["tested"]})


def op_profile(s, results, data):
    """Data quality of a table: missing values per column, exact duplicates, impossible values (as judged by the
    validity rules), dates after today."""
    from app.analyst.scope import eligibility
    df = _table(results, s.get("from", "data"), data, s["id"])
    cols = [c for c in (s.get("columns") or df.columns) if c in df.columns]
    _, reasons = eligibility(df, cols, historical=True)
    missing = {str(c): int(df[c].isna().sum()) for c in cols if int(df[c].isna().sum())}
    return Result("series", {"rows": int(len(df)), "missing": missing, "issues": reasons},
                  provenance={"op": "profile", "columns": [str(c) for c in cols], "rows": int(len(df))})


def op_ratio(s, results, data):
    """a / b: two single values, or two per-group results divided group by group (groups missing from either side,
    or with b = 0, have no value and are listed)."""
    a, b = results.get(s["a"]), results.get(s.get("b"))
    if a is None or b is None:
        raise ProgramError(f"step '{s['id']}': ratio needs two earlier results a and b")
    unit = f"{a.unit or 'value'} per {b.unit or 'value'}"
    if a.kind == "series" and b.kind == "series":
        out, missing = {}, []
        for k in a.value:
            av, bv = a.value.get(k), b.value.get(k)
            if av is None or bv in (None, 0):
                missing.append(k)
            else:
                out[k] = av / bv
        if not out:
            raise ProgramError(f"step '{s['id']}': '{s['a']}' and '{s['b']}' have no groups in common")
        return Result("series", out, unit=unit, hypothetical=a.hypothetical or b.hypothetical,
                      provenance={"op": "ratio", "a": s["a"], "b": s["b"], "groups": len(out), "no_value": missing})
    if a.kind == "series" or b.kind == "series":
        raise ProgramError(f"step '{s['id']}': cannot divide a per-group result by a single value; give both per group "
                           "or both as single values")
    av, bv = _value(a), _value(b)
    return Result("scalar", av / bv if bv else None, unit=unit, hypothetical=a.hypothetical or b.hypothetical,
                  provenance={"op": "ratio", "a": s["a"], "b": s["b"]})


RUN = {"select": op_select, "derive": op_derive, "aggregate": op_aggregate, "project": op_project,
       "compare": op_compare, "scenario": op_scenario, "rank": op_rank, "anomalies": op_anomalies,
       "associate": op_associate, "profile": op_profile, "ratio": op_ratio}


def validate(program: list[dict], known=()) -> None:
    """known: ids of results supplied from earlier in the conversation (run(..., inputs=...))."""
    seen = set(known)
    for s in program:
        if s.get("op") not in OPS:
            raise ProgramError(f"step '{s.get('id')}': unknown operation '{s.get('op')}'")
        if not s.get("id") or s["id"] in seen or s["id"] == "data":
            raise ProgramError(f"step ids must be unique and not 'data' ('{s.get('id')}')")
        for ref in [s.get("from"), s.get("a"), s.get("b"), s.get("base")]:
            if ref and ref != "data" and ref not in seen:
                raise ProgramError(f"step '{s['id']}' uses '{ref}', which is not an earlier step")
        seen.add(s["id"])


def run(program: list[dict], data: pd.DataFrame, inputs: dict[str, Result] | None = None) -> dict[str, Result]:
    validate(program, known=inputs or {})
    results: dict[str, Result] = dict(inputs or {})
    for s in program:
        r = RUN[s["op"]](s, results, data)
        # anything computed from a hypothetical result is hypothetical too: a total over a scenario table is not an
        # observed total
        if any(isinstance(results.get(s.get(k)), Result) and results[s[k]].hypothetical for k in ("from", "a", "b", "base")):
            r.hypothetical = True
        results[s["id"]] = r
    return results
