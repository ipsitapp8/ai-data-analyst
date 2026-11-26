"""Answer every deliverable of a compound request in one turn: execute, validate, compose, check completeness.

Execution: deliverables run in dependency order; a failure is caught per deliverable, so one failed step never
erases the others; a deliverable whose dependency failed is either computed without it (said so) or marked
blocked with the reason. Every number comes from Silt's validated tools or deterministic pandas code and is
recomputed independently where that is meaningful. Composition keeps the user's order, gives each deliverable
its own section with its own scope, and ends with a checklist of every requested item and where it is answered;
the answer is never marked complete while any requested item is missing.
"""
from __future__ import annotations

import math
import re
import traceback

import numpy as np
import pandas as pd

from app.analyst import forecasting as F
from app.analyst import overview, present, tools
from app.analyst.data import kind_of
from app.analyst.deliverables import Deliverable
from app.analyst.explain import f as fmt
from app.analyst.semantics import additive, aggregation_problem, profile

AGG_WORD = {"avg": "average", "var": "variance", "std": "standard deviation", "sum": "total", "min": "minimum",
            "max": "maximum", "median": "median", "count": "count"}
PANDAS_AGG = {"avg": "mean", "var": "var", "std": "std", "sum": "sum", "min": "min", "max": "max", "median": "median"}
DAYS = {"D": 1.0, "W": 7.0, "M": 365.25 / 12, "Q": 365.25 / 4, "Y": 365.25}
PERIOD = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}


class StepError(Exception):
    """A deliverable that cannot be completed; the message says why in plain words."""


class NeedsInput(StepError):
    """The item is too ambiguous to compute: Silt asks, and every other item is still answered."""



# --------------------------------------------------------------------- helpers --
def _scope(df: pd.DataFrame, entities: dict) -> tuple[pd.DataFrame, str, list]:
    d, parts, filters = df, [], []
    for col, val in (entities or {}).items():
        vals = val if isinstance(val, list) else [val]
        d = d[d[col].astype(str).str.lower().isin([str(v).lower() for v in vals])]
        parts.append(f"{col} = {', '.join(map(str, vals))}")
        filters += [{"column": col, "op": "in", "value": vals}] if len(vals) > 1 else [{"column": col, "op": "==", "value": vals[0]}]
    if (entities or {}) and d.empty:
        raise StepError(f"no records match {', '.join(parts)}")
    return d, ("; ".join(parts) if parts else "all records"), filters


def _date_col(df, prof) -> str | None:
    return prof["dates"][0] if prof["dates"] else None


def _window(df, prof, d) -> pd.DataFrame:
    """The records inside the deliverable's own time window (inclusive)."""
    date = _date_col(df, prof)
    if not date:
        raise StepError("a time window was asked for, but the data has no date column")
    t = pd.to_datetime(df[date], errors="coerce")
    keep = pd.Series(True, index=df.index)
    if d.date_from:
        keep &= t >= pd.Timestamp(d.date_from)
    if d.date_to:
        keep &= t <= pd.Timestamp(d.date_to) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    out = df[keep]
    if out.empty:
        raise StepError(f"no records between {d.date_from or 'the start'} and {d.date_to or 'the latest date'} "
                        f"(the data runs from {t.min():%Y-%m-%d} to {t.max():%Y-%m-%d})")
    return out


def _people_col(df, prof, ents) -> str | None:
    return next(iter(ents), None) if ents else prof.get("entity")


def _series_for(df, prof, measure, entities, want_freq=None):
    """(series, freq, grain, how): the measure per period for the scoped records, at the data's own grain or a
    coarser requested one (daily records summed to months); never finer than the data."""
    date = _date_col(df, prof)
    if not date:
        raise StepError("the data has no date column, so values cannot be placed in time")
    d, scope, _ = _scope(df, entities)
    ent_col = next(iter(entities), None) if entities else None
    grain = F.infer_grain(df, date, ent_col or prof.get("entity"))
    if not grain["freq"]:
        raise StepError("there are not enough dated records to tell how often values are recorded")
    order = "DWMQY"
    freq = grain["freq"]
    if want_freq and order.index(want_freq) > order.index(freq):
        freq = want_freq                     # aggregate up (days -> months); never split a monthly total into days
    role = profile(d)["roles"].get(measure, {})
    agg = "sum" if additive(d, measure, role) or grain["one_per_period"] else "mean"
    if not grain["one_per_period"] and not additive(d, measure, role):
        agg = "mean"
    s = F.to_series(d, date, measure, freq, agg)
    how = ("one record per period: each value is that period's total" if grain["one_per_period"] and freq == grain["freq"]
           else f"records {'summed' if agg == 'sum' else 'averaged'} per {PERIOD[freq]}")
    return s, freq, grain, how, scope


def _metric(df, measure, agg, filters, group_by=None, order=None):
    args = {"aggregation": agg, "column": measure}
    if filters:
        args["filters"] = filters
    if group_by:
        args["group_by"] = group_by
    if order:
        args["order"] = order
    call = tools.validate_call({"tool": "compute_metric", "args": args}, df)
    return call, tools.run(call, df, {})


def _close(a, b) -> bool:
    return a is not None and b is not None and (abs(a - b) <= 1e-6 * max(1.0, abs(b)))


def _tbl(rows, cols) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|\n"
    return head + "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)


# ------------------------------------------------------------------- executors --
def run_summary(d: Deliverable, df, ctx):
    r = overview.understand(df, ctx.get("filename"))
    ctx["summary_evidence"] = r
    text = present.understand(r)
    text = re.split(r"\n\*\*Columns that could be predicted|\n\*\*Questions worth asking next", text)[0].strip()
    text = text.replace("### What's in this dataset\n", "")
    return {"text": text, "values": {"rows": int(len(df)), "columns": int(df.shape[1])}, "verified": True,
            "summary": r, "calls": [{"tool": "describe_dataset", "status": "ok"}]}


def run_stat(d: Deliverable, df, ctx):
    if not d.measures:
        raise StepError(d.detail or "no measure was named")
    sub, scope, filters = _scope(df, d.entities)
    rows, values, calls, skipped = [], {}, [], []
    for m in d.measures:
        if m not in df.columns or kind_of(df[m]) != "numeric":
            skipped.append(f"{m} (not a numeric column)")
            continue
        fn = PANDAS_AGG[d.stat]
        prob = aggregation_problem(df, m, {"avg": "mean"}.get(d.stat, d.stat))
        if prob:
            skipped.append(f"{m} ({prob})")
            continue
        call, r = _metric(df, m, d.stat, filters)
        v = r.get("value")
        check = float(getattr(pd.to_numeric(sub[m], errors="coerce").dropna(), fn)())
        ok = _close(v, check)
        values[m] = v
        calls.append({**call, "status": "ok", "result": {"value": v, "rows": r.get("rows")}})
        rows.append({"measure": m, AGG_WORD[d.stat]: fmt(v), "records": r.get("rows"), "check": "✓" if ok else "✗"})
    if not rows:
        raise StepError("no numeric measure could be used: " + "; ".join(skipped))
    text = (f"Scope: {scope}.\n\n" if d.entities else "") + _tbl(rows, ["measure", AGG_WORD[d.stat], "records", "check"])
    if d.stat == "var":
        text += "\n\n_Sample variance (n − 1 in the denominator); its unit is the measure's unit squared._"
    if skipped:
        text += "\n\n_Not computed: " + "; ".join(skipped) + "._"
    return {"text": text, "values": values, "verified": all(r["check"] == "✓" for r in rows), "calls": calls,
            "partial": bool(skipped)}


def run_breakdown(d: Deliverable, df, ctx, ranking=False):
    m = d.measures[0] if d.measures else None
    if not m:
        raise StepError("no measure was named")
    sub, scope, filters = _scope(df, d.entities)
    role = profile(df)["roles"].get(m, {})
    agg = d.stat or ("sum" if additive(df, m, role) else "avg")
    order = d.order_dir if ranking else None
    call, r = _metric(df, m, agg, filters, group_by=d.group_by, order=order or "desc")
    groups = [g for g in r.get("groups", []) if g.get("value") is not None]
    check = getattr(sub.groupby(d.group_by)[m], PANDAS_AGG[agg])()
    ok = all(_close(g["value"], float(check[g["group"]])) for g in groups if g["group"] in check.index)
    if ranking and d.top:
        groups = groups[: d.top]
    rows = [{"rank" if ranking else d.group_by: (i + 1 if ranking else g["group"]),
             **({d.group_by: g["group"]} if ranking else {}), AGG_WORD[agg] + " " + m: fmt(g["value"])}
            for i, g in enumerate(groups)]
    cols = (["rank", d.group_by] if ranking else [d.group_by]) + [AGG_WORD[agg] + " " + m]
    text = (f"Scope: {scope}.\n\n" if d.entities else "") + _tbl(rows, cols)
    if ranking and groups:
        text += f"\n\n**{'Highest' if d.order_dir == 'desc' else 'Lowest'}: {groups[0]['group']}** ({fmt(groups[0]['value'])})."
    chart = {"type": "bar", "title": f"{AGG_WORD[agg]} {m} by {d.group_by}" + (f" ({scope})" if d.entities else ""),
             "x": d.group_by, "y": m, "data": [{d.group_by: g["group"], m: g["value"]} for g in groups]}
    return {"text": text, "values": {g["group"]: g["value"] for g in groups}, "verified": ok,
            "calls": [{**call, "status": "ok", "result": {"groups": groups}}], "chart": chart}


def run_series(d: Deliverable, df, ctx):
    m = d.measures[0] if d.measures else None
    if not m:
        raise StepError("no measure was named")
    prof = ctx["prof"]
    s, freq, grain, how, scope = _series_for(df, prof, m, d.entities, d.period)
    s = s.dropna()
    rows = [{PERIOD[freq]: str(p), m: fmt(float(v))} for p, v in s.items()]
    text = f"Scope: {scope}. {how.capitalize()}.\n\n" + _tbl(rows, [PERIOD[freq], m])
    chart = {"type": "line", "title": f"{m} per {PERIOD[freq]}" + (f" — {scope}" if d.entities else ""),
             "x": "period", "y": m, "data": [{"period": str(p), m: float(v)} for p, v in s.items()]}
    return {"text": text, "values": {str(p): float(v) for p, v in s.items()}, "verified": True, "chart": chart,
            "series": s, "freq": freq}


def run_forecast(d: Deliverable, df, ctx):
    m = d.measures[0] if d.measures else None
    if not m:
        raise StepError("no measure was named to forecast")
    prof = ctx["prof"]
    want = d.period or None
    s, freq, grain, how, scope = _series_for(df, prof, m, d.entities, want)
    target = d.target
    if target is not None and d.period is None:
        tper = next((x.period for x in ctx["all"] if x.kind == "target_check" and x.target == target), "M")
        if tper and tper != freq:
            target = target * DAYS[freq] / DAYS[tper]     # a monthly target, weekly data: the weekly equivalent
    fc = F.forecast(s, d.horizon or 1, freq, target)
    unit = d.target_unit or ""
    sub, _, _ = _scope(df, d.entities)
    checks = F.series_checks(sub[_date_col(df, prof)], sub[m], freq, s)
    fc["data_checks"] = checks
    check_line = "**Data checks:** " + F.describe_checks(checks, freq) + "."
    if fc["status"] != "ok":
        res = {"text": f"Scope: {scope}.\n\n**No forecast:** {fc['reason']}. The history is shown so nothing is "
                       f"invented.\n\n{check_line}", "values": {}, "verified": True, "forecast": fc, "insufficient": True,
               "chart": _forecast_chart(m, scope, fc, None, unit, freq)}
        return res
    rows = []
    for p in fc["forecasts"]:
        rng = (f"{fmt(p['lower'])} – {fmt(p['upper'])}" if p["lower"] is not None else "not enough history for a range")
        row = {PERIOD[freq]: p["period"], "forecast": fmt(p["forecast"]), f"likely range ({int(fc['level'] * 100)}%)": rng}
        if target is not None:
            row["vs target"] = (f"{p['target_verdict']} ({'+' if p['gap_to_target'] >= 0 else '−'}"
                                f"{fmt(abs(p['gap_to_target']))})")
        rows.append(row)
    cols = list(rows[0].keys())
    top = fc["backtest"][: 4]
    bt = "; ".join(f"{b['label']}: MAE {fmt(b['mae'])}" for b in top)
    last = fc["history"][-1]
    text = (f"Scope: {scope}. History: {fc['points']} {PERIOD[freq]}s ({how}); last actual {last['period']}: "
            f"{fmt(last['value'])}.\n\n" + _tbl(rows, cols) +
            f"\n\n**Method:** {fc['method_label']} — {fc['selection']} ({fc['validation']}). Out-of-sample error: "
            f"MAE {fmt(fc['backtest_mae'])}, RMSE {fmt(fc['backtest_rmse'])}"
            + (f", MASE {fc['mase']:.2f}" if fc.get("mase") is not None else "") +
            f"; naive \"same as last {PERIOD[freq]}\" baseline MAE {fmt(fc['naive_mae'])}. Candidates: {bt}."
            f"\n\n{check_line}"
            "\n\n_The range comes from the spread of past forecast errors; it assumes the coming periods behave like "
            "the history. A forecast is an estimate, not a certainty._")
    if target is not None and d.target is not None and abs(target - d.target) > 1e-9:
        text += f"\n\n_Target converted to the data's grain: {fmt(d.target)} per month ≈ {fmt(target)} per {PERIOD[freq]}._"
    return {"text": text, "values": {p["period"]: p["forecast"] for p in fc["forecasts"]}, "verified": True,
            "forecast": fc, "chart": _forecast_chart(m, scope, fc, target, unit, freq), "freq": freq, "target_used": target}


def _forecast_chart(m, scope, fc, target, unit, freq):
    data = [{"period": h["period"], "value": h["value"], "kind": "actual", "lower": None, "upper": None}
            for h in fc["history"]]
    for p in fc.get("forecasts") or []:
        data.append({"period": p["period"], "value": p["forecast"], "kind": "forecast", "lower": p["lower"],
                     "upper": p["upper"]})
    chart = {"type": "forecast", "title": f"{m}: actual and forecast — {scope}", "x": "period", "y": "value",
             "y_title": f"{m}" + (f" ({unit} per {PERIOD[freq]})" if unit else f" per {PERIOD[freq]}"),
             "data": data, "series": "kind"}
    if target is not None:
        chart["target"] = float(target)
        chart["target_label"] = f"target {fmt(target)}" + (f" {unit}" if unit else "") + f" per {PERIOD[freq]}"
    if fc.get("forecasts") and fc["forecasts"][0].get("lower") is not None:
        chart["interval_level"] = fc.get("level")
    return chart


def run_target(d: Deliverable, df, ctx):
    if d.target is None:
        raise StepError("no target value was given")
    m = d.measures[0] if d.measures else None
    if not m:
        raise StepError("the measure the target refers to could not be identified")
    prof = ctx["prof"]
    s, freq, grain, how, scope = _series_for(df, prof, m, d.entities, d.period)
    target = d.target
    note = ""
    if d.period and freq != d.period:
        target = d.target * DAYS[freq] / DAYS[d.period]
        note = (f"_The data is recorded per {PERIOD[freq]}, so the target of {fmt(d.target)} per "
                f"{PERIOD[d.period]} is compared as {fmt(target)} per {PERIOD[freq]}._")
    s = s.dropna()
    if s.empty:
        raise StepError("there is no history for this scope")
    last_p, last_v = str(s.index[-1]), float(s.iloc[-1])
    met = int((s >= target).sum())
    recent = s.iloc[-3:] if len(s) >= 3 else s
    lines = [f"Scope: {scope}. {how.capitalize()}.", "",
             _tbl([{"": f"latest {PERIOD[freq]} ({last_p})", "value": fmt(last_v)},
                   {"": "gap to target", "value": ("+" if last_v >= target else "−") + fmt(abs(last_v - target))},
                   {"": f"average of the last {len(recent)} {PERIOD[freq]}s", "value": fmt(float(recent.mean()))},
                   {"": f"{PERIOD[freq]}s at or above target", "value": f"{met} of {len(s)}"}], ["", "value"])]
    future = None
    dep = next((ctx["results"].get(i) for i in d.depends_on if ctx["results"].get(i)), None)
    if dep is None and re.search(r"\b(will|would|going to|on track|make it|by the end)\b", d.clause, re.I):
        # a question about the future: the evidence is a tested forecast of the next period, not the past average
        own = F.forecast(s, 1, freq, target)
        dep = {"forecast": own, "internal": True}
    if dep and dep.get("forecast") and dep["forecast"].get("status") == "ok":
        p = dep["forecast"]["forecasts"][0]
        future = p
        chance = p.get("chance_at_or_above_target")
        verdict = p["target_verdict"]
        if dep.get("internal"):
            fcx = dep["forecast"]
            lines += ["", f"_Forecast used for the answer: {fcx['method_label']}, the lowest out-of-sample error of the "
                          f"methods tested ({fcx['validation']}); naive-baseline error {fmt(fcx['naive_mae'])}, chosen "
                          f"method {fmt(fcx['backtest_mae'])}._"]
        lines += ["", f"**Next {PERIOD[freq]} ({p['period']}): forecast {fmt(p['forecast'])} vs target "
                      f"{fmt(target)} — {verdict}"
                      + (f" ({chance * 100:.0f}% of past forecast errors would put it at or above the target)" if chance is not None else "")
                      + ".**"]
        answer = "yes" if verdict in ("likely", "at or above") else "no" if verdict in ("unlikely", "below") else "uncertain"
    else:
        answer = "yes" if last_v >= target else "no"
        if dep is not None:
            why = (dep.get("forecast") or {}).get("reason") or "the forecast could not be made"
            lines += ["", f"_Whether the target will be met needs a forecast, and none could be made ({why}); this "
                          "compares the latest actual with the target._"]
        lines += ["", f"**Latest {PERIOD[freq]}: {fmt(last_v)} — {'at or above' if last_v >= target else 'below'} the target of {fmt(target)}.**"]
    if note:
        lines += ["", note]
    # the answer to the question as asked: "Is he below the target? If so ..." -> yes means below
    if re.search(r"\b(below|under|short of|miss(?:es|ed)?|fall(?:s|ing)? short|behind)\b", d.clause, re.I) and answer in ("yes", "no"):
        answer = "no" if answer == "yes" else "yes"
    return {"text": "\n".join(lines), "values": {"latest": last_v, "target": target, "met": met, "gap": last_v - target,
                                                 "recent_average": float(recent.mean()), "periods": int(len(s))},
            "verified": True,
            "answer": answer, "below": last_v < target if future is None else future["forecast"] < target,
            "future": future, "freq": freq}


def run_pace(d: Deliverable, df, ctx):
    """The average needed per pace unit to reach a target per period, and the recent actual pace -- both measured on
    the same basis: when daily records exist only on some days (working days), a daily pace is per recorded day; the
    recent pace comes from complete pace units of real data; calendar conversion only when the data are coarser."""
    if d.target is None:
        raise StepError("no target value was given to compute the pace for")
    m = d.measures[0] if d.measures else None
    if not m:
        raise StepError("the measure the target refers to could not be identified")
    prof = ctx["prof"]
    tp = d.period or "M"
    pu = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}[d.pace_unit or "week"]
    s, freq, grain, how, scope = _series_for(df, prof, m, d.entities, None)
    s = s.dropna()
    date = _date_col(df, prof)
    sub, _, _ = _scope(df, d.entities)
    dates = pd.to_datetime(sub[date], errors="coerce")
    order = "DWMQY"
    basis = None
    if pu == "D" and freq == "D":
        per_tp = dates.dt.to_period(tp).groupby(dates.dt.to_period(tp)).apply(lambda x: x.index.size)
        rec_days = dates.dt.normalize().groupby(dates.dt.to_period(tp)).nunique()
        full = rec_days.iloc[1:-1] if len(rec_days) > 2 else rec_days        # complete periods only
        days_per_tp = float(full.median()) if len(full) else DAYS[tp]
        if days_per_tp < DAYS[tp] - 0.5:
            conv = days_per_tp
            basis = (f"the data has records on about {fmt(days_per_tp)} days per {PERIOD[tp]}, so the daily pace is per "
                     "recorded (working) day")
        else:
            conv = DAYS[tp]
        daily = pd.to_numeric(sub[m], errors="coerce").groupby(dates.dt.normalize()).sum().sort_index()
        recent = daily.iloc[-3:]
        recent_per_unit = float(recent.mean())
        recent_desc = f"the last {len(recent)} recorded days averaged **{fmt(recent_per_unit)} per day**"
    elif order.index(freq) <= order.index(pu):
        conv = DAYS[tp] / DAYS[pu]
        per_u = pd.to_numeric(sub[m], errors="coerce").groupby(dates.dt.to_period(pu)).sum().sort_index()
        if len(per_u) > 1 and freq != pu:
            last_n = dates.dt.normalize().groupby(dates.dt.to_period(pu)).nunique()
            if last_n.iloc[-1] < last_n.median():
                per_u = per_u.iloc[:-1]                                         # the latest unit is incomplete
        recent = per_u.iloc[-3:]
        recent_per_unit = float(recent.mean())
        recent_desc = (f"the last {len(recent)} complete {PERIOD[pu]}s averaged **{fmt(recent_per_unit)} per "
                       f"{PERIOD[pu]}**")
    else:
        conv = DAYS[tp] / DAYS[pu]
        recent = s.iloc[-3:] if len(s) >= 3 else s
        recent_per_unit = float(recent.mean()) * DAYS[pu] / DAYS[freq]
        recent_desc = (f"the last {len(recent)} {PERIOD[freq]}s averaged {fmt(float(recent.mean()))} per {PERIOD[freq]} "
                       f"≈ **{fmt(recent_per_unit)} per {PERIOD[pu]}** (calendar conversion: the data are recorded per "
                       f"{PERIOD[freq]})")
    per_unit_needed = d.target / conv
    gap = per_unit_needed - recent_per_unit
    text = (f"Scope: {scope}.\n\n"
            f"To reach **{fmt(d.target)} per {PERIOD[tp]}**, the required average is **{fmt(per_unit_needed)} per "
            f"{PERIOD[pu]}** ({fmt(d.target)} ÷ {fmt(conv)} {'recorded days' if basis else PERIOD[pu] + 's'} in a "
            f"{PERIOD[tp]})." + (f" _{basis[0].upper() + basis[1:]}; per calendar day that is "
                                 f"{fmt(d.target / DAYS[tp])}._" if basis else "") + "\n\n"
            f"Recent pace: {recent_desc}, so the "
            + (f"required pace is **{fmt(gap)} per {PERIOD[pu]} higher** than recently." if gap > 0 else
               f"recent pace is already **{fmt(-gap)} per {PERIOD[pu]} above** what is needed."))
    return {"text": text, "values": {"required": per_unit_needed, "recent": recent_per_unit, "gap": gap,
                                     "units_per_period": conv, "target": d.target}, "verified": True}



def _groups_frame(df, d):
    if not d.group_by or d.group_by not in df.columns:
        raise StepError("no column to compare groups by was named")
    sub, scope, filters = _scope(df, d.entities)
    if d.groups:
        want = {str(g).lower() for g in d.groups}
        sub = sub[sub[d.group_by].astype(str).str.lower().isin(want)]
        missing = sorted(want - set(sub[d.group_by].astype(str).str.lower()))
        if missing:
            raise StepError(f"no records for {', '.join(missing)} in {d.group_by}")
    return sub, scope, filters


def run_compare(d: Deliverable, df, ctx):
    """A measure compared between groups: each group's value, the difference, the percentage difference and, for
    record-level data, whether the difference is statistically meaningful (Welch's t / Welch ANOVA)."""
    m = d.measures[0] if d.measures else None
    if not m:
        raise StepError("no measure was named to compare")
    sub, scope, filters = _groups_frame(df, d)
    role = profile(df)["roles"].get(m, {})
    agg = d.stat or "avg"
    prob = aggregation_problem(df, m, {"avg": "mean"}.get(agg, agg))
    if prob:
        raise StepError(prob)
    vals = getattr(sub.groupby(d.group_by)[m], PANDAS_AGG[agg])().dropna()
    order = [g for g in (d.groups or list(vals.sort_values(ascending=False).index)) if str(g).lower() in
             {str(i).lower() for i in vals.index}]
    keyed = {str(i).lower(): (str(i), float(v)) for i, v in vals.items()}
    rows_v = [keyed[str(g).lower()] for g in order]
    if len(rows_v) < 2:
        raise StepError("fewer than two groups have values to compare")
    # the same numbers through the analytical tool (independent of the pandas line above)
    call = tools.validate_call({"tool": "compute_metric", "args": {"aggregation": agg, "column": m, "group_by": d.group_by,
                                                                    **({"filters": filters} if filters else {})}}, df)
    r = tools.run(call, df, {})
    tool_vals = {str(g["group"]).lower(): g["value"] for g in r.get("groups", [])}
    ok = all(_close(tool_vals.get(name.lower()), v) for name, v in rows_v)
    lines = [{d.group_by: name, AGG_WORD[agg] + " " + m: fmt(v), "records": int((sub[d.group_by].astype(str) == name).sum())}
             for name, v in rows_v]
    text = (f"Scope: {scope}.\n\n" if d.entities else "") + _tbl(lines, [d.group_by, AGG_WORD[agg] + " " + m, "records"])
    values = {name: v for name, v in rows_v}
    hi, lo = max(rows_v, key=lambda x: x[1]), min(rows_v, key=lambda x: x[1])
    diff = hi[1] - lo[1]
    pct = (diff / abs(lo[1]) * 100) if lo[1] else None
    text += (f"\n\n**{hi[0]}** is higher than **{lo[0]}** by {fmt(diff)}" +
             (f" ({pct:,.1f}% of {lo[0]}'s value)." if pct is not None else "."))
    values["difference"] = diff
    if pct is not None:
        values["pct_difference"] = pct
    test_note = ""
    if agg in ("avg", "median") and kind_of(df[m]) == "numeric":
        tcall = tools.validate_call({"tool": "compare_groups", "args": {
            "outcome": m, "group": d.group_by, "filters": filters + ([{"column": d.group_by, "op": "in",
                                                                        "value": [n for n, _ in rows_v]}])}}, df)
        tr = tools.run(tcall, df, {})
        t = (tr or {}).get("test") or {}
        if t.get("p_value") is not None:
            p = float(t["p_value"])
            eff = t.get("hedges_g") if t.get("hedges_g") is not None else t.get("eta_squared")
            effn = "Hedges' g" if t.get("hedges_g") is not None else "η²"
            meaningful = p < 0.05
            test_note = (f"\n\nIs the difference meaningful? {t['name']}: p = {p:.3g}" +
                         (f", {effn} = {eff:.2f}" if eff is not None else "") + ". " +
                         ("The difference is statistically significant at the 5% level" if meaningful else
                          "The difference is not statistically significant at the 5% level: it could be chance") +
                         "; significance is not the same as practical importance, and it does not show a cause.")
            values["p_value"] = p
            if eff is not None:          # every number the answer states is in the result, so it can be traced
                values["hedges_g" if t.get("hedges_g") is not None else "eta_squared"] = float(eff)
        elif (tr or {}).get("note"):
            test_note = f"\n\nNo significance test: {tr['note']}."
    chart = {"type": "bar", "title": f"{AGG_WORD[agg]} {m} by {d.group_by}" + (f" ({scope})" if d.entities else ""),
             "x": d.group_by, "y": m, "data": [{d.group_by: n, m: v} for n, v in rows_v]}
    return {"text": text + test_note, "values": values, "verified": ok, "chart": chart,
            "calls": [{**call, "status": "ok"}]}


def run_growth(d: Deliverable, df, ctx):
    """Change of a measure over time, per group when a group column is named: first vs last period and the
    average period-on-period change, with the greatest growth and the greatest decline named."""
    m = d.measures[0] if d.measures else None
    if not m:
        raise StepError("no measure was named")
    prof = ctx["prof"]
    if d.group_by:
        sub, scope, _ = _groups_frame(df, d)
        names = [str(g) for g in (d.groups or sorted(sub[d.group_by].dropna().astype(str).unique()))]
    else:
        sub, scope, _ = _scope(df, d.entities)
        names = [None]
    if len(names) > 30:
        raise StepError(f"{d.group_by} has {len(names)} groups; name the ones to compare")
    rows, values, series_pts, freq_used = [], {}, [], None
    for name in names:
        ents = dict(d.entities)
        if name is not None:
            ents[d.group_by] = name
        s, freq, grain, how, _ = _series_for(df, prof, m, ents, d.period)
        s = s.dropna()
        freq_used = freq
        if len(s) < 2:
            raise StepError(f"{name or 'the data'} has fewer than two periods, so growth cannot be measured")
        first, last = float(s.iloc[0]), float(s.iloc[-1])
        chg = (last - first) / abs(first) * 100 if first else None
        pop = s.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
        avg_pop = float(pop.mean() * 100) if len(pop) else None
        key = name or m
        values[key] = chg
        rows.append({"group": key, "first": f"{fmt(first)} ({s.index[0]})", "last": f"{fmt(last)} ({s.index[-1]})",
                     "change": f"{chg:+,.1f}%" if chg is not None else "n/a",
                     f"avg change per {PERIOD[freq]}": f"{avg_pop:+,.1f}%" if avg_pop is not None else "n/a"})
        series_pts += [{"period": str(p), m: float(v), "group": key} for p, v in s.items()]
    cols = ["group", "first", "last", "change", f"avg change per {PERIOD[freq_used]}"]
    text = (f"Scope: {scope}.\n\n" if d.entities else "") + _tbl(rows, cols)
    ranked = sorted([(k, v) for k, v in values.items() if v is not None], key=lambda x: x[1])
    if len(ranked) >= 2:
        text += (f"\n\nGreatest decline: **{ranked[0][0]}** ({ranked[0][1]:+,.1f}%)" if ranked[0][1] < 0 else
                 f"\n\nNo group declined; the smallest change is **{ranked[0][0]}** ({ranked[0][1]:+,.1f}%)") + \
                f". Greatest growth: **{ranked[-1][0]}** ({ranked[-1][1]:+,.1f}%)."
    text += ("\n\n_Change = last period vs first period. Growth figures describe what happened; they do not explain "
             "why._")
    chart = {"type": "line", "title": f"{m} per {PERIOD[freq_used]}" + (f" by {d.group_by}" if d.group_by else ""),
             "x": "period", "y": m, "data": series_pts, **({"series": "group"} if d.group_by else {})}
    return {"text": text, "values": values, "verified": True, "chart": chart}


def run_correlation(d: Deliverable, df, ctx):
    if len(d.measures) != 2:
        raise StepError("a correlation needs two numeric columns")
    sub, scope, _ = _scope(df, d.entities)
    x, y = d.measures
    call = tools.validate_call({"tool": "correlation", "args": {"x": x, "y": y}}, sub)
    r = tools.run(call, sub, {})
    if r.get("pearson_r") is None:
        raise StepError(r.get("note") or "the correlation could not be computed")
    pair = sub[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    ok = _close(float(pair[x].corr(pair[y])), r["pearson_r"])
    strength = abs(r["pearson_r"])
    word = "strong" if strength >= 0.7 else "moderate" if strength >= 0.4 else "weak" if strength >= 0.2 else "negligible"
    text = ((f"Scope: {scope}.\n\n" if d.entities else "") +
            f"Pearson r = {r['pearson_r']:.3f} (95% CI {r['pearson_ci'][0]:.3f} to {r['pearson_ci'][1]:.3f}, p = "
            f"{r['pearson_p']:.3g}); Spearman ρ = {r['spearman_rho']:.3f}; {r['n']} complete pairs. A {word} "
            f"{'positive' if r['pearson_r'] > 0 else 'negative'} association. Correlation is not causation.")
    chart = {"type": "scatter", "title": f"{y} vs {x}", "x": x, "y": y,
             "data": pair.head(2000).to_dict("records")}
    return {"text": text, "values": {"pearson_r": r["pearson_r"]}, "verified": ok, "chart": chart,
            "calls": [{**call, "status": "ok"}]}


def run_chart(d: Deliverable, df, ctx):
    subj = ctx["results"].get(d.chart_for)
    if not subj:
        raise StepError("the result it should draw could not be computed")
    chart = subj.get("chart")
    if not chart or not chart.get("data"):
        raise StepError("there was no data to draw")
    problems = validate_chart(chart, subj, ctx["by_id"][d.chart_for])
    if problems:
        raise StepError("the chart failed validation: " + "; ".join(problems))
    note = ""
    if d.chart_type and d.chart_type != chart.get("type"):
        fits = {"bar": {"bar", "pie"}, "pie": {"bar", "pie"}, "line": {"line", "bar"}, "scatter": {"scatter"}}
        if chart.get("type") in fits.get(d.chart_type, set()) and not (d.chart_type == "pie" and len(chart["data"]) > 12):
            chart = {**chart, "type": d.chart_type}
        else:
            note = (f" A {d.chart_type} chart does not suit this result, so it is drawn as a {chart.get('type')} "
                    "chart.")
    return {"text": f"Chart: **{chart['title']}** (shown below).{note}", "note": note.strip(), "values": {"points": len(chart["data"])},
            "verified": True, "chart_spec": chart}


def validate_chart(chart, subj, subject: Deliverable) -> list:
    """The chart draws the subject's own numbers: same values, the scoped records only, the target when one was
    used, and forecast points marked as forecasts."""
    probs = []
    if subject.kind == "forecast":
        fc = subj.get("forecast") or {}
        hist = [x for x in chart["data"] if x["kind"] == "actual"]
        fut = [x for x in chart["data"] if x["kind"] == "forecast"]
        if [h["value"] for h in hist] != [h["value"] for h in fc.get("history", [])]:
            probs.append("the actual values differ from the history used")
        if fc.get("status") == "ok" and [x["value"] for x in fut] != [p["forecast"] for p in fc["forecasts"]]:
            probs.append("the forecast points differ from the forecast")
        if subj.get("target_used") is not None and chart.get("target") != subj["target_used"]:
            probs.append("the target line is missing")
    elif subject.kind == "series":
        if [x[chart["y"]] for x in chart["data"]] != list(subj["values"].values()):
            probs.append("the plotted values differ from the series")
    elif subject.kind == "growth":
        if len(chart["data"]) < 2:
            probs.append("fewer than two points to draw")
    elif subject.kind == "correlation":
        pass
    elif subject.kind in ("breakdown", "ranking", "compare"):
        if {str(x[chart["x"]]): x[chart["y"]] for x in chart["data"]} != {str(k): v for k, v in subj["values"].items()
                                                                          if k not in ("difference", "pct_difference",
                                                                                       "p_value", "hedges_g",
                                                                                       "eta_squared")}:
            probs.append("the bars differ from the computed values")
    return probs


def period_totals(df: pd.DataFrame, ds: list) -> bool:
    """True when the records are one per entity per week / month / quarter / year (period totals): a target per
    period is then compared with those totals, never projected from a daily pace."""
    prof = profile(df)
    date = _date_col(df, prof)
    if not date:
        return False
    ent = next((c for d in ds for c in d.entities), None) or prof.get("entity") or _people_col(df, prof, {})
    g = F.infer_grain(df, date, ent)
    return bool(g["one_per_period"] and g["freq"] in ("W", "M", "Q", "Y"))


RUN = {"summary": run_summary, "stat": run_stat, "breakdown": run_breakdown,
       "ranking": lambda d, df, ctx: run_breakdown(d, df, ctx, ranking=True), "series": run_series,
       "compare": run_compare, "growth": run_growth, "correlation": run_correlation,
       "forecast": run_forecast, "target_check": run_target, "pace": run_pace, "chart": run_chart,
       "unparsed": lambda d, df, ctx: (_ for _ in ()).throw(StepError(d.detail or "this part could not be read")),
       "clarify": lambda d, df, ctx: (_ for _ in ()).throw(NeedsInput(d.detail or "this part needs more detail"))}


# ----------------------------------------------------------------------- engine --
def _topo(ds: list[Deliverable]) -> list[Deliverable]:
    done, out, by = set(), [], {d.id: d for d in ds}
    def visit(d, stack=()):
        if d.id in done or d.id in stack:
            return
        for i in d.depends_on:
            if i in by:
                visit(by[i], stack + (d.id,))
        done.add(d.id)
        out.append(d)
    for d in ds:
        visit(d)
    return out


def execute(ds: list[Deliverable], df: pd.DataFrame, filename: str | None = None) -> dict:
    prof = profile(df)
    ctx = {"prof": prof, "results": {}, "all": ds, "by_id": {d.id: d for d in ds}, "filename": filename}
    for d in _topo(ds):
        blocked = [i for i in d.depends_on if ctx["by_id"].get(i) and ctx["by_id"][i].status in ("failed", "blocked")
                   and d.kind == "chart"]
        if blocked:
            d.status, d.detail = "blocked", f"it depends on item {blocked[0]}, which could not be completed"
            continue
        from app.analyst.deliverables import plan_mismatch
        why = plan_mismatch(d, df, prof)
        if why:                              # the plan would answer a different question: ask, compute nothing
            d.status, d.detail = "needs_input", why
            continue
        try:
            dfd = df
            if d.date_from or d.date_to:
                dfd = _window(df, ctx["prof"], d)
            res = RUN[d.kind](d, dfd, ctx)
            if (d.date_from or d.date_to) and d.kind != "chart" and res.get("text"):
                res["text"] = f"Period: {d.date_from or 'start of the data'} to {d.date_to or 'latest record'}." + \
                    f" ({len(dfd):,} of {len(df):,} records)\n\n" + res["text"]
            ctx["results"][d.id] = res
            d.status = "partial" if res.get("partial") or res.get("insufficient") else "done"
            d.validation = "recomputed independently" if res.get("verified") else "computed; independent check failed"
            if res.get("verified") is False:
                d.status = "partial"
        except NeedsInput as e:
            d.status, d.detail = "needs_input", str(e)
        except StepError as e:
            d.status, d.detail = "failed", str(e)
        except tools.InvalidCall as e:
            d.status, d.detail = "failed", str(e)
        except Exception as e:  # noqa: BLE001 - one step's bug never takes the other steps down
            d.status, d.detail = "failed", f"{type(e).__name__}: {e}"
            ctx.setdefault("errors", []).append(traceback.format_exc(limit=3))
    # display conditions ("If yes, how many hours ...") are evaluated after everything ran
    for d in ds:
        if d.condition in ("yes", "no", "not") and d.status in ("done", "partial"):
            tc = ctx["results"].get(d.condition_on) if d.condition_on else None
            if tc and tc.get("answer") in ("yes", "no"):
                want = "yes" if d.condition == "yes" else "no"
                if tc["answer"] != want:
                    d.condition_met = False
    return ctx


def compose(ds: list[Deliverable], ctx: dict) -> tuple[str, list, list, list]:
    """(answer, charts, calls, incomplete labels) -- sections in the user's order, then the checklist."""
    sections, charts, calls, incomplete = [], [], [], []
    shown = [d for d in sorted(ds, key=lambda x: (x.order, x.id))]
    n = 0
    numbering = {}
    for d in shown:
        res = ctx["results"].get(d.id)
        if d.kind == "chart":
            if res and res.get("chart_spec"):
                charts.append(res["chart_spec"])
            continue
        n += 1
        numbering[d.id] = n
        head = f"### {n}. {d.label()}"
        if d.entity_from_context:
            pm = re.search(r"\b(he|she|him|her|his|hers|they|them|their|it|its)\b", d.clause, re.I)
            head += f" _(“{pm.group(1) if pm else 'this'}” read as {', '.join(map(str, d.entities.values()))})_"
        if d.status in ("done", "partial") and res:
            body = res["text"]
            if getattr(d, "condition_met", True) is False:
                body = (f"_You asked for this only if the answer above was “{d.condition}”; it was not, but the result "
                        "is shown for completeness._\n\n") + body
            if d.status == "partial":
                incomplete.append(d.label() + " (partly)")
        elif d.status == "needs_input":
            body = f"**Needs your input:** {d.detail}"
            incomplete.append(d.label() + " (needs your input)")
        else:
            body = f"**Not completed:** {d.detail or 'it could not be computed'}."
            incomplete.append(d.label())
        # the same scope sentence in consecutive sections is said once
        first = body.split("\n", 1)[0]
        if first.startswith("Scope:") and sections and first in sections[-1]:
            body = "Scope: as above." + (body[len(first):] if len(body) > len(first) else "")
        body = body.replace("_The range comes from the spread of past forecast errors; it assumes the coming periods "
                            "behave like the history. A forecast is an estimate, not a certainty._", "") \
            if any("_The range comes from the spread" in x for x in sections) else body
        sections.append(f"{head}\n\n{body.strip()}")
        calls += (res or {}).get("calls") or []
    # charts: report chart items in the checklist, attached to their subject's section number
    lines = []
    for d in shown:
        if d.kind == "chart":
            subj = ctx["by_id"].get(d.chart_for)
            where = f"below, for item {numbering.get(d.chart_for, '?')}" if subj else "below"
            ok = d.status == "done"
            cnote = ((ctx["results"].get(d.id) or {}).get("note") or "")
            lines.append(f"- {'✓' if ok else '✗'} Chart of {subj.label()[0].lower() + subj.label()[1:] if subj else 'the result'} — "
                         + (where + (f" ({cnote[0].lower() + cnote[1:]})" if cnote else "") if ok else f"not drawn: {d.detail}"))
            if not ok:
                incomplete.append("chart" + (f" of {subj.label().lower()}" if subj else ""))
            continue
        mark = "✓" if d.status == "done" else "◐" if d.status == "partial" else "?" if d.status == "needs_input" else "✗"
        lines.append(f"- {mark} {d.label()} — " + (f"section {numbering[d.id]}" if d.status in ("done", "partial")
                                                    else f"needs your input (section {numbering[d.id]})" if d.status == "needs_input"
                                                    else f"not completed: {d.detail}"))
    total = len(ds)
    verdict = (f"_All {total} requested items are answered._" if not incomplete else
               f"_Not complete: {len(incomplete)} of {total} requested items could not be fully answered "
               f"({'; '.join(incomplete)}). Everything else above is complete._")
    answer = "\n\n".join(sections) + "\n\n**What you asked, and where it is answered**\n" + "\n".join(lines) + "\n\n" + verdict
    return answer, charts, calls, incomplete


def answer(message: str, ds: list[Deliverable], df: pd.DataFrame, filename: str | None = None) -> dict:
    ctx = execute(ds, df, filename)
    text, charts, calls, incomplete = compose(ds, ctx)
    verification = [{"check": f"item {d.id}: {d.label()}", "ok": d.status == "done",
                     "detail": d.validation or d.detail or d.status} for d in ds]
    return {"answer": text, "charts": charts, "calls": calls, "incomplete": incomplete, "verification": verification,
            "deliverables": [{**d.to_dict(), "condition_met": getattr(d, "condition_met", True),
                              "result_values": _jsonable((ctx["results"].get(d.id) or {}).get("values")),
                              "evidence": _jsonable({k: v for k, v in (ctx["results"].get(d.id) or {}).items()
                                                     if k in ("forecast", "values", "summary", "chart", "chart_spec")})}
                             for d in ds],
            "results": ctx["results"], "errors": ctx.get("errors", [])}


_NUMTOK = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")


def _shows(text: str, v: float) -> bool:
    """`text` states the value `v` (rounded at the precision it is written with)."""
    for tok in _NUMTOK.findall(text or ""):
        try:
            x = float(tok.replace(",", ""))
        except ValueError:
            continue
        dp = len(tok.split(".")[1]) if "." in tok else 0
        if dp == 0 and abs(v) < 100 and not float(v).is_integer():
            continue
        if abs(round(v, dp) - x) <= 0.5 * 10 ** -dp + 1e-9:
            return True
    return False


def covers(legacy: dict, ds: list, df: pd.DataFrame) -> tuple[bool, list]:
    """Does another path's answer deliver every requested statistic / ranking / breakdown, with the right numbers?
    Each deliverable is computed independently here and looked for in that answer. (covered, missing labels)"""
    if not legacy or (legacy.get("plan") or {}).get("type") in ("clarify", "decline", "chat"):
        return False, [d.label() for d in ds]
    text = legacy.get("answer") or ""
    charts = legacy.get("charts") or []
    ctx = execute([d for d in ds], df)
    missing = []
    for d in ds:
        res = ctx["results"].get(d.id)
        if d.status not in ("done", "partial") or not res:
            # a part nobody can compute must at least be acknowledged, never silently dropped
            asked = re.search(r"“(.+?)”", d.detail or "")
            word = (asked.group(1).split()[-1] if asked else "")
            if d.kind == "chart" or not word or "no column for" not in (d.detail or ""):
                continue                  # only a genuinely missing column must be acknowledged by the other path
                #                           (the rule reader marks it as a statistic with no measure, the model
                #                           decomposer as an unparsed part: both say "no column for ...")
            if not re.search(rf"(no (?:column|data|field)[^.\n]*{re.escape(word)}|{re.escape(word)}[^.\n]*(?:not (?:in|found|"
                             rf"available)|no column|isn.t|doesn.t|cannot|can.t)|(?:couldn.t|could not|can.t|cannot)[^.\n]*"
                             rf"{re.escape(word)})", text, re.I):
                missing.append(d.label())
            continue
        if d.kind == "summary":
            if not (re.search(r"\b(rows?|records?)\b", text, re.I) and re.search(r"\bcolumns?\b", text, re.I)):
                missing.append(d.label())
            continue
        if d.kind in ("series", "chart", "forecast"):
            want = res.get("chart_spec") or res.get("chart") or {}
            ys = sorted(round(float(x.get(want.get("y")) or x.get("value") or 0), 6) for x in want.get("data", [])
                        if x.get("kind", "actual") == "actual")
            got = [sorted(round(float(x.get(c.get("y")) or x.get("value") or 0), 6) for x in c.get("data", [])
                          if x.get("kind", "actual") == "actual") for c in charts]
            if d.kind != "forecast" and ys not in got:
                missing.append(d.label())
            if d.kind == "forecast" and not any(c.get("type") == "forecast" for c in charts):
                missing.append(d.label())
            continue
        vals = [v for v in (res.get("values") or {}).values() if isinstance(v, (int, float)) and math.isfinite(v)]
        if d.kind in ("ranking", "breakdown"):
            names_ok = all(str(k) in text for k in (res.get("values") or {}))
            if not names_ok or not all(_shows(text, v) for v in vals):
                missing.append(d.label())
        elif not all(_shows(text, v) for v in vals):
            missing.append(d.label())
    return not missing, missing


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x
