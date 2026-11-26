"""Row-level calculations on an uploaded table. Each takes the table, the concept -> column mapping, the template's fixed
parameters and the user's values, and returns the same shape:

    {"status": "ok" | "insufficient_evidence", "metrics": [{key, label, baseline, scenario, delta, pct_change, unit}],
     "primary": key, "details": {...}, "assumptions": [...], "held_constant": [...], "warnings": [...], "rows_used": n,
     "uncertainty": {...}, "reason": str (when not ok)}

Nothing here is random and nothing here calls a model. Missing values are left out and counted, never filled. A relationship
is estimated only when the data supports it; otherwise the operation says so and returns no number.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

MIN_ASSOCIATION_ROWS = 30
MIN_TREND_PERIODS = 8
ALPHA = 0.05


class OperationError(ValueError):
    """The values cannot be used; the message says why."""


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _metric(key: str, label: str, baseline: float | None, scenario: float | None, unit: str = "") -> dict:
    delta = None if baseline is None or scenario is None else scenario - baseline
    pct = None if delta is None or not baseline else delta / abs(baseline) * 100
    return {"key": key, "label": label, "baseline": baseline, "scenario": scenario, "delta": delta,
            "pct_change": pct, "unit": unit}


def _finite(v: float | None) -> float | None:
    return None if v is None or not math.isfinite(float(v)) else float(v)


def _single(cols: dict, concept: str) -> str:
    got = cols.get(concept)
    if not got:
        raise OperationError(f"No column is mapped to '{concept}'")
    return got[0] if isinstance(got, list) else got


def _many(cols: dict, concept: str) -> list[str]:
    got = cols.get(concept)
    if not got:
        raise OperationError(f"No column is mapped to '{concept}'")
    return list(got) if isinstance(got, list) else [got]


def _is_fraction(x: pd.Series) -> bool:
    return bool(len(x)) and float(x.max()) <= 1.0


# ---------------------------------------------------------------- thresholds --

def threshold(df: pd.DataFrame, cols: dict, params: dict, values: dict) -> dict:
    """How many records meet a cut-off before and after it moves. Exact: it counts the records in the table.

    A record meets the cut-off when its value is >= the threshold. A column stored as a 0-1 fraction is compared with the
    threshold divided by 100 (the threshold is always typed in percent for a percentage concept); this is stated in the
    assumptions so it is never silent. Records with no value are excluded and counted."""
    concept, noun, rule = params["concept"], params.get("noun", "records"), params.get("rule", "the threshold")
    old, new = float(values["current_threshold"]), float(values["new_threshold"])
    columns = _many(cols, concept)
    metrics, details, notes, rows_used = [], {"columns": {}}, [], 0
    masks_old, masks_new = [], []
    for c in columns:
        x = _num(df, c)
        ok = x.notna()
        if not ok.any():
            raise OperationError(f"Column '{c}' has no numeric values")
        scale = 100.0 if params.get("percent_scale") and _is_fraction(x[ok]) else 1.0
        if scale != 1.0:
            notes.append(f"'{c}' is stored as a 0-1 fraction, so thresholds typed as percentages were divided by 100.")
        o, n = old / scale, new / scale
        le = params.get("direction") == "le"          # "within the target" rules (waiting time) are met by values at or below it
        meet_o, meet_n = ((x <= o) if le else (x >= o)) & ok, ((x <= n) if le else (x >= n)) & ok
        total = int(ok.sum())
        rows_used = max(rows_used, total)
        masks_old.append(meet_o.where(ok))
        masks_new.append(meet_n.where(ok))
        pre = f"{c}: " if len(columns) > 1 else ""
        metrics += [_metric(f"{c}__count", f"{pre}{noun} meeting {rule}", float(meet_o.sum()), float(meet_n.sum()), noun),
                    _metric(f"{c}__rate", f"{pre}share of {noun} meeting {rule}", meet_o.sum() / total * 100, meet_n.sum() / total * 100, "%")]
        details["columns"][c] = {"checked": total, "missing": int((~ok).sum()), "lost": int((meet_o & ~meet_n).sum()),
                                 "gained": int((~meet_o & meet_n).sum()), "observed_min": float(x[ok].min()), "observed_max": float(x[ok].max())}
        if not le and n > float(x[ok].max()):
            notes.append(f"The new threshold is above every observed value in '{c}', so no record would meet it.")
    if len(columns) > 1:
        both = pd.concat(masks_old, axis=1).dropna()
        both_new = pd.concat(masks_new, axis=1).dropna()
        if len(both):
            metrics.insert(0, _metric("all__count", f"{noun} meeting {rule} in every listed column",
                                      float(both.all(axis=1).sum()), float(both_new.all(axis=1).sum()), noun))
            rows_used = int(len(both))
    word = "at most" if params.get("direction") == "le" else "at least"
    primary = metrics[0]["key"]
    missing = sum(d["missing"] for d in details["columns"].values())
    return {"status": "ok", "metrics": metrics, "primary": primary, "details": details, "rows_used": rows_used,
            "assumptions": [f"A record meets the rule when its value is {word} the threshold ({old:g} before, {new:g} after).", *notes],
            "held_constant": [f"Every record's own value in {', '.join(columns)}"],
            "warnings": ([f"{missing} blank value(s) were left out of the counts, not filled."] if missing else []),
            "uncertainty": {"kind": "none", "note": "An exact count over the records in the table; there is no sampling error to estimate."}}


# --------------------------------------------------------------- shift a metric --

def shift_metric(df: pd.DataFrame, cols: dict, params: dict, values: dict) -> dict:
    """Move every record's value by a percentage or an amount and recompute the average (and, when a cut-off is given, how
    many records then meet it). A value is held inside its natural range when the concept has one (a percentage cannot pass 100).
    It applies the same change to every record: a what-if on the numbers, not a prediction that behaviour will change."""
    concept, noun = params["concept"], params.get("noun", "records")
    column = _single(cols, concept)
    change = float(values["change"])
    x = _num(df, column)
    ok = x.notna()
    if not ok.any():
        raise OperationError(f"Column '{column}' has no numeric values")
    xs = x[ok]
    mode = params.get("mode", "pct")
    new = xs * (1 + change / 100) if mode == "pct" else xs + change
    notes, clipped = [], 0
    if params.get("cap") == "percent_scale":
        top = 1.0 if _is_fraction(xs) else 100.0
        clipped = int((new > top).sum())
        new = new.clip(lower=0.0, upper=top)
        if clipped:
            notes.append(f"{clipped} record(s) would pass the maximum ({top:g}); they were held at it.")
    metrics = [_metric("mean", f"Average {column}", float(xs.mean()), float(new.mean()), params.get("unit", "")),
               _metric("total", f"Total {column}", float(xs.sum()), float(new.sum()), params.get("unit", ""))]
    thr = values.get("threshold")
    if thr is not None:
        scale = 100.0 if params.get("percent_scale") and _is_fraction(xs) else 1.0
        t = float(thr) / scale
        before, after = int((xs >= t).sum()), int((new >= t).sum())
        metrics += [_metric("meeting_count", f"{noun} at or above {float(thr):g}", float(before), float(after), noun),
                    _metric("meeting_rate", f"Share of {noun} at or above {float(thr):g}", before / len(xs) * 100, after / len(xs) * 100, "%")]
    verb = f"{change:g}% {'higher' if change >= 0 else 'lower'}" if mode == "pct" else f"{change:+g} {params.get('unit', '')}".strip()
    return {"status": "ok", "metrics": metrics, "primary": metrics[-1]["key"] if thr is not None else "mean",
            "details": {"column": column, "clipped": clipped}, "rows_used": int(ok.sum()),
            "assumptions": [f"Every record's {column} is {verb}; nothing else is allowed to react.", *notes],
            "held_constant": [f"All columns other than {column}"],
            "warnings": [f"{int((~ok).sum())} blank value(s) were left out."] if (~ok).any() else [],
            "uncertainty": {"kind": "none", "note": "Exact arithmetic on the records in the table, given the change you set. It says nothing about whether the change is achievable."}}


# ------------------------------------------------------------------ association --

def association(df: pd.DataFrame, cols: dict, params: dict, values: dict) -> dict:
    """Estimate what a change in X goes with in Y from the table's own history (simple least squares).

    Only done when there are at least 30 complete pairs and the slope differs from zero at the 5% level; otherwise the
    result is `insufficient_evidence` and carries no estimate. The interval is the 95% confidence interval of the slope,
    scaled by the size of the change. The estimate describes an association in past data: it does not show that changing X
    causes Y to change, it assumes the relationship holds beyond the observed range, and other influences are ignored."""
    xc, yc = _single(cols, params["x"]), _single(cols, params["y"])
    pair = pd.DataFrame({"x": _num(df, xc), "y": _num(df, yc)}).dropna()
    n = len(pair)
    base = {"details": {"x": xc, "y": yc, "pairs": n}, "rows_used": n, "metrics": [], "primary": None}
    if n < MIN_ASSOCIATION_ROWS:
        return {**base, "status": "insufficient_evidence",
                "reason": f"Only {n} complete rows of {xc} and {yc}; at least {MIN_ASSOCIATION_ROWS} are needed to estimate a relationship."}
    if pair["x"].nunique() < 3:
        return {**base, "status": "insufficient_evidence", "reason": f"{xc} takes fewer than three distinct values, so its effect cannot be estimated."}
    fit = stats.linregress(pair["x"], pair["y"])
    if not math.isfinite(fit.pvalue) or fit.pvalue >= ALPHA:
        return {**base, "status": "insufficient_evidence", "details": {**base["details"], "r": _finite(fit.rvalue), "p_value": _finite(fit.pvalue)},
                "reason": (f"In this data {yc} shows no reliable relationship with {xc} (correlation {fit.rvalue:.2f}, p = {fit.pvalue:.2f}), "
                           f"so an effect of changing {xc} on {yc} cannot be estimated from it.")}
    change = float(values["change"])
    pct_mode = params.get("mode", "pct") == "pct"
    mean_x, mean_y = float(pair["x"].mean()), float(pair["y"].mean())
    dx = mean_x * change / 100 if pct_mode else change
    crit = stats.t.ppf(1 - ALPHA / 2, n - 2)
    lo_s, hi_s = fit.slope - crit * fit.stderr, fit.slope + crit * fit.stderr
    est = fit.slope * dx
    lo, hi = sorted((lo_s * dx, hi_s * dx))
    moved = pair["x"] * (1 + change / 100) if pct_mode else pair["x"] + change
    beyond = int((moved > pair["x"].max()).sum())
    warnings = []
    if beyond:
        warnings.append(f"{beyond} of {n} records would move beyond the highest {xc} ever observed; the estimate extrapolates there.")
    metrics = [_metric("mean_x", f"Average {xc}", mean_x, mean_x + dx, params.get("x_unit", "")),
               _metric("mean_y", f"Average {yc} per record (estimated)", mean_y, mean_y + est, params.get("y_unit", "")),
               _metric("total_y", f"Total {yc} across {n} records (estimated)", mean_y * n, (mean_y + est) * n, params.get("y_unit", ""))]
    return {"status": "ok", "metrics": metrics, "primary": "mean_y", "rows_used": n,
            "details": {"x": xc, "y": yc, "pairs": n, "slope": float(fit.slope), "r": float(fit.rvalue), "r_squared": float(fit.rvalue ** 2),
                        "p_value": float(fit.pvalue), "slope_ci95": [float(lo_s), float(hi_s)], "change_in_x": float(dx)},
            "assumptions": [f"The linear relationship seen in the {n} past records ({yc} changes by {fit.slope:.4g} per unit of {xc}) continues.",
                            f"Nothing other than {xc} changes."],
            "held_constant": [f"Every column other than {xc}"], "warnings": warnings,
            "uncertainty": {"kind": "confidence_interval", "level": 0.95, "target": "mean_y", "change_low": float(lo), "change_high": float(hi),
                            "note": "95% confidence interval of the fitted slope, scaled by the change. It reflects sampling error only, not omitted influences or a change in the relationship."}}


# --------------------------------------------------------------------- forecast --

def forecast_trend(df: pd.DataFrame, cols: dict, params: dict, values: dict) -> dict:
    """Project a count (or total) per period forward with a straight-line trend, with a 95% prediction interval.

    Needs at least 8 periods. The forecast assumes the past trend continues; it says nothing about a policy change, a seasonal
    pattern, or a one-off event. It is labelled a forecast and never an exact value."""
    dc = _single(cols, params["date"])
    d = pd.to_datetime(df[dc], errors="coerce")
    freq = params.get("freq", "M")
    if params.get("value"):
        vc = _single(cols, params["value"])
        series = pd.to_numeric(df[vc], errors="coerce").groupby(d.dt.to_period(freq)).sum()
        what = f"Total {vc}"
    else:
        series = df.groupby(d.dt.to_period(freq)).size()
        what = params.get("noun", "Records")
    series = series[series.index.notna()]
    horizon = int(values.get("periods_ahead", 1))
    base = {"details": {"periods": int(len(series))}, "metrics": [], "primary": None, "rows_used": int(d.notna().sum())}
    if horizon < 1 or horizon > 24:
        raise OperationError("Periods ahead must be between 1 and 24")
    if len(series) < MIN_TREND_PERIODS:
        return {**base, "status": "insufficient_evidence",
                "reason": f"Only {len(series)} period(s) of history; at least {MIN_TREND_PERIODS} are needed for a trend forecast."}
    t = np.arange(len(series), dtype=float)
    y = series.to_numpy(dtype=float)
    fit = stats.linregress(t, y)
    resid = y - (fit.intercept + fit.slope * t)
    dof = len(y) - 2
    s = math.sqrt(float((resid ** 2).sum()) / dof)
    t_new = len(y) - 1 + horizon
    se = s * math.sqrt(1 + 1 / len(y) + (t_new - t.mean()) ** 2 / float(((t - t.mean()) ** 2).sum()))
    crit = stats.t.ppf(1 - ALPHA / 2, dof)
    point = float(fit.intercept + fit.slope * t_new)
    m = _metric("forecast", f"{what} in period +{horizon}", float(y[-1]), point, params.get("unit", ""))
    return {"status": "ok", "metrics": [m], "primary": "forecast", "rows_used": base["rows_used"],
            "details": {"periods": int(len(y)), "slope_per_period": float(fit.slope), "last_period": str(series.index[-1]), "horizon": horizon,
                        "history": [{"period": str(p), "value": float(v)} for p, v in series.items()]},
            "assumptions": [f"The straight-line trend of the last {len(y)} periods continues for {horizon} more.",
                            "No policy change, seasonal pattern or one-off event is included."],
            "held_constant": ["Everything that did not change the historical trend"], "warnings": [],
            "uncertainty": {"kind": "prediction_interval", "level": 0.95, "target": "forecast", "low": point - crit * se, "high": point + crit * se,
                            "note": "95% prediction interval of a linear trend; it assumes the trend and its scatter stay the same."}}


OPERATIONS = {"threshold": threshold, "shift_metric": shift_metric, "association": association, "forecast_trend": forecast_trend}


# ----------------------------------------------------------- derived baselines --

def derive_input(rule: str, df: pd.DataFrame, cols: dict, profile: dict) -> tuple[float | None, str | None]:
    """A baseline value read from the table, with how it was read. None when the table cannot supply it (never guessed)."""
    kind, _, arg = rule.partition(":")
    try:
        if kind == "row_count":
            return float(len(df)), f"{len(df):,} rows"
        if kind == "entity_count":
            ent = (profile.get("entities") or [{}])[0].get("column")
            if ent and ent in df.columns:
                return float(df[ent].nunique(dropna=True)), f"distinct values of {ent}"
            return float(len(df)), "one row per record"
        if kind in ("mean", "sum"):
            c = _single(cols, arg)
            x = _num(df, c).dropna()
            return (float(x.mean() if kind == "mean" else x.sum()), f"{kind} of {c}") if len(x) else (None, None)
        if kind == "rows_per_day":
            d = pd.to_datetime(df[_single(cols, arg)], errors="coerce").dropna()
            days = (d.dt.normalize().max() - d.dt.normalize().min()).days + 1 if len(d) else 0
            return (float(len(d) / days), f"{len(d):,} records over {days} days") if days else (None, None)
    except (OperationError, KeyError):
        return None, None
    return None, None
