"""Time-series forecasting for a measure over regular periods (one entity, a group, or the whole table).

What it guarantees:
  * the data's own grain is read first (one row per entity per month is a MONTHLY total, never a day);
  * candidate methods are compared on chronological, rolling-origin backtests (each forecast uses only earlier
    periods, so nothing leaks from the future), against the naive "same as last period" baseline;
  * the method with the lowest backtest error is chosen; its skill against the naive baseline is reported;
  * a prediction range is given only when there are enough backtest errors to make one defensible (empirical
    quantiles of those errors, by horizon); otherwise it says so;
  * with too little history it returns an honest "insufficient" result instead of a number.
Methods: naive, historical mean, drift (random walk with drift), linear trend (least squares on time),
simple exponential smoothing (alpha chosen on the backtest), seasonal naive (when two full seasons exist).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_POINTS = 6           # fewer periods than this: no forecast, said plainly
MIN_INTERVAL_ERRORS = 5  # backtest errors needed per horizon step before a range is shown
LEVEL = 0.8              # the prediction range covers the middle 80% of past forecast errors
FREQ_NAMES = {"D": "day", "W": "week", "M": "month", "Q": "quarter", "Y": "year"}
FREQ_ALIASES = {"D": "D", "W": "W-MON", "M": "MS", "Q": "QS", "Y": "YS"}
SEASON = {"M": 12, "Q": 4, "W": 52, "D": 7}


# ------------------------------------------------------------------- grain --
def infer_grain(df: pd.DataFrame, date: str, entity: str | None = None) -> dict:
    """How often records occur: the typical step between consecutive dates (per entity when there is one), and
    whether each entity has one record per period (period totals) or many (events to be summed)."""
    d = pd.to_datetime(df[date], errors="coerce")
    frame = pd.DataFrame({"d": d, "e": df[entity] if entity and entity in df.columns else "_all"}).dropna(subset=["d"])
    steps = []
    for _, g in frame.groupby("e"):
        u = np.sort(g["d"].unique())
        if len(u) >= 2:
            steps += list(np.diff(u).astype("timedelta64[D]").astype(float))
    if not steps:
        return {"freq": None, "label": None, "one_per_period": False, "step_days": None}
    med = float(np.median(steps))
    freq = "D" if med <= 1.5 else "W" if med <= 8 else "M" if med <= 35 else "Q" if med <= 100 else "Y"
    per = frame.assign(p=frame["d"].dt.to_period(freq if freq != "W" else "W")).groupby(["e", "p"]).size()
    return {"freq": freq, "label": FREQ_NAMES[freq], "one_per_period": bool((per == 1).mean() >= 0.95),
            "step_days": med}


def to_series(df: pd.DataFrame, date: str, value: str, freq: str, agg: str = "sum") -> pd.Series:
    """The measure per period, on a regular calendar from first to last period. Periods with no records are 0 for a
    sum of events and missing for anything else (a missing month is not a month of zero)."""
    d = pd.to_datetime(df[date], errors="coerce")
    v = pd.to_numeric(df[value], errors="coerce")
    frame = pd.DataFrame({"p": d.dt.to_period(freq), "v": v}).dropna()
    s = frame.groupby("p")["v"].agg(agg)
    if s.empty:
        return s
    full = pd.period_range(s.index.min(), s.index.max(), freq=freq)
    s = s.reindex(full)
    return s.fillna(0.0) if agg == "sum" else s


# ----------------------------------------------------------------- methods --
MIN_SKILL = 0.05          # a method replaces the naive baseline only if its out-of-sample MAE is >= 5% lower


def _ma(y, h, k=3, **_):
    return np.repeat(float(np.mean(y[-k:])), h)


def _naive(y, h, **_):
    return np.repeat(y[-1], h)


def _mean(y, h, **_):
    return np.repeat(np.mean(y), h)


def _drift(y, h, **_):
    slope = (y[-1] - y[0]) / max(1, len(y) - 1)
    return y[-1] + slope * np.arange(1, h + 1)


def _trend(y, h, **_):
    t = np.arange(len(y))
    b, a = np.polyfit(t, y, 1)
    return a + b * np.arange(len(y), len(y) + h)


def _ses(y, h, alpha=0.5, **_):
    level = y[0]
    for x in y[1:]:
        level = alpha * x + (1 - alpha) * level
    return np.repeat(level, h)


def _snaive(y, h, m=12, **_):
    return np.array([y[len(y) - m + (i % m)] for i in range(h)])


@dataclass
class Candidate:
    name: str
    label: str
    fn: object
    kw: dict


def _candidates(n: int, freq: str) -> list[Candidate]:
    c = [Candidate("naive", "same as the last period (naive baseline)", _naive, {}),
         Candidate("mean", "historical average", _mean, {}),
         Candidate("moving_average", "average of the last 3 periods (moving average)", _ma, {"k": 3}),
         Candidate("drift", "last value plus the average change per period (drift)", _drift, {}),
         Candidate("trend", "straight-line trend (least squares)", _trend, {})]
    c += [Candidate(f"ses_{a}", f"exponential smoothing (alpha {a})", _ses, {"alpha": a}) for a in (0.3, 0.6)]
    m = SEASON.get(freq)
    if m and n >= 2 * m + 2:
        c.append(Candidate("seasonal_naive", f"same period last {FREQ_NAMES.get(freq, 'season')}-cycle (seasonal naive)",
                           _snaive, {"m": m}))
    return c


# ------------------------------------------------------------------ backtest --
def backtest(y: np.ndarray, h: int, cands: list[Candidate]) -> dict:
    """Rolling-origin evaluation: for every origin from about half the history, fit on y[:origin] only and forecast
    the next h periods. Returns, per method, the mean absolute error and the signed errors by horizon step."""
    n = len(y)
    start = max(4, n // 2)
    out = {}
    for c in cands:
        errs = {k: [] for k in range(1, h + 1)}
        for o in range(start, n):
            train = y[:o]
            if c.name == "seasonal_naive" and len(train) < c.kw["m"]:
                continue
            steps = min(h, n - o)
            f = c.fn(train, steps, **c.kw)
            for k in range(steps):
                errs[k + 1].append(float(y[o + k] - f[k]))
        flat = [abs(e) for v in errs.values() for e in v]
        if flat:
            out[c.name] = {"mae": float(np.mean(flat)), "rmse": float(np.sqrt(np.mean(np.square(flat)))),
                           "errors": errs, "n": len(flat)}
    return out


def forecast(s: pd.Series, horizon: int, freq: str, target: float | None = None) -> dict:
    """Forecast `horizon` periods after the last one. Returns a JSON-safe dict with history, chosen method, the
    backtest table (every candidate vs the naive baseline), forecasts with ranges when defensible, and -- with a
    target -- for each forecast period whether the target is likely, unlikely or within the range."""
    s = s.dropna()
    hist = [{"period": str(p), "value": float(v)} for p, v in s.items()]
    if len(s) < MIN_POINTS:
        return {"status": "insufficient", "history": hist, "points": len(s), "needed": MIN_POINTS,
                "reason": f"only {len(s)} {FREQ_NAMES.get(freq, 'period')}s of history; at least {MIN_POINTS} are needed "
                          "to test a forecasting method on past data before trusting it"}
    y = s.to_numpy(float)
    h = max(1, int(horizon))
    cands = _candidates(len(y), freq)
    bt = backtest(y, h, cands)
    if "naive" not in bt:
        return {"status": "insufficient", "history": hist, "points": len(s), "needed": MIN_POINTS,
                "reason": "not enough history to test any method out of sample"}
    lowest = min(bt, key=lambda k: (bt[k]["mae"], k != "naive"))
    naive_mae = bt["naive"]["mae"]
    # selection is honest: with several candidates the lowest backtest error beats the baseline by chance alone, so
    # a method must beat the naive baseline by a margin before it is preferred to it
    if lowest != "naive" and naive_mae > 0 and bt[lowest]["mae"] > (1 - MIN_SKILL) * naive_mae:
        best = "naive"
        selection = (f"no method beat the naive baseline by at least {MIN_SKILL:.0%} on past data (best: "
                     f"{next(c.label for c in cands if c.name == lowest)}), so the naive baseline is used")
    else:
        best = lowest
        selection = ("the naive baseline had the lowest error on past data" if best == "naive" else
                     f"lowest error on past data, {1 - bt[best]['mae'] / naive_mae:.0%} below the naive baseline"
                     if naive_mae > 0 else "lowest error on past data")
    chosen = next(c for c in cands if c.name == best)
    pred = chosen.fn(y, h, **chosen.kw)
    last = s.index[-1]
    future = [last + k for k in range(1, h + 1)]
    lo_q, hi_q = (1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2
    points = []
    for k, (p, f) in enumerate(zip(future, pred), 1):
        e = bt[best]["errors"].get(k) or []
        pt = {"period": str(p), "forecast": float(f), "lower": None, "upper": None, "errors_used": len(e)}
        if len(e) >= MIN_INTERVAL_ERRORS:
            pt["lower"] = float(f + np.quantile(e, lo_q))
            pt["upper"] = float(f + np.quantile(e, hi_q))
        if target is not None:
            if len(e) >= MIN_INTERVAL_ERRORS:
                share = float(np.mean([(f + x) >= target for x in e]))
                pt["chance_at_or_above_target"] = share
                pt["target_verdict"] = ("likely" if share >= 0.8 else "unlikely" if share <= 0.2 else "uncertain")
            else:
                pt["target_verdict"] = "at or above" if f >= target else "below"
                pt["chance_at_or_above_target"] = None
            pt["gap_to_target"] = float(f - target)
        points.append(pt)
    scale = float(np.mean(np.abs(np.diff(y)))) if len(y) > 1 else 0.0     # in-sample one-step naive error (MASE scale)
    table = sorted(({"method": c.name, "label": c.label, "mae": bt[c.name]["mae"], "rmse": bt[c.name]["rmse"],
                     "mase": (bt[c.name]["mae"] / scale) if scale > 0 else None,
                     "vs_naive": (bt[c.name]["mae"] / naive_mae) if naive_mae > 0 else None}
                    for c in cands if c.name in bt), key=lambda r: r["mae"])
    return {"status": "ok", "freq": freq, "unit_label": FREQ_NAMES.get(freq), "history": hist, "points": len(s),
            "horizon": h, "method": best, "method_label": chosen.label, "backtest": table,
            "backtest_mae": bt[best]["mae"], "naive_mae": naive_mae, "backtest_rmse": bt[best]["rmse"],
            "mase": (bt[best]["mae"] / scale) if scale > 0 else None, "selection": selection,
            "skill_vs_naive": (1 - bt[best]["mae"] / naive_mae) if naive_mae > 0 else 0.0,
            "level": LEVEL, "forecasts": points, "target": target,
            "validation": f"rolling-origin backtest: {bt[best]['n']} out-of-sample forecasts from {len(s)} periods; "
                          "each used only earlier periods"}


def finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def series_checks(dates: pd.Series, values: pd.Series, freq: str, s: pd.Series) -> dict:
    """What was done to the data before forecasting -- reported, never silent: records combined per period, periods
    with no records, missing values dropped, and outlying periods (robust z > 3.5; kept, not removed)."""
    d = pd.to_datetime(dates, errors="coerce")
    v = pd.to_numeric(values, errors="coerce")
    per = d.dt.to_period(freq)
    counts = per.value_counts()
    full = pd.period_range(per.min(), per.max(), freq=freq) if per.notna().any() else []
    missing = [str(p) for p in full if p not in counts.index]
    vals = s.dropna().to_numpy(float)
    med = float(np.median(vals)) if len(vals) else 0.0
    mad = float(np.median(np.abs(vals - med))) if len(vals) else 0.0
    out_idx = [str(p) for p, x in s.dropna().items() if mad > 0 and abs(0.6745 * (x - med) / mad) > 3.5]
    return {"records": int(v.notna().sum()), "periods": int(len(full)), "multi_record_periods": int((counts > 1).sum()),
            "missing_periods": missing, "missing_values": int(v.isna().sum()), "outliers": out_idx}


def describe_checks(c: dict, freq: str) -> str:
    name = FREQ_NAMES.get(freq, "period")
    parts = [f"{c['records']} records over {c['periods']} {name}s"]
    if c["multi_record_periods"]:
        parts.append(f"{c['multi_record_periods']} {name}s had several records, combined into one value per {name}")
    if c["missing_periods"]:
        mp = c["missing_periods"]
        parts.append(f"{len(mp)} {name}(s) with no records ({', '.join(mp[:3])}{'…' if len(mp) > 3 else ''})")
    if c["missing_values"]:
        parts.append(f"{c['missing_values']} missing values left out")
    if c["outliers"]:
        parts.append(f"unusual {name}(s) kept in the history: {', '.join(c['outliers'][:3])}")
    return "; ".join(parts)
