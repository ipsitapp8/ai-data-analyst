"""Price elasticity of demand, estimated from a dataset's own history.

The what-if engine takes price elasticity as a number the user types. This module can *suggest* one from observed
price and quantity, with the uncertainty that goes with it. It never changes a scenario by itself: the manual value
stays authoritative until the user chooses the estimate.

Model
-----
A log-log regression, ln(quantity) = a + e x ln(price) + controls, so the coefficient `e` on ln(price) is a
constant-elasticity estimate (a 1% price rise changes quantity by about e%).

* Observations. With a date column, rows are aggregated to periods (day, week, month or quarter): the period's
  quantity is the sum, its price the quantity-weighted average. Without one, each row is an observation, which mixes
  products and customers and is flagged as such. With a segment column, observations are (period, segment) and each
  segment gets its own intercept, so price differences between products are not read as demand curves.
* Controls. A linear time trend when there are at least 12 observations, month-of-year (or quarter) dummies when the
  data spans at least two years, segment intercepts. When a control cannot be supported (too few observations, or it is
  collinear with price) it is dropped and the result says so. Seasonality that is not controlled is a stated caveat.
* Inference. Least squares (app/stats/series.py, the same code the root-cause analysis uses) with Newey-West
  standard errors for a single time series, HC3 otherwise, and t-based intervals.
* Refusals. Too few usable observations, no price variation, or an unestimable design returns
  `status: "not_estimated"` with the reasons. Nothing is estimated from unsuitable data.

What it does not establish: elasticity is estimated from observational history, so promotions, competitor moves,
stock-outs and anything else that changed with price are not separated out. The estimate applies inside the observed
price range and says so.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as st

from app import metric_engine as me
from app.scenarios import ScenarioError
from app.stats import series as ts

ELASTICITY_VERSION = "elasticity.1"
MIN_OBSERVATIONS = 8
RECOMMENDED_OBSERVATIONS = 24
MIN_PRICE_VARIATION = 0.02          # standard deviation of ln(price): under 2% there is nothing to learn from
TREND_MIN_OBSERVATIONS = 12
SEASON_MIN_MONTHS = 24
MAX_SEGMENTS = 20
MIN_PER_SEGMENT = 3
OUTLIER_T = 3.0
WIDE_INTERVAL = 1.5
GRANULARITIES = ("auto", "day", "week", "month", "quarter")
ELASTICITY_BOUNDS = (-10.0, 10.0)   # the range the what-if engine accepts for the assumption
AGGREGATION_CHECK_MAX_ROWS = 200_000
LIMITATIONS = [
    "Observational history: promotions, competitor prices, stock-outs, seasonality and anything else that moved with price "
    "are not separated out unless listed under controls.",
    "A single constant elasticity is fitted (log-log). Real demand can be more or less sensitive at different price levels.",
    "The estimate describes the observed price range only. Do not apply it far outside that range.",
    "Without a segment column, everything sold in a period is combined into one quantity and one average price, so a change "
    "in the mix between products or regions looks like a change in price.",
]


class ElasticityError(ScenarioError):
    """The request cannot be run (bad column, bad option); the message says why."""


def _numeric(df: pd.DataFrame, col: str, what: str) -> pd.Series:
    if col not in df.columns:
        raise ElasticityError(f"The {what} column '{col}' does not exist in this dataset")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ElasticityError(f"The {what} column '{col}' is not numeric")
    return df[col].astype(float)


def _period_key(dates: pd.Series, granularity: str) -> pd.Series:
    if granularity == "day":
        return dates.dt.normalize()
    freq = {"week": "W-SUN", "month": "M", "quarter": "Q"}[granularity]
    return dates.dt.to_period(freq).dt.start_time


def choose_granularity(dates: pd.Series) -> str:
    """The coarsest period that still gives at least RECOMMENDED_OBSERVATIONS periods; the finest if none does."""
    for g in ("quarter", "month", "week"):
        if _period_key(dates, g).nunique() >= RECOMMENDED_OBSERVATIONS:
            return g
    return "day"


def prepare(df: pd.DataFrame, price_column: str, quantity_column: str, date_column: str | None,
            segment_column: str | None, granularity: str, filters: list[dict] | None) -> dict:
    """Clean and aggregate to the observations the regression uses, recording what was left out and why."""
    if granularity not in GRANULARITIES:
        raise ElasticityError(f"granularity must be one of: {', '.join(GRANULARITIES)}")
    if price_column == quantity_column:
        raise ElasticityError("The price and quantity columns must be different columns")
    try:
        data = me.apply_filters(df, filters)
    except me.FormulaError as e:
        raise ElasticityError(str(e)) from e
    price, qty = _numeric(data, price_column, "price"), _numeric(data, quantity_column, "quantity")
    frame = pd.DataFrame({"price": price, "qty": qty})
    if date_column:
        if date_column not in data.columns:
            raise ElasticityError(f"The date column '{date_column}' does not exist in this dataset")
        dates = pd.to_datetime(data[date_column], errors="coerce")
        if dates.notna().mean() < 0.9:
            raise ElasticityError(f"The date column '{date_column}' is not made of dates")
        frame["date"] = dates
    if segment_column:
        if segment_column not in data.columns:
            raise ElasticityError(f"The segment column '{segment_column}' does not exist in this dataset")
        frame["segment"] = data[segment_column].astype(str)
    excluded = {"rows_in": int(len(frame)), "missing_value": 0, "non_finite": 0, "price_not_positive": 0,
                "quantity_zero_or_negative": 0}
    required = ["price", "qty"] + (["date"] if date_column else [])
    missing = frame[required].isna().any(axis=1)
    excluded["missing_value"] = int(missing.sum())
    frame = frame[~missing]
    finite = np.isfinite(frame["price"]) & np.isfinite(frame["qty"])
    excluded["non_finite"] = int((~finite).sum())
    frame = frame[finite]
    bad_price = frame["price"] <= 0
    excluded["price_not_positive"] = int(bad_price.sum())
    frame = frame[~bad_price]
    bad_qty = frame["qty"] <= 0
    excluded["quantity_zero_or_negative"] = int(bad_qty.sum())
    frame = frame[~bad_qty]
    out: dict[str, Any] = {"excluded": excluded, "granularity": None, "unit": "row"}
    if date_column:
        gran = choose_granularity(frame["date"]) if granularity == "auto" and len(frame) else (
            "month" if granularity == "auto" else granularity)
        out["granularity"], out["unit"] = gran, gran
        frame = frame.assign(period=_period_key(frame["date"], gran))
        keys = ["period"] + (["segment"] if segment_column else [])
        frame = frame.assign(value=frame["price"] * frame["qty"])
        agg = frame.groupby(keys, sort=True).agg(qty=("qty", "sum"), value=("value", "sum"), rows=("qty", "size")).reset_index()
        agg["price"] = agg["value"] / agg["qty"]
        obs = agg.drop(columns=["value"])
    else:
        obs = frame.assign(rows=1, period=pd.NaT).reset_index(drop=True)
    out["observations"] = obs.reset_index(drop=True)
    return out


def _design(obs: pd.DataFrame, date_column: str | None, segment_column: str | None, gran: str | None) -> dict:
    """Design matrix: intercept, ln(price), then whichever controls the data supports."""
    n = len(obs)
    x = np.log(obs["price"].to_numpy(float))
    y = np.log(obs["qty"].to_numpy(float))
    cols, names = [np.ones(n), x], ["intercept", "ln_price"]
    notes: dict[str, Any] = {"trend": "not_used", "seasonality": "not_applicable", "segments": "not_used"}
    keep = np.ones(n, dtype=bool)
    if segment_column:
        counts = obs["segment"].value_counts()
        usable = [s for s in counts.index if counts[s] >= MIN_PER_SEGMENT][:MAX_SEGMENTS]
        dropped = int((~obs["segment"].isin(usable)).sum())
        keep = obs["segment"].isin(usable).to_numpy()
        if dropped:
            notes["segments_dropped_observations"] = dropped
        if len(usable) >= 2:
            for s in usable[1:]:
                cols.append((obs["segment"] == s).to_numpy(float))
                names.append(f"segment={s}")
            notes["segments"] = f"{len(usable)} segment intercepts"
        else:
            notes["segments"] = "only one segment has enough observations; no intercepts needed"
    if date_column:
        periods = pd.to_datetime(obs["period"])
        span_months = (periods.max().year - periods.min().year) * 12 + periods.max().month - periods.min().month + 1
        if n >= TREND_MIN_OBSERVATIONS:
            ordinal = (periods - periods.min()).dt.days.to_numpy(float)
            cols.append(ordinal / max(ordinal.max(), 1.0))
            names.append("trend")
            notes["trend"] = "linear time trend"
        else:
            notes["trend"] = f"not controlled: fewer than {TREND_MIN_OBSERVATIONS} observations"
        if span_months >= SEASON_MIN_MONTHS and gran in ("day", "week", "month", "quarter"):
            label = periods.dt.quarter if gran == "quarter" else periods.dt.month
            levels = sorted(label.unique())
            if n >= len(cols) + len(levels) + 4 and len(levels) > 1:
                for lv in levels[1:]:
                    cols.append((label == lv).to_numpy(float))
                    names.append(f"{'quarter' if gran == 'quarter' else 'month'}={lv}")
                notes["seasonality"] = f"{'quarter' if gran == 'quarter' else 'month'}-of-year dummies"
            else:
                notes["seasonality"] = "not controlled: too few observations for seasonal dummies"
        else:
            notes["seasonality"] = (f"not controlled: the data spans {span_months} months and at least {SEASON_MIN_MONTHS} "
                                    "are needed to tell seasonality from price")
    X = np.column_stack(cols)
    return {"X": X[keep], "y": y[keep], "names": names, "notes": notes, "keep": keep, "x_price": x[keep]}


def _drop_collinear(d: dict) -> dict:
    """If the design is rank deficient, drop whole control groups (seasonality, then trend, then segment intercepts)
    until it has full rank. The intercept and ln(price) are never dropped."""
    X, names, notes = d["X"], list(d["names"]), dict(d["notes"])
    groups = (("seasonality", ("month=", "quarter=")), ("trend", ("trend",)), ("segments", ("segment=",)))
    for key, prefixes in groups:
        if np.linalg.matrix_rank(X) >= X.shape[1]:
            break
        idx = [i for i, n in enumerate(names) if n.startswith(prefixes)]
        if not idx:
            continue
        X = np.delete(X, idx, axis=1)
        names = [n for i, n in enumerate(names) if i not in idx]
        notes[key] = "dropped: collinear with price or another control"
    return {**d, "X": X, "names": names, "notes": notes}


def _fit(d: dict, level: float, ordered: bool) -> dict:
    X, y = d["X"], d["y"]
    n = len(y)
    cov = "HAC" if ordered else "HC3"
    lags = ts.newey_west_lag(n) if ordered else 0
    fit = ts.ols(X, y, cov=cov, lags=lags)
    coef = ts.coefficient(fit, 1, level)
    k = X.shape[1]
    r2 = fit["r_squared"]
    adj = None if r2 is None or n - k <= 0 else 1 - (1 - r2) * (n - 1) / (n - k)
    xtx_inv = np.linalg.inv(X.T @ X)
    s2 = float(fit["resid"] @ fit["resid"]) / (n - k)
    lev = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    with np.errstate(divide="ignore", invalid="ignore"):
        studentized = fit["resid"] / np.sqrt(s2 * np.clip(1 - lev, 1e-12, None)) if s2 > 0 else np.zeros(n)
    cooks = studentized ** 2 * lev / (k * np.clip(1 - lev, 1e-12, None))
    return {"fit": fit, "coef": coef, "cov": cov, "lags": lags, "r2": r2, "adj_r2": adj, "studentized": studentized,
            "cooks": cooks, "leverage": lev, "n": n, "k": k}


def estimate(df: pd.DataFrame, price_column: str, quantity_column: str, *, date_column: str | None = None,
             segment_column: str | None = None, granularity: str = "auto", filters: list[dict] | None = None,
             exclude_outliers: bool = False, level: float = 0.95, verify: bool = True) -> dict:
    if isinstance(level, bool) or not isinstance(level, (int, float)) or not 0.5 <= level < 1:
        raise ElasticityError("level must be between 0.5 and 1 (for example 0.95)")
    prep = prepare(df, price_column, quantity_column, date_column, segment_column, granularity, filters)
    obs, gran = prep["observations"], prep["granularity"]
    base = {"kind": "elasticity_estimate", "data_kind": "statistical_estimate", "engine_version": ELASTICITY_VERSION,
            "inputs": {"price_column": price_column, "quantity_column": quantity_column, "date_column": date_column,
                       "segment_column": segment_column, "granularity": gran or "none (each row is an observation)",
                       "level": level, "exclude_outliers": bool(exclude_outliers), "filters": filters or []},
            "data": {**prep["excluded"], "observations": int(len(obs)), "unit": prep["unit"]},
            "limitations": list(LIMITATIONS)}
    reasons = _insufficient(obs, prep)
    if reasons:
        return {**base, "status": "not_estimated", "reasons": reasons, "elasticity": None,
                "reliability": {"level": "not_estimated", "flags": reasons,
                                "recommendation": "Do not use an estimate: " + reasons[0]},
                "verification": None}
    design = _drop_collinear(_design(obs, date_column, segment_column, gran))
    ordered = bool(date_column) and not segment_column
    try:
        main = _fit(design, level, ordered)
    except ts.RegressionError as e:
        reasons = [f"the regression cannot be estimated from this data ({e})"]
        return {**base, "status": "not_estimated", "reasons": reasons, "elasticity": None,
                "reliability": {"level": "not_estimated", "flags": reasons, "recommendation": "Do not use an estimate."},
                "verification": None}
    flagged = _flag_outliers(design, level, ordered, main)
    outliers = {"count": int(flagged.sum()), "threshold": f"|studentized residual| > {OUTLIER_T:g}, re-checked after each removal",
                "high_influence": int((main["cooks"] > 4 / main["n"]).sum()), "without_outliers": None,
                "excluded_from_primary": False}
    used, used_design = main, design
    if flagged.any() and main["n"] - int(flagged.sum()) >= MIN_OBSERVATIONS:
        trimmed = {**design, "X": design["X"][~flagged], "y": design["y"][~flagged]}
        try:
            alt = _fit(trimmed, level, ordered)
            outliers["without_outliers"] = {"estimate": alt["coef"]["estimate"], "ci_low": alt["coef"]["low"],
                                            "ci_high": alt["coef"]["high"], "n": alt["n"]}
            if exclude_outliers:
                used, used_design = alt, trimmed
                outliers["excluded_from_primary"] = True
        except ts.RegressionError:
            pass
    coef = used["coef"]
    names = used_design["names"]
    resid = used["fit"]["resid"]
    r1 = ts.autocorr1(resid) if ordered else None
    price_x = used_design["x_price"] if not (outliers["excluded_from_primary"]) else used_design["x_price"][~flagged]
    sd_lnp = float(np.std(price_x, ddof=1)) if len(price_x) > 1 else 0.0
    diagnostics = {
        "observations": used["n"], "parameters": used["k"], "r_squared": used["r2"], "adjusted_r_squared": used["adj_r2"],
        "standard_errors": ("Newey-West (lags " + str(used["lags"]) + ")") if used["cov"] == "HAC" else "HC3 (robust to unequal variance)",
        "residual_autocorrelation_lag1": r1, "price_variation_sd_ln": sd_lnp,
        "price_range": [float(np.exp(price_x.min())), float(np.exp(price_x.max()))],
        "condition_number": used["fit"]["condition_number"], "outliers": outliers,
        "controls": {"trend": used_design["notes"]["trend"], "seasonality": used_design["notes"]["seasonality"],
                     "segments": used_design["notes"]["segments"]},
        "residual_normality_p": (float(st.shapiro(resid).pvalue) if 3 <= len(resid) <= 5000 and np.ptp(resid) > 0 else None),
        "period_start": None if not date_column else str(pd.to_datetime(obs["period"]).min().date()),
        "period_end": None if not date_column else str(pd.to_datetime(obs["period"]).max().date()),
    }
    result = {**base, "status": "estimated",
              "elasticity": {"estimate": coef["estimate"], "standard_error": coef["se"], "ci_low": coef["low"],
                             "ci_high": coef["high"], "level": level, "p_value": coef["p_value"], "df": coef["df"],
                             "method": "log-log least squares, " + diagnostics["standard_errors"]},
              "diagnostics": diagnostics, "reliability": _reliability(coef, used, diagnostics, prep, date_column),
              "fit_curve": _curve(used_design, used, price_x), "scatter": _scatter(obs, used_design, flagged, used_design is not design),
              "limitations": list(LIMITATIONS)}
    if verify:
        from app import scenario_verify
        result["verification"] = scenario_verify.safe(scenario_verify.verify_elasticity, result, used_design, used, obs, df if len(df) <= AGGREGATION_CHECK_MAX_ROWS else None)
    return result


def _flag_outliers(design: dict, level: float, ordered: bool, main: dict, rounds: int = 3) -> np.ndarray:
    """Observations with a large studentized residual. One large residual inflates the fit's error and can hide a
    second, so the fit is repeated without what was flagged (at most `rounds` times) and the flags are combined."""
    n = len(design["y"])
    flagged = np.zeros(n, dtype=bool)
    current = main
    live = np.arange(n)
    for _ in range(rounds):
        new = np.abs(current["studentized"]) > OUTLIER_T
        if not new.any() or len(live) - int(new.sum()) < MIN_OBSERVATIONS:
            break
        flagged[live[new]] = True
        live = live[~new]
        try:
            current = _fit({**design, "X": design["X"][live], "y": design["y"][live]}, level, ordered)
        except ts.RegressionError:
            break
    return flagged


def _insufficient(obs: pd.DataFrame, prep: dict) -> list[str]:
    reasons = []
    n = len(obs)
    if n < MIN_OBSERVATIONS:
        e = prep["excluded"]
        why = (f" ({e['rows_in']} rows in, {e['missing_value']} missing, {e['price_not_positive']} with a price of zero or "
               f"less, {e['quantity_zero_or_negative']} with zero or negative quantity)" if e["rows_in"] else "")
        reasons.append(f"only {n} usable observation(s); at least {MIN_OBSERVATIONS} are needed{why}")
        return reasons
    p = obs["price"].to_numpy(float)
    if len(np.unique(np.round(p, 9))) < 3:
        reasons.append("the price takes fewer than 3 distinct values, so there is no price movement to learn from")
    elif np.std(np.log(p), ddof=1) < MIN_PRICE_VARIATION:
        reasons.append(f"the price barely changes (its spread is under {MIN_PRICE_VARIATION * 100:g}%), so elasticity cannot be identified")
    return reasons


def _reliability(coef: dict, fit: dict, diag: dict, prep: dict, date_column: str | None) -> dict:
    e, lo, hi = coef["estimate"], coef["low"], coef["high"]
    flags: list[str] = []
    blockers: list[str] = []
    if e >= 0:
        blockers.append("the estimate is not negative: quantity did not fall as price rose, which usually means something "
                        "else (promotions, mix, supply limits) moved with price")
    if lo is None or hi is None:
        blockers.append("no confidence interval could be computed")
    elif lo <= 0 <= hi:
        blockers.append("the confidence interval includes zero: the data cannot tell this from no price sensitivity")
    elif hi - lo > WIDE_INTERVAL:
        flags.append(f"the confidence interval is wide ({lo:.2f} to {hi:.2f})")
    if fit["n"] < RECOMMENDED_OBSERVATIONS:
        flags.append(f"only {fit['n']} observations (at least {RECOMMENDED_OBSERVATIONS} recommended)")
    if fit["r2"] is not None and fit["r2"] < 0.1:
        flags.append(f"price explains little of the variation in quantity (R² {fit['r2']:.2f})")
    r1 = diag["residual_autocorrelation_lag1"]
    if r1 is not None and abs(r1) > 0.5:
        flags.append(f"residuals are strongly autocorrelated ({r1:+.2f}); intervals allow for it but the model is missing structure")
    if diag["price_variation_sd_ln"] < 0.05:
        flags.append("price varied only a little")
    out = diag["outliers"]
    if out["count"] and out["count"] / fit["n"] > 0.1:
        flags.append(f"{out['count']} observations ({out['count'] / fit['n'] * 100:.0f}%) are outliers")
    elif out["without_outliers"] and abs(out["without_outliers"]["estimate"] - e) > 0.25:
        flags.append("the estimate moves noticeably when outliers are removed")
    if date_column and diag["controls"]["seasonality"].startswith(("not controlled", "dropped")):
        flags.append("seasonality is not controlled, so a seasonal pattern that moves with price can bias the estimate")
    if not date_column:
        flags.append("each row is an observation, so differences between products or customers can be mistaken for price response")
    if blockers:
        level, rec = "not_recommended", "Do not apply this estimate: " + "; ".join(blockers) + "."
    elif flags:
        level, rec = "use_with_caution", "Usable as a starting point, with these cautions: " + "; ".join(flags) + "."
    else:
        level, rec = "usable", ("The estimate is negative, clearly different from zero and not flagged by the diagnostics. "
                                "It is still observational.")
    return {"level": level, "flags": blockers + flags, "recommendation": rec,
            "can_apply": level in ("usable", "use_with_caution") and ELASTICITY_BOUNDS[0] <= e <= ELASTICITY_BOUNDS[1]}


def _curve(design: dict, fit: dict, price_x: np.ndarray, points: int = 40) -> dict:
    """Predicted quantity across the observed price range, other variables at their average."""
    X, beta = design["X"], fit["fit"]["beta"]
    control_mean = float((X[:, 2:] @ beta[2:]).mean()) if X.shape[1] > 2 else 0.0
    grid = np.linspace(price_x.min(), price_x.max(), points)
    return {"price": np.exp(grid).tolist(), "quantity": np.exp(beta[0] + control_mean + beta[1] * grid).tolist()}


def _scatter(obs: pd.DataFrame, design: dict, flagged: np.ndarray, trimmed: bool, limit: int = 500) -> list[dict]:
    kept = obs[design["keep"]].reset_index(drop=True)
    flags = np.zeros(len(kept), dtype=bool)
    if not trimmed and len(flagged) == len(kept):
        flags = flagged
    step = max(1, math.ceil(len(kept) / limit))
    rows = []
    for i in range(0, len(kept), step):
        r = kept.iloc[i]
        rows.append({"price": float(r["price"]), "quantity": float(r["qty"]), "outlier": bool(flags[i]) if i < len(flags) else False,
                     "period": None if pd.isna(r["period"]) else str(pd.Timestamp(r["period"]).date())})
    return rows
