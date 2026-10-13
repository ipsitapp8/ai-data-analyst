"""Least squares with robust standard errors, and the time-series tools built
on it. Pure numerical code: callers pass arrays in time order.

Consecutive days are not independent observations. Wherever a series is
analysed here, uncertainty comes from Newey-West (HAC) standard errors, which
allow for autocorrelation, rather than from a formula that assumes it away."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st


# Residual sum of squares, relative to that of the outcome, below which a fit
# is exact: a residual norm under 1e-12 of the outcome's is floating-point noise.
EXACT_FIT_TOLERANCE = 1e-24


class RegressionError(ValueError):
    """The model cannot be estimated from this data."""


def newey_west_lag(n: int) -> int:
    """The usual automatic bandwidth, floor(4 * (n / 100)^(2/9))."""
    return int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def autocorr1(x: np.ndarray) -> float | None:
    """Lag-1 autocorrelation."""
    if len(x) < 3:
        return None
    d = x - x.mean()
    denom = float((d * d).sum())
    return None if denom == 0 else float((d[1:] * d[:-1]).sum() / denom)


def ols(X: np.ndarray, y: np.ndarray, cov: str = "HC3", lags: int = 0) -> dict:
    """Ordinary least squares. cov is "classical", "HC3" (robust to unequal
    variances) or "HAC" (Newey-West, robust to autocorrelation as well)."""
    n, k = X.shape
    if n <= k:
        raise RegressionError(f"{n} observations cannot estimate {k} coefficients")
    if np.linalg.matrix_rank(X) < k:
        raise RegressionError("the explanatory variables are perfectly collinear")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    # An exact fit leaves rounding noise, not zeros, in the residuals. Left in,
    # it gives a tiny non-zero standard error and a test built on that noise.
    if float(e @ e) <= EXACT_FIT_TOLERANCE * float(y @ y):
        e = np.zeros_like(e)
    if cov == "classical":
        V = xtx_inv * float(e @ e) / (n - k)
    elif cov == "HC3":
        leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
        u = np.where(1 - leverage > 1e-10, e / np.maximum(1 - leverage, 1e-10), 0.0)
        xu = X * u[:, None]
        V = xtx_inv @ (xu.T @ xu) @ xtx_inv
    elif cov == "HAC":
        xe = X * e[:, None]
        S = xe.T @ xe
        for lag in range(1, min(lags, n - 1) + 1):
            G = xe[lag:].T @ xe[:-lag]
            S += (1 - lag / (lags + 1)) * (G + G.T)
        V = xtx_inv @ S @ xtx_inv * n / (n - k)
    else:
        raise ValueError(f"unknown covariance '{cov}'")
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    sst = float(((y - y.mean()) ** 2).sum())
    return {"beta": beta, "se": se, "df": n - k, "resid": e, "n": n, "k": k,
            "r_squared": None if sst == 0 else 1 - float(e @ e) / sst, "condition_number": float(np.linalg.cond(X))}


def coefficient(fit: dict, i: int, level: float) -> dict:
    """Estimate, standard error, t-based interval and p-value for coefficient i."""
    b, se, df = float(fit["beta"][i]), float(fit["se"][i]), fit["df"]
    if se == 0 or not math.isfinite(se):
        return {"estimate": b, "se": None, "low": None, "high": None, "statistic": None, "p_value": None, "df": df}
    half = st.t.ppf(0.5 + level / 2, df) * se
    t = b / se
    return {"estimate": b, "se": se, "low": b - half, "high": b + half, "statistic": t,
            "p_value": float(2 * st.t.sf(abs(t), df)), "df": df}


def level_shift(a: np.ndarray, b: np.ndarray, level: float) -> dict:
    """Difference between the mean level of two consecutive stretches of a
    series, with Newey-West standard errors."""
    y = np.concatenate([a, b])
    n = len(y)
    X = np.column_stack([np.ones(n), np.r_[np.zeros(len(a)), np.ones(len(b))]])
    lags = newey_west_lag(n)
    fit = ols(X, y, cov="HAC", lags=lags)
    out = coefficient(fit, 1, level)
    out.update(lags=lags, autocorrelation=autocorr1(fit["resid"]), n=n)
    return out


def linear_trend(y: np.ndarray, level: float) -> dict:
    """Slope per time step, with Newey-West standard errors."""
    n = len(y)
    t = np.arange(n, dtype=float)
    lags = newey_west_lag(n)
    fit = ols(np.column_stack([np.ones(n), t]), y, cov="HAC", lags=lags)
    out = coefficient(fit, 1, level)
    out.update(lags=lags, autocorrelation=autocorr1(fit["resid"]), n=n, intercept=float(fit["beta"][0]))
    return out


def pettitt(x: np.ndarray) -> dict:
    """Pettitt's test for a single shift in the level of a series. Returns the
    position after which the level changes (0-based index of the last point
    before the change) and the approximate p-value."""
    n = len(x)
    ranks = st.rankdata(x)
    u = 2 * np.cumsum(ranks)[:-1] - np.arange(1, n) * (n + 1)
    idx = int(np.argmax(np.abs(u)))
    k = float(abs(u[idx]))
    return {"statistic": k, "index": idx, "p_value": float(min(1.0, 2 * math.exp(-6 * k * k / (n ** 3 + n ** 2)))), "n": n}


def prewhiten(x: np.ndarray, r1: float) -> np.ndarray:
    """Remove lag-1 autocorrelation before a test that assumes independence."""
    return x[1:] - r1 * x[:-1]
