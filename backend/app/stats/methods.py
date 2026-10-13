"""Numerical methods. Each returns the pieces of a result (estimate, interval,
test, effects, warnings); `selector` decides which one to call and assembles
the full result. Resampling methods take a seed, so every result is
reproducible.

Differences are always "later minus earlier" / "second group minus first"."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as st

from app.stats.result import num

RESAMPLES = 9999
RESAMPLES_LARGE = 1999          # used above LARGE_N observations, to bound the work
LARGE_N = 50_000


def _resamples(n: int) -> int:
    return RESAMPLES_LARGE if n > LARGE_N else RESAMPLES


def clean(values: Any) -> tuple[np.ndarray, int]:
    """Finite floats and the number of values dropped (missing or non-numeric).
    Nothing is imputed and no outlier is removed."""
    import pandas as pd

    raw = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    keep = np.isfinite(raw)
    return raw[keep], int((~keep).sum())


# ------------------------------------------------------------ descriptive --

def describe(x: np.ndarray) -> dict:
    """Summary of one group. `sd` is the sample standard deviation (n - 1)."""
    n = int(len(x))
    if n == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "q1": None, "q3": None, "min": None, "max": None,
                "skewness": None, "outliers": 0, "extreme_outliers": 0}
    q1, med, q3 = (float(v) for v in np.percentile(x, [25, 50, 75]))
    iqr = q3 - q1
    sd = float(x.std(ddof=1)) if n > 1 else None
    skew = num(st.skew(x, bias=False)) if n >= 3 and sd else None
    return {"n": n, "mean": float(x.mean()), "median": med, "sd": sd, "q1": q1, "q3": q3,
            "min": float(x.min()), "max": float(x.max()), "skewness": skew,
            # Tukey's fences: flagged for the reader, never removed.
            "outliers": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0,
            "extreme_outliers": int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum()) if iqr > 0 else 0}


def cochran_n(skewness: float | None) -> float | None:
    """Cochran's rule: a mean's sampling distribution is close enough to normal
    for a t-based interval once n >= 25 * skewness^2."""
    return None if skewness is None else 25.0 * skewness * skewness


def mean_interval(x: np.ndarray, level: float) -> tuple[float, float] | None:
    n = len(x)
    if n < 2 or x.std(ddof=1) == 0:
        return None
    half = st.t.ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean() - half), float(x.mean() + half)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float | None:
    """Standardised mean difference with the small-sample correction."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return None
    return float((b.mean() - a.mean()) / pooled * (1 - 3 / (4 * (na + nb) - 9)))


# ----------------------------------------------------- two independent groups --

def _mean_diff(x, y, axis=-1):
    return np.mean(y, axis=axis) - np.mean(x, axis=axis)


def _median_diff(x, y, axis=-1):
    return np.median(y, axis=axis) - np.median(x, axis=axis)


def _bootstrap(data: tuple, statistic, level: float, seed: int, paired: bool = False) -> tuple[float, float, str] | None:
    n = sum(len(d) for d in data)
    rng = np.random.default_rng(seed)
    for method in ("BCa", "percentile"):
        try:
            with np.errstate(all="ignore"):
                res = st.bootstrap(data, statistic, vectorized=True, paired=paired, confidence_level=level,
                                   n_resamples=_resamples(n), method=method, rng=rng)
        except Exception:  # noqa: BLE001 - degenerate resamples; try the simpler interval
            continue
        low, high = num(res.confidence_interval.low), num(res.confidence_interval.high)
        if low is not None and high is not None and low <= high:
            return low, high, f"bootstrap ({method}, {_resamples(n)} resamples)"
    return None


def welch(a: np.ndarray, b: np.ndarray, level: float) -> dict:
    r = st.ttest_ind(b, a, equal_var=False)
    ci = r.confidence_interval(level)
    return {"estimate": float(b.mean() - a.mean()),
            "interval": {"low": float(ci.low), "high": float(ci.high), "level": level,
                         "method": "t interval with Welch-Satterthwaite degrees of freedom"},
            "test": {"name": "Welch's t-test", "statistic_name": "t", "statistic": float(r.statistic), "df": float(r.df),
                     "p_value": float(r.pvalue)}}


def permutation_means(a: np.ndarray, b: np.ndarray, level: float, seed: int) -> dict:
    """The same estimand as Welch (difference in means) without relying on the
    means being normally distributed."""
    n = len(a) + len(b)
    r = st.permutation_test((a, b), _mean_diff, vectorized=True, n_resamples=_resamples(n), alternative="two-sided",
                            rng=np.random.default_rng(seed))
    exact = len(r.null_distribution) < _resamples(n)
    boot = _bootstrap((a, b), _mean_diff, level, seed + 1)
    arrangements = math.comb(n, len(a))
    out = {"estimate": float(b.mean() - a.mean()), "interval": None,
           "test": {"name": "permutation test of the difference in means", "statistic_name": "difference in means",
                    "statistic": float(r.statistic), "p_value": float(r.pvalue), "exact": exact,
                    "resamples": int(len(r.null_distribution))},
           "smallest_attainable_p": min(1.0, 2.0 / arrangements)}
    if boot:
        out["interval"] = {"low": boot[0], "high": boot[1], "level": level, "method": boot[2]}
    return out


def rank_sum(a: np.ndarray, b: np.ndarray, level: float, seed: int) -> dict:
    """Mann-Whitney: whether values in one group tend to be higher. Its
    estimand is not the mean, so it is used only when that is what is asked."""
    r = st.mannwhitneyu(b, a, alternative="two-sided")
    boot = _bootstrap((a, b), _median_diff, level, seed + 1)
    out = {"estimate": float(np.median(b) - np.median(a)), "interval": None,
           "test": {"name": "Mann-Whitney U test", "statistic_name": "U", "statistic": float(r.statistic),
                    "p_value": float(r.pvalue)},
           "probability_of_superiority": float(r.statistic / (len(a) * len(b)))}
    if boot:
        out["interval"] = {"low": boot[0], "high": boot[1], "level": level, "method": boot[2]}
    return out


# ------------------------------------------------------------------- paired --

def paired_t(d: np.ndarray, level: float) -> dict:
    r = st.ttest_1samp(d, 0.0)
    ci = r.confidence_interval(level)
    return {"estimate": float(d.mean()),
            "interval": {"low": float(ci.low), "high": float(ci.high), "level": level, "method": "paired t interval"},
            "test": {"name": "paired t-test", "statistic_name": "t", "statistic": float(r.statistic), "df": float(len(d) - 1),
                     "p_value": float(r.pvalue)}}


def sign_flip(d: np.ndarray, level: float, seed: int) -> dict:
    """Paired permutation test: under 'no change', each unit's difference is as
    likely to be positive as negative. Exact when there are few pairs."""
    n = len(d)
    r = st.permutation_test((d,), lambda x, axis=-1: np.mean(x, axis=axis), vectorized=True, permutation_type="samples",
                            n_resamples=_resamples(n), alternative="two-sided", rng=np.random.default_rng(seed))
    boot = _bootstrap((d,), lambda x, axis=-1: np.mean(x, axis=axis), level, seed + 1) if n >= 3 else None
    out = {"estimate": float(d.mean()), "interval": None,
           "test": {"name": "paired sign-flip permutation test", "statistic_name": "mean difference",
                    "statistic": float(r.statistic), "p_value": float(r.pvalue),
                    "exact": len(r.null_distribution) < _resamples(n), "resamples": int(len(r.null_distribution))},
           "smallest_attainable_p": min(1.0, 2.0 / (2 ** n))}
    if boot:
        out["interval"] = {"low": boot[0], "high": boot[1], "level": level, "method": boot[2]}
    return out


def signed_rank(d: np.ndarray, level: float, seed: int) -> dict:
    r = st.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    boot = _bootstrap((d,), lambda x, axis=-1: np.median(x, axis=axis), level, seed + 1) if len(d) >= 3 else None
    out = {"estimate": float(np.median(d)), "interval": None,
           "test": {"name": "Wilcoxon signed-rank test", "statistic_name": "W", "statistic": float(r.statistic),
                    "p_value": float(r.pvalue)}}
    if boot:
        out["interval"] = {"low": boot[0], "high": boot[1], "level": level, "method": boot[2]}
    return out


# -------------------------------------------------------------- proportions --

def wilson(x: int, n: int, level: float) -> tuple[float, float] | None:
    """Wilson score interval for one proportion: valid near 0 and 1, where the
    textbook normal interval is not."""
    if n <= 0:
        return None
    z = st.norm.ppf(0.5 + level / 2)
    p = x / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(x_a: int, n_a: int, x_b: int, n_b: int, level: float) -> tuple[float, float] | None:
    """Newcombe's hybrid score interval for the difference p_b - p_a."""
    wa, wb = wilson(x_a, n_a, level), wilson(x_b, n_b, level)
    if wa is None or wb is None:
        return None
    pa, pb = x_a / n_a, x_b / n_b
    d = pb - pa
    return (d - math.sqrt((pb - wb[0]) ** 2 + (wa[1] - pa) ** 2), d + math.sqrt((wb[1] - pb) ** 2 + (pa - wa[0]) ** 2))


def expected_counts(table: np.ndarray) -> np.ndarray:
    return np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()


def cochran_expected_ok(expected: np.ndarray) -> bool:
    """Cochran's condition for the chi-square approximation: no expected count
    below 1 and at most 20% below 5 (for a 2x2 table: all at least 5)."""
    if expected.size == 4:
        return bool((expected >= 5).all())
    return bool((expected >= 1).all() and (expected < 5).mean() <= 0.2)


def ratio_effects(x_a: int, n_a: int, x_b: int, n_b: int, level: float) -> list[dict]:
    """Risk ratio and odds ratio (later relative to earlier) with log-scale
    intervals. Unavailable, not zero or infinite, when a cell is empty."""
    z = st.norm.ppf(0.5 + level / 2)
    out = []
    if x_a > 0 and x_b > 0:
        rr = (x_b / n_b) / (x_a / n_a)
        se = math.sqrt(1 / x_b - 1 / n_b + 1 / x_a - 1 / n_a)
        out.append({"name": "risk ratio", "value": rr, "interval": {"low": rr * math.exp(-z * se), "high": rr * math.exp(z * se),
                                                                    "level": level, "method": "log (Katz) interval"}})
    else:
        out.append({"name": "risk ratio", "value": None, "note": "not defined: one group has no matching rows"})
    cells = (x_a, n_a - x_a, x_b, n_b - x_b)
    if all(c > 0 for c in cells):
        odds = (x_b * (n_a - x_a)) / (x_a * (n_b - x_b))
        se = math.sqrt(sum(1 / c for c in cells))
        out.append({"name": "odds ratio", "value": odds, "interval": {"low": odds * math.exp(-z * se), "high": odds * math.exp(z * se),
                                                                      "level": level, "method": "log (Woolf) interval"}})
    else:
        out.append({"name": "odds ratio", "value": None, "note": "not defined: the table has an empty cell"})
    return out


def chi_square(table: np.ndarray) -> dict:
    r = st.chi2_contingency(table, correction=False)
    return {"name": "chi-square test of independence", "statistic_name": "chi-square", "statistic": float(r.statistic),
            "df": float(r.dof), "p_value": float(r.pvalue)}


def fisher(table: np.ndarray) -> dict:
    r = st.fisher_exact(table, alternative="two-sided")
    return {"name": "Fisher's exact test", "p_value": float(r.pvalue), "exact": True}


def permutation_chi_square(rows: np.ndarray, cols: np.ndarray, seed: int) -> dict:
    """Chi-square statistic with its null distribution from shuffling one
    variable: valid for sparse tables larger than 2x2, where the chi-square
    approximation is not."""
    r_codes, c_codes = np.unique(rows, return_inverse=True)[1], np.unique(cols, return_inverse=True)[1]
    nr, nc = r_codes.max() + 1, c_codes.max() + 1

    def stat(c):
        table = np.zeros((nr, nc))
        np.add.at(table, (r_codes, c), 1)
        exp = expected_counts(table)
        return float(((table - exp) ** 2 / exp).sum())

    observed = stat(c_codes)
    rng = np.random.default_rng(seed)
    n_res = _resamples(len(rows))
    hits = sum(stat(rng.permutation(c_codes)) >= observed - 1e-12 for _ in range(n_res))
    return {"name": "permutation chi-square test", "statistic_name": "chi-square", "statistic": observed,
            "p_value": (hits + 1) / (n_res + 1), "resamples": n_res}


def cramers_v(chi2: float, n: int, shape: tuple[int, int]) -> float | None:
    k = min(shape[0] - 1, shape[1] - 1)
    return None if n <= 0 or k <= 0 else float(math.sqrt(chi2 / (n * k)))


# ---------------------------------------------------------------- k groups --

def welch_anova(groups: list[np.ndarray]) -> dict:
    """Welch's one-way test of equal means, which does not assume equal variances."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    m = np.array([g.mean() for g in groups])
    w = n / np.array([g.var(ddof=1) for g in groups])
    grand = (w * m).sum() / w.sum()
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    f = ((w * (m - grand) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k * k - 1) * lam)
    df2 = (k * k - 1) / (3 * lam)
    return {"name": "Welch's one-way test", "statistic_name": "F", "statistic": float(f), "df": [float(k - 1), float(df2)],
            "p_value": float(st.f.sf(f, k - 1, df2))}


def kruskal(groups: list[np.ndarray]) -> dict:
    r = st.kruskal(*groups)
    return {"name": "Kruskal-Wallis test", "statistic_name": "H", "statistic": float(r.statistic), "df": float(len(groups) - 1),
            "p_value": float(r.pvalue)}


def eta_squared(groups: list[np.ndarray]) -> float | None:
    """Share of the outcome's variance that lies between the groups."""
    values = np.concatenate(groups)
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return None
    return float(sum(len(g) * (g.mean() - values.mean()) ** 2 for g in groups) / total)


# -------------------------------------------------------------- correlation --

def pearson(x: np.ndarray, y: np.ndarray, level: float) -> dict:
    r = st.pearsonr(x, y)
    ci = r.confidence_interval(level)
    return {"estimate": float(r.statistic),
            "interval": {"low": float(ci.low), "high": float(ci.high), "level": level, "method": "Fisher z interval"},
            "test": {"name": "test of zero Pearson correlation", "statistic_name": "r", "statistic": float(r.statistic),
                     "df": float(len(x) - 2), "p_value": float(r.pvalue)}}


def spearman(x: np.ndarray, y: np.ndarray, level: float) -> dict:
    r = st.spearmanr(x, y)
    rho, n = float(r.statistic), len(x)
    out = {"estimate": rho, "interval": None,
           "test": {"name": "test of zero Spearman rank correlation", "statistic_name": "rho", "statistic": rho,
                    "p_value": float(r.pvalue)}}
    if n > 3 and abs(rho) < 1:
        z, half = math.atanh(rho), st.norm.ppf(0.5 + level / 2) * 1.03 / math.sqrt(n - 3)
        out["interval"] = {"low": math.tanh(z - half), "high": math.tanh(z + half), "level": level,
                           "method": "Fisher z interval (approximate for ranks)"}
    return out


# ------------------------------------------------------------- multiplicity --

def holm(p_values: list[float]) -> list[float]:
    """Holm's step-down adjustment: controls the chance of any false positive
    in the family, with no assumption about how the tests depend on each other."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjustment: controls the expected share of false
    positives among the results declared significant."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i], reverse=True)
    adjusted, running = [0.0] * m, 1.0
    for rank, i in enumerate(order):
        running = min(running, p_values[i] * m / (m - rank))
        adjusted[i] = running
    return adjusted
