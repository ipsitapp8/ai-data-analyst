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
# A test-inversion interval needs every resample's group sums, which is rows x resamples of work.
# Above this many observations it is not attempted and no interval is reported.
INVERSION_MAX_N = 5_000
SHIFT_INTERVAL = "permutation test inversion (assumes the two groups have the same shape)"
SYMMETRY_INTERVAL = "sign-flip test inversion (assumes the differences are symmetric about their centre)"


def _resamples(n: int) -> int:
    return RESAMPLES_LARGE if n > LARGE_N else RESAMPLES


def _smallest_p(arrangements: int, mirrored: bool = True) -> float:
    """The smallest two-sided p-value an exact test with this many arrangements
    can produce. `mirrored`: every arrangement has a mirror image equally far
    from zero (sign patterns; two groups of the same size), so at least two
    are always as extreme as the one observed. The count is an integer of any
    size (it passes 1e308 at about 1,030 observations), so it is compared
    before it is divided."""
    return 0.0 if arrangements > 10 ** 300 else min(1.0, (2.0 if mirrored else 1.0) / arrangements)


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


# ------------------------------------------------- intervals by test inversion --
#
# The interval that goes with a permutation test is the set of values the test does not
# reject. For two groups: every shift d for which "a against b - d" is not rejected at
# 1 - level. It is exact when the groups differ only by a shift, which is also what the
# p-value assumes, so the test and the interval say the same thing under one assumption.
#
# The statistic permuted is Welch's t, not the raw difference in means. Both are exact when
# the groups are exchangeable. They part company when the groups differ in spread: with a
# smaller, more variable group the raw difference rejected a true null in about 25% of
# simulations, Welch's t in 6% to 8% (backend/evals/reports/permutation_audit.json).
#
# This replaces the bootstrap (BCa) interval. In simulation that interval held the true
# value in 84% to 89% of small samples instead of 95%: a dozen values cannot stand in for
# the tails of their distribution, however often they are resampled. See
# backend/evals/study_small_sample_intervals.py and docs/COMPARISON_WORKFLOW.md.

def _invert(p_value, estimate: float, scale: float, alpha: float) -> tuple[float, float] | None:
    """The interval {v : p_value(v) >= alpha} around `estimate`, by bisection on
    each side. None when it is unbounded: the test cannot reject any value."""
    bounds = []
    for sign in (-1.0, 1.0):
        step, far = scale, estimate + sign * scale
        while p_value(far) >= alpha:
            step *= 2
            if step > scale * 1e4:           # far beyond the data: no value is going to be rejected
                return None
            far = estimate + sign * step
        near = estimate
        for _ in range(80):
            mid = (near + far) / 2
            if p_value(mid) >= alpha:
                near = mid
            else:
                far = mid
        bounds.append(near)
    return bounds[0], bounds[1]


def _two_sided(below: int, above: int, total: int, exact: bool) -> float:
    """Twice the smaller tail, as scipy's permutation test defines "two-sided";
    a Monte Carlo tail counts the observed arrangement as one of the draws.
    Used for the sign-flip test, whose null distribution is symmetric by
    construction, so the choice of rule cannot matter there."""
    if exact:
        return min(1.0, 2.0 * min(below, above) / total)
    return min(1.0, 2.0 * (min(below, above) + 1) / (total + 1))


def _as_extreme(hits: int, total: int, exact: bool) -> float:
    """Share of arrangements at least as far from zero as the one observed.

    This is the two-sided rule for the two-group permutation test. "Twice the
    smaller tail" is the other common rule and equally exact when the groups
    are exchangeable; when they differ in spread, a single sample's permutation
    distribution is lopsided and that rule takes the favourable side each time.
    Measured with 24 and 8 values and a fourfold spread ratio: 13.0% false
    positives against 7.5% for this rule."""
    return min(1.0, hits / total) if exact else min(1.0, (hits + 1) / (total + 1))


def abs_welch_statistic(x, y, axis=-1):
    return np.abs(welch_statistic(x, y, axis=axis))


def welch_statistic(x, y, axis=-1):
    """Welch's t for y minus x. A split with no variation in either part has
    no standard error; it is counted as infinitely far out on the side of its
    difference, which is what a zero-variance group implies."""
    diff = np.mean(y, axis=axis) - np.mean(x, axis=axis)
    se = np.sqrt(np.var(x, axis=axis, ddof=1) / np.shape(x)[axis] + np.var(y, axis=axis, ddof=1) / np.shape(y)[axis])
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, diff / se, np.where(diff == 0, 0.0, np.copysign(np.inf, diff)))


def shift_interval(a: np.ndarray, b: np.ndarray, level: float, seed: int) -> tuple[float, float] | None:
    """Interval for the difference in means (b minus a) by inverting the
    permutation test of Welch's t: every shift d for which "a against b - d"
    is not rejected.

    For a resample that puts the pooled positions L in the second group, the
    group sums, sums of squares and counts of values that came from b are
    computed once. Subtracting d from the b values changes each group's mean
    and variance by an exact formula in those, so every candidate d is judged
    from the same resamples without drawing them again."""
    na, nb = len(a), len(b)
    n = na + nb
    if n > INVERSION_MAX_N or min(na, nb) < 2:
        return None
    centre = float(np.concatenate([a, b]).mean())                 # centring keeps the sums of squares accurate
    z, was_b = np.concatenate([a, b]) - centre, np.concatenate([np.zeros(na), np.ones(nb)])
    arrangements = math.comb(n, nb)
    exact = arrangements <= RESAMPLES
    if _smallest_p(arrangements, mirrored=na == nb) >= 1 - level:
        return None                          # no shift can be rejected, so the interval is the whole line
    if exact:
        from itertools import combinations

        chosen = np.fromiter((i for combo in combinations(range(n), nb) for i in combo), dtype=np.int64,
                             count=arrangements * nb).reshape(arrangements, nb)
    else:
        rng = np.random.default_rng(seed)
        chosen = np.empty((RESAMPLES, nb), dtype=np.int64)
        for start in range(0, RESAMPLES, 500):                       # bounded memory: 500 x n keys at a time
            keys = rng.random((min(500, RESAMPLES - start), n))
            chosen[start:start + len(keys)] = np.argpartition(keys, nb - 1, axis=1)[:, :nb]
    # second group of each resample: sum, sum of squares, how many came from b, and the sum of those
    s2, q2 = z[chosen].sum(axis=1), (z[chosen] ** 2).sum(axis=1)
    c2, p2 = was_b[chosen].sum(axis=1), (z[chosen] * was_b[chosen]).sum(axis=1)
    s1, q1, c1, p1 = z.sum() - s2, (z ** 2).sum() - q2, nb - c2, float((z * was_b).sum()) - p2
    estimate, total = float(b.mean() - a.mean()), len(chosen)
    se_observed = math.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    if se_observed == 0:
        return None

    def p_value(d: float) -> float:
        # after subtracting d from the values that came from b
        m1, m2 = (s1 - d * c1) / na, (s2 - d * c2) / nb
        v1 = np.maximum((q1 - 2 * d * p1 + d * d * c1) - na * m1 * m1, 0.0) / (na - 1)
        v2 = np.maximum((q2 - 2 * d * p2 + d * d * c2) - nb * m2 * m2, 0.0) / (nb - 1)
        se = np.sqrt(v1 / na + v2 / nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            permuted = np.where(se > 1e-12 * (abs(d) + 1), (m2 - m1) / se, np.where(m2 == m1, 0.0, np.copysign(np.inf, m2 - m1)))
        observed = abs(estimate - d) / se_observed
        return _as_extreme(int((np.abs(permuted) >= observed * (1 - 1e-9) - 1e-12).sum()), total, exact)

    return _invert(p_value, estimate, max(float(z.std(ddof=1)), 1e-12), 1 - level)


def symmetry_interval(d: np.ndarray, level: float, seed: int) -> tuple[float, float] | None:
    """Interval for the mean paired difference by inverting the sign-flip
    test: every c for which "d - c" is not rejected. mean(s * (d - c)) is
    A(s) - c * B(s) for a sign pattern s."""
    n = len(d)
    if n > INVERSION_MAX_N or n < 2:
        return None
    exact = 2 ** n <= RESAMPLES
    if _smallest_p(2 ** n) >= 1 - level:
        return None                          # no value can be rejected, so the interval is the whole line
    if exact:
        signs = 1.0 - 2.0 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
        a_s, b_s = (signs * d).mean(axis=1), signs.mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        a_s, b_s = np.empty(RESAMPLES), np.empty(RESAMPLES)
        for start in range(0, RESAMPLES, 500):
            signs = rng.choice((1.0, -1.0), (min(500, RESAMPLES - start), n))
            a_s[start:start + len(signs)], b_s[start:start + len(signs)] = (signs * d).mean(axis=1), signs.mean(axis=1)
    estimate, total = float(d.mean()), len(a_s)

    def p_value(c: float) -> float:
        flipped, observed = a_s - c * b_s, estimate - c
        tol = 1e-12 * max(1.0, abs(observed))
        return _two_sided(int((flipped <= observed + tol).sum()), int((flipped >= observed - tol).sum()), total, exact)

    return _invert(p_value, estimate, max(float(d.std(ddof=1)), 1e-12), 1 - level)


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
    means being normally distributed. The labels are permuted and Welch's t is
    recomputed each time: exact when the groups are exchangeable, and far less
    sensitive to unequal spreads than permuting the raw difference."""
    n = len(a) + len(b)
    # |t| with the upper tail: "at least as far from zero as observed" (see _as_extreme).
    r = st.permutation_test((a, b), abs_welch_statistic, vectorized=True, n_resamples=_resamples(n), alternative="greater",
                            rng=np.random.default_rng(seed))
    exact = len(r.null_distribution) < _resamples(n)
    bounds = shift_interval(a, b, level, seed + 1)
    arrangements = math.comb(n, len(a))
    out = {"estimate": float(b.mean() - a.mean()), "interval": None,
           "test": {"name": "permutation test of the difference in means (Welch's t permuted)", "statistic_name": "Welch's t",
                    "statistic": float(welch_statistic(a, b)), "p_value": float(r.pvalue), "exact": exact,
                    "resamples": int(len(r.null_distribution))},
           "smallest_attainable_p": _smallest_p(arrangements, mirrored=len(a) == len(b)),
           "interval_unavailable": None}
    if bounds:
        out["interval"] = {"low": bounds[0], "high": bounds[1], "level": level, "method": SHIFT_INTERVAL}
    else:
        out["interval_unavailable"] = (f"more than {INVERSION_MAX_N:,} observations: the test-inversion interval is not computed "
                                       "at this size" if n > INVERSION_MAX_N else
                                       "the permutation test cannot reject any difference with this few observations, so the "
                                       "interval would be unbounded")
    return out


# Below this many values in the smaller group the Brunner-Munzel statistic is referred to its
# permutation distribution instead of a t distribution. Measured (backend/evals/reports/
# method_decisions_v3.json): with 5 values in a group the t form rejected a true null in 7% to
# 8.5% of simulations; from 30 per group it was within 4.7% to 5.4%.
BM_PERMUTATION_BELOW = 30


def _brunner_munzel_statistics(za: np.ndarray, zb: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> tuple:
    """Brunner-Munzel for many splits at once, one per row. `ra`, `rb` are the
    pooled (mid)ranks of the values in `za`, `zb`. Returns (statistic,
    probability of superiority, standard error, degrees of freedom)."""
    na, nb = za.shape[1], zb.shape[1]
    wa, wb = st.rankdata(za, axis=1), st.rankdata(zb, axis=1)
    p_hat = (rb.mean(axis=1) - (nb + 1) / 2) / na
    sa = ((ra - wa - ra.mean(axis=1, keepdims=True) + (na + 1) / 2) ** 2).sum(axis=1) / (na - 1)
    sb = ((rb - wb - rb.mean(axis=1, keepdims=True) + (nb + 1) / 2) ** 2).sum(axis=1) / (nb - 1)
    v = na * sa + nb * sb
    se = np.sqrt(v) / (na * nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = np.where(se > 0, (p_hat - 0.5) / se, np.where(p_hat == 0.5, 0.0, np.copysign(np.inf, p_hat - 0.5)))
        df = np.where(v > 0, v * v / ((na * sa) ** 2 / (na - 1) + (nb * sb) ** 2 / (nb - 1)), np.nan)
    return statistic, p_hat, se, df


def brunner_munzel(a: np.ndarray, b: np.ndarray, level: float, seed: int) -> dict:
    """Whether values in one group tend to be higher than in the other.

    The quantity is p = P(a value from b exceeds one from a) + half the chance
    of a tie, and the null is p = 1/2: neither group tends to be higher. Unlike
    the Mann-Whitney test, which tests that null only when the two groups have
    the same shape, this one allows them to differ in spread. Ties are handled
    by midranks.

    The reported estimate stays the difference in medians, with a resampled
    interval, as a description of how far apart typical values are; the test
    and the effect size are about p."""
    na, nb = len(a), len(b)
    n = na + nb
    z = np.concatenate([a, b])
    pooled = st.rankdata(z)
    statistic, p_hat, se, df = (float(x[0]) for x in _brunner_munzel_statistics(a[None, :], b[None, :], pooled[None, :na], pooled[None, na:]))
    permute = (min(na, nb) < BM_PERMUTATION_BELOW or se == 0) and n <= INVERSION_MAX_N
    if permute:
        arrangements = math.comb(n, na)
        exact = arrangements <= RESAMPLES
        hits = total = 0
        if exact:
            from itertools import combinations

            first = np.array(list(combinations(range(n), na)))
            chunks = [first[i:i + 2000] for i in range(0, len(first), 2000)]
        else:
            rng = np.random.default_rng(seed)
            chunks = [np.argpartition(rng.random((min(500, RESAMPLES - i), n)), na - 1, axis=1)[:, :na] for i in range(0, RESAMPLES, 500)]
        for first in chunks:
            mask = np.ones((len(first), n), bool)
            np.put_along_axis(mask, first, False, axis=1)
            second = np.nonzero(mask)[1].reshape(len(first), nb)
            stats = np.abs(_brunner_munzel_statistics(z[first], z[second], pooled[first], pooled[second])[0])
            hits += int((stats >= abs(statistic) * (1 - 1e-9)).sum())
            total += len(first)
        p_value = _as_extreme(hits, total, exact)
        test = {"name": "Brunner-Munzel test (permutation)", "statistic_name": "Brunner-Munzel statistic",
                "statistic": statistic if math.isfinite(statistic) else None, "p_value": float(p_value), "exact": exact,
                "resamples": int(total)}
        floor = _smallest_p(arrangements, mirrored=na == nb)
    elif se == 0:
        # Too large to permute and nothing to estimate a standard error from: every value in one
        # group is on the same side of every value in the other. No p-value is invented.
        test, floor = None, None
    else:
        test = {"name": "Brunner-Munzel test", "statistic_name": "Brunner-Munzel statistic", "statistic": statistic,
                "df": df, "p_value": float(2 * st.t.sf(abs(statistic), df))}
        floor = None
    boot = _bootstrap((a, b), _median_diff, level, seed + 1)
    out = {"estimate": float(np.median(b) - np.median(a)), "interval": None, "test": test,
           "probability_of_superiority": p_hat, "smallest_attainable_p": floor, "interval_unavailable": None}
    if boot:
        out["interval"] = {"low": boot[0], "high": boot[1], "level": level, "method": boot[2]}
    else:
        out["interval_unavailable"] = "the median difference could not be resampled (too few distinct values)"
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
    bounds = symmetry_interval(d, level, seed + 1)
    out = {"estimate": float(d.mean()), "interval": None,
           "test": {"name": "paired sign-flip permutation test", "statistic_name": "mean difference",
                    "statistic": float(r.statistic), "p_value": float(r.pvalue),
                    "exact": len(r.null_distribution) < _resamples(n), "resamples": int(len(r.null_distribution))},
           "smallest_attainable_p": _smallest_p(2 ** n),
           "interval_unavailable": None}
    if bounds:
        out["interval"] = {"low": bounds[0], "high": bounds[1], "level": level, "method": SYMMETRY_INTERVAL}
    else:
        out["interval_unavailable"] = (f"more than {INVERSION_MAX_N:,} pairs: the test-inversion interval is not computed at "
                                       "this size" if n > INVERSION_MAX_N else
                                       "the sign-flip test cannot reject any value with this few pairs, so the interval "
                                       "would be unbounded")
    return out


def sign_test(d: np.ndarray, level: float) -> dict:
    """Is a change as likely to be up as down? Exact (binomial) whatever the
    shape of the changes; unchanged units carry no information and are left
    out. The interval is for the median change, from order statistics."""
    up, down = int((d > 0).sum()), int((d < 0).sum())
    p_value = float(st.binomtest(up, up + down, 0.5).pvalue) if up + down else None
    iv = median_interval(d, level)
    return {"estimate": float(np.median(d)), "increases": up, "decreases": down, "unchanged": int(len(d) - up - down), "p_value": p_value,
            "interval": None if iv is None else {"low": iv[0], "high": iv[1], "level": level,
                                                 "method": "order-statistic interval for the median (distribution-free)"}}


def signed_rank(d: np.ndarray, level: float, seed: int) -> dict:
    r = st.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    # The interval is for the median change and comes from order statistics, not from the
    # signed-rank test and not from a bootstrap: a bootstrap of the median held the true value
    # in 92% to 94% of simulations at 8 to 30 pairs (backend/evals/reports/permutation_audit_before.json).
    bounds = median_interval(d, level)
    out = {"estimate": float(np.median(d)), "interval": None,
           "test": {"name": "Wilcoxon signed-rank test", "statistic_name": "W", "statistic": float(r.statistic),
                    "p_value": float(r.pvalue)},
           "interval_unavailable": None}
    if bounds:
        out["interval"] = {"low": bounds[0], "high": bounds[1], "level": level,
                           "method": "order-statistic interval for the median (distribution-free, at least the stated level)"}
    else:
        out["interval_unavailable"] = (f"{len(d)} pairs are too few: even the smallest and largest changes bracket the "
                                       f"median with less than {level * 100:g}% probability")
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


def welch_f_statistic(*groups, axis=-1):
    """Welch's one-way F for any number of groups, along `axis`. This is the
    statistic the permutation one-way test permutes: like Welch's t for two
    groups, it weighs each group by its own variance, so a small, variable
    group does not masquerade as a difference in means."""
    k = len(groups)
    n = np.array([np.shape(g)[axis] for g in groups], float)
    shape = [1] * np.ndim(np.mean(groups[0], axis=axis)) + [k]
    means = np.stack([np.mean(g, axis=axis) for g in groups], axis=-1)
    variances = np.stack([np.var(g, axis=axis, ddof=1) for g in groups], axis=-1)
    w = n.reshape(shape) / np.maximum(variances, 1e-300)
    total = w.sum(axis=-1, keepdims=True)
    grand = (w * means).sum(axis=-1, keepdims=True) / total
    lam = (((1 - w / total) ** 2) / (n.reshape(shape) - 1)).sum(axis=-1)
    return ((w * (means - grand) ** 2).sum(axis=-1) / (k - 1)) / (1 + 2 * (k - 2) / (k * k - 1) * lam)


def median_interval(d: np.ndarray, level: float) -> tuple[float, float] | None:
    """Distribution-free interval for a median, from order statistics.

    The number of values below the true median is Binomial(n, 1/2) whatever
    the distribution, so the k-th smallest and k-th largest values bracket the
    median with a probability that can be computed exactly. k is the largest
    for which that probability is at least `level`. Nothing is assumed about
    shape or symmetry, and nothing is resampled. None when even the smallest
    and largest values do not reach the level (fewer than 6 values at 95%)."""
    n = len(d)
    if n < 1:
        return None
    ordered = np.sort(d)
    best = None
    for k in range(1, n // 2 + 1):
        if 1 - 2 * st.binom.cdf(k - 1, n, 0.5) >= level:
            best = k
        else:
            break
    return None if best is None else (float(ordered[best - 1]), float(ordered[n - best]))


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
        # Fieller, Hartley and Pearson (1957): on the Fisher z scale a rank correlation has variance
        # 1.06 / (n - 3). Its square root is 1.0296; an earlier version used the rounded 1.03.
        z, half = math.atanh(rho), st.norm.ppf(0.5 + level / 2) * math.sqrt(1.06 / (n - 3))
        out["interval"] = {"low": math.tanh(z - half), "high": math.tanh(z + half), "level": level,
                           "method": "Fisher z interval with Fieller's variance for ranks (approximate)"}
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
