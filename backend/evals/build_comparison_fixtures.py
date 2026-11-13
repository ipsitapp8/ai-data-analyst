"""Fixtures and independent reference answers for the comparison workflow.

    python backend/evals/build_comparison_fixtures.py

Writes one CSV per scenario to fixtures/comparison/ and the cases, with their
expected values, to cases/comparison.json.

Everything here is the Python standard library. The data comes from
`random.Random(seed)`; the expected statistics come from textbook formulas
written out below, with this file's own t, F, chi-square and normal
distribution functions, and from enumerating every permutation where a test is
exact. No numpy, no scipy, no pandas and no application code: the application
computes the same quantities through scipy.stats, so agreement is agreement
between two implementations that share nothing.

A scenario fixes which method SHOULD be chosen (from the design and the data's
shape) and what every number should be. Where the application's choice of
approximation differs from the textbook one used here, the case will fail and
say so; that is what it is for.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

VERSION = "2026.10.5"
HERE = Path(__file__).resolve().parent
OUT = HERE / "fixtures" / "comparison"
LEVEL = 0.95
Z = statistics.NormalDist().inv_cdf(0.5 + LEVEL / 2)


# ------------------------------------------------------------ distributions --

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 2000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided(t: float, df: float) -> float:
    return betai(df / 2.0, 0.5, df / (df + t * t))


def t_quantile(p: float, df: float) -> float:
    """Upper quantile: the t with P(T <= t) = p, for p > 0.5, by bisection."""
    lo, hi = 0.0, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1.0 - t_two_sided(mid, df) / 2.0 < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def f_upper(f: float, d1: float, d2: float) -> float:
    return betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def chi2_upper(x: float, k: float) -> float:
    """Upper tail of chi-square: the regularised upper incomplete gamma Q(k/2, x/2)."""
    a, x = k / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1.0:                                   # series for P, then Q = 1 - P
        term = total = 1.0 / a
        n = a
        for _ in range(5000):
            n += 1.0
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-17:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300                                     # continued fraction for Q (Lentz)
    b = x + 1.0 - a
    c, d = 1.0 / tiny, 1.0 / b
    h = d
    for i in range(1, 5000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


# --------------------------------------------------------------- statistics --

def mean(xs):
    return math.fsum(xs) / len(xs)


def var(xs):
    m = mean(xs)
    return math.fsum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def welch(a, b) -> dict:
    """b minus a, Welch-Satterthwaite degrees of freedom."""
    va, vb = var(a) / len(a), var(b) / len(b)
    est, se = mean(b) - mean(a), math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t_quantile(0.5 + LEVEL / 2, df) * se
    return {"estimate": est, "statistic": est / se, "df": df, "p_value": t_two_sided(est / se, df),
            "interval_low": est - half, "interval_high": est + half}


def paired_t(first, second) -> dict:
    d = [y - x for x, y in zip(first, second)]
    est, se, df = mean(d), math.sqrt(var(d) / len(d)), len(d) - 1
    half = t_quantile(0.5 + LEVEL / 2, df) * se
    return {"estimate": est, "statistic": est / se, "df": df, "p_value": t_two_sided(est / se, df),
            "interval_low": est - half, "interval_high": est + half}


def _welch_t(a, b) -> float:
    """Welch's t for b minus a. No variation in either part: infinitely far out on the side of the difference."""
    diff = mean(b) - mean(a)
    se = math.sqrt(var(a) / len(a) + var(b) / len(b))
    return diff / se if se > 0 else math.copysign(math.inf, diff) if diff else 0.0


def exact_permutation_means(a, b) -> dict:
    """Every way of splitting the pooled values into groups of these sizes,
    with Welch's t recomputed for each split. Two-sided: the share of splits
    whose t is at least as far from zero as the one observed."""
    pooled, na = a + b, len(a)
    observed = _welch_t(a, b)
    count = hits = 0
    for combo in itertools.combinations(range(len(pooled)), na):
        chosen = set(combo)
        t = _welch_t([pooled[i] for i in combo], [pooled[i] for i in range(len(pooled)) if i not in chosen])
        count += 1
        hits += abs(t) >= abs(observed) * (1 - 1e-9)
    return {"estimate": mean(b) - mean(a), "statistic": observed, "p_value": hits / count, "arrangements": count}


def _inverted(p_value, estimate: float, scale: float) -> tuple[float, float]:
    """The interval of values a test does not reject at 1 - LEVEL: on each
    side of the estimate, the furthest value whose p-value is still >= alpha."""
    alpha, bounds = 1 - LEVEL, []
    for sign in (-1.0, 1.0):
        step = scale
        while p_value(estimate + sign * step) >= alpha:
            step *= 2
        near, far = estimate, estimate + sign * step
        for _ in range(100):
            mid = (near + far) / 2
            if p_value(mid) >= alpha:
                near = mid
            else:
                far = mid
        bounds.append(near)
    return bounds[0], bounds[1]


def exact_shift_interval(a, b) -> dict:
    """Every shift d for which the exact permutation test of a against b - d,
    with Welch's t as its statistic, does not reject. Each split is recomputed
    from the values for each d; nothing is shared with the application's
    closed-form shortcut."""
    na, nb = len(a), len(b)
    splits = [(combo, [i for i in range(na + nb) if i not in set(combo)]) for combo in itertools.combinations(range(na + nb), na)]

    def p_value(d: float) -> float:
        shifted = [v - d for v in b]
        pooled, observed = a + shifted, _welch_t(a, shifted)
        hits = sum(abs(_welch_t([pooled[i] for i in first], [pooled[i] for i in second])) >= abs(observed) * (1 - 1e-9)
                   for first, second in splits)
        return hits / len(splits)

    low, high = _inverted(p_value, mean(b) - mean(a), math.sqrt(var(a + b)))
    return {"interval_low": low, "interval_high": high}


def exact_symmetry_interval(first, second) -> dict:
    """Every c for which the exact sign-flip test of (differences - c) does not reject."""
    d = [y - x for x, y in zip(first, second)]
    n = len(d)
    patterns = list(itertools.product((1, -1), repeat=n))

    def p_value(c: float) -> float:
        shifted = [v - c for v in d]
        observed = mean(shifted)
        below = above = 0
        for signs in patterns:
            flipped = math.fsum(s * v for s, v in zip(signs, shifted)) / n
            below += flipped <= observed + 1e-12
            above += flipped >= observed - 1e-12
        return min(1.0, 2 * min(below, above) / len(patterns))

    low, high = _inverted(p_value, mean(d), math.sqrt(var(d)))
    return {"interval_low": low, "interval_high": high}


def exact_sign_flip(first, second) -> dict:
    d = [y - x for x, y in zip(first, second)]
    observed, n = mean(d), len(d)
    hits = sum(abs(math.fsum(s * v for s, v in zip(signs, d)) / n) >= abs(observed) - 1e-12
               for signs in itertools.product((1, -1), repeat=n))
    return {"estimate": observed, "p_value": hits / 2 ** n, "arrangements": 2 ** n}


def exact_mann_whitney(a, b) -> dict:
    """U counts the pairs in which the b value is the larger. Exact null
    distribution by enumeration; valid here because the values have no ties."""
    assert len(set(a + b)) == len(a + b), "exact Mann-Whitney needs untied values"
    order = sorted(a + b)
    rank = {v: i + 1 for i, v in enumerate(order)}
    na, nb = len(a), len(b)
    u_obs = sum(rank[v] for v in b) - nb * (nb + 1) / 2
    centre, count, hits = na * nb / 2, 0, 0
    for combo in itertools.combinations(range(1, na + nb + 1), nb):
        u = sum(combo) - nb * (nb + 1) / 2
        count += 1
        hits += abs(u - centre) >= abs(u_obs - centre) - 1e-12
    return {"estimate": statistics.median(b) - statistics.median(a), "statistic": u_obs, "p_value": hits / count,
            "arrangements": count}


def _midranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _brunner_munzel_statistic(a, b) -> tuple[float, float]:
    """(statistic, estimated probability that a b value exceeds an a value, ties counted half)."""
    na, nb = len(a), len(b)
    pooled = _midranks(a + b)
    ra, rb, wa, wb = pooled[:na], pooled[na:], _midranks(a), _midranks(b)
    p_hat = (mean(rb) - (nb + 1) / 2) / na
    sa = math.fsum((r - w - mean(ra) + (na + 1) / 2) ** 2 for r, w in zip(ra, wa)) / (na - 1)
    sb = math.fsum((r - w - mean(rb) + (nb + 1) / 2) ** 2 for r, w in zip(rb, wb)) / (nb - 1)
    se = math.sqrt(na * sa + nb * sb) / (na * nb)
    if se == 0:
        return (0.0 if p_hat == 0.5 else math.copysign(math.inf, p_hat - 0.5)), p_hat
    return (p_hat - 0.5) / se, p_hat


def exact_brunner_munzel(a, b) -> dict:
    """The Brunner-Munzel statistic from its definition, referred to every way
    of splitting the pooled values into groups of these sizes."""
    z, na = a + b, len(a)
    observed, p_hat = _brunner_munzel_statistic(a, b)
    count = hits = 0
    for first in itertools.combinations(range(len(z)), na):
        chosen = set(first)
        t, _ = _brunner_munzel_statistic([z[i] for i in first], [z[i] for i in range(len(z)) if i not in chosen])
        count += 1
        hits += abs(t) >= abs(observed) * (1 - 1e-9)
    return {"estimate": statistics.median(b) - statistics.median(a), "statistic": observed, "p_value": hits / count,
            "probability_of_superiority": p_hat, "arrangements": count}


def sample_skewness(x) -> float:
    """The adjusted Fisher-Pearson coefficient (G1)."""
    n, mu = len(x), mean(x)
    m2, m3 = math.fsum((v - mu) ** 2 for v in x) / n, math.fsum((v - mu) ** 3 for v in x) / n
    return m3 / m2 ** 1.5 * math.sqrt(n * (n - 1)) / (n - 2)


def shows_skew(x) -> bool:
    """The sample itself fails Cochran's rule (n >= 25 * skewness^2)."""
    return len(x) < 25 * sample_skewness(x) ** 2


def welch_anova(groups) -> dict:
    k = len(groups)
    n = [len(g) for g in groups]
    m = [mean(g) for g in groups]
    w = [ni / var(g) for ni, g in zip(n, groups)]
    sw = math.fsum(w)
    grand = math.fsum(wi * mi for wi, mi in zip(w, m)) / sw
    lam = math.fsum((1 - wi / sw) ** 2 / (ni - 1) for wi, ni in zip(w, n))
    f = (math.fsum(wi * (mi - grand) ** 2 for wi, mi in zip(w, m)) / (k - 1)) / (1 + 2 * (k - 2) / (k * k - 1) * lam)
    df2 = (k * k - 1) / (3 * lam)
    return {"statistic": f, "df": [k - 1, df2], "p_value": f_upper(f, k - 1, df2)}


def _f_statistic(groups) -> float:
    values = [v for g in groups for v in g]
    grand, k, n = mean(values), len(groups), len(values)
    between = math.fsum(len(g) * (mean(g) - grand) ** 2 for g in groups) / (k - 1)
    within = math.fsum((v - mean(g)) ** 2 for g in groups for v in g) / (n - k)
    return between / within


def exact_permutation_anova(groups) -> dict:
    """Every way of dealing the pooled values into groups of these sizes, with
    Welch's F recomputed for each deal."""
    pooled = [v for g in groups for v in g]
    welch_f = lambda gs: welch_anova(gs)["statistic"]  # noqa: E731
    sizes, observed = [len(g) for g in groups], welch_f(groups)
    count = hits = 0

    def deal(remaining: tuple[int, ...], which: int, built: list):
        nonlocal count, hits
        if which == len(sizes) - 1:
            f = welch_f(built + [[pooled[i] for i in remaining]])
            count += 1
            hits += f >= observed * (1 - 1e-9)
            return
        for combo in itertools.combinations(remaining, sizes[which]):
            chosen = set(combo)
            deal(tuple(i for i in remaining if i not in chosen), which + 1, built + [[pooled[i] for i in combo]])

    deal(tuple(range(len(pooled))), 0, [])
    return {"statistic": observed, "p_value": hits / count, "arrangements": count}


def pearson(x, y) -> dict:
    n, mx, my = len(x), mean(x), mean(y)
    r = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(
        math.fsum((a - mx) ** 2 for a in x) * math.fsum((b - my) ** 2 for b in y))
    t = r * math.sqrt((n - 2) / (1 - r * r))
    half = Z / math.sqrt(n - 3)
    return {"estimate": r, "df": n - 2, "p_value": t_two_sided(t, n - 2),
            "interval_low": math.tanh(math.atanh(r) - half), "interval_high": math.tanh(math.atanh(r) + half)}


def _ranks(values) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks, i = [0.0] * len(values), 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y) -> dict:
    """Rho is Pearson's r on ranks; the test uses the t approximation. The
    interval is Fisher's z with Fieller's variance for ranks, 1.06 / (n - 3)."""
    out = pearson(_ranks(x), _ranks(y))
    n, rho = len(x), out["estimate"]
    half = Z * math.sqrt(1.06 / (n - 3))
    return {"estimate": rho, "p_value": out["p_value"],
            "interval_low": math.tanh(math.atanh(rho) - half), "interval_high": math.tanh(math.atanh(rho) + half)}


def _wilson(x: int, n: int) -> tuple[float, float]:
    p = x / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def chi_square(table) -> dict:
    """Pearson's chi-square, no continuity correction."""
    rows, cols = [sum(r) for r in table], [sum(c) for c in zip(*table)]
    n = sum(rows)
    stat = math.fsum((table[i][j] - rows[i] * cols[j] / n) ** 2 / (rows[i] * cols[j] / n)
                     for i in range(len(rows)) for j in range(len(cols)))
    df = (len(rows) - 1) * (len(cols) - 1)
    return {"statistic": stat, "df": df, "p_value": chi2_upper(stat, df), "n": n,
            "cramers_v": math.sqrt(stat / (n * min(len(rows) - 1, len(cols) - 1)))}


def fisher_exact(table) -> dict:
    """Two-sided: the total probability of every table, with these margins,
    that is no more likely than the one observed."""
    (a, b), (c, d) = table
    r1, r2, c1, n = a + b, c + d, a + c, a + b + c + d
    prob = lambda k: math.comb(r1, k) * math.comb(r2, c1 - k) / math.comb(n, c1)
    observed = prob(a)
    p = math.fsum(prob(k) for k in range(max(0, c1 - r2), min(r1, c1) + 1) if prob(k) <= observed * (1 + 1e-7))
    return {"p_value": min(1.0, p)}


def two_proportions(x_a: int, n_a: int, x_b: int, n_b: int) -> dict:
    """b minus a, with Newcombe's hybrid score interval."""
    pa, pb = x_a / n_a, x_b / n_b
    (la, ua), (lb, ub) = _wilson(x_a, n_a), _wilson(x_b, n_b)
    d = pb - pa
    return {"estimate": d, "interval_low": d - math.sqrt((pb - lb) ** 2 + (ua - pa) ** 2),
            "interval_high": d + math.sqrt((ub - pb) ** 2 + (pa - la) ** 2)}


# ----------------------------------------------------------------- scenarios --

def _write(name: str, header: list[str], rows: list[list]) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return f"comparison/{name}"


def _normal(rng, n, mu, sd):
    return [round(rng.gauss(mu, sd), 4) for _ in range(n)]


def _skewed(rng, n, scale):
    return [round(scale * rng.lognormvariate(0, 1.1), 4) for _ in range(n)]


def _two_group_rows(a, b, labels=("A", "B")):
    rows = [[labels[0], v] for v in a] + [[labels[1], v] for v in b]
    return rows


def scenarios() -> list[dict]:
    cases = []

    def add(case_id, seed, fixture, plan, method, expect, question=None, n=None, why=""):
        cases.append({"id": f"cmp-{case_id}", "component": "comparison", "fixture": fixture, "plan": plan,
                      **({"question": question} if question else {}), "seed": seed, "n": n, "why": why,
                      "expect": {"method": method, **expect}, "rel_tolerance": 1e-6})

    # 1. two groups, roughly normal, a planted difference of 8 on sd 15
    seed = 101
    rng = random.Random(seed)
    a, b = _normal(rng, 60, 100, 15), _normal(rng, 60, 108, 15)
    add("welch-planted-difference", seed, _write("s01_two_groups_effect.csv", ["segment", "spend"], _two_group_rows(a, b)),
        {"outcome": "spend", "explanatory": "segment"}, "welch_t", {**welch(a, b), "significant": True},
        question="Is spend different between segment groups?", n=[60, 60],
        why="two independent groups, means, both satisfy Cochran's rule")

    # 2. the same design with no difference planted
    seed = 102
    rng = random.Random(seed)
    a, b = _normal(rng, 60, 100, 15), _normal(rng, 60, 100, 15)
    add("welch-no-difference", seed, _write("s02_two_groups_null.csv", ["segment", "spend"], _two_group_rows(a, b)),
        {"outcome": "spend", "explanatory": "segment"}, "welch_t", {**welch(a, b), "significant": False}, n=[60, 60],
        why="as above; nothing planted, so the test should not reject")

    # 3. two small, strongly skewed groups: no test of means is supported. The difference is still measured.
    seed = 103
    rng = random.Random(seed)
    a, b = _skewed(rng, 7, 10), _skewed(rng, 7, 30)
    assert shows_skew(a) or shows_skew(b)
    add("small-skewed-not-tested", seed, _write("s03_two_groups_skewed.csv", ["segment", "spend"], _two_group_rows(a, b)),
        {"outcome": "spend", "explanatory": "segment"}, "none",
        {"status": "unavailable", "inference_unavailable": "too_skewed_for_sample_size", "estimate": mean(b) - mean(a)}, n=[7, 7],
        why="7 and 7 values, visibly skewed: the mean difference is reported, and no test or interval, because the only "
            "test left assumes the groups have the same shape")

    # 3a. two small groups that do not look skewed: a t interval cannot be justified, the permutation test is used
    seed = 1030
    rng = random.Random(seed)
    a, b = _normal(rng, 7, 100, 12), _normal(rng, 7, 118, 12)
    assert not shows_skew(a) and not shows_skew(b)
    ref = exact_permutation_means(a, b)
    add("permutation-small", seed, _write("s03a_two_groups_small.csv", ["segment", "spend"], _two_group_rows(a, b)),
        {"outcome": "spend", "explanatory": "segment"}, "permutation_means",
        {"estimate": ref["estimate"], "statistic": ref["statistic"], "p_value": ref["p_value"], "exact": True,
         "rejects_method": "welch_t", **exact_shift_interval(a, b), "interval_method": "permutation test inversion"}, n=[7, 7],
        why=f"7 skewed values a side: Cochran's rule fails; all {ref['arrangements']} splits are enumerated")

    # 3b. unequal sizes, the smaller group more variable: where permuting the raw difference went wrong
    seed = 1047
    rng = random.Random(seed)
    a, b = _normal(rng, 9, 100, 5), _normal(rng, 4, 100, 25)
    assert not shows_skew(a) and not shows_skew(b)      # the seed is one whose samples do not look skewed
    ref = exact_permutation_means(a, b)
    add("permutation-unequal-sizes-and-spread", seed, _write("s03b_two_groups_unequal.csv", ["segment", "spend"], _two_group_rows(a, b)),
        {"outcome": "spend", "explanatory": "segment"}, "permutation_means",
        {"estimate": ref["estimate"], "statistic": ref["statistic"], "p_value": ref["p_value"], "exact": True,
         **exact_shift_interval(a, b), "interval_method": "permutation test inversion"}, n=[9, 4],
        why=f"9 and 4 values with unequal spread: Welch's t is permuted over all {ref['arrangements']} splits")

    # 4. the question is about typical values, not means
    seed = 104
    rng = random.Random(seed)
    a, b = _skewed(rng, 7, 10), _skewed(rng, 7, 25)
    ref = exact_brunner_munzel(a, b)
    add("brunner-munzel-typical", seed, _write("s04_two_groups_typical.csv", ["segment", "spend"], _two_group_rows(a, b)),
        {"outcome": "spend", "explanatory": "segment", "estimand": "typical"}, "brunner_munzel",
        {"estimate": ref["estimate"], "statistic": ref["statistic"], "p_value": ref["p_value"], "exact": True,
         "probability_of_superiority": ref["probability_of_superiority"]}, n=[7, 7],
        why=f"the plan asks about typical values: the Brunner-Munzel statistic is referred to all {ref['arrangements']} splits")

    # 4b. the same question with tied ratings and groups of different size and spread
    seed = 1041
    rng = random.Random(seed)
    a, b = [float(rng.choice([2, 3, 3, 3, 4])) for _ in range(9)], [float(rng.choice([1, 1, 2, 4, 5, 5])) for _ in range(5)]
    ref = exact_brunner_munzel(a, b)
    add("brunner-munzel-ties", seed, _write("s04b_two_groups_ratings.csv", ["segment", "rating"], _two_group_rows(a, b)),
        {"outcome": "rating", "explanatory": "segment", "estimand": "typical"}, "brunner_munzel",
        {"estimate": ref["estimate"], "statistic": ref["statistic"], "p_value": ref["p_value"], "exact": True,
         "probability_of_superiority": ref["probability_of_superiority"]}, n=[9, 5],
        why="tied ratings, unequal sizes and spread: ties are handled by midranks")

    # 5. the same units measured twice
    seed = 105
    rng = random.Random(seed)
    before = _normal(rng, 40, 50, 10)
    after = [round(x + rng.gauss(3, 4), 4) for x in before]
    rows = [[f"u{i:02d}", "before", x] for i, x in enumerate(before)] + [[f"u{i:02d}", "after", y] for i, y in enumerate(after)]
    add("paired-t", seed, _write("s05_paired.csv", ["store", "period", "sales"], rows),
        {"outcome": "sales", "explanatory": "period", "levels": ["before", "after"], "design": "paired", "unit": "store"},
        "paired_t", {**paired_t(before, after), "significant": True}, n=[40],
        why="each store appears once in each period: the differences are analysed, not the two groups")

    # 6. few pairs with skewed differences
    seed = 106
    rng = random.Random(seed)
    before = _skewed(rng, 10, 20)
    after = [round(x + rng.lognormvariate(1.0, 1.0), 4) for x in before]
    rows = [[f"u{i:02d}", "before", x] for i, x in enumerate(before)] + [[f"u{i:02d}", "after", y] for i, y in enumerate(after)]
    assert shows_skew([y - x for x, y in zip(before, after)])
    add("paired-skewed-not-tested", seed, _write("s06_paired_skewed.csv", ["store", "period", "sales"], rows),
        {"outcome": "sales", "explanatory": "period", "levels": ["before", "after"], "design": "paired", "unit": "store"},
        "none", {"status": "unavailable", "inference_unavailable": "too_skewed_for_sample_size",
                 "estimate": mean([y - x for x, y in zip(before, after)])}, n=[10],
        why="10 visibly skewed differences: the mean change is reported, and no test or interval, because the sign-flip "
            "test assumes symmetric changes")

    # 6a. few pairs whose differences do not look skewed
    seed = 1062
    rng = random.Random(seed)
    before = _normal(rng, 10, 50, 10)
    after = [round(x + rng.gauss(4, 5), 4) for x in before]
    assert not shows_skew([y - x for x, y in zip(before, after)])
    rows = [[f"u{i:02d}", "before", x] for i, x in enumerate(before)] + [[f"u{i:02d}", "after", y] for i, y in enumerate(after)]
    ref = exact_sign_flip(before, after)
    add("sign-flip-small", seed, _write("s06a_paired_small.csv", ["store", "period", "sales"], rows),
        {"outcome": "sales", "explanatory": "period", "levels": ["before", "after"], "design": "paired", "unit": "store"},
        "sign_flip_permutation", {"estimate": ref["estimate"], "p_value": ref["p_value"], "exact": True,
                                  **exact_symmetry_interval(before, after), "interval_method": "sign-flip test inversion"}, n=[10],
        why=f"10 differences, too few for a t interval: all {ref['arrangements']} sign patterns are enumerated")

    # 6b. five pairs: 32 sign patterns, so the smallest two-sided p-value is 0.0625. No value can be
    # rejected at 0.05, the interval is the whole line, and the right answer is "no interval".
    seed = 1070
    rng = random.Random(seed)
    before = _normal(rng, 5, 50, 10)
    after = [round(x + rng.gauss(6, 3), 4) for x in before]
    assert not shows_skew([y - x for x, y in zip(before, after)])
    rows = [[f"u{i:02d}", "before", x] for i, x in enumerate(before)] + [[f"u{i:02d}", "after", y] for i, y in enumerate(after)]
    ref = exact_sign_flip(before, after)
    add("sign-flip-no-interval", seed, _write("s06b_paired_five.csv", ["store", "period", "sales"], rows),
        {"outcome": "sales", "explanatory": "period", "levels": ["before", "after"], "design": "paired", "unit": "store"},
        "sign_flip_permutation", {"estimate": ref["estimate"], "p_value": ref["p_value"], "exact": True,
                                  "interval_available": False, "significant": False}, n=[5],
        why="5 pairs allow 32 sign patterns: no value can be rejected at 0.05, so no interval is reported")

    # 7. four groups, roughly normal
    seed = 107
    rng = random.Random(seed)
    groups = {"North": _normal(rng, 50, 100, 12), "South": _normal(rng, 50, 104, 18), "East": _normal(rng, 50, 100, 9),
              "West": _normal(rng, 50, 110, 14)}
    rows = [[k, v] for k, vs in groups.items() for v in vs]
    ordered = [groups[k] for k in sorted(groups)]
    add("welch-anova", seed, _write("s07_four_groups.csv", ["region", "spend"], rows),
        {"outcome": "spend", "explanatory": "region"}, "welch_anova", {**welch_anova(ordered), "significant": True},
        question="Does spend differ by region?", n=[50, 50, 50, 50],
        why="several groups with unequal variances, each satisfying Cochran's rule")

    # 8. three tiny skewed groups
    seed = 108
    rng = random.Random(seed)
    groups = {"a": _skewed(rng, 3, 10), "b": _skewed(rng, 3, 12), "c": _skewed(rng, 4, 40)}
    rows = [[k, v] for k, vs in groups.items() for v in vs]
    assert any(shows_skew(g) for g in groups.values())
    add("three-groups-skewed-not-tested", seed, _write("s08_three_groups_small.csv", ["tier", "spend"], rows),
        {"outcome": "spend", "explanatory": "tier"}, "none",
        {"status": "unavailable", "inference_unavailable": "too_skewed_for_sample_size"}, n=[3, 3, 4],
        why="groups of 3, 3 and 4, visibly skewed: the group means are reported, and no test, because the only test left "
            "assumes every group has the same shape")

    # 8a. three tiny groups that do not look skewed
    seed = 1295
    rng = random.Random(seed)
    groups = {"a": _normal(rng, 3, 100, 10), "b": _normal(rng, 3, 104, 10), "c": _normal(rng, 4, 125, 10)}
    assert not any(shows_skew(g) for g in groups.values())      # the seed is one whose samples do not look skewed
    rows = [[k, v] for k, vs in groups.items() for v in vs]
    ref = exact_permutation_anova([groups[k] for k in sorted(groups)])
    add("permutation-anova-small", seed, _write("s08a_three_groups_small.csv", ["tier", "spend"], rows),
        {"outcome": "spend", "explanatory": "tier"}, "permutation_anova",
        {"statistic": ref["statistic"], "p_value": ref["p_value"], "rejects_method": "welch_anova"}, n=[3, 3, 4],
        why=f"groups of 3, 3 and 4: an F distribution is not justified; all {ref['arrangements']} deals are enumerated")

    # 9. two numeric variables, linear
    seed = 109
    rng = random.Random(seed)
    x = _normal(rng, 80, 50, 10)
    y = [round(0.6 * v + rng.gauss(0, 8), 4) for v in x]
    add("pearson", seed, _write("s09_linear.csv", ["price", "units"], [[a, b] for a, b in zip(x, y)]),
        {"outcome": "units", "explanatory": "price"}, "pearson", {**pearson(x, y), "significant": True},
        question="Is units related to price?", n=[80], why="two numeric variables, no extreme values")

    # 10. the same, with extreme values that would dominate a linear correlation
    seed = 110
    rng = random.Random(seed)
    x = _normal(rng, 60, 50, 10)
    y = [round(0.5 * v + rng.gauss(0, 6), 4) for v in x]
    x[:3], y[:3] = [400.0, 450.0, 500.0], [5.0, 4.0, 3.0]
    add("spearman-extreme-values", seed, _write("s10_outliers.csv", ["price", "units"], [[a, b] for a, b in zip(x, y)]),
        {"outcome": "units", "explanatory": "price"}, "spearman", {**spearman(x, y), "rejects_method": "pearson"}, n=[60],
        why="three extreme prices: ranks are used so that they cannot decide the answer")

    # 11. a yes/no outcome in two groups, large counts
    seed = 111
    rng = random.Random(seed)
    counts = {"control": (200, 0.30), "variant": (200, 0.42)}
    rows, table = [], {}
    for group, (n, p) in counts.items():
        flags = ["yes" if rng.random() < p else "no" for _ in range(n)]
        table[group] = (flags.count("yes"), n)
        rows += [[group, f] for f in flags]
    (xa, na), (xb, nb) = table["control"], table["variant"]
    ref = chi_square([[xa, na - xa], [xb, nb - xb]])
    add("two-proportions-chi-square", seed, _write("s11_proportions.csv", ["arm", "converted"], rows),
        {"outcome": "converted", "explanatory": "arm", "outcome_level": "yes"}, "chi_square",
        {**two_proportions(xa, na, xb, nb), "statistic": ref["statistic"], "df": ref["df"], "p_value": ref["p_value"]},
        n=[na, nb], why="two proportions, every expected count at least 5")

    # 12. the same with too few rows for the chi-square approximation
    seed = 112
    rows = [["control", "yes"]] * 1 + [["control", "no"]] * 9 + [["variant", "yes"]] * 6 + [["variant", "no"]] * 4
    add("two-proportions-fisher", seed, _write("s12_proportions_sparse.csv", ["arm", "converted"], [list(r) for r in rows]),
        {"outcome": "converted", "explanatory": "arm", "outcome_level": "yes"}, "fisher_exact",
        {**two_proportions(1, 10, 6, 10), **fisher_exact([[1, 9], [6, 4]]), "rejects_method": "chi_square"}, n=[10, 10],
        why="an expected count below 5: the exact test replaces the approximation")

    # 13. two categorical variables, more than two levels each
    seed = 113
    rng = random.Random(seed)
    weights = {"North": (0.5, 0.3, 0.2), "South": (0.3, 0.4, 0.3), "West": (0.2, 0.3, 0.5)}
    plans, rows = ("basic", "plus", "pro"), []
    for region, w in weights.items():
        rows += [[region, rng.choices(plans, w)[0]] for _ in range(120)]
    table = [[sum(1 for r, p in rows if r == region and p == plan) for plan in plans] for region in sorted(weights)]
    ref = chi_square(table)
    add("contingency-chi-square", seed, _write("s13_table.csv", ["region", "plan"], rows),
        {"outcome": "plan", "explanatory": "region"}, "chi_square",
        {"estimate": ref["cramers_v"], "statistic": ref["statistic"], "df": ref["df"], "p_value": ref["p_value"],
         "significant": True}, n=[360], why="a 3 x 3 table with adequate expected counts; Cramér's V is the effect size")
    # The whole route on the shared sales fixture: question in, published dashboard out, with no
    # model call and no code run. Group means are recomputed here from the CSV with the csv module.
    sums: dict[str, list[float]] = {}
    with open(HERE / "fixtures" / "sales.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sums.setdefault(row["region"], []).append(float(row["revenue"]))
    means = {region: mean(values) for region, values in sums.items()}
    top, low = max(means, key=means.get), min(means, key=means.get)
    cases.append({"id": "cmp-pipeline-question-to-dashboard", "component": "pipeline", "fixture": "sales.csv",
                  "question": "Is average revenue different between regions?",
                  "expect": {"route": "comparison", "status": "verified", "verdict_state": "VERIFIED_WITH_CAVEATS",
                             "kpi_values": {f"Mean revenue: {top}": means[top], f"Mean revenue: {low}": means[low],
                                            "Number of region groups": len(means)},
                             "kpi_evidence": {f"Mean revenue: {top}": "verified", f"Mean revenue: {low}": "verified"},
                             "charts": 0, "max_llm_calls": 0},
                  "tolerance": 0.005})
    return cases


def main() -> None:
    cases = scenarios()
    out = {"version": VERSION, "level": LEVEL,
           "description": "Comparison workflow: expected method and values from build_comparison_fixtures.py (standard library only).",
           "cases": cases}
    (HERE / "cases" / "comparison.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"comparison {VERSION}: {len(cases)} cases")
    for c in cases:
        e = c["expect"]
        if "method" not in e:
            continue
        print(f"  {c['id']:34s} {e['method']:22s} {'p=' + format(e['p_value'], '.6g') if 'p_value' in e else 'not tested'}" + (f" est={e['estimate']:.6g}" if "estimate" in e else ""))


if __name__ == "__main__":
    main()
