"""Which interval should accompany a permutation test on a small sample?

    python backend/evals/study_small_sample_intervals.py [--sims 2000]

The comparison workflow's calibration run found that the bootstrap (BCa)
interval reported beside the permutation tests held the true value in 86.0%
(two groups of 12, skewed) and 87.2% (8 pairs) of simulations, against a
nominal 95%. This study measures that interval and four alternatives on the
same two designs and on four harder ones, including designs that break each
alternative's assumption.

It is independent of the application: every interval here is implemented in
this file with numpy and scipy's distribution functions. Nothing is imported
from `app`. Seeds are fixed; simulation i of design d uses BASE_SEED + 100000 d + i.

Candidates, two independent groups (difference in means, b minus a):
    bca         bias-corrected and accelerated bootstrap (what the engine reported)
    percentile  percentile bootstrap
    boot_t      studentised bootstrap, Welch standard error
    welch       Welch's t interval
    inversion   every shift d for which the permutation test of "a versus b - d"
                does not reject. Exact under a location shift; that is its assumption.

Candidates, paired differences (mean difference):
    bca, percentile, boot_t (one-sample t), paired_t,
    inversion   every d for which the sign-flip test of "differences - d" does
                not reject. Exact when the differences are symmetric about their centre.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats as st

BASE_SEED = 20_261_101
LEVEL, ALPHA = 0.95, 0.05
BOOT = 1999          # bootstrap resamples per interval in this study
PERMS = 4999         # Monte Carlo permutations when full enumeration is too large
HERE = Path(__file__).resolve().parent


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


# -------------------------------------------------------- two-group intervals --

def _boot_two(a, b, rng):
    ia, ib = rng.integers(0, len(a), (BOOT, len(a))), rng.integers(0, len(b), (BOOT, len(b)))
    return a[ia], b[ib]


def two_percentile(a, b, rng):
    ra, rb = _boot_two(a, b, rng)
    d = rb.mean(1) - ra.mean(1)
    return float(np.quantile(d, ALPHA / 2)), float(np.quantile(d, 1 - ALPHA / 2))


def two_bca(a, b, rng):
    ra, rb = _boot_two(a, b, rng)
    d, est = rb.mean(1) - ra.mean(1), b.mean() - a.mean()
    z0 = st.norm.ppf(np.clip((d < est).mean(), 1e-6, 1 - 1e-6))
    jack = np.r_[[b.mean() - np.delete(a, i).mean() for i in range(len(a))],
                 [np.delete(b, i).mean() - a.mean() for i in range(len(b))]]
    u = jack.mean() - jack
    acc = (u ** 3).sum() / (6 * ((u ** 2).sum() ** 1.5)) if (u ** 2).sum() > 0 else 0.0
    out = []
    for q in (ALPHA / 2, 1 - ALPHA / 2):
        z = st.norm.ppf(q)
        out.append(float(np.quantile(d, st.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))))
    return out[0], out[1]


def two_boot_t(a, b, rng):
    ra, rb = _boot_two(a, b, rng)
    est = b.mean() - a.mean()
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    se_star = np.sqrt(ra.var(axis=1, ddof=1) / len(a) + rb.var(axis=1, ddof=1) / len(b))
    ok = se_star > 0
    t = ((rb.mean(1) - ra.mean(1)) - est)[ok] / se_star[ok]
    return float(est - np.quantile(t, 1 - ALPHA / 2) * se), float(est - np.quantile(t, ALPHA / 2) * se)


def two_welch(a, b, rng):
    ci = st.ttest_ind(b, a, equal_var=False).confidence_interval(LEVEL)
    return float(ci.low), float(ci.high)


def _labels(na: int, nb: int, rng) -> np.ndarray:
    """Rows: assignments of the pooled positions to the second group (True)."""
    n = na + nb
    if math.comb(n, nb) <= PERMS:
        out = np.zeros((math.comb(n, nb), n), bool)
        for r, combo in enumerate(itertools.combinations(range(n), nb)):
            out[r, list(combo)] = True
        return out
    out = np.zeros((PERMS, n), bool)
    for r in range(PERMS):
        out[r, rng.permutation(n)[:nb]] = True
    return out


def two_inversion(a, b, rng):
    """{d : the permutation test of a against (b - d) has p >= alpha}.

    With pooled values z and group-two membership L, the permuted difference in
    means for shift d is S(L) - d * W(L): linear in d, so no test is re-run."""
    na, nb = len(a), len(b)
    labels = _labels(na, nb, rng)
    exact = len(labels) == math.comb(na + nb, nb)
    z = np.r_[a, b]
    was_b = np.r_[np.zeros(na), np.ones(nb)]
    s0 = (labels * z).sum(1) / nb - (~labels * z).sum(1) / na
    w = (labels * was_b).sum(1) / nb - (~labels * was_b).sum(1) / na
    est = b.mean() - a.mean()

    def p(d: float) -> float:
        hits = (np.abs(s0 - d * w) >= abs(est - d) - 1e-12).sum()
        return hits / len(labels) if exact else (hits + 1) / (len(labels) + 1)

    return _invert(p, est, max(float(np.r_[a, b].std(ddof=1)), 1e-9))


def _invert(p, est: float, scale: float) -> tuple[float, float]:
    out = []
    for sign in (-1.0, 1.0):
        step = scale
        far = est + sign * step
        while p(far) >= ALPHA and step < scale * 1e6:
            step *= 2
            far = est + sign * step
        near = est
        for _ in range(60):
            mid = (near + far) / 2
            if p(mid) >= ALPHA:
                near = mid
            else:
                far = mid
        out.append(near)
    return out[0], out[1]


# ----------------------------------------------------------- paired intervals --

def paired_percentile(d, rng):
    m = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1)
    return float(np.quantile(m, ALPHA / 2)), float(np.quantile(m, 1 - ALPHA / 2))


def paired_bca(d, rng):
    m, est = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1), d.mean()
    z0 = st.norm.ppf(np.clip((m < est).mean(), 1e-6, 1 - 1e-6))
    jack = np.array([np.delete(d, i).mean() for i in range(len(d))])
    u = jack.mean() - jack
    acc = (u ** 3).sum() / (6 * ((u ** 2).sum() ** 1.5)) if (u ** 2).sum() > 0 else 0.0
    out = []
    for q in (ALPHA / 2, 1 - ALPHA / 2):
        z = st.norm.ppf(q)
        out.append(float(np.quantile(m, st.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))))))
    return out[0], out[1]


def paired_boot_t(d, rng):
    r = d[rng.integers(0, len(d), (BOOT, len(d)))]
    est, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
    se_star = r.std(axis=1, ddof=1) / math.sqrt(len(d))
    ok = se_star > 0
    t = (r.mean(1) - est)[ok] / se_star[ok]
    return float(est - np.quantile(t, 1 - ALPHA / 2) * se), float(est - np.quantile(t, ALPHA / 2) * se)


def paired_t(d, rng):
    ci = st.ttest_1samp(d, 0.0).confidence_interval(LEVEL)
    return float(ci.low), float(ci.high)


def paired_inversion(d, rng):
    """{c : the sign-flip test of (d - c) has p >= alpha}. mean(s * (d - c)) = A(s) - c * B(s)."""
    n = len(d)
    if 2 ** n <= PERMS:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        exact = True
    else:
        signs, exact = rng.choice((1.0, -1.0), (PERMS, n)), False
    a_s, b_s, est = (signs * d).mean(1), signs.mean(1), d.mean()

    def p(c: float) -> float:
        hits = (np.abs(a_s - c * b_s) >= abs(est - c) - 1e-12).sum()
        return hits / len(signs) if exact else (hits + 1) / (len(signs) + 1)

    return _invert(p, est, max(float(d.std(ddof=1)), 1e-9))


SKEW_GUARD = 1.0     # |sample skewness| above this: the guarded variants give no interval


def _skew(x) -> float:
    return float(st.skew(x, bias=False)) if len(x) >= 3 and x.std() > 0 else 0.0


def two_welch_guarded(a, b, rng):
    """Welch's interval unless either group looks strongly skewed, then none."""
    return None if max(abs(_skew(a)), abs(_skew(b))) > SKEW_GUARD else two_welch(a, b, rng)


def paired_inversion_guarded(d, rng):
    """The sign-flip interval unless the differences look asymmetric, then none."""
    return None if abs(_skew(d)) > SKEW_GUARD else paired_inversion(d, rng)


def paired_t_guarded(d, rng):
    return None if abs(_skew(d)) > SKEW_GUARD else paired_t(d, rng)


TWO = {"bca": two_bca, "percentile": two_percentile, "boot_t": two_boot_t, "welch": two_welch, "inversion": two_inversion,
       "welch_guarded": two_welch_guarded}
PAIRED = {"bca": paired_bca, "percentile": paired_percentile, "boot_t": paired_boot_t, "paired_t": paired_t,
          "inversion": paired_inversion, "inversion_guarded": paired_inversion_guarded, "paired_t_guarded": paired_t_guarded}


# -------------------------------------------------------------------- designs --

LOGNORMAL_MEAN = lambda mu, sigma: math.exp(mu + sigma * sigma / 2)  # noqa: E731


def d_skewed_shift(rng):          # the calibration run's design: 86.0% for BCa
    return rng.lognormal(3, 0.9, 12), rng.lognormal(3, 0.9, 12) + 15.0, 15.0


def d_skewed_shift_unequal(rng):
    return rng.lognormal(3, 0.9, 8), rng.lognormal(3, 0.9, 20) + 15.0, 15.0


def d_normal_tiny(rng):
    return rng.normal(100, 10, 6), rng.normal(108, 10, 6), 8.0


def d_skewed_scale(rng):          # NOT a shift: the second group is a rescaled version of the first
    return rng.lognormal(3, 0.9, 12), rng.lognormal(3.4, 0.9, 12), LOGNORMAL_MEAN(3.4, 0.9) - LOGNORMAL_MEAN(3, 0.9)


def d_unequal_spread(rng):        # NOT a shift: same shape, different spread, unequal sizes
    return rng.normal(100, 5, 16), rng.normal(104, 20, 8), 4.0


def p_normal_few(rng):            # the calibration run's design: 87.2% for BCa
    return rng.normal(4, 4, 8), 4.0


def p_skewed(rng):                # NOT symmetric: differences are skewed about their mean
    return rng.lognormal(1, 0.9, 10) - LOGNORMAL_MEAN(1, 0.9) + 3.0, 3.0


def p_heavy(rng):                 # symmetric, heavy tails
    return rng.standard_t(3, 12) * 4 + 3.0, 3.0


def d_very_skewed_shift(rng):
    return rng.lognormal(3, 1.5, 10), rng.lognormal(3, 1.5, 10) + 40.0, 40.0


def d_very_skewed_tiny(rng):
    return rng.lognormal(3, 1.2, 5), rng.lognormal(3, 1.2, 5) + 30.0, 30.0


def d_exponential_scale(rng):     # NOT a shift
    return rng.exponential(10, 8), rng.exponential(18, 8), 8.0


def p_very_skewed(rng):           # NOT symmetric
    return rng.lognormal(1, 1.3, 8) - LOGNORMAL_MEAN(1, 1.3) + 3.0, 3.0


def p_normal_tiny(rng):
    return rng.normal(4, 4, 5), 4.0


TWO_DESIGNS = [
    ("two groups 12 + 12, skewed, shift (calibration design)", d_skewed_shift, "shift holds"),
    ("two groups 8 + 20, skewed, shift", d_skewed_shift_unequal, "shift holds"),
    ("two groups 6 + 6, normal, shift", d_normal_tiny, "shift holds"),
    ("two groups 12 + 12, skewed, rescaled", d_skewed_scale, "shift VIOLATED"),
    ("two groups 16 + 8, normal, unequal spread", d_unequal_spread, "shift VIOLATED"),
    ("two groups 10 + 10, very skewed, shift", d_very_skewed_shift, "shift holds"),
    ("two groups 5 + 5, very skewed, shift", d_very_skewed_tiny, "shift holds"),
    ("two groups 8 + 8, exponential, rescaled", d_exponential_scale, "shift VIOLATED"),
]
PAIRED_DESIGNS = [
    ("8 pairs, normal differences (calibration design)", p_normal_few, "symmetry holds"),
    ("12 pairs, heavy-tailed symmetric differences", p_heavy, "symmetry holds"),
    ("10 pairs, skewed differences", p_skewed, "symmetry VIOLATED"),
    ("8 pairs, very skewed differences", p_very_skewed, "symmetry VIOLATED"),
    ("5 pairs, normal differences", p_normal_tiny, "symmetry holds"),
]


def study(sims: int) -> dict:
    rows = []
    for d, (name, design, assumption) in enumerate(TWO_DESIGNS + PAIRED_DESIGNS):
        methods = TWO if d < len(TWO_DESIGNS) else PAIRED
        covered = {m: 0 for m in methods}
        given = {m: 0 for m in methods}
        width = {m: [] for m in methods}
        for i in range(sims):
            rng = np.random.default_rng(BASE_SEED + 100_000 * d + i)
            drawn = design(rng)
            truth, data = drawn[-1], drawn[:-1]
            for m, fn in methods.items():
                out = fn(*data, np.random.default_rng(BASE_SEED + 100_000 * d + i + 50_000))
                if out is None:                       # a guarded method declined to give an interval
                    continue
                low, high = out
                given[m] += 1
                covered[m] += low <= truth <= high
                width[m].append(high - low)
        rows.append({"design": name, "assumption": assumption, "kind": "two groups" if methods is TWO else "paired",
                     "simulations": sims, "seed_first": BASE_SEED + 100_000 * d,
                     "methods": {m: {"intervals_given": given[m], "share_given": given[m] / sims,
                                     "coverage": covered[m] / given[m] if given[m] else None,
                                     "coverage_interval": wilson(covered[m], given[m]) if given[m] else None,
                                     "median_width": float(np.median(width[m])) if width[m] else None} for m in methods}})
    return {"base_seed": BASE_SEED, "simulations_per_design": sims, "level": LEVEL, "bootstrap_resamples": BOOT,
            "permutations": PERMS, "designs": rows}


def to_text(report: dict) -> str:
    lines = [f"seed {report['base_seed']}, {report['simulations_per_design']} simulations per design, nominal {report['level']:.0%}"]
    for row in report["designs"]:
        lines.append(f"\n{row['design']}  [{row['assumption']}]")
        for m, r in row["methods"].items():
            lo, hi = r["coverage_interval"]
            lines.append(f"    {m:18s} given {r['share_given'] * 100:5.1f}%  coverage {r['coverage'] * 100:5.1f}%  "
                         f"[{lo * 100:4.1f}, {hi * 100:4.1f}]   median width {r['median_width']:8.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--json", default=str(HERE / "reports" / "small_sample_interval_study.json"))
    args = parser.parse_args()
    out = study(args.sims)
    print(to_text(out))
    Path(args.json).write_text(json.dumps(out, indent=1), encoding="utf-8")
