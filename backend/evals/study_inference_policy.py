"""Which samples should get a test of means, and which test. Report version 4.

    cd backend && python evals/study_inference_policy.py --part policies --job N --parts-dir DIR   # one family
    cd backend && python evals/study_inference_policy.py --part policies-merge --parts-dir DIR
    cd backend && python evals/study_inference_policy.py --part audit
    cd backend && python evals/study_inference_policy.py --part workflow --job N --parts-dir DIR   # the engine
    cd backend && python evals/study_inference_policy.py --part workflow-merge --parts-dir DIR

Writes evals/reports/inference_policy_v4.json. No earlier report is read for
its numbers or written to.

WHAT IS BEING DECIDED

The estimand is a difference in means (or a mean change). Report 3 withheld
the test whenever a sample failed Cochran's rule on its own skewness. That
held the workflow's error down, and it also left 45% of normal 12 + 12 samples
and nearly all same-shape lognormal samples untested, where the permutation
test is exact. This report compares policies on the same seeded datasets.

WHAT EACH TEST ASSUMES (nothing here is exact merely for being a permutation)

  Welch's t            H0: equal means. Valid when each mean is close to normal.
                       Skewness enters through the skewness of the DIFFERENCE of
                       the two means, k3_b/n_b^2 - k3_a/n_a^2 over its sd cubed:
                       equal shapes and equal sizes cancel, a small skewed group
                       beside a large mild one does not.
  Permuted Welch's t   Exact for H0: the two groups have the same distribution
                       (exchangeability). For the weaker H0 of equal means it is
                       only asymptotically valid, and that is what studentizing
                       buys: with unequal variances the permuted Welch statistic
                       still has the right limit, the permuted raw difference
                       does not. In small samples it is wrong when the groups
                       differ in skewness, and nothing fixes that: no test of a
                       mean is valid over all distributions (Bahadur and Savage).
  Sign-flip (pairs)    Exact for H0: the changes are symmetric about zero. A test
                       of the mean change only under symmetry.
  Bootstrap t          H0: equal means, second-order accurate under skewness;
                       included as an alternative fallback.

POLICIES (who is tested, and how). Every constant is one the engine already
uses: Cochran's n >= 25 skewness^2, i.e. |skewness of the mean| <= 0.2, and
the allowance of two standard errors, sqrt(6 / n), on a measured skewness.
No constant is fitted to these simulations.

  off        Cochran-with-allowance in both groups -> Welch; otherwise permuted Welch's t.
             (the engine before report 3)
  v3         as off, but a sample in which either group fails Cochran's rule on its measured
             skewness is reported without a test. (the engine at report 3)
  diff       Cochran's rule applied to the quantity tested. g = estimated skewness of the
             difference in means. |g| + 2 se <= 0.2 -> Welch. Otherwise |g| > 0.2 -> no test.
             Otherwise permuted Welch's t.
  mismatch   Welch as in "diff". Otherwise the permutation test is withheld when the skewness of
             the difference estimated from each group's own moments differs by more than 0.2
             from what permuting the pooled sample implies, i.e. when the sample shows the
             groups differing in shape in the way that breaks the permutation test. Equal shapes
             are not penalised for being skewed, whatever the sizes.
  shape      as v3, except that a visibly skewed sample is still tested when the two groups do not
             visibly differ in spread (Brown-Forsythe test of equal spread, p >= alpha). The
             permutation test needs the groups to have the same distribution, not a symmetric
             one, so skewness alone is not evidence against it; skewness together with unequal
             spread is. ADDED AFTER the first comparison had been read (v3 was the only policy
             without widespread C1 failures and left most same-shape samples untested). The
             acceptance policy was not changed, and the workflow part uses seeds this
             candidate was not chosen on.
  welch      Welch for every sample. (reference)
  boot_t     as off, with a bootstrap-t test in place of the permutation test. (reference)

A low measured skewness is never taken as evidence that a population is
symmetric: every gate withholds on evidence against, and the report gives the
error among the samples that were still tested.

ACCEPTANCE POLICY, written before any of this was run and not changed after

  For a rate r with 95% Wilson interval (lo, hi) and a limit L:
  "pass" if hi is on the right side of L, "fail" if lo is on the wrong side, "undetermined" otherwise.

  C1  Type I error of the workflow, over all samples (a sample given no test is not a
      rejection): at most 7.5% in every null cell (Bradley's liberal criterion, 1.5 alpha).
  C2  Type I error among the samples given a p-value, cells with at least 200 such
      samples: at most 10%.
  C3  Coverage where an interval is given, cells with at least 200 intervals: at least 92.5%.
  C4  Share tested, in cells where the fallback test's own assumption holds (identical
      distributions; symmetric changes) and the smaller group has at least 12 values: at least 80%.
  C5  Power kept, in power cells with at least 12 values in the smaller group: at least
      0.9 of the power of "off".

  Choice between policies: fewest C1 failures; then most C4 passes; then fewest C2 failures.
  A criterion that fails stays failed in the report.

Seeds: dataset i of family f at its s-th size uses BASE_SEED + PART + 10_000 (100 f + s) + i,
PART = 0 for policies, 300_000_000 for several groups, 600_000_000 for the workflow.
Nothing here says anything about designs that are not listed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as st

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPORT = HERE / "reports" / "inference_policy_v4.json"
BASE_SEED = 20_270_201
ALPHA, LEVEL = 0.05, 0.95
B = 999                                  # permutations / bootstrap resamples per dataset in the policy part
TOLERANCE = 0.2                          # Cochran: n >= 25 skewness^2  <=>  |skewness of the mean| <= 0.2
LIMITS = {"C1": 0.075, "C2": 0.10, "C3": 0.925, "C4": 0.80, "C5": 0.90}
POLICIES = ("off", "v3", "diff", "mismatch", "shape", "welch", "boot_t")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def rate(k: int, n: int) -> dict | None:
    return None if not n else {"rate": k / n, "interval": list(wilson(int(k), int(n))), "count": int(k), "simulations": int(n)}


def verdict(r: dict | None, limit: float, at_most: bool) -> str | None:
    if r is None:
        return None
    lo, hi = r["interval"]
    if at_most:
        return "pass" if hi <= limit else "fail" if lo > limit else "undetermined"
    return "pass" if lo >= limit else "fail" if hi < limit else "undetermined"


# ------------------------------------------------------------------ statistics --

def skewness(x: np.ndarray) -> float:
    """Adjusted Fisher-Pearson coefficient G1; 0 when it cannot be measured."""
    n = len(x)
    if n < 3:
        return 0.0
    d = x - x.mean()
    m2 = (d * d).mean()
    if m2 <= 0:
        return 0.0
    return float((d ** 3).mean() / m2 ** 1.5 * math.sqrt(n * (n - 1)) / (n - 2))


def cochran_with_allowance(x: np.ndarray) -> bool:
    n = len(x)
    return n >= 3 and n >= 25 * (abs(skewness(x)) + 2 * math.sqrt(6 / n)) ** 2


def shows_skew(x: np.ndarray) -> bool:
    return len(x) < 25 * skewness(x) ** 2


def difference_skewness(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """(skewness of mean(b) - mean(a) from each group's own moments, its
    standard error, the same skewness if both groups had the pooled shape)."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    total = va / na + vb / nb
    if total <= 0:
        return 0.0, 0.0, 0.0
    ga, gb = skewness(a), skewness(b)
    wa, wb = (va / na) / total, (vb / nb) / total
    own = gb * wb ** 1.5 / math.sqrt(nb) - ga * wa ** 1.5 / math.sqrt(na)
    se = math.sqrt(wa ** 3 * 6 / na ** 2 + wb ** 3 * 6 / nb ** 2)
    resid = np.concatenate([a - a.mean(), b - b.mean()])
    vp = (resid * resid).sum() / (na + nb - 2)
    gp = skewness(resid)
    pooled = gp * vp ** 1.5 * (1 / nb ** 2 - 1 / na ** 2) / (vp * (1 / na + 1 / nb)) ** 1.5 if vp > 0 else 0.0
    return own, se, pooled


def spread_differs(a: np.ndarray, b: np.ndarray) -> bool:
    """Brown-Forsythe: a one-way test on absolute deviations from each group's median."""
    da, db = np.abs(a - np.median(a)), np.abs(b - np.median(b))
    if da.var() + db.var() == 0:
        return bool(da.mean() != db.mean())
    return bool(st.ttest_ind(da, db, equal_var=True).pvalue < ALPHA)


def welch_t(a: np.ndarray, b: np.ndarray):
    """Welch's t, along the last axis. No variation: 0 if the means agree, else infinite."""
    va, vb = a.var(axis=-1, ddof=1) / a.shape[-1], b.var(axis=-1, ddof=1) / b.shape[-1]
    diff = b.mean(axis=-1) - a.mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(va + vb > 0, diff / np.sqrt(va + vb), np.where(diff == 0, 0.0, np.copysign(np.inf, diff)))


def permutation_p(a: np.ndarray, b: np.ndarray, rng) -> float:
    z, na = np.concatenate([a, b]), len(a)
    zp = z[np.argsort(rng.random((B, len(z))), axis=1)]
    t = np.abs(welch_t(zp[:, :na], zp[:, na:]))
    return float((1 + (t >= abs(float(welch_t(a, b))) * (1 - 1e-9)).sum()) / (B + 1))


def bootstrap_t_p(a: np.ndarray, b: np.ndarray, rng) -> float:
    """Each group resampled from itself, centred at its own mean: the null
    distribution of Welch's t without assuming the groups share a shape."""
    ra = a[rng.integers(0, len(a), (B, len(a)))] - a.mean()
    rb = b[rng.integers(0, len(b), (B, len(b)))] - b.mean()
    t = np.abs(welch_t(ra, rb))
    return float((1 + (t >= abs(float(welch_t(a, b))) * (1 - 1e-9)).sum()) / (B + 1))


def decide(a: np.ndarray, b: np.ndarray) -> dict:
    """For each policy: "welch", "fallback" or "none" (no test)."""
    eligible = cochran_with_allowance(a) and cochran_with_allowance(b)
    own, se, pooled = difference_skewness(a, b)
    eligible_diff = min(len(a), len(b)) >= 3 and abs(own) + 2 * se <= TOLERANCE
    return {"off": "welch" if eligible else "fallback",
            "v3": "welch" if eligible else "none" if shows_skew(a) or shows_skew(b) else "fallback",
            "diff": "welch" if eligible_diff else "none" if abs(own) > TOLERANCE else "fallback",
            "mismatch": "welch" if eligible_diff else "none" if abs(own - pooled) > TOLERANCE else "fallback",
            "shape": "welch" if eligible else "none" if (shows_skew(a) or shows_skew(b)) and spread_differs(a, b) else "fallback",
            "welch": "welch", "boot_t": "welch" if eligible else "fallback"}


# -------------------------------------------------------------------- designs --

def _ln(rng, n, sigma, mean=30.0):
    return rng.lognormal(math.log(mean) - sigma * sigma / 2, sigma, n)


def _contaminated(rng, n, mean=30.0):
    """95% normal, 5% a second normal 60 higher; the overall mean is `mean`."""
    high = rng.random(n) < 0.05
    return np.where(high, rng.normal(mean - 3 + 60, 6, n), rng.normal(mean - 3, 6, n))


RATINGS, FLAT = np.array([1.0, 2, 3, 4, 5]), np.array([0.2, 0.2, 0.2, 0.2, 0.2])
LN_SD = 30.0 * math.sqrt(math.e - 1)
# (family, what it is, generator(rng, n_small, n_large) -> (a, b), does the permutation test's own assumption hold,
#  true difference in means)
TWO = [
    ("normal, same spread", "identical", lambda r, s, l: (r.normal(30, 6, l), r.normal(30, 6, s)), True, 0.0),
    ("lognormal, same shape", "identical, skewed", lambda r, s, l: (_ln(r, l, 1.0), _ln(r, s, 1.0)), True, 0.0),
    ("ratings 1 to 5, same distribution", "identical, ties", lambda r, s, l: (r.choice(RATINGS, l, p=FLAT), r.choice(RATINGS, s, p=FLAT)), True, 0.0),
    ("normal with 5% outliers, same in both", "identical, outliers", lambda r, s, l: (_contaminated(r, l), _contaminated(r, s)), True, 0.0),
    ("normal, smaller group 3x the spread", "unequal variances", lambda r, s, l: (r.normal(30, 4, l), r.normal(30, 12, s)), False, 0.0),
    ("normal, larger group 3x the spread", "unequal variances", lambda r, s, l: (r.normal(30, 12, l), r.normal(30, 4, s)), False, 0.0),
    ("smaller group strongly skewed, larger mildly", "different shapes", lambda r, s, l: (_ln(r, l, 0.4), _ln(r, s, 1.2)), False, 0.0),
    ("smaller group exponential, larger normal", "different shapes", lambda r, s, l: (r.normal(30, 6, l), r.exponential(30, s)), False, 0.0),
    ("larger group strongly skewed, smaller mildly", "different shapes", lambda r, s, l: (_ln(r, l, 1.2), _ln(r, s, 0.4)), False, 0.0),
    ("smaller group exponential, larger normal with the same variance", "different shapes, equal variances",
     lambda r, s, l: (r.normal(30, 30, l), r.exponential(30, s)), False, 0.0),
    ("smaller group has 5% outliers, larger normal", "different shapes, outliers", lambda r, s, l: (r.normal(30, 6, l), _contaminated(r, s)), False, 0.0),
    ("normal, shifted 0.8 sd", "power", lambda r, s, l: (r.normal(30, 6, l), r.normal(30, 6, s) + 4.8), True, 4.8),
    ("lognormal, same shape, shifted 0.8 sd", "power", lambda r, s, l: (_ln(r, l, 1.0), _ln(r, s, 1.0) + 0.8 * LN_SD), True, 0.8 * LN_SD),
    ("normal, smaller group 3x the spread, shifted 0.8 of its sd", "power", lambda r, s, l: (r.normal(30, 4, l), r.normal(30, 12, s) + 9.6), False, 9.6),
]
TWO_SIZES = [(5, 5), (5, 15), (8, 8), (8, 20), (12, 12), (12, 30), (20, 20), (20, 50), (30, 30), (30, 75), (50, 50), (80, 80)]


def policies_job(index: int, sims: int) -> dict:
    family, what, gen, holds, truth = TWO[index]
    rows = []
    for si, (n_small, n_large) in enumerate(TWO_SIZES):
        seed_first = BASE_SEED + 10_000 * (100 * index + si)
        count = {p: {"welch": 0, "fallback": 0, "none": 0, "rejected": 0, "rejected_welch": 0, "rejected_fallback": 0} for p in POLICIES}
        for i in range(sims):
            rng = np.random.default_rng(seed_first + i)
            a, b = gen(rng, n_small, n_large)
            if np.ptp(np.concatenate([a, b])) == 0:
                for p in POLICIES:
                    count[p]["none"] += 1
                continue
            routes = decide(a, b)
            p_welch = float(st.ttest_ind(a, b, equal_var=False).pvalue) if a.var() + b.var() > 0 else 1.0
            p_perm = permutation_p(a, b, rng) if "fallback" in [routes[p] for p in POLICIES if p != "boot_t"] else None
            p_boot = bootstrap_t_p(a, b, rng) if routes["boot_t"] == "fallback" else None
            for p in POLICIES:
                route = routes[p]
                count[p][route] += 1
                if route == "none":
                    continue
                value = p_welch if route == "welch" else p_boot if p == "boot_t" else p_perm
                hit = value < ALPHA
                count[p]["rejected"] += hit
                count[p][f"rejected_{route}"] += hit
        key = "type_one_error" if truth == 0 else "power"
        out = {}
        for p in POLICIES:
            c = count[p]
            tested = c["welch"] + c["fallback"]
            out[p] = {f"{key}_of_all_samples": rate(c["rejected"], sims), f"{key}_of_tested_samples": rate(c["rejected"], tested),
                      "tested": tested / sims, "by_welch": c["welch"], "by_fallback": c["fallback"], "not_tested": c["none"],
                      "rejected_of_welch": rate(c["rejected_welch"], c["welch"]), "rejected_of_fallback": rate(c["rejected_fallback"], c["fallback"])}
        rows.append({"family": family, "what": what, "fallback_assumption_holds": holds, "null_is_true": truth == 0, "true_difference": truth,
                     "smaller": n_small, "larger": n_large, "seed_first": seed_first, "simulations": sims, "policies": out})
        f = lambda r: "  -  " if not r else f"{r['rate'] * 100:5.1f}"  # noqa: E731
        print(f"{family[:44]:44s} {n_small:3d}+{n_large:<3d} " + "  ".join(
            f"{p} {f(out[p][key + '_of_all_samples'])}/{f(out[p][key + '_of_tested_samples'])} t{out[p]['tested'] * 100:3.0f}" for p in POLICIES), flush=True)
    return {"index": index, "rows": rows}


def judge_policies(rows: list[dict]) -> dict:
    """The acceptance policy applied to every policy. Counts of pass / fail / undetermined per criterion."""
    off_power = {(r["family"], r["smaller"], r["larger"]): r["policies"]["off"].get("power_of_all_samples") for r in rows if not r["null_is_true"]}
    summary = {}
    for p in POLICIES:
        tally = {c: {"pass": 0, "fail": 0, "undetermined": 0, "failed_cells": []} for c in ("C1", "C2", "C4", "C5")}

        def add(c, v, r):
            if v:
                tally[c][v] += 1
                if v == "fail":
                    tally[c]["failed_cells"].append(f"{r['family']} {r['smaller']}+{r['larger']}")
        for r in rows:
            x = r["policies"][p]
            if r["null_is_true"]:
                add("C1", verdict(x["type_one_error_of_all_samples"], LIMITS["C1"], True), r)
                t = x["type_one_error_of_tested_samples"]
                if t and t["simulations"] >= 200:
                    add("C2", verdict(t, LIMITS["C2"], True), r)
                if r["fallback_assumption_holds"] and r["smaller"] >= 12:
                    add("C4", "pass" if x["tested"] >= LIMITS["C4"] else "fail", r)
            elif r["smaller"] >= 12:
                base = off_power[(r["family"], r["smaller"], r["larger"])]["rate"]
                add("C5", "pass" if x["power_of_all_samples"]["rate"] >= LIMITS["C5"] * base else "fail", r)
        summary[p] = tally
    testing = [p for p in POLICIES if p not in ("welch", "boot_t")]            # the references are not candidates
    order = sorted(testing, key=lambda p: (summary[p]["C1"]["fail"], -summary[p]["C4"]["pass"], summary[p]["C2"]["fail"]))
    return {"by_policy": summary, "candidates": testing, "chosen_by_the_declared_rule": order[0], "order": order}


# ------------------------------------------------------------- several groups --

def welch_f(groups: list[np.ndarray]):
    """Welch's F along the last axis of each group array."""
    k = len(groups)
    n = np.array([g.shape[-1] for g in groups], float)
    means = np.stack([g.mean(axis=-1) for g in groups])
    var = np.stack([g.var(axis=-1, ddof=1) for g in groups])
    with np.errstate(divide="ignore", invalid="ignore"):
        w = n[:, None] / var if var.ndim > 1 else n / var
        total = w.sum(axis=0)
        grand = (w * means).sum(axis=0) / total
        num = (w * (means - grand) ** 2).sum(axis=0) / (k - 1)
        lam = (((1 - w / total) ** 2) / ((n[:, None] if var.ndim > 1 else n) - 1)).sum(axis=0)
        f = num / (1 + 2 * (k - 2) / (k * k - 1) * lam)
    return np.nan_to_num(f, nan=0.0, posinf=np.inf), k - 1, (k * k - 1) / (3 * lam)


def groups_p(groups: list[np.ndarray], rng) -> tuple[float, float]:
    """(Welch's F p-value, permuted Welch's F p-value)."""
    f, d1, d2 = welch_f(groups)
    p_f = float(st.f.sf(float(f), d1, float(d2))) if np.isfinite(d2) else 1.0
    z = np.concatenate(groups)
    cuts = np.cumsum([len(g) for g in groups])[:-1]
    zp = z[np.argsort(rng.random((B, len(z))), axis=1)]
    fp = welch_f(np.split(zp, cuts, axis=1))[0]
    return p_f, float((1 + (fp >= float(f) * (1 - 1e-9)).sum()) / (B + 1))


# (family, what, generator(rng, sizes) -> list of groups, does the permutation test's assumption hold, null true)
GROUPS = [
    ("three normal groups, same spread", "identical", lambda r, n: [r.normal(30, 6, k) for k in n], True, True),
    ("three lognormal groups, same shape", "identical, skewed", lambda r, n: [_ln(r, k, 1.0) for k in n], True, True),
    ("three normal groups, last one 4x the spread", "unequal variances", lambda r, n: [r.normal(30, 4, n[0]), r.normal(30, 4, n[1]), r.normal(30, 16, n[2])], False, True),
    ("two mild lognormal groups and one strongly skewed (last)", "different shapes", lambda r, n: [_ln(r, n[0], 0.4), _ln(r, n[1], 0.4), _ln(r, n[2], 1.2)], False, True),
    ("two normal groups and one exponential (last)", "different shapes", lambda r, n: [r.normal(30, 6, n[0]), r.normal(30, 6, n[1]), r.exponential(30, n[2])], False, True),
    ("three normal groups, last shifted 0.8 sd", "power", lambda r, n: [r.normal(30, 6, n[0]), r.normal(30, 6, n[1]), r.normal(34.8, 6, n[2])], True, False),
    ("three lognormal groups, last shifted 0.8 sd", "power", lambda r, n: [_ln(r, n[0], 1.0), _ln(r, n[1], 1.0), _ln(r, n[2], 1.0) + 0.8 * LN_SD], True, False),
]
GROUP_SIZES = [(6, 6, 6), (10, 10, 10), (20, 20, 6), (20, 20, 20), (40, 40, 12), (40, 40, 40), (80, 80, 80)]
GROUP_POLICIES = ("off", "v3")


def groups_job(index: int, sims: int) -> dict:
    family, what, gen, holds, null = GROUPS[index]
    rows = []
    for si, sizes in enumerate(GROUP_SIZES):
        seed_first = BASE_SEED + 300_000_000 + 10_000 * (100 * index + si)
        count = {p: {"welch": 0, "fallback": 0, "none": 0, "rejected": 0} for p in GROUP_POLICIES}
        for i in range(sims):
            rng = np.random.default_rng(seed_first + i)
            g = gen(rng, sizes)
            eligible = all(cochran_with_allowance(x) for x in g)
            shown = any(shows_skew(x) for x in g)
            p_f, p_perm = groups_p(g, rng)
            for p in GROUP_POLICIES:
                route = "welch" if eligible else "none" if p == "v3" and shown else "fallback"
                count[p][route] += 1
                if route != "none":
                    count[p]["rejected"] += (p_f if route == "welch" else p_perm) < ALPHA
        key = "type_one_error" if null else "power"
        out = {p: {f"{key}_of_all_samples": rate(c["rejected"], sims), f"{key}_of_tested_samples": rate(c["rejected"], c["welch"] + c["fallback"]),
                   "tested": (c["welch"] + c["fallback"]) / sims, "by_welch": c["welch"], "by_fallback": c["fallback"], "not_tested": c["none"]}
               for p, c in count.items()}
        rows.append({"family": family, "what": what, "fallback_assumption_holds": holds, "null_is_true": null, "sizes": list(sizes),
                     "smaller": min(sizes), "larger": max(sizes), "seed_first": seed_first, "simulations": sims, "policies": out})
        f = lambda r: "  -  " if not r else f"{r['rate'] * 100:5.1f}"  # noqa: E731
        print(f"{family[:52]:52s} {str(sizes):14s} " + "  ".join(
            f"{p} {f(out[p][key + '_of_all_samples'])}/{f(out[p][key + '_of_tested_samples'])} t{out[p]['tested'] * 100:3.0f}" for p in GROUP_POLICIES), flush=True)
    return {"index": index, "rows": rows}


# ----------------------------------------------------------------------- audit --

def audit(sims: int) -> dict:
    """Larger runs of the cells report 3 left in doubt. Independent of the application
    except where the engine's own interval is the thing being audited."""
    import study_method_decisions as v3
    from app.stats import methods as m

    out = {"simulations": sims}
    # 1. Brunner-Munzel, 5 against 15, the smaller group three times as variable (and its neighbours)
    rows = []
    designs = [("5 (sd 12) + 15 (sd 4)", lambda r: (r.normal(30, 4, 15), r.normal(30, 12, 5))),
               ("5 (sd 4) + 15 (sd 12)", lambda r: (r.normal(30, 12, 15), r.normal(30, 4, 5))),
               ("5 + 15, same sd", lambda r: (r.normal(30, 6, 15), r.normal(30, 6, 5))),
               ("8 (sd 12) + 20 (sd 4)", lambda r: (r.normal(30, 4, 20), r.normal(30, 12, 8))),
               ("12 (sd 12) + 30 (sd 4)", lambda r: (r.normal(30, 4, 30), r.normal(30, 12, 12)))]
    for d, (name, gen) in enumerate(designs):
        hits_perm = hits_t = hits_mw = 0
        for i in range(sims):
            rng = np.random.default_rng(BASE_SEED + 900_000_000 + 100_000 * d + i)
            a, b = gen(rng)
            r = v3.brunner_munzel(a, b)
            pt = v3.bm_t_p(r)
            hits_t += pt is not None and pt < ALPHA
            hits_perm += v3.bm_permutation_p(a, b, rng) < ALPHA
            hits_mw += st.mannwhitneyu(b, a, alternative="two-sided").pvalue < ALPHA
        rows.append({"design": name, "null_is_true": True, "seed_first": BASE_SEED + 900_000_000 + 100_000 * d,
                     "brunner_munzel_permutation": rate(hits_perm, sims), "brunner_munzel_t": rate(hits_t, sims), "mann_whitney": rate(hits_mw, sims)})
        print("[audit BM]", name, {k: round(v["rate"] * 100, 2) for k, v in rows[-1].items() if isinstance(v, dict)}, flush=True)
    out["brunner_munzel_unequal_spread"] = rows
    # 2. the engine's bootstrap interval for the difference in medians (true difference 0)
    rows = []
    for d, (name, gen) in enumerate(designs[:1] + designs[3:] + [("12 + 12 lognormal, same shape", lambda r: (_ln(r, 12, 1.0), _ln(r, 12, 1.0))),
                                                                    ("5 + 15 ratings", lambda r: (r.choice(RATINGS, 15, p=FLAT), r.choice(RATINGS, 5, p=FLAT)))]):
        covered = given = 0
        n = max(sims // 4, 1000)
        for i in range(n):
            rng = np.random.default_rng(BASE_SEED + 910_000_000 + 100_000 * d + i)
            a, b = gen(rng)
            iv = m._bootstrap((a, b), m._median_diff, LEVEL, int(rng.integers(1 << 30)))
            if iv:
                given += 1
                covered += iv[0] <= 0.0 <= iv[1]
        rows.append({"design": name, "seed_first": BASE_SEED + 910_000_000 + 100_000 * d, "coverage": rate(covered, given), "intervals_not_given": n - given})
        print("[audit median interval]", name, round(rows[-1]["coverage"]["rate"] * 100, 2), flush=True)
    out["median_difference_interval"] = rows
    # 3. pairs: mostly small losses, a few large gains (true mean change 0), by how many gains the sample holds
    rows = []
    for d, n in enumerate((8, 12, 20)):
        by = {}
        signs = None
        for i in range(sims):
            rng = np.random.default_rng(BASE_SEED + 920_000_000 + 100_000 * d + i)
            gain = rng.random(n) < 0.2
            x = np.where(gain, rng.normal(8, 3, n), rng.normal(-2, 1, n))
            flips = rng.choice([-1.0, 1.0], (B, n))
            p = (1 + (np.abs((flips * x).mean(axis=1)) >= abs(x.mean()) * (1 - 1e-9)).sum()) / (B + 1)
            c = by.setdefault(min(int(gain.sum()), 3), {"samples": 0, "shows_skew": 0, "rejected": 0, "tested_and_rejected": 0})
            c["samples"] += 1
            c["shows_skew"] += shows_skew(x)
            c["rejected"] += p < ALPHA
            c["tested_and_rejected"] += (not shows_skew(x)) and p < ALPHA
        rows.append({"pairs": n, "seed_first": BASE_SEED + 920_000_000 + 100_000 * d,
                     "by_large_gains_in_the_sample": {("3 or more" if k == 3 else str(k)): {
                         "share_of_samples": v["samples"] / sims, "reported_without_a_test": v["shows_skew"] / v["samples"],
                         "sign_flip_rejects": v["rejected"] / v["samples"], "tested_and_rejected_share_of_all_samples": v["tested_and_rejected"] / sims}
                         for k, v in sorted(by.items())},
                     "workflow_type_one_error_of_all_samples": rate(sum(v["tested_and_rejected"] for v in by.values()), sims)})
        print("[audit pairs]", n, json.dumps(rows[-1]["by_large_gains_in_the_sample"])[:400], round(rows[-1]["workflow_type_one_error_of_all_samples"]["rate"] * 100, 2), flush=True)
    out["pairs_small_losses_large_gains"] = rows
    return out


# ------------------------------------------------- the engine, sample by sample --

def _ln_changes(rng, n, sigma, centre):
    x = rng.lognormal(1, sigma, n)
    return x - (math.exp(1 + sigma * sigma / 2) if centre == "mean" else math.e)


# (family, what, generator(rng, n) -> changes, true mean change, does the sign-flip test's assumption hold,
#  is the mean change zero, is a change as likely to be up as down)
PAIRED = [
    ("normal changes", "symmetric", lambda r, n: r.normal(0, 4, n), 0.0, True, True, True),
    ("heavy-tailed changes, symmetric", "symmetric", lambda r, n: r.standard_t(3, n) * 4, 0.0, True, True, True),
    ("whole-number changes, symmetric", "symmetric, ties", lambda r, n: r.choice([-2.0, -1, 0, 1, 2], n, p=[.1, .2, .4, .2, .1]), 0.0, True, True, True),
    ("normal changes with 5% large ones either way", "symmetric, outliers",
     lambda r, n: np.where(r.random(n) < 0.05, r.normal(0, 40, n), r.normal(0, 4, n)), 0.0, True, True, True),
    ("skewed changes, mean zero (lognormal, shape 0.9)", "asymmetric", lambda r, n: _ln_changes(r, n, 0.9, "mean"), 0.0, False, True, False),
    ("mildly skewed changes, mean zero (lognormal, shape 0.4)", "asymmetric", lambda r, n: _ln_changes(r, n, 0.4, "mean"), 0.0, False, True, False),
    ("mostly small losses, a few large gains, mean zero", "asymmetric, outliers",
     lambda r, n: np.where(r.random(n) < 0.2, r.normal(8, 3, n), r.normal(-2, 1, n)), 0.0, False, True, False),
    ("skewed changes, median zero (lognormal, shape 0.9)", "asymmetric; the mean change is not zero",
     lambda r, n: _ln_changes(r, n, 0.9, "median"), math.exp(1 + 0.405) - math.e, False, False, True),
    ("normal changes, mean 0.8 sd", "power", lambda r, n: r.normal(3.2, 4, n), 3.2, True, False, False),
]
PAIRED_SIZES = [5, 8, 12, 20, 30, 50, 80]
TENDENCY_NULL = ("identical", "identical, skewed", "identical, ties", "identical, outliers", "unequal variances")


def workflow_jobs() -> list[tuple[str, int]]:
    return [("two", i) for i in range(len(TWO))] + [("paired", i) for i in range(len(PAIRED))] + [("groups", i) for i in range(len(GROUPS))]


def workflow_job(index: int, sims: int) -> dict:
    """One family, every size, through the engine's selector as the application calls it."""
    from app.stats import selector

    kind, f = workflow_jobs()[index]
    if kind == "two":
        family, what, gen, holds, truth = TWO[f]
        sizes, mean_null, other_null = TWO_SIZES, truth == 0, what in TENDENCY_NULL
    elif kind == "paired":
        family, what, gen, truth, holds, mean_null, other_null = PAIRED[f]
        sizes = [(n, n) for n in PAIRED_SIZES]
    else:
        family, what, gen, holds, mean_null = GROUPS[f]
        sizes, truth, other_null = GROUP_SIZES, None, False
    rows = []
    for si, size in enumerate(sizes):
        seed_first = BASE_SEED + 600_000_000 + 10_000 * (100 * index + si)
        c = {"tested": 0, "rejected": 0, "not_tested": 0, "unavailable": 0, "intervals": 0, "covered": 0, "other_given": 0, "other_rejected": 0}
        methods, reasons = {}, {}
        for i in range(sims):
            rng = np.random.default_rng(seed_first + i)
            if kind == "two":
                a, b = gen(rng, size[0], size[1])
                r = selector.two_groups(a, b, what="y", labels=("a", "b"))
            elif kind == "paired":
                r = selector.paired(np.zeros(size[0]), gen(rng, size[0]), what="y", unit_name="unit")
            else:
                r = selector.k_groups({str(k): v for k, v in enumerate(gen(rng, size))}, what="y", group_name="g")
            methods[r["method"]] = methods.get(r["method"], 0) + 1
            p = (r.get("test") or {}).get("p_value")
            withheld = r.get("inference_unavailable")
            if p is not None:
                c["tested"] += 1
                c["rejected"] += p < ALPHA
            elif withheld:
                c["not_tested"] += 1
                reasons[withheld["code"]] = reasons.get(withheld["code"], 0) + 1
                other = withheld.get("supported_analysis")
                if other:
                    c["other_given"] += 1
                    c["other_rejected"] += other["p_value"] < ALPHA
            else:
                c["unavailable"] += 1
                reasons["no test possible"] = reasons.get("no test possible", 0) + 1
            iv = r.get("interval")
            if iv is not None and truth is not None:
                c["intervals"] += 1
                c["covered"] += iv["low"] <= truth <= iv["high"]
        key = "type_one_error" if mean_null else "rejections_mean_not_zero" if kind == "paired" and "median zero" in family else "power"
        row = {"design": kind, "family": family, "what": what, "fallback_assumption_holds": holds, "mean_null_is_true": mean_null,
               "true_difference": truth, "sizes": list(size) if kind != "paired" else [size[0]], "smaller": min(size), "larger": max(size),
               "seed_first": seed_first, "simulations": sims, "tested": c["tested"] / sims, "not_tested": c["not_tested"],
               "no_test_possible": c["unavailable"], "reasons_not_tested": reasons, "methods": methods,
               f"{key}_of_all_samples": rate(c["rejected"], sims), f"{key}_of_tested_samples": rate(c["rejected"], c["tested"]),
               "coverage_where_an_interval_was_given": rate(c["covered"], c["intervals"]), "intervals_given": c["intervals"],
               "different_question": {"analysis": "which group tends to be higher (Brunner-Munzel)" if kind == "two" else "is a change as likely up as down (sign test)" if kind == "paired" else None,
                                      "given": c["other_given"], "its_null_is_true": other_null,
                                      "rejections_among_samples_given_it": rate(c["other_rejected"], c["other_given"])}}
        rows.append(row)
        f1 = lambda x: "  -  " if not x else f"{x['rate'] * 100:5.1f}"  # noqa: E731
        print(f"[{kind:6s}] {family[:46]:46s} {str(row['sizes']):12s} {key[:6]} all {f1(row[key + '_of_all_samples'])} tested-only {f1(row[key + '_of_tested_samples'])} "
              f"tested {row['tested'] * 100:3.0f}% cover {f1(row['coverage_where_an_interval_was_given'])} other {f1(row['different_question']['rejections_among_samples_given_it'])} "
              f"({c['other_given']}, null {other_null})", flush=True)
    return {"index": index, "rows": rows}


def judge_workflow(rows: list[dict]) -> dict:
    tally = {c: {"pass": 0, "fail": 0, "undetermined": 0, "failed_cells": [], "undetermined_cells": []} for c in ("C1", "C2", "C3", "C4", "D1")}

    def add(c, v, r):
        if v:
            tally[c][v] += 1
            if v != "pass":
                tally[c]["failed_cells" if v == "fail" else "undetermined_cells"].append(
                    f"{r['design']}: {r['family']} {'+'.join(str(x) for x in r['sizes'])}")
    for r in rows:
        if r["mean_null_is_true"]:
            add("C1", verdict(r["type_one_error_of_all_samples"], LIMITS["C1"], True), r)
            t = r["type_one_error_of_tested_samples"]
            if t and t["simulations"] >= 200:
                add("C2", verdict(t, LIMITS["C2"], True), r)
            if r["fallback_assumption_holds"] and r["smaller"] >= 12:
                add("C4", "pass" if r["tested"] >= LIMITS["C4"] else "fail", r)
        cov = r["coverage_where_an_interval_was_given"]
        if cov and cov["simulations"] >= 200:
            add("C3", verdict(cov, LIMITS["C3"], False), r)
        d = r["different_question"]
        if d["its_null_is_true"] and d["rejections_among_samples_given_it"] and d["given"] >= 200:
            # the same limit as C1, applied to the different-question analysis among the samples that were given it
            add("D1", verdict(d["rejections_among_samples_given_it"], LIMITS["C1"], True), r)
    return tally


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", required=True)
    parser.add_argument("--job", type=int)
    parser.add_argument("--sims", type=int, default=4000)
    parser.add_argument("--parts-dir")
    args = parser.parse_args()
    started = time.time()
    if args.part == "policies":
        out = policies_job(args.job, args.sims)
        Path(args.parts_dir, f"policies_{args.job:02d}.json").write_text(json.dumps(out), encoding="utf-8")
        print(f"job {args.job} done in {time.time() - started:.0f}s")
        sys.exit(0)
    if args.part == "groups":
        out = groups_job(args.job, args.sims)
        Path(args.parts_dir, f"groups_{args.job:02d}.json").write_text(json.dumps(out), encoding="utf-8")
        sys.exit(0)
    if args.part == "workflow":
        out = workflow_job(args.job, args.sims)
        Path(args.parts_dir, f"workflow_{args.job:02d}.json").write_text(json.dumps(out), encoding="utf-8")
        sys.exit(0)
    if args.part == "audit":
        Path(args.parts_dir, "audit.json").write_text(json.dumps(audit(args.sims)), encoding="utf-8")
        sys.exit(0)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {
        "report_version": 4, "base_seed": BASE_SEED, "alpha": ALPHA, "acceptance_limits": LIMITS,
        "acceptance_policy": __doc__.split("ACCEPTANCE POLICY")[1].split("Seeds:")[0].strip()}
    if args.part == "policies-merge":
        parts = [json.loads(p.read_text()) for p in sorted(Path(args.parts_dir).glob("policies_*.json"))]
        assert [p["index"] for p in parts] == list(range(len(TWO))), "every family must have run"
        rows = [r for p in parts for r in p["rows"]]
        report["two_groups_policies"] = {"simulations_per_cell": args.sims, "resamples_per_dataset": B, "policies": list(POLICIES),
                                         "judgement": judge_policies(rows), "cells": rows}
    if args.part == "groups-merge":
        parts = [json.loads(p.read_text()) for p in sorted(Path(args.parts_dir).glob("groups_*.json"))]
        assert [p["index"] for p in parts] == list(range(len(GROUPS))), "every family must have run"
        rows = [r for p in parts for r in p["rows"]]
        report["several_groups_policies"] = {"simulations_per_cell": args.sims, "resamples_per_dataset": B, "policies": list(GROUP_POLICIES), "cells": rows}
        report["audit"] = json.loads(Path(args.parts_dir, "audit.json").read_text())
    if args.part == "workflow-merge":
        parts = [json.loads(p.read_text()) for p in sorted(Path(args.parts_dir).glob("workflow_*.json"))]
        assert [p["index"] for p in parts] == list(range(len(workflow_jobs()))), "every family must have run"
        rows = [r for p in parts for r in p["rows"]]
        report["engine_workflow"] = {
            "simulations_per_cell": args.sims, "seed_offset": 600_000_000,
            "note": "the engine's selector on seeds no policy was chosen on. D1 is C1's limit applied to the different-question "
                    "analysis among the samples given it, where its own null is true; it was added with that analysis.",
            "judgement": judge_workflow(rows), "cells": rows}
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {REPORT} in {time.time() - started:.0f}s")
