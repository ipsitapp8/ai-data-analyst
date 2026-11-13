"""Evidence for three method decisions. Report version 3.

    cd backend && python evals/study_method_decisions.py --part rank      # independent of the application
    # the engine's selector, one family per job (jobs can run side by side), then one merge per estimand
    cd backend && python evals/study_method_decisions.py --part workflow-job --job 0 --parts-dir DIR [--estimand typical]
    cd backend && python evals/study_method_decisions.py --part workflow-merge --parts-dir DIR [--estimand typical]

Writes evals/reports/method_decisions_v3.json. Earlier reports
(comparison_calibration*.json, permutation_audit*.json) are not touched.

PART "rank": Mann-Whitney against Brunner-Munzel.

  What each tests.
    Mann-Whitney     H0: the two distributions are identical. It is sensitive
                     to one group tending to be higher, but its p-value is
                     only right for "neither tends to be higher" when the two
                     groups have the same shape.
    Brunner-Munzel   H0: p = 1/2, where p = P(a value from B exceeds one from A)
                     + half the chance of a tie. That is "neither tends to be
                     higher", and nothing else: the groups may differ in spread.

  Both are about the same quantity, p, the probability of superiority. Neither
  is a test of medians.

  Everything in this part is implemented here (numpy for arithmetic, scipy only
  for the t distribution and for Mann-Whitney itself). Brunner-Munzel is written
  out from its definition, in a t-approximation and a permutation form.

PART "workflow": the complete selector, as a user's data meets it.

  For a grid of sample sizes and population shapes with a true difference of
  zero, the engine's selector is called and whatever it decides is recorded:
  which methods were chosen and how often, how often a sample was reported
  without a test, the Type I error (of all samples, and of those given a
  p-value), interval coverage; and, with a planted difference, power.

  Each sample is put through two workflows: the engine as it stands, and the
  same engine with the rule that reports a visibly skewed small sample without
  a test switched off, which is what the engine did before this report.

Seeds: rank part, simulation i of design c uses BASE_SEED + 10_000 c + i. Workflow part,
simulation i of family j at its s-th size uses BASE_SEED + 400_000_000 + 10_000 (100 j + s) + i.
Nothing here says anything about designs that are not listed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as st

BASE_SEED = 20_270_101
ALPHA = 0.05
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
REPORT = HERE / "reports" / "method_decisions_v3.json"

LN_MEAN = lambda mu, s: math.exp(mu + s * s / 2)   # noqa: E731


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(k, n) -> dict | None:
    return None if not n else {"rate": float(k) / n, "interval": wilson(int(k), int(n)), "count": int(k), "simulations": int(n)}


# ------------------------------------------------------------ Brunner-Munzel --

def brunner_munzel(a: np.ndarray, b: np.ndarray) -> dict:
    """From the definition (Brunner and Munzel, 2000). Ranks are midranks.

    p_hat = (mean rank of b in the pooled sample - (n_b + 1) / 2) / n_a
    The variance uses, for each value, its pooled rank minus its rank within
    its own group. statistic = (p_hat - 1/2) / se, referred to a t distribution
    with Satterthwaite degrees of freedom."""
    na, nb = len(a), len(b)
    pooled = st.rankdata(np.r_[a, b])
    ra, rb = pooled[:na], pooled[na:]
    wa, wb = st.rankdata(a), st.rankdata(b)
    p_hat = (rb.mean() - (nb + 1) / 2) / na
    sa = ((ra - wa - ra.mean() + (na + 1) / 2) ** 2).sum() / (na - 1)
    sb = ((rb - wb - rb.mean() + (nb + 1) / 2) ** 2).sum() / (nb - 1)
    v = na * sa + nb * sb
    se = math.sqrt(v) / (na * nb)
    if se == 0:
        return {"p_hat": p_hat, "se": 0.0, "statistic": math.copysign(math.inf, p_hat - 0.5) if p_hat != 0.5 else 0.0, "df": None}
    df = v * v / ((na * sa) ** 2 / (na - 1) + (nb * sb) ** 2 / (nb - 1))
    return {"p_hat": p_hat, "se": se, "statistic": (p_hat - 0.5) / se, "df": df}


def bm_t_p(r: dict) -> float | None:
    if r["df"] is None or not math.isfinite(r["statistic"]):
        return None
    return float(2 * st.t.sf(abs(r["statistic"]), r["df"]))


def _bm_statistics(za: np.ndarray, zb: np.ndarray, ra: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """The Brunner-Munzel statistic for many splits at once (one per row).
    `ra`, `rb` are the pooled ranks of the values in `za`, `zb`."""
    na, nb = za.shape[1], zb.shape[1]
    wa, wb = st.rankdata(za, axis=1), st.rankdata(zb, axis=1)
    p_hat = (rb.mean(axis=1) - (nb + 1) / 2) / na
    sa = ((ra - wa - ra.mean(axis=1, keepdims=True) + (na + 1) / 2) ** 2).sum(axis=1) / (na - 1)
    sb = ((rb - wb - rb.mean(axis=1, keepdims=True) + (nb + 1) / 2) ** 2).sum(axis=1) / (nb - 1)
    se = np.sqrt(na * sa + nb * sb) / (na * nb)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(se > 0, (p_hat - 0.5) / se, np.where(p_hat == 0.5, 0.0, np.copysign(np.inf, p_hat - 0.5)))


def bm_permutation_p(a: np.ndarray, b: np.ndarray, rng, resamples: int = 999) -> float:
    """The same statistic with its null distribution from permuting group
    labels. Exact when the groups are exchangeable; two-sided by absolute value."""
    na, z = len(a), np.r_[a, b]
    n = len(z)
    pooled = st.rankdata(z)
    observed = abs(float(_bm_statistics(a[None, :], b[None, :], pooled[None, :na], pooled[None, na:])[0]))
    exact = math.comb(n, na) <= resamples
    if exact:
        first = np.array(list(itertools.combinations(range(n), na)))
        mask = np.ones((len(first), n), bool)
        np.put_along_axis(mask, first, False, axis=1)
        second = np.nonzero(mask)[1].reshape(len(first), n - na)
    else:
        order = np.argsort(rng.random((resamples, n)), axis=1)
        first, second = order[:, :na], order[:, na:]
    stats = np.abs(_bm_statistics(z[first], z[second], pooled[first], pooled[second]))
    hits = int((stats >= observed * (1 - 1e-9)).sum())
    return hits / len(stats) if exact else (hits + 1) / (resamples + 1)


def bm_interval(r: dict, level: float = 0.95) -> tuple[float, float] | None:
    if r["df"] is None:
        return None
    half = st.t.ppf(0.5 + level / 2, r["df"]) * r["se"]
    return max(0.0, r["p_hat"] - half), min(1.0, r["p_hat"] + half)


# ---------------------------------------------------------------- rank designs --
# (name, group, sizes, generator -> (a, b), true probability of superiority)

def _normal_p(delta, s1, s2):
    return float(st.norm.cdf(delta / math.sqrt(s1 * s1 + s2 * s2)))


RATINGS = np.array([1.0, 2, 3, 4, 5])
FLAT, PEAKED = np.array([.2, .2, .2, .2, .2]), np.array([.05, .2, .5, .2, .05])
HIGHER = np.array([.05, .15, .30, .30, .20])


def _discrete_p(pa, pb):
    return float(sum(pa[i] * pb[j] * (1.0 if j > i else 0.5 if j == i else 0.0) for i in range(5) for j in range(5)))


RANK_DESIGNS = []
for na, nb in ((5, 5), (8, 8), (15, 15), (30, 30), (5, 15), (10, 30)):
    RANK_DESIGNS.append((f"identical normal {na} + {nb}", "equal distributions",
                         lambda r, na=na, nb=nb: (r.normal(0, 1, na), r.normal(0, 1, nb)), 0.5))
    RANK_DESIGNS.append((f"identical lognormal {na} + {nb}", "equal distributions, skewed",
                         lambda r, na=na, nb=nb: (r.lognormal(0, 1, na), r.lognormal(0, 1, nb)), 0.5))
for na, nb, sa, sb in ((12, 12, 1, 4), (24, 8, 1, 4), (8, 24, 1, 4), (30, 10, 1, 4), (15, 5, 1, 4), (40, 40, 1, 4), (60, 20, 1, 3)):
    RANK_DESIGNS.append((f"normal, same centre, spread {sa} vs {sb}, {na} + {nb}", "unequal variances",
                         lambda r, na=na, nb=nb, sa=sa, sb=sb: (r.normal(0, sa, na), r.normal(0, sb, nb)), 0.5))
for na, nb in ((20, 8), (8, 20), (30, 30)):
    RANK_DESIGNS.append((f"lognormal, same median, shape 0.4 vs 1.2, {na} + {nb}", "skewed, different shapes",
                         lambda r, na=na, nb=nb: (r.lognormal(3, 0.4, na), r.lognormal(3, 1.2, nb)), 0.5))
for na, nb in ((10, 10), (20, 20), (12, 36)):
    RANK_DESIGNS.append((f"ratings 1 to 5, identical, {na} + {nb}", "ties",
                         lambda r, na=na, nb=nb: (r.choice(RATINGS, na, p=FLAT), r.choice(RATINGS, nb, p=FLAT)), 0.5))
    RANK_DESIGNS.append((f"ratings 1 to 5, flat vs peaked (same centre), {na} + {nb}", "ties, unequal spread",
                         lambda r, na=na, nb=nb: (r.choice(RATINGS, na, p=FLAT), r.choice(RATINGS, nb, p=PEAKED)), _discrete_p(FLAT, PEAKED)))
# alternatives: the probability of superiority is not 1/2
for na, nb in ((8, 8), (15, 15), (30, 30), (10, 30)):
    RANK_DESIGNS.append((f"normal shifted by 0.8 sd, {na} + {nb}", "power",
                         lambda r, na=na, nb=nb: (r.normal(0, 1, na), r.normal(0.8, 1, nb)), _normal_p(0.8, 1, 1)))
    RANK_DESIGNS.append((f"lognormal scaled by 1.8, {na} + {nb}", "power, skewed",
                         lambda r, na=na, nb=nb: (r.lognormal(0, 1, na), r.lognormal(math.log(1.8), 1, nb)), _normal_p(math.log(1.8), 1, 1)))
RANK_DESIGNS.append(("normal shifted, spread 1 vs 3, 24 + 8", "power, unequal variances",
                     lambda r: (r.normal(0, 1, 24), r.normal(1.5, 3, 8)), _normal_p(1.5, 1, 3)))
RANK_DESIGNS.append(("ratings 1 to 5, flat vs higher, 20 + 20", "power, ties",
                     lambda r: (r.choice(RATINGS, 20, p=FLAT), r.choice(RATINGS, 20, p=HIGHER)), _discrete_p(FLAT, HIGHER)))


def rank_part(sims: int) -> dict:
    rows = []
    for c, (name, group, gen, truth) in enumerate(RANK_DESIGNS):
        k = {"mann_whitney": 0, "brunner_munzel_t": 0, "brunner_munzel_permutation": 0}
        undefined = covered = with_interval = 0
        for i in range(sims):
            rng = np.random.default_rng(BASE_SEED + 10_000 * c + i)
            a, b = gen(rng)
            if np.ptp(np.r_[a, b]) == 0:            # every value identical: nothing to test
                undefined += 1
                continue
            k["mann_whitney"] += st.mannwhitneyu(b, a, alternative="two-sided").pvalue < ALPHA
            r = brunner_munzel(a, b)
            p_t = bm_t_p(r)
            k["brunner_munzel_t"] += (p_t is not None and p_t < ALPHA) or (p_t is None and r["statistic"] != 0)
            k["brunner_munzel_permutation"] += bm_permutation_p(a, b, rng) < ALPHA
            iv = bm_interval(r)
            if iv is not None:
                with_interval += 1
                covered += iv[0] <= truth <= iv[1]
        n = sims - undefined
        rows.append({"design": name, "group": group, "true_probability_of_superiority": truth, "null_is_true": abs(truth - 0.5) < 1e-12,
                     "simulations": sims, "seed_first": BASE_SEED + 10_000 * c, "degenerate_samples": undefined,
                     "rejections": {m: rate(v, n) for m, v in k.items()},
                     "interval_for_p": {"coverage": rate(covered, with_interval), "not_available": n - with_interval}})
        print(f"[rank] {name:58s} MW {k['mann_whitney'] / n * 100:5.1f}%  BM-t {k['brunner_munzel_t'] / n * 100:5.1f}%  "
              f"BM-perm {k['brunner_munzel_permutation'] / n * 100:5.1f}%  cover {covered / max(with_interval, 1) * 100:5.1f}%  "
              f"(no interval {n - with_interval})", flush=True)
    return {"simulations_per_design": sims, "permutations": 999, "designs": rows}


# ------------------------------------------------------------ workflow designs --

def _matched_lognormal(rng, n, sigma, mean=30.0):
    return rng.lognormal(math.log(mean) - sigma * sigma / 2, sigma, n)


# two groups, equal means. (family, note, generator(rng, n_small, n_large) -> (a, b))
TWO_FAMILIES = [
    ("normal, same spread", "exchangeable", lambda r, s, l: (r.normal(30, 6, l), r.normal(30, 6, s))),
    ("lognormal, same shape", "exchangeable, skewed", lambda r, s, l: (_matched_lognormal(r, l, 1.0), _matched_lognormal(r, s, 1.0))),
    ("normal, smaller group 3x the spread", "unequal variances", lambda r, s, l: (r.normal(30, 4, l), r.normal(30, 12, s))),
    ("smaller group strongly skewed, larger mildly", "different shapes", lambda r, s, l: (_matched_lognormal(r, l, 0.4), _matched_lognormal(r, s, 1.2))),
    ("smaller group exponential, larger normal", "different shapes", lambda r, s, l: (r.normal(30, 6, l), r.exponential(30, s))),
    ("larger group strongly skewed, smaller mildly", "different shapes", lambda r, s, l: (_matched_lognormal(r, l, 1.2), _matched_lognormal(r, s, 0.4))),
    ("ratings 1 to 5, same distribution", "ties", lambda r, s, l: (r.choice(RATINGS, l, p=FLAT), r.choice(RATINGS, s, p=FLAT))),
]
TWO_SIZES = [(5, 5), (5, 15), (8, 8), (8, 20), (12, 12), (12, 30), (20, 20), (20, 50), (30, 30), (30, 75), (50, 50), (80, 80)]

# paired differences with a true mean of zero. (family, note, generator(rng, n))
PAIRED_FAMILIES = [
    ("normal", "symmetric", lambda r, n: r.normal(0, 4, n)),
    ("heavy-tailed, symmetric", "symmetric", lambda r, n: r.standard_t(3, n) * 4),
    ("skewed (lognormal, shape 0.9)", "asymmetric", lambda r, n: r.lognormal(1, 0.9, n) - LN_MEAN(1, 0.9)),
    ("mildly skewed (lognormal, shape 0.4)", "asymmetric", lambda r, n: r.lognormal(1, 0.4, n) - LN_MEAN(1, 0.4)),
    ("mostly small losses, a few large gains", "asymmetric", lambda r, n: np.where(r.random(n) < 0.8, r.normal(-2, 1, n), r.normal(8, 3, n))),
    ("whole numbers, symmetric", "ties", lambda r, n: r.choice([-2.0, -1, 0, 1, 2], n, p=[.1, .2, .4, .2, .1])),
]
PAIRED_SIZES = [5, 8, 12, 20, 30, 50, 80]


# power: the same generators with the second group (or the changes) moved by 0.8 standard deviations
LN_SD = 30.0 * math.sqrt(math.e - 1)
POWER_FAMILIES = [
    ("two", "normal, same spread, shifted 0.8 sd", lambda r, s, l: (r.normal(30, 6, l), r.normal(30, 6, s) + 4.8), 4.8),
    ("two", "lognormal, same shape, shifted 0.8 sd", lambda r, s, l: (_matched_lognormal(r, l, 1.0), _matched_lognormal(r, s, 1.0) + 0.8 * LN_SD), 0.8 * LN_SD),
    ("paired", "normal changes, mean 0.8 sd", lambda r, n: r.normal(3.2, 4, n), 3.2),
]
IDENTITY_CHECKS = 25


def jobs() -> list[dict]:
    """One job per family, so that families can run side by side. A job's seeds depend only on its index."""
    out = [{"kind": "two", "family": f, "tests": note, "gen": gen, "truth": 0.0, "null": True} for f, note, gen in TWO_FAMILIES]
    out += [{"kind": "paired", "family": f, "tests": note, "gen": gen, "truth": 0.0, "null": True} for f, note, gen in PAIRED_FAMILIES]
    out += [{"kind": kind, "family": f, "tests": "power", "gen": gen, "truth": truth, "null": False} for kind, f, gen, truth in POWER_FAMILIES]
    return out


def _cell() -> dict:
    return {"runs": 0, "tested": 0, "rejected": 0, "with_interval": 0, "covered": 0, "not_tested": 0, "unavailable": 0, "methods": {}}


def _record(cell: dict, r: dict, truth: float) -> None:
    cell["runs"] += 1
    m = cell["methods"].setdefault(r["method"], {"chosen": 0, "tested": 0, "rejected": 0})
    m["chosen"] += 1
    p = (r.get("test") or {}).get("p_value")
    if r.get("inference_unavailable"):
        cell["not_tested"] += 1
    elif r["status"] != "ok" or p is None:
        cell["unavailable"] += 1
    if p is not None:
        cell["tested"] += 1
        cell["rejected"] += p < ALPHA
        m["tested"] += 1
        m["rejected"] += p < ALPHA
    iv = r.get("interval")
    if iv is not None:
        cell["with_interval"] += 1
        cell["covered"] += iv["low"] <= truth <= iv["high"]


def _cell_out(cell: dict, null: bool) -> dict:
    key = "type_one_error" if null else "power"
    return {"runs": cell["runs"], "reported_without_a_test": cell["not_tested"], "otherwise_unavailable": cell["unavailable"],
            # of all samples: a sample reported without a test is not a rejection. This is the workflow's error rate.
            f"{key}_of_all_samples": rate(cell["rejected"], cell["runs"]),
            # of the samples that were given a p-value
            f"{key}_of_tested_samples": rate(cell["rejected"], cell["tested"]),
            "coverage_where_an_interval_was_given": rate(cell["covered"], cell["with_interval"]),
            "intervals_given": cell["with_interval"],
            "methods": {k: {"chosen": v["chosen"], "rejected_of_tested": rate(v["rejected"], v["tested"])} for k, v in sorted(cell["methods"].items())}}


def workflow_job(index: int, sims: int, estimand: str = "mean") -> dict:
    """One family at every size, through the selector exactly as the application calls it.

    Two workflows are measured on the same samples: the engine as it stands, and
    the same engine with the rule that withholds inference from visibly skewed
    small samples switched off (which is what the engine did before this
    report). For a sample the rule does not stop the two are the same
    computation; that is checked on the first samples of every cell."""
    from app.stats import selector

    job = jobs()[index]
    sizes = TWO_SIZES if job["kind"] == "two" else [(n, n) for n in PAIRED_SIZES]
    rule = selector._shows_skew

    def call(off: bool, x, y):
        selector._shows_skew = (lambda d: False) if off else rule
        try:
            if job["kind"] == "two":
                return selector.two_groups(x, y, what="y", labels=("a", "b"), estimand=estimand)
            return selector.paired(x, y, what="y", unit_name="unit", estimand=estimand)
        finally:
            selector._shows_skew = rule

    rows = []
    for si, (n_small, n_large) in enumerate(sizes):
        seed_first = BASE_SEED + 400_000_000 + 10_000 * (100 * index + si)
        with_rule, rule_off, checked = _cell(), _cell(), 0
        for i in range(sims):
            rng = np.random.default_rng(seed_first + i)
            x, y = job["gen"](rng, n_small, n_large) if job["kind"] == "two" else (np.zeros(n_small), job["gen"](rng, n_small))
            r = call(False, x, y)
            if r.get("inference_unavailable") or i < IDENTITY_CHECKS:
                r_off = call(True, x, y)
                if not r.get("inference_unavailable"):
                    assert (r_off["method"], r_off.get("test"), r_off.get("interval")) == (r["method"], r.get("test"), r.get("interval"))
                    checked += 1
            else:
                r_off = r
            _record(with_rule, r, job["truth"])
            _record(rule_off, r_off, job["truth"])
        row = {"design": job["kind"], "family": job["family"], "tests": job["tests"], "null_is_true": job["null"],
               **({"smaller": n_small, "larger": n_large} if job["kind"] == "two" else {"pairs": n_small}),
               "seed_first": seed_first, "simulations": sims, "identical_when_tested_checked_on": checked,
               "engine": _cell_out(with_rule, job["null"]), "rule_switched_off": _cell_out(rule_off, job["null"])}
        rows.append(row)
        key = "type_one_error" if job["null"] else "power"
        f = lambda c: "  -  " if not c else f"{c['rate'] * 100:5.1f}%"  # noqa: E731
        print(f"[{job['kind']:6s}] {job['family']:46s} {n_small:3d}{'+' + str(n_large) if job['kind'] == 'two' else ' pairs':7s} "
              f"{key}: off {f(row['rule_switched_off'][key + '_of_all_samples'])} -> engine {f(row['engine'][key + '_of_all_samples'])} of all, "
              f"{f(row['engine'][key + '_of_tested_samples'])} of tested; not tested {row['engine']['reported_without_a_test']:4d}; "
              f"coverage off {f(row['rule_switched_off']['coverage_where_an_interval_was_given'])} -> {f(row['engine']['coverage_where_an_interval_was_given'])}",
              flush=True)
    return {"index": index, "estimand": estimand, "rows": rows}


def edge_cases(estimand: str = "mean") -> list[dict]:
    """Zero variance and other edges: what the selector returns, once each."""
    from app.stats import selector

    two = {
        "both groups constant and equal": (np.full(6, 5.0), np.full(6, 5.0)),
        "both groups constant, different": (np.full(6, 5.0), np.full(6, 7.0)),
        "one group constant": (np.full(8, 5.0), np.array([4.0, 5, 6, 5, 7, 3, 5, 6])),
        "one value in a group": (np.array([5.0]), np.array([4.0, 5, 6])),
        "complete separation, 4 + 4": (np.array([1.0, 2, 3, 4]), np.array([10.0, 11, 12, 13])),
        "one extreme value among 16": (np.array([10.0, 11, 12, 9, 10, 11, 12, 10, 11, 9, 10, 11, 12, 10, 11, 9]),
                                       np.array([10.0, 11, 12, 9, 10, 11, 12, 10, 11, 9, 10, 11, 12, 10, 11, 400])),
    }
    pairs = {
        "every unit changed by the same amount": np.full(8, 2.0),
        "no unit changed": np.zeros(8),
        "seven small losses and one large gain": np.array([-2.0, -1.5, -2.5, -1.0, -2.2, -1.8, -2.1, 14.0]),
    }
    out = []
    for name, (a, b) in two.items():
        out.append((name, "two groups", selector.two_groups(a, b, what="y", labels=("a", "b"), estimand=estimand)))
    for name, d in pairs.items():
        out.append((name, "paired", selector.paired(np.zeros(len(d)), d, what="y", unit_name="unit", estimand=estimand)))
    return [{"case": name, "design": design, "status": r["status"], "method": r["method"], "reason": r.get("reason"),
             "estimate": r.get("estimate"), "sample": r["sample"].get("groups") or r["sample"].get("pairs"),
             "p_value": (r.get("test") or {}).get("p_value"), "interval": r.get("interval") is not None,
             "interval_unavailable": r.get("interval_unavailable"),
             "inference_unavailable": (r.get("inference_unavailable") or {}).get("code")} for name, design, r in out]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["rank", "workflow-job", "workflow-merge"], required=True)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--job", type=int, help="index of the family to run (see jobs())")
    parser.add_argument("--parts-dir", help="where workflow jobs leave their results until they are merged")
    parser.add_argument("--estimand", default="mean")
    args = parser.parse_args()
    started = time.time()
    if args.part == "workflow-job":
        out = workflow_job(args.job, args.sims, args.estimand)
        Path(args.parts_dir, f"{args.estimand}_{args.job:02d}.json").write_text(json.dumps(out), encoding="utf-8")
        print(f"job {args.job} done in {time.time() - started:.0f}s")
        sys.exit(0)
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {"report_version": 3, "base_seed": BASE_SEED, "alpha": ALPHA}
    if args.part == "rank":
        report["rank_tests"] = rank_part(args.sims)
    else:
        parts = [json.loads(p.read_text()) for p in sorted(Path(args.parts_dir).glob(f"{args.estimand}_*.json"))]
        assert [p["index"] for p in parts] == list(range(len(jobs()))), "every family must have run"
        left_out = []
        if args.estimand == "typical":
            # The families were built with equal means. For a question about typical values that is the
            # null only where neither group tends to be higher (identical groups, or symmetric about the
            # same centre) and, for pairs, where the changes are symmetric about zero. The other families
            # are not a true null for these tests, so their rejections are not errors and are left out.
            keep = {i for i, j in enumerate(jobs()) if not j["null"] or j["tests"] in ("exchangeable", "exchangeable, skewed",
                                                                                         "unequal variances", "ties", "symmetric")}
            left_out = [jobs()[p["index"]]["family"] for p in parts if p["index"] not in keep]
            parts = [p for p in parts if p["index"] in keep]
        report.setdefault("workflow", {})[args.estimand] = {
            "simulations_per_cell": args.sims, "estimand": args.estimand, "seed_offset": 400_000_000,
            "what_is_compared": "the engine as it stands, and the same engine with the rule that withholds inference from "
                                "visibly skewed small samples switched off; both on the same samples",
            "families_left_out_because_their_null_is_not_true_for_this_estimand": left_out,
            "cells": [row for p in parts for row in p["rows"]], "edge_cases": edge_cases(args.estimand)}
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {REPORT} in {time.time() - started:.0f}s")
