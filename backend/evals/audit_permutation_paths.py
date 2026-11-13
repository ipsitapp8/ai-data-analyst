"""Audit of the permutation, test-inversion and rank-test paths of the statistics engine.

    cd backend && python evals/audit_permutation_paths.py [--part engine|alternatives|all]

Part "engine" calls the engine itself (app.stats.selector) on simulated data
with a known truth and records, per design and per method the engine chose:

    Type I error    how often a true "no difference" is called significant
    coverage        how often the reported interval contains the true value
    no interval     how often the engine reported none

Part "alternatives" is independent of the application (numpy and scipy only).
For the two-group designs it compares the engine's test (a permutation test of
the difference in means) with a permutation test of Welch's t statistic and
with Welch's t-test, to see whether an alternative would do better where the
engine's test does badly.

Designs are null designs: the quantity each method estimates is truly zero.
Every interval method here shifts with the data (adding a constant to one
group moves the estimate and both bounds by that constant), so coverage of a
true difference d equals coverage of zero in the corresponding null design;
both are still measured directly, not inferred.

Seeds: simulation i of design d uses BASE_SEED + 100_000 * d + i. Nothing here
generalises beyond the designs listed.
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

BASE_SEED = 20_261_201
ALPHA = 0.05
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LN_MEAN = lambda mu, s: math.exp(mu + s * s / 2)          # noqa: E731
LN_MEDIAN = lambda mu: math.exp(mu)                       # noqa: E731


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(k: int, n: int) -> dict | None:
    return None if not n else {"rate": k / n, "interval": wilson(k, n), "count": int(k), "simulations": int(n)}


# ------------------------------------------------------- two independent groups --
# Each design: (name, what it tests, generator -> (a, b)). All have equal means.

def _matched_lognormal(rng, n, sigma, mean=30.0):
    return rng.lognormal(math.log(mean) - sigma * sigma / 2, sigma, n)


TWO_GROUP = [
    ("normal 12 + 12, equal spread", "control: exchangeable",
     lambda r: (r.normal(100, 10, 12), r.normal(100, 10, 12))),
    ("normal 6 + 18, equal spread", "control: unequal sizes only",
     lambda r: (r.normal(100, 10, 6), r.normal(100, 10, 18))),
    ("lognormal 12 + 12, same shape", "control: skewed, exchangeable",
     lambda r: (r.lognormal(3, 0.9, 12), r.lognormal(3, 0.9, 12))),
    ("normal 12 + 12, spread 5 vs 20", "unequal variances, equal sizes",
     lambda r: (r.normal(100, 5, 12), r.normal(100, 20, 12))),
    ("normal 8 + 24, larger group has spread 20 (smaller 5)", "unequal sizes, LARGER group more variable",
     lambda r: (r.normal(100, 5, 8), r.normal(100, 20, 24))),
    ("normal 24 + 8, smaller group has spread 20 (larger 5)", "unequal sizes, SMALLER group more variable",
     lambda r: (r.normal(100, 5, 24), r.normal(100, 20, 8))),
    ("normal 30 + 10, smaller group has spread 20 (larger 5)", "unequal sizes, SMALLER group more variable",
     lambda r: (r.normal(100, 5, 30), r.normal(100, 20, 10))),
    ("normal 15 + 5, smaller group has spread 20 (larger 5)", "small sample, SMALLER group more variable",
     lambda r: (r.normal(100, 5, 15), r.normal(100, 20, 5))),
    ("normal 16 + 8, smaller group has spread 20 (larger 10)", "milder: spread ratio 2",
     lambda r: (r.normal(100, 10, 16), r.normal(100, 20, 8))),
    ("lognormal 20 + 8, equal means, smaller group more skewed", "skewed, SMALLER group more variable",
     lambda r: (_matched_lognormal(r, 20, 0.5), _matched_lognormal(r, 8, 1.2))),
    ("lognormal 8 + 20, equal means, larger group more skewed", "skewed, LARGER group more variable",
     lambda r: (_matched_lognormal(r, 8, 0.5), _matched_lognormal(r, 20, 1.2))),
    ("exponential 8 vs normal 20, equal means", "different shapes, smaller group skewed",
     lambda r: (r.normal(10, 2, 20), r.exponential(10, 8))),
    ("normal 5 + 5, spread 5 vs 20", "very small, unequal variances",
     lambda r: (r.normal(100, 5, 5), r.normal(100, 20, 5))),
]

# Several groups, all with the same mean. (name, what it tests, generator -> {label: values})
K_GROUP = [
    ("3 groups of 8, normal, equal spread", "control: exchangeable",
     lambda r: {"a": r.normal(100, 10, 8), "b": r.normal(100, 10, 8), "c": r.normal(100, 10, 8)}),
    ("3 groups 20 + 20 + 6, smallest has spread 20 (others 5)", "unequal sizes, SMALLEST group most variable",
     lambda r: {"a": r.normal(100, 5, 20), "b": r.normal(100, 5, 20), "c": r.normal(100, 20, 6)}),
    ("3 groups 6 + 6 + 20, largest has spread 20 (others 5)", "unequal sizes, LARGEST group most variable",
     lambda r: {"a": r.normal(100, 5, 6), "b": r.normal(100, 5, 6), "c": r.normal(100, 20, 20)}),
]

# --------------------------------------------------------------- paired differences --
# Each design: differences with a true mean of zero.

PAIRED = [
    ("8 pairs, normal", "control: symmetric", lambda r: r.normal(0, 4, 8)),
    ("12 pairs, heavy-tailed symmetric", "control: symmetric", lambda r: r.standard_t(3, 12) * 4),
    ("8 pairs, skewed", "symmetry violated", lambda r: r.lognormal(1, 0.9, 8) - LN_MEAN(1, 0.9)),
    ("12 pairs, skewed", "symmetry violated", lambda r: r.lognormal(1, 0.9, 12) - LN_MEAN(1, 0.9)),
    ("20 pairs, skewed", "symmetry violated", lambda r: r.lognormal(1, 0.9, 20) - LN_MEAN(1, 0.9)),
    ("40 pairs, skewed", "symmetry violated", lambda r: r.lognormal(1, 0.9, 40) - LN_MEAN(1, 0.9)),
    ("15 pairs, mostly small losses and a few large gains", "symmetry violated",
     lambda r: np.where(r.random(15) < 0.8, r.normal(-2, 1, 15), r.normal(8, 3, 15))),
]

# ------------------------------------------------------------- typical values --
# Mann-Whitney: (name, note, generator -> (a, b), true difference in medians, is the test's null true?)

MANN_WHITNEY = [
    ("lognormal 8 + 8, identical", "null true", lambda r: (r.lognormal(3, 0.9, 8), r.lognormal(3, 0.9, 8)), 0.0, True),
    ("lognormal 15 + 15, identical", "null true", lambda r: (r.lognormal(3, 0.9, 15), r.lognormal(3, 0.9, 15)), 0.0, True),
    ("lognormal 30 + 30, identical", "null true", lambda r: (r.lognormal(3, 0.9, 30), r.lognormal(3, 0.9, 30)), 0.0, True),
    ("lognormal 10 + 30, identical", "null true, unequal sizes", lambda r: (r.lognormal(3, 0.9, 10), r.lognormal(3, 0.9, 30)), 0.0, True),
    ("lognormal 8 + 8, shifted by 10", "interval coverage", lambda r: (r.lognormal(3, 0.9, 8), r.lognormal(3, 0.9, 8) + 10), 10.0, False),
    ("lognormal 15 + 15, shifted by 10", "interval coverage", lambda r: (r.lognormal(3, 0.9, 15), r.lognormal(3, 0.9, 15) + 10), 10.0, False),
    ("lognormal 30 + 30, shifted by 10", "interval coverage", lambda r: (r.lognormal(3, 0.9, 30), r.lognormal(3, 0.9, 30) + 10), 10.0, False),
    ("normal 6 + 6, shifted by 8", "very small", lambda r: (r.normal(100, 10, 6), r.normal(108, 10, 6)), 8.0, False),
    ("ratings 1 to 5, 20 + 20, identical", "tied, discrete data",
     lambda r: (r.choice([1, 2, 3, 4, 5], 20, p=[.1, .2, .3, .25, .15]).astype(float),
                r.choice([1, 2, 3, 4, 5], 20, p=[.1, .2, .3, .25, .15]).astype(float)), 0.0, True),
    ("normal 12 + 12, same median, spread 1 vs 4", "equal medians, unequal spread",
     lambda r: (r.normal(50, 1, 12), r.normal(50, 4, 12)), 0.0, True),
    ("normal 24 + 8, same median, smaller group spread 4", "equal medians, SMALLER group more variable",
     lambda r: (r.normal(50, 1, 24), r.normal(50, 4, 8)), 0.0, True),
    ("lognormal 20 + 8, same median, smaller group more skewed", "equal medians, different shape",
     lambda r: (r.lognormal(3, 0.4, 20), r.lognormal(3, 1.2, 8)), 0.0, False),
]

# Wilcoxon signed-rank: (name, note, generator -> differences, true median, is the test's null true?)
WILCOXON = [
    ("8 pairs, normal, median 0", "null true", lambda r: r.normal(0, 4, 8), 0.0, True),
    ("15 pairs, normal, median 0", "null true", lambda r: r.normal(0, 4, 15), 0.0, True),
    ("30 pairs, normal, median 0", "null true", lambda r: r.normal(0, 4, 30), 0.0, True),
    ("8 pairs, normal, median 3", "interval coverage", lambda r: r.normal(3, 4, 8), 3.0, False),
    ("15 pairs, normal, median 3", "interval coverage", lambda r: r.normal(3, 4, 15), 3.0, False),
    ("30 pairs, normal, median 3", "interval coverage", lambda r: r.normal(3, 4, 30), 3.0, False),
    ("12 pairs, skewed, median 0", "median zero but not symmetric", lambda r: r.lognormal(1, 0.9, 12) - LN_MEDIAN(1), 0.0, False),
    ("30 pairs, skewed, median 0", "median zero but not symmetric", lambda r: r.lognormal(1, 0.9, 30) - LN_MEDIAN(1), 0.0, False),
    ("20 pairs, whole numbers -2 to 3", "ties and zeros",
     lambda r: r.choice([-2, -1, 0, 1, 2, 3], 20, p=[.1, .2, .25, .2, .15, .1]).astype(float), 0.0, False),
]


def _tally() -> dict:
    return {"n": 0, "rejected": 0, "with_interval": 0, "covered": 0, "no_interval": 0, "unavailable": 0}


def _record(cell: dict, result: dict, truth: float) -> None:
    if result["status"] != "ok" or (result.get("test") or {}).get("p_value") is None:
        cell["unavailable"] += 1
        return
    cell["n"] += 1
    cell["rejected"] += result["test"]["p_value"] < ALPHA
    iv = result.get("interval")
    if iv is None:
        cell["no_interval"] += 1
    else:
        cell["with_interval"] += 1
        cell["covered"] += iv["low"] <= truth <= iv["high"]


def _summary(cells: dict) -> dict:
    return {m: {"runs": c["n"], "type_one_error": rate(c["rejected"], c["n"]), "coverage": rate(c["covered"], c["with_interval"]),
                "no_interval": c["no_interval"], "unavailable": c["unavailable"]} for m, c in sorted(cells.items())}


def engine_part(sims: int, typical_sims: int) -> dict:
    from app.stats import selector

    out = {"two_groups": [], "k_groups": [], "paired": [], "mann_whitney": [], "wilcoxon": []}
    d = 0
    for name, note, gen in TWO_GROUP:
        cells: dict = {}
        for i in range(sims):
            a, b = gen(np.random.default_rng(BASE_SEED + 100_000 * d + i))
            r = selector.two_groups(a, b, what="y", labels=("a", "b"))
            _record(cells.setdefault(r["method"], _tally()), r, 0.0)
        out["two_groups"].append({"design": name, "tests": note, "seed_first": BASE_SEED + 100_000 * d, "simulations": sims,
                                  "by_method": _summary(cells)})
        print(f"[two groups] {name}: " + "; ".join(f"{m} type I {v['type_one_error']['rate'] * 100:.1f}%" for m, v in out["two_groups"][-1]["by_method"].items() if v["type_one_error"]), flush=True)
        d += 1
    d = 500                                              # k-group designs take their seeds from here
    for name, note, gen in K_GROUP:
        cells = {}
        for i in range(sims):
            r = selector.k_groups(gen(np.random.default_rng(BASE_SEED + 100_000 * d + i)), what="y", group_name="g")
            _record(cells.setdefault(r["method"], _tally()), {**r, "interval": None}, 0.0)
        out["k_groups"].append({"design": name, "tests": note, "seed_first": BASE_SEED + 100_000 * d, "simulations": sims,
                                "by_method": _summary(cells)})
        print(f"[k groups] {name}: " + "; ".join(f"{m} type I {v['type_one_error']['rate'] * 100:.1f}%" for m, v in out["k_groups"][-1]["by_method"].items() if v["type_one_error"]), flush=True)
        d += 1
    d = len(TWO_GROUP)
    for name, note, gen in PAIRED:
        cells = {}
        for i in range(sims):
            diff = gen(np.random.default_rng(BASE_SEED + 100_000 * d + i))
            r = selector.paired(np.zeros(len(diff)), diff, what="y", unit_name="unit")
            _record(cells.setdefault(r["method"], _tally()), r, 0.0)
        out["paired"].append({"design": name, "tests": note, "seed_first": BASE_SEED + 100_000 * d, "simulations": sims,
                              "by_method": _summary(cells)})
        print(f"[paired] {name}: " + "; ".join(f"{m} type I {v['type_one_error']['rate'] * 100:.1f}%" for m, v in out["paired"][-1]["by_method"].items() if v["type_one_error"]), flush=True)
        d += 1
    for name, note, gen, truth, null_true in MANN_WHITNEY:
        cells = {}
        for i in range(typical_sims):
            a, b = gen(np.random.default_rng(BASE_SEED + 100_000 * d + i))
            r = selector.two_groups(a, b, what="y", labels=("a", "b"), estimand="typical")
            _record(cells.setdefault(r["method"], _tally()), r, truth)
        out["mann_whitney"].append({"design": name, "tests": note, "true_median_difference": truth, "test_null_is_true": null_true,
                                    "seed_first": BASE_SEED + 100_000 * d, "simulations": typical_sims, "by_method": _summary(cells)})
        print(f"[mann-whitney] {name}: " + "; ".join(
            f"reject {v['type_one_error']['rate'] * 100:.1f}%, cover {'-' if not v['coverage'] else format(v['coverage']['rate'] * 100, '.1f') + '%'}, no interval {v['no_interval']}"
            for v in out["mann_whitney"][-1]["by_method"].values() if v["type_one_error"]), flush=True)
        d += 1
    for name, note, gen, truth, null_true in WILCOXON:
        cells = {}
        for i in range(typical_sims):
            diff = gen(np.random.default_rng(BASE_SEED + 100_000 * d + i))
            r = selector.paired(np.zeros(len(diff)), diff, what="y", unit_name="unit", estimand="typical")
            _record(cells.setdefault(r["method"], _tally()), r, truth)
        out["wilcoxon"].append({"design": name, "tests": note, "true_median": truth, "test_null_is_true": null_true,
                                "seed_first": BASE_SEED + 100_000 * d, "simulations": typical_sims, "by_method": _summary(cells)})
        print(f"[wilcoxon] {name}: " + "; ".join(
            f"reject {v['type_one_error']['rate'] * 100:.1f}%, cover {'-' if not v['coverage'] else format(v['coverage']['rate'] * 100, '.1f') + '%'}, no interval {v['no_interval']}"
            for v in out["wilcoxon"][-1]["by_method"].values() if v["type_one_error"]), flush=True)
        d += 1
    return out


# ------------------------------------------------- alternatives (independent) --

def _perm_p(a, b, rng, studentised: bool, resamples: int = 1999) -> float:
    """Monte Carlo permutation p-value, two-sided by absolute value, for the
    difference in means or for Welch's t statistic."""
    na, z = len(a), np.r_[a, b]
    idx = np.argsort(rng.random((resamples, len(z))), axis=1)
    pa, pb = z[idx[:, :na]], z[idx[:, na:]]

    def stat(x, y):
        diff = y.mean(-1) - x.mean(-1)
        if not studentised:
            return diff
        se = np.sqrt(x.var(-1, ddof=1) / x.shape[-1] + y.var(-1, ddof=1) / y.shape[-1])
        return np.divide(diff, se, out=np.zeros_like(diff), where=se > 0)

    observed = stat(a, b)
    return float(((np.abs(stat(pa, pb)) >= abs(observed) - 1e-12).sum() + 1) / (resamples + 1))


def alternatives_part(sims: int) -> list[dict]:
    rows = []
    for d, (name, note, gen) in enumerate(TWO_GROUP):
        hits = {"permutation of the mean difference (engine's test)": 0, "permutation of Welch's t": 0, "Welch's t-test": 0}
        for i in range(sims):
            a, b = gen(np.random.default_rng(BASE_SEED + 100_000 * d + i))
            rng = np.random.default_rng(BASE_SEED + 900_000_000 + 100_000 * d + i)
            hits["permutation of the mean difference (engine's test)"] += _perm_p(a, b, rng, False) < ALPHA
            hits["permutation of Welch's t"] += _perm_p(a, b, rng, True) < ALPHA
            hits["Welch's t-test"] += st.ttest_ind(b, a, equal_var=False).pvalue < ALPHA
        rows.append({"design": name, "tests": note, "simulations": sims, "type_one_error": {m: rate(k, sims) for m, k in hits.items()}})
        print(f"[alternatives] {name}: " + "; ".join(f"{m.split(' (')[0]} {k / sims * 100:.1f}%" for m, k in hits.items()), flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["engine", "alternatives", "all"], default="all")
    parser.add_argument("--sims", type=int, default=3000)
    parser.add_argument("--typical-sims", type=int, default=2000)
    parser.add_argument("--json", default=str(HERE / "reports" / "permutation_audit.json"))
    args = parser.parse_args()
    path = Path(args.json)
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({"base_seed": BASE_SEED, "alpha": ALPHA})
    started = time.time()
    if args.part in ("alternatives", "all"):
        report["alternatives"] = {"simulations_per_design": args.sims, "permutations": 1999, "designs": alternatives_part(args.sims)}
    if args.part in ("engine", "all"):
        report["engine"] = {"simulations_per_design": args.sims, "typical_simulations_per_design": args.typical_sims,
                            **engine_part(args.sims, args.typical_sims)}
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {path} in {time.time() - started:.0f}s")
