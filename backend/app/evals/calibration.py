"""Calibration of the comparison workflow by simulation.

    cd backend && python -m app.evals.calibration [--json PATH]

For each design, datasets are drawn from a distribution whose truth is known
by construction, the workflow is run on each exactly as it runs on a user's
data (plan -> validate -> statistics engine), and three things are counted:

  false-positive rate   how often a true "no difference" is called significant
  coverage              how often the interval contains the true value
  power                 how often a planted difference is detected

Results are grouped by the method the engine actually chose, because that is
what a reader of a result is told.

Two runs, both with fixed seeds:

  primary        400 simulations per design, with and without a planted effect.
                 Rates, coverage and power for every method.
  confirmation   4,000 simulations per design with no effect. Enough to tell a
                 miscalibrated test from sampling variation: at 400 a true 5% rate
                 is measured to about +/- 2 points, at 4,000 to about +/- 0.7.

Criteria. One band for every method ("between 3% and 7%") cannot be right: an
exact test on a small table cannot reject 5% of the time by construction, and
a band that wide says little about a test simulated 4,000 times. So:

  approximate tests (t, F, chi-square, correlation, large-sample rank tests)
      claim a false-positive rate of alpha. Criterion: alpha lies inside the
      95% Wilson interval of the observed rate in the confirmation run.

  exact and permutation tests
      guarantee a rate of AT MOST alpha, and fall below it when the data allow
      only a few distinct p-values. Criterion: no evidence of a rate above
      alpha (the Wilson interval's lower limit is at or below alpha). Where
      the test's exact size can be worked out from the design, the observed
      rate must also agree with it (the size lies inside the Wilson interval):
      that is what shows the conservatism is the test's, not a defect.

  intervals
      Criterion: no evidence of covering less often than stated (the Wilson
      interval's upper limit is at or above the level). Covering more often is
      reported as conservatism. An interval the engine declines to report is
      counted and shown, not left out.

The original fixed bands (3% to 7%, 93% to 97%) are still evaluated and
reported beside these, and the first run's report is kept unchanged in
evals/reports/comparison_calibration_v1.json.

Seeds: simulation i of design d uses BASE_SEED + 100_000 * d + i (+ 50_000 with
a planted effect); the confirmation run adds CONFIRMATION_OFFSET.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from app import comparison

BASE_SEED = 20_261_013
CONFIRMATION_OFFSET = 50_000_000
SIMS = 400
CONFIRMATION_SIMS = 4000
ALPHA = 0.05
LEVEL = 0.95
FPR_GATE = (0.03, 0.07)                # the original fixed bands, still reported
COVERAGE_GATE = (0.93, 0.97)
# What each test claims about its false-positive rate.
EXACT = {"fisher_exact", "sign_flip_permutation", "permutation_means", "permutation_anova"}
APPROXIMATE = {"welch_t", "paired_t", "welch_anova", "chi_square", "pearson", "spearman", "brunner_munzel"}


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def _profile(df: pd.DataFrame) -> dict:
    from app.profiling import profile_dataframe

    return profile_dataframe(df)


# ------------------------------------------------------------------ designs --
# Each returns (frame, plan, true value of the estimate or None).

def two_groups_normal(rng, effect):
    n = 80
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n, "y": np.r_[rng.normal(100, 10, n), rng.normal(100 + effect, 10, n)]})
    return df, {"outcome": "y", "explanatory": "g"}, float(effect)


def two_groups_skewed(rng, effect):
    n = 12
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n, "y": np.r_[rng.lognormal(3, 0.9, n), rng.lognormal(3, 0.9, n) + effect]})
    return df, {"outcome": "y", "explanatory": "g"}, float(effect)


def two_groups_small(rng, effect):
    """Small and symmetric: the samples of the skewed design above are mostly reported without a
    test now, so the permutation test is calibrated where it is still used."""
    n = 12
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n, "y": np.r_[rng.normal(100, 12, n), rng.normal(100 + effect, 12, n)]})
    return df, {"outcome": "y", "explanatory": "g"}, float(effect)


def two_groups_typical(rng, effect):
    n = 15
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n, "y": np.r_[rng.lognormal(3, 0.9, n), rng.lognormal(3, 0.9, n) + effect]})
    return df, {"outcome": "y", "explanatory": "g", "estimand": "typical"}, float(effect)


def paired_normal(rng, effect):
    n = 80
    before = rng.normal(50, 10, n)
    after = before + rng.normal(effect, 4, n)
    df = pd.DataFrame({"u": list(range(n)) * 2, "g": ["before"] * n + ["after"] * n, "y": np.r_[before, after]})
    return df, {"outcome": "y", "explanatory": "g", "levels": ["before", "after"], "design": "paired", "unit": "u"}, float(effect)


def paired_small(rng, effect):
    n = 8
    before = rng.normal(50, 10, n)
    after = before + rng.normal(effect, 4, n)
    df = pd.DataFrame({"u": list(range(n)) * 2, "g": ["before"] * n + ["after"] * n, "y": np.r_[before, after]})
    return df, {"outcome": "y", "explanatory": "g", "levels": ["before", "after"], "design": "paired", "unit": "u"}, float(effect)


def three_groups_normal(rng, effect):
    n = 90
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n + ["c"] * n,
                       "y": np.r_[rng.normal(100, 10, n), rng.normal(100, 14, n), rng.normal(100 + effect, 8, n)]})
    return df, {"outcome": "y", "explanatory": "g"}, None


def three_groups_small(rng, effect):
    n = 6
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n + ["c"] * n,
                       "y": np.r_[rng.normal(100, 10, n), rng.normal(100, 10, n), rng.normal(100 + effect, 10, n)]})
    return df, {"outcome": "y", "explanatory": "g"}, None


def three_groups_moderate(rng, effect):
    """Twenty per group: too few for Welch's F by Cochran's rule, so the permutation test is used on the
    samples that do not look skewed. Added when several-group comparisons got the same rule as two."""
    n = 20
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n + ["c"] * n,
                       "y": np.r_[rng.normal(100, 10, n), rng.normal(100, 10, n), rng.normal(100 + effect, 10, n)]})
    return df, {"outcome": "y", "explanatory": "g"}, None


def linear(rng, effect):
    """`effect` is the population correlation."""
    n = 50
    x = rng.normal(0, 1, n)
    y = effect * x + math.sqrt(1 - effect ** 2) * rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "y": y}), {"outcome": "y", "explanatory": "x"}, float(effect)


def heavy_tails(rng, effect):
    """Independent heavy-tailed variables: extreme values are common, so ranks
    are used. Only the null is simulated; the true rank correlation is zero."""
    n = 60
    return (pd.DataFrame({"x": rng.standard_t(2, n), "y": rng.standard_t(2, n)}), {"outcome": "y", "explanatory": "x"}, 0.0)


def proportions_large(rng, effect):
    n, p = 150, 0.30
    flags = np.r_[rng.random(n) < p, rng.random(n) < p + effect]
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n, "hit": np.where(flags, "yes", "no")})
    return df, {"outcome": "hit", "explanatory": "g", "outcome_level": "yes"}, float(effect)


def proportions_small(rng, effect):
    n, p = 12, 0.30
    flags = np.r_[rng.random(n) < p, rng.random(n) < p + effect]
    df = pd.DataFrame({"g": ["a"] * n + ["b"] * n, "hit": np.where(flags, "yes", "no")})
    return df, {"outcome": "hit", "explanatory": "g", "outcome_level": "yes"}, float(effect)


def table_3x3(rng, effect):
    n = 100
    levels = np.array(["low", "mid", "high"])
    base = np.array([1 / 3, 1 / 3, 1 / 3])
    shifted = np.array([1 / 3 - effect, 1 / 3, 1 / 3 + effect])
    cols = np.r_[rng.choice(levels, n, p=base), rng.choice(levels, n, p=base), rng.choice(levels, n, p=shifted)]
    return pd.DataFrame({"g": ["a"] * n + ["b"] * n + ["c"] * n, "tier": cols}), {"outcome": "tier", "explanatory": "g"}, None


# (name, generator, planted effect for the power and coverage runs or None, sample size as text)
DESIGNS = [
    ("two groups, normal", two_groups_normal, 4.0, "80 + 80"),
    ("two groups, skewed, small", two_groups_skewed, 15.0, "12 + 12"),
    ("two groups, typical values", two_groups_typical, 15.0, "15 + 15"),
    ("paired, normal differences", paired_normal, 1.2, "80 pairs"),
    ("paired, few pairs", paired_small, 4.0, "8 pairs"),
    ("three groups, normal", three_groups_normal, 4.0, "90 x 3"),
    ("three groups, small", three_groups_small, 15.0, "6 x 3"),
    ("two numeric, linear", linear, 0.4, "50 rows"),
    ("two numeric, heavy tails", heavy_tails, None, "60 rows"),
    ("two proportions, large", proportions_large, 0.15, "150 + 150"),
    ("two proportions, small", proportions_small, 0.40, "12 + 12"),
    ("3 x 3 table", table_3x3, 0.15, "100 x 3"),
    ("two groups, small", two_groups_small, 15.0, "12 + 12"),       # added last: earlier designs keep their seeds
    ("three groups, moderate", three_groups_moderate, 8.0, "20 x 3"),
]

# Designs in which a share of the samples is visibly skewed and so reported without a test. They are
# simulated this many times over, so that the test used on the rest is still judged on enough samples.
REPEATS = {"paired, few pairs": 2, "two groups, small": 2, "three groups, moderate": 2}


def _one(generator, effect: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    df, plan, truth = generator(rng, effect)
    try:
        clean = comparison.validate({**plan, "source": "simulation"}, _profile(df))
        stat = comparison.run(df, clean)["statistic"]
    except comparison.PlanError as e:
        # The data contradicts the plan (in a small table, no row has the outcome value the plan
        # names). The workflow refuses it, as it would for a user; that is counted, not hidden.
        return {"refused": e.code}
    interval, test = stat.get("interval"), stat.get("test") or {}
    return {"method": stat["method"], "status": stat["status"], "p": test.get("p_value"), "has_interval": interval is not None,
            "covers": None if interval is None or truth is None else interval["low"] <= truth <= interval["high"],
            "truth_known": truth is not None}


def simulate(sims: int, designs: list, conditions: tuple[str, ...] = ("null", "effect"), offset: int = 0) -> dict:
    """Counts per method. `designs` are (index, entry) pairs so that a subset
    keeps the seeds it has in the full run."""
    by_method: dict[str, dict] = {}
    by_design: dict[str, dict] = {}
    refused: dict[str, dict] = {}
    for d, (name, generator, planted, size) in designs:
        for kind in conditions:
            effect = 0.0 if kind == "null" else planted
            if effect is None:
                continue
            for i in range(sims * REPEATS.get(name, 1)):
                out = _one(generator, effect, BASE_SEED + offset + 100_000 * d + i + (50_000 if kind == "effect" else 0))
                if "refused" in out:
                    refused.setdefault(name, {}).setdefault(out["refused"], 0)
                    refused[name][out["refused"]] += 1
                    continue
                b = by_method.setdefault(out["method"], {
                    "null_n": 0, "null_rejected": 0, "effect_n": 0, "effect_rejected": 0, "cover_n": 0, "covered": 0,
                    "interval_expected": 0, "interval_missing": 0, "unavailable": 0, "designs": {}})
                b["designs"][name] = size
                cell = by_design.setdefault(name, {}).setdefault(out["method"], {"null_n": 0, "null_rejected": 0})
                if kind == "null" and out["status"] == "ok" and out["p"] is not None:
                    cell["null_n"] += 1
                    cell["null_rejected"] += int(out["p"] < ALPHA)
                if out["status"] != "ok" or out["p"] is None:
                    b["unavailable"] += 1
                    continue
                b[f"{kind}_n"] += 1
                b[f"{kind}_rejected"] += out["p"] < ALPHA
                if out["truth_known"]:
                    b["interval_expected"] += 1
                    b["interval_missing"] += not out["has_interval"]
                if out["covers"] is not None:
                    b["cover_n"] += 1
                    b["covered"] += bool(out["covers"])
    by_method["__by_design__"] = by_design
    by_method["__refused__"] = refused
    return by_method


def _rate(k: int, n: int) -> dict | None:
    return None if not n else {"rate": k / n, "interval": list(wilson(k, n)), "count": k, "simulations": n}


def exact_sizes() -> dict:
    """The false-positive rate each discrete exact test must have in its
    design, worked out without simulating. Standard library only."""
    # Two proportions, 12 + 12, both true shares 0.30. Every possible pair of counts, its
    # probability, and whether Fisher's exact test (used when an expected count is below 5,
    # as the engine does) rejects at alpha.
    n, p = 12, 0.30
    binom = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    used = rejected = 0.0
    for xa in range(n + 1):
        for xb in range(n + 1):
            total = xa + xb
            if total in (0, 2 * n):                              # the outcome does not vary: no test
                continue
            if min(total, 2 * n - total) * n / (2 * n) >= 5:     # every expected count >= 5: chi-square is used instead
                continue
            prob = binom[xa] * binom[xb]
            table = lambda k: math.comb(n, k) * math.comb(n, total - k) / math.comb(2 * n, total)  # noqa: E731
            observed = table(xa)
            p_value = math.fsum(table(k) for k in range(max(0, total - n), min(n, total) + 1) if table(k) <= observed * (1 + 1e-7))
            used += prob
            rejected += prob * (p_value < ALPHA)
    # Sign-flip test, 8 pairs, continuous symmetric differences: 2^8 equally likely sign patterns.
    # The two-sided p-value is 2k / 256 for k = 1, 2, ...; it is below alpha for k <= 6.
    patterns = 2 ** 8
    k_max = max(k for k in range(1, patterns) if 2 * k / patterns < ALPHA)
    return {"fisher_exact": {"design": "two proportions, small", "size": rejected / used,
                             "share_of_datasets_tested_this_way": used,
                             "how": "enumeration of all 169 pairs of counts with their binomial probabilities"},
            "sign_flip_permutation": {"design": "paired, few pairs", "size": 2 * k_max / patterns,
                                      "how": f"256 sign patterns: the p-value is below {ALPHA} for the {2 * k_max} most extreme"}}


def judge(method: str, primary: dict, confirmation: dict | None, sizes: dict) -> dict:
    """The method-appropriate criteria, and the original bands beside them."""
    fpr_primary = _rate(primary["null_rejected"], primary["null_n"])
    fpr = _rate(confirmation["null_rejected"], confirmation["null_n"]) if confirmation else fpr_primary
    coverage = _rate(primary["covered"], primary["cover_n"])
    out = {"claim": "exact: at most alpha" if method in EXACT else "approximate: alpha",
           "false_positive_rate": fpr_primary, "false_positive_rate_confirmation": fpr if confirmation else None,
           "coverage": coverage, "power": _rate(primary["effect_rejected"], primary["effect_n"]),
           "intervals_not_reported": {"count": primary["interval_missing"], "of": primary["interval_expected"]},
           "original_gates": {
               "false_positive_rate_3_to_7_percent": None if not fpr_primary else FPR_GATE[0] <= fpr_primary["rate"] <= FPR_GATE[1],
               "coverage_93_to_97_percent": None if not coverage else COVERAGE_GATE[0] <= coverage["rate"] <= COVERAGE_GATE[1]},
           "criteria": {}}
    if fpr:
        low, high = fpr["interval"]
        if method in EXACT:
            out["criteria"]["no_excess_false_positives"] = low <= ALPHA
            if method in sizes:
                out["exact_size"] = sizes[method]["size"]
                out["criteria"]["agrees_with_exact_size"] = low <= sizes[method]["size"] <= high
        else:
            out["criteria"]["consistent_with_alpha"] = low <= ALPHA <= high
    if coverage:
        out["criteria"]["no_undercoverage"] = coverage["interval"][1] >= LEVEL
        out["conservative_interval"] = coverage["interval"][0] > LEVEL
    out["meets_criteria"] = all(out["criteria"].values())
    return out


def run(sims: int = SIMS, confirmation_sims: int = CONFIRMATION_SIMS, designs: list | None = None) -> dict:
    """The full report: primary run, confirmation run, criteria per method."""
    chosen = list(enumerate(DESIGNS)) if designs is None else designs
    primary = simulate(sims, chosen)
    confirmation = simulate(confirmation_sims, chosen, conditions=("null",), offset=CONFIRMATION_OFFSET) if confirmation_sims else {}
    primary.pop("__by_design__")
    refused = {"primary": primary.pop("__refused__"), "confirmation": confirmation.pop("__refused__", {})}
    confirmation_by_design = confirmation.pop("__by_design__", {})
    sizes = exact_sizes()
    methods = []
    for method in sorted(primary):
        row = {"method": method, "designs": primary[method]["designs"], "unavailable": primary[method]["unavailable"],
               **judge(method, primary[method], confirmation.get(method), sizes)}
        methods.append(row)
    return {"base_seed": BASE_SEED, "confirmation_seed_offset": CONFIRMATION_OFFSET, "alpha": ALPHA, "level": LEVEL,
            "simulations_per_design_and_condition": sims, "confirmation_null_simulations_per_design": confirmation_sims,
            "original_gates": {"false_positive_rate": FPR_GATE, "coverage": COVERAGE_GATE},
            "designs_simulated_more_than_once": REPEATS,
            "exact_sizes": sizes,
            "designs": [{"index": d, "name": n, "sample": s, "planted_effect": e} for d, (n, _, e, s) in chosen],
            "confirmation_by_design": confirmation_by_design, "plans_refused_by_the_workflow": refused, "methods": methods}


def to_text(report: dict) -> str:
    pct = lambda r: "    -            -      " if r is None else f"{r['rate'] * 100:5.1f}% [{r['interval'][0] * 100:4.1f}, {r['interval'][1] * 100:4.1f}] n={r['simulations']:<5d}"  # noqa: E731
    lines = [f"seed {report['base_seed']}; primary {report['simulations_per_design_and_condition']} per design and condition; "
             f"confirmation {report['confirmation_null_simulations_per_design']} null per design; alpha {report['alpha']}",
             f"{'method':22s} {'false positives, primary':30s} {'false positives, confirmation':30s} {'coverage':30s} {'power':>6s}  criteria"]
    for r in report["methods"]:
        power = "   -  " if not r["power"] else f"{r['power']['rate'] * 100:5.1f}%"
        verdict = ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in r["criteria"].items())
        lines.append(f"{r['method']:22s} {pct(r['false_positive_rate'])} {pct(r['false_positive_rate_confirmation'])} "
                     f"{pct(r['coverage'])} {power}  {verdict}")
        if r.get("exact_size") is not None:
            lines.append(f"{'':22s} exact size of the test in this design: {r['exact_size'] * 100:.2f}%")
        if r["intervals_not_reported"]["count"]:
            lines.append(f"{'':22s} interval not reported in {r['intervals_not_reported']['count']} of {r['intervals_not_reported']['of']} runs")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evals.calibration")
    parser.add_argument("--sims", type=int, default=SIMS)
    parser.add_argument("--confirmation-sims", type=int, default=CONFIRMATION_SIMS)
    parser.add_argument("--json", metavar="PATH")
    args = parser.parse_args(argv)
    report = run(args.sims, args.confirmation_sims)
    print(to_text(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
