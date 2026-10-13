"""Statistical reasoning: numerical correctness, method selection, the result
model, time-series and regression inference, and integration with the
hypothesis engine, the API and evidence verification.

Expected values are published examples, closed-form results computed here
from first principles, or (for coverage) seeded simulations with tolerances."""
from __future__ import annotations

import copy
import csv
import io
import math

import numpy as np
import pandas as pd
import pytest
from platform_helpers import FIXTURES, SALES_CSV, ask, question_row, run_job, workspace
from scipy import stats as st

from app import analysis_ops as ops
from app import config, hypothesis_engine as he, runtime
from app import metric_engine as me
from app.evals.scripted import ScriptedProvider
from app.profiling import load_and_profile_csv, profile_dataframe
from app.stats import methods as m
from app.stats import result as sr
from app.stats import selector as sel
from app.stats import series as ts

SPLIT = "2025-03-01"
ROWS = list(csv.DictReader(io.StringIO(SALES_CSV.decode())))
RNG = lambda seed: np.random.default_rng(seed)  # noqa: E731
LAST_EARLIER = max(r["order_date"] for r in ROWS if r["order_date"] < SPLIT)      # the fixture has rows every second day
FIRST_LATER = min(r["order_date"] for r in ROWS if r["order_date"] >= SPLIT)


# ============================================================ result model ==

def _ok(**over):
    base = dict(status="ok", method="welch_t", method_label="Welch's t-test", design="independent_groups", estimand="difference in mean x",
                alpha=0.05, estimate=1.5, interval={"low": 0.5, "high": 2.5, "level": 0.95, "method": "t"},
                test={"name": "Welch's t-test", "null": "the means are equal", "alternative": "they differ", "p_value": 0.01},
                significant=True)
    base.update(over)
    return base


@pytest.mark.parametrize("bad", [
    {"test": {"name": "t", "null": "n", "alternative": "a", "p_value": 1.2}},
    {"test": {"name": "t", "null": "n", "alternative": "a", "p_value": -0.1}},
    {"interval": {"low": 3.0, "high": 1.0, "level": 0.95, "method": "t"}},
    {"interval": {"low": 1.0, "high": 2.0, "level": 95, "method": "t"}},
    {"significant": False},                                   # contradicts p = 0.01 at alpha 0.05
    {"significant": None},                                    # a p-value exists, so significance must be stated
    {"test": None},                                           # significance claimed without a p-value
    {"status": "unavailable"},                                # no reason given
    {"effects": [{"name": "g", "value": float("inf")}]},
    {"assumptions": [{"name": "x", "status": "probably fine", "detail": ""}]},
])
def test_an_impossible_statistical_result_is_refused(bad):
    with pytest.raises(sr.StatResultError):
        sr.build(**_ok(**bad))


def test_unavailable_values_stay_unavailable_and_the_wording_is_fixed():
    r = sr.build(**_ok(estimate=float("nan"), interval=None))
    assert r["estimate"] is None and r["interval"] is None and r["schema_version"] == sr.SCHEMA_VERSION
    gone = sr.unavailable(method="none", method_label="no test", design="paired", estimand="x", reason="too few pairs", alpha=0.05)
    assert gone["test"] is None and gone["estimate"] is None and gone["significant"] is None
    assert gone["interpretation"] == ["No statistical conclusion: too few pairs"]

    significant = " ".join(sr.build(**_ok())["interpretation"])
    assert "not a statement about how large or how important" in significant and "or about its cause" in significant
    assert "not a 95% probability that the true value lies inside" in significant
    weak = sr.build(**_ok(test={"name": "t", "null": "the means are equal", "alternative": "a", "p_value": 0.4}, significant=False))
    text = " ".join(weak["interpretation"])
    assert "not evidence that there is no difference" in text and "no effect" not in text
    adjusted = sr.with_adjustment(sr.build(**_ok()), 0.04, "Holm", "tests of this investigation", 4)
    assert adjusted["adjusted"] == {"p_value": 0.04, "method": "Holm", "family": "tests of this investigation", "comparisons": 4}
    assert adjusted["test"]["p_value"] == 0.01, "the raw p-value is kept"
    assert "After Holm correction across 4 tests" in " ".join(adjusted["interpretation"])
    assert sr.p_text(0.00001) == "p < 0.001" and sr.p_text(None) == "no p-value" and sr.num("x") is None


# ====================================================== numerical methods ==

def test_welch_matches_the_closed_form():
    a, b = RNG(1).normal(100, 15, 80), RNG(2).normal(108, 25, 60)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (b.mean() - a.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = st.t.ppf(0.975, df) * math.sqrt(va + vb)
    r = sel.two_groups(a, b, what="order value", units="USD")
    assert r["method"] == "welch_t" and r["estimate"] == pytest.approx(b.mean() - a.mean())
    assert r["test"]["statistic"] == pytest.approx(t) and r["test"]["df"] == pytest.approx(df)
    assert r["test"]["p_value"] == pytest.approx(2 * st.t.sf(abs(t), df))
    assert (r["interval"]["low"], r["interval"]["high"]) == pytest.approx((b.mean() - a.mean() - half, b.mean() - a.mean() + half))
    pooled = math.sqrt(((79) * a.var(ddof=1) + (59) * b.var(ddof=1)) / 138)
    effects = {e["name"]: e["value"] for e in r["effects"]}
    assert effects["Hedges' g (standardised mean difference)"] == pytest.approx((b.mean() - a.mean()) / pooled * (1 - 3 / (4 * 140 - 9)))
    assert effects["relative difference, percent"] == pytest.approx((b.mean() - a.mean()) / a.mean() * 100)
    assert r["sample"]["groups"] == {"earlier": 80, "later": 60} and r["units"] == "USD"


def test_proportion_intervals_match_published_values():
    assert m.wilson(0, 10, 0.95) == pytest.approx((0.0, 0.2775), abs=5e-4)
    assert m.wilson(5, 10, 0.95) == pytest.approx((0.2366, 0.7634), abs=5e-4)
    # Newcombe (1998), method 10: 56/70 against 48/80.
    assert m.newcombe(48, 80, 56, 70, 0.95) == pytest.approx((0.0524, 0.3339), abs=5e-4)
    r = sel.proportions(48, 80, 56, 70, what="response rate")
    assert r["estimate"] == pytest.approx(0.2) and r["method"] == "chi_square"
    table = np.array([[48, 32], [56, 14]], float)
    expected = np.outer(table.sum(1), table.sum(0)) / table.sum()
    chi2 = ((table - expected) ** 2 / expected).sum()
    assert r["test"]["statistic"] == pytest.approx(chi2) and r["test"]["p_value"] == pytest.approx(st.chi2.sf(chi2, 1))
    effects = {e["name"]: e for e in r["effects"]}
    assert effects["risk ratio"]["value"] == pytest.approx((56 / 70) / (48 / 80))
    assert effects["odds ratio"]["value"] == pytest.approx((56 * 32) / (48 * 14))
    assert effects["risk difference, percentage points"]["value"] == pytest.approx(20.0)
    assert effects["odds ratio"]["interval"]["low"] < effects["odds ratio"]["value"] < effects["odds ratio"]["interval"]["high"]


def test_fisher_exact_and_empty_cells():
    tea = sel.proportions(1, 4, 3, 4, what="correct guesses")                 # the lady tasting tea
    assert tea["method"] == "fisher_exact" and tea["test"]["p_value"] == pytest.approx(0.4857, abs=1e-4)
    assert tea["test"]["exact"] is True and tea["test"]["statistic"] is None, "Fisher's test has no test statistic to report"
    assert [a["status"] for a in tea["assumptions"] if a["name"] == "expected counts"] == ["violated"]

    empty = sel.proportions(0, 30, 6, 30, what="refund share")
    effects = {e["name"]: e for e in empty["effects"]}
    assert effects["risk ratio"]["value"] is None and "not defined" in effects["risk ratio"]["note"], "undefined is not zero"
    assert effects["odds ratio"]["value"] is None and empty["interval"]["low"] > 0

    flat = sel.proportions(0, 12, 0, 11, what="refund share")
    assert flat["status"] == "ok" and flat["test"] is None and flat["significant"] is None and flat["estimate"] == 0.0
    assert sel.proportions(3, 0, 2, 10, what="x")["status"] == "unavailable"


def test_paired_methods_and_exact_small_samples():
    d = RNG(4).normal(1.0, 2.0, 40)
    r = sel.paired(np.zeros(40), d, what="revenue", unit_name="store")
    t = d.mean() / (d.std(ddof=1) / math.sqrt(40))
    assert r["method"] == "paired_t" and r["test"]["statistic"] == pytest.approx(t) and r["test"]["df"] == 39
    assert r["test"]["p_value"] == pytest.approx(2 * st.t.sf(abs(t), 39))
    assert any(x["method"] == "welch_t" and "not independent" in x["reason"] for x in r["rejected_methods"])

    five = sel.paired([0, 0, 0, 0, 0], [1, 2, 3, 4, 5], what="revenue", unit_name="store")
    assert five["method"] == "sign_flip_permutation" and five["test"]["exact"] and five["test"]["p_value"] == pytest.approx(2 / 32)
    assert five["significant"] is False and five["interval"] is None
    assert any("cannot detect an effect of any size" in w for w in five["warnings"]), "2/32 > 0.05: no data could be significant"

    ties = sel.paired([1, 2, 3], [2, 3, 4], what="revenue", unit_name="store")
    assert ties["status"] == "unavailable" and "exactly the same amount" in ties["reason"]
    assert sel.paired([1], [2], what="x", unit_name="store")["status"] == "unavailable"
    typical = sel.paired(np.zeros(30), RNG(5).normal(1, 1, 30), what="revenue", unit_name="store", estimand="typical")
    assert typical["method"] == "wilcoxon_signed_rank" and "median change" in typical["estimand"]


def test_k_group_test_reduces_to_welch_and_pairwise_is_holm_adjusted():
    a, b = RNG(1).normal(0, 1, 40), RNG(2).normal(0.5, 2, 55)
    two = st.ttest_ind(b, a, equal_var=False)
    w = m.welch_anova([a, b])
    assert w["statistic"] == pytest.approx(two.statistic ** 2) and w["p_value"] == pytest.approx(two.pvalue)
    assert w["df"][1] == pytest.approx(two.df)

    groups = {"a": RNG(1).normal(0, 1, 40), "b": RNG(2).normal(0.2, 2, 50), "c": RNG(3).normal(1, 1, 45)}
    r = sel.k_groups(groups, what="units", group_name="region")
    raw = [c["p_value"] for c in r["comparisons"]]
    assert r["method"] == "welch_anova" and len(raw) == 3 and [c["p_adjusted"] for c in r["comparisons"]] == pytest.approx(m.holm(raw))
    assert all(c["p_adjusted"] >= c["p_value"] for c in r["comparisons"]) and r["details"]["pairwise_correction"] == "Holm"
    assert "all pairwise comparisons of region" in r["details"]["pairwise_family"]
    many = sel.k_groups({str(i): RNG(i).normal(0, 1, 20) for i in range(8)}, what="units", group_name="store")
    assert many["comparisons"] == [] and any("pairwise comparisons were not run" in x for x in many["warnings"])
    skewed = sel.k_groups({"a": RNG(1).lognormal(0, 1.5, 12), "b": RNG(2).lognormal(0, 1.5, 12), "c": RNG(3).lognormal(0, 1.5, 12)},
                          what="units", group_name="region")
    assert skewed["method"] == "permutation_anova"
    assert sel.k_groups({"a": [1, 2, 3], "b": [4]}, what="u", group_name="g")["status"] == "unavailable"
    total = sum(len(v) for v in groups.values())
    values = np.concatenate(list(groups.values()))
    between = sum(len(v) * (v.mean() - values.mean()) ** 2 for v in groups.values())
    assert r["effects"][0]["value"] == pytest.approx(between / ((values - values.mean()) ** 2).sum()) and r["sample"]["n"] == total


def test_multiplicity_corrections():
    p = [0.01, 0.04, 0.03, 0.005]
    assert m.holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert m.benjamini_hochberg(p) == pytest.approx(list(st.false_discovery_control(p)))
    assert m.holm([0.2]) == [0.2] and m.holm([0.9, 0.9]) == [1.0, 1.0], "never above 1"
    assert sel.adjust_family(p, "bh") == pytest.approx(m.benjamini_hochberg(p))
    with pytest.raises(ValueError):
        sel.adjust_family(p, "bonferroni-ish")


def test_association_measures():
    perfect = sel.contingency(["a"] * 30 + ["b"] * 30, ["x"] * 30 + ["y"] * 30, row_name="group", col_name="outcome")
    assert perfect["estimate"] == pytest.approx(1.0) and perfect["method"] == "chi_square"
    sparse = sel.contingency(list("aaabbbcccd"), list("xyxyxyxyxy"), row_name="r", col_name="c")
    assert sparse["method"] == "permutation_chi_square" and 0 < sparse["test"]["p_value"] <= 1
    assert sel.contingency(["a"] * 5, list("xyxyx"), row_name="r", col_name="c")["status"] == "unavailable"

    x = RNG(6).normal(0, 1, 100)
    y = 0.5 * x + RNG(7).normal(0, 1, 100)
    r = sel.correlation(x, y, x_name="x", y_name="y")
    rho = float(np.corrcoef(x, y)[0, 1])
    half = st.norm.ppf(0.975) / math.sqrt(97)
    assert r["method"] == "pearson" and r["estimate"] == pytest.approx(rho)
    assert (r["interval"]["low"], r["interval"]["high"]) == pytest.approx((math.tanh(math.atanh(rho) - half), math.tanh(math.atanh(rho) + half)))
    wild = x.copy()
    wild[0] = 500.0
    assert sel.correlation(wild, y, x_name="x", y_name="y")["method"] == "spearman", "one extreme value would dominate Pearson"
    assert sel.correlation([1, 2, 3], [1, 2, 3], x_name="x", y_name="y")["status"] == "unavailable"
    assert sel.correlation([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], x_name="x", y_name="y")["status"] == "unavailable"


def test_robust_standard_errors_match_their_closed_forms():
    a, b = RNG(1).normal(10, 1, 30), RNG(2).normal(12, 4, 50)
    y = np.concatenate([a, b])
    X = np.column_stack([np.ones(80), np.r_[np.zeros(30), np.ones(50)]])
    hc3 = ts.ols(X, y, cov="HC3")
    assert hc3["beta"][1] == pytest.approx(b.mean() - a.mean())
    assert hc3["se"][1] ** 2 == pytest.approx(a.var(ddof=1) / 29 + b.var(ddof=1) / 49), "HC3 for a two-group comparison"
    hac0 = ts.ols(X, y, cov="HAC", lags=0)
    hc0 = ((a - a.mean()) ** 2).sum() / 30 ** 2 + ((b - b.mean()) ** 2).sum() / 50 ** 2
    assert hac0["se"][1] ** 2 == pytest.approx(hc0 * 80 / 78), "with no lags, Newey-West is the heteroskedasticity-robust estimate"
    x = RNG(3).normal(0, 1, 60)
    yy = 2 + 3 * x + RNG(4).normal(0, 1, 60)
    classical = ts.ols(np.column_stack([np.ones(60), x]), yy, cov="classical")
    line = st.linregress(x, yy)
    assert classical["beta"][1] == pytest.approx(line.slope) and classical["se"][1] == pytest.approx(line.stderr)
    assert classical["r_squared"] == pytest.approx(line.rvalue ** 2)
    with pytest.raises(ts.RegressionError, match="collinear"):
        ts.ols(np.column_stack([np.ones(10), np.ones(10)]), np.arange(10.0))
    with pytest.raises(ts.RegressionError, match="cannot estimate"):
        ts.ols(np.ones((2, 2)), np.ones(2))
    assert ts.newey_west_lag(100) == 4 and ts.newey_west_lag(12) == 2


# ============================================================ selection ====

def test_the_selector_chooses_by_the_data_and_records_why():
    normal = sel.two_groups(RNG(1).normal(0, 1, 50), RNG(2).normal(0, 1, 50), what="x")
    assert normal["method"] == "welch_t" and all(a["status"] == "met" for a in normal["assumptions"][:2])
    assert any("Cochran" in r for r in normal["selection"])

    skew_a, skew_b = RNG(3).lognormal(0, 1.5, 25), RNG(4).lognormal(0.4, 1.5, 25)
    skewed = sel.two_groups(skew_a, skew_b, what="x")
    assert skewed["method"] == "permutation_means" and skewed["interval"]["method"].startswith("bootstrap")
    assert "difference in mean" in skewed["estimand"], "the fallback keeps the estimand; it does not switch to ranks"
    assert {r["method"] for r in skewed["rejected_methods"]} == {"welch_t", "mann_whitney"}
    assert any(a["status"] == "violated" and "Cochran's rule needs n >=" in a["detail"] for a in skewed["assumptions"])
    again = sel.two_groups(skew_a, skew_b, what="x")
    assert (again["test"]["p_value"], again["interval"]) == (skewed["test"]["p_value"], skewed["interval"]), "resampling is seeded"

    big = RNG(5).lognormal(0, 1.5, 4000)
    assert sel.two_groups(big, RNG(6).lognormal(0, 1.5, 4000), what="x")["method"] in ("welch_t", "permutation_means")
    mild = RNG(7).gamma(8, 1, 400)                          # skewness about 0.7: 400 >= 25 * 0.5
    assert sel.two_groups(mild, RNG(8).gamma(8, 1, 400), what="x")["method"] == "welch_t", "the rule scales with skewness, not a fixed n"

    typical = sel.two_groups(RNG(1).normal(0, 1, 50), RNG(2).normal(1, 1, 50), what="x", estimand="typical")
    assert typical["method"] == "mann_whitney" and "median" in typical["estimand"]
    apart = sel.two_groups([1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18], what="x", estimand="typical")
    assert apart["effects"][0]["value"] == pytest.approx(1.0), "every later value exceeds every earlier one"

    tiny = sel.two_groups([1, 2, 3], [2, 3, 9], what="x")
    assert tiny["method"] == "permutation_means" and tiny["test"]["exact"] and tiny["interval"] is None
    assert any("smallest p-value any data could produce is 0.100" in w for w in tiny["warnings"])
    assert sel.two_groups([1], [2, 3], what="x")["reason"].startswith("each group needs at least 2 values")
    assert "nothing to test" in sel.two_groups([4, 4, 4], [5, 5, 5], what="x")["reason"]
    assert sel.two_groups([], [], what="x")["status"] == "unavailable"


@pytest.mark.parametrize("suggested,outcome", [("t-test", "accepted"), ("Welch", "accepted"), ("mann_whitney", "overridden"),
                                              ("chi-square", "overridden"), ("quantum annealing", "unknown"), ("", None), (None, None)])
def test_a_suggested_method_is_advice_not_an_instruction(suggested, outcome):
    a, b = RNG(1).normal(0, 1, 50), RNG(2).normal(0.3, 1, 50)
    r = sel.two_groups(a, b, what="x", suggested=suggested)
    assert r["method"] == "welch_t", "the data decides"
    assert (r["suggested_method"] or {}).get("outcome") == outcome
    assert r["test"]["p_value"] == sel.two_groups(a, b, what="x")["test"]["p_value"], "a suggestion never changes a number"


def test_missing_values_and_outliers_are_reported_not_silently_handled():
    a = [10, 11, None, 12, "n/a", 9, 10, 11, 12, 10, 11, 9, 10, 11, 12, 10]
    b = [10, 11, 12, 9, 10, 11, 12, 10, 11, 9, 10, 11, 12, 10, 11, 400]
    r = sel.two_groups(a, b, what="x")
    assert r["sample"]["excluded"]["missing_or_non_numeric"] == {"earlier": 2, "later": 0}
    assert r["sample"]["groups"] == {"earlier": 14, "later": 16} and "nothing is imputed" in r["sample"]["exclusion_rule"]
    assert r["descriptives"]["later"]["extreme_outliers"] == 1 and r["descriptives"]["later"]["max"] == 400, "the outlier is kept"
    assert any("extreme value(s) in 'later'" in w and "they were kept" in w for w in r["warnings"])
    assert r["method"] == "permutation_means", "one extreme value makes the mean too skewed for a t interval"
    d = m.describe(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert (d["n"], d["median"], d["outliers"]) == (5, 3.0, 1) and d["sd"] == pytest.approx(np.std([1, 2, 3, 4, 100], ddof=1))
    assert m.describe(np.array([]))["mean"] is None and m.describe(np.array([5.0]))["sd"] is None


# ============================================================= coverage ====

def test_welch_interval_covers_the_true_difference_about_95_percent_of_the_time():
    rng, hits, reps = RNG(11), 0, 400
    for _ in range(reps):
        r = m.welch(rng.normal(10, 2, 25), rng.normal(11, 5, 40), 0.95)
        hits += r["interval"]["low"] <= 1.0 <= r["interval"]["high"]
    assert 0.92 <= hits / reps <= 0.98


def test_wilson_interval_holds_its_coverage_where_the_textbook_interval_fails():
    rng, n, p, reps = RNG(12), 30, 0.05, 2000
    wilson = wald = 0
    for x in rng.binomial(n, p, reps):
        lo, hi = m.wilson(int(x), n, 0.95)
        wilson += lo <= p <= hi
        ph = x / n
        half = 1.96 * math.sqrt(ph * (1 - ph) / n)
        wald += ph - half <= p <= ph + half
    assert wilson / reps >= 0.93 and wald / reps < 0.85


def test_autocorrelation_aware_test_does_not_cry_wolf_on_a_series_with_no_shift():
    """Daily values that drift (AR(1), no real change): a test treating days as
    independent finds 'significant' differences far too often."""
    rng, reps, naive, robust = RNG(13), 300, 0, 0
    for _ in range(reps):
        e = rng.normal(0, 1, 120)
        x = np.zeros(120)
        for i in range(1, 120):
            x[i] = 0.7 * x[i - 1] + e[i]
        naive += st.ttest_ind(x[60:], x[:60], equal_var=False).pvalue < 0.05
        robust += ts.level_shift(x[:60], x[60:], 0.95)["p_value"] < 0.05
    assert naive / reps > 0.25, "the naive test is badly miscalibrated here"
    assert robust / reps < naive / reps / 2 and robust / reps < 0.2, "Newey-West errors remove most of the false alarms"


# ===================================================== series and regression ==

def test_level_shift_trend_and_change_point_on_known_series():
    rng = RNG(21)
    flat = rng.normal(100, 5, 40)
    shifted = sel.level_shift(flat, rng.normal(90, 5, 40), what="revenue", bucket="day", units="USD")
    assert shifted["method"] == "ols_newey_west" and shifted["significant"] and -13 < shifted["estimate"] < -7
    assert shifted["interval"]["low"] < shifted["estimate"] < shifted["interval"]["high"] < 0
    assert any(r["method"] == "welch_t" and "independent" in r["reason"] for r in shifted["rejected_methods"])
    assert any("not independent" in l for l in shifted["limitations"])
    short = sel.level_shift([1, 2, 3], [2, 3, 4], what="revenue", bucket="day")
    assert short["status"] == "unavailable" and f"at least {config.STAT_MIN_SERIES_POINTS} are needed" in short["reason"]
    assert sel.level_shift([5] * 10, [5] * 10, what="r", bucket="day")["status"] == "unavailable"
    assert sel.level_shift([5] * 10, [7] * 10, what="r", bucket="day")["status"] == "unavailable", "an exact difference needs no test"

    falling = 200 - 2.0 * np.arange(40) + rng.normal(0, 3, 40)
    later = 200 - 2.0 * np.arange(40, 80) + rng.normal(0, 3, 40)
    trend = sel.pre_trend(falling, later, what="revenue", bucket="day")
    assert trend["estimate"] == pytest.approx(-2.0, abs=0.15) and trend["significant"]
    assert trend["details"]["share_explained_by_trend"] == pytest.approx(1.0, abs=0.1), "the later level is where the trend was heading"
    none = sel.pre_trend(flat, rng.normal(90, 5, 40), what="revenue", bucket="day")
    assert not none["significant"] and abs(none["details"]["share_explained_by_trend"]) < 0.5
    assert any("extrapolation, not a forecast" in l for l in none["limitations"])

    step = np.r_[rng.normal(10, 1, 35), rng.normal(7, 1, 45)]
    labels = [f"d{i}" for i in range(80)]
    cp = sel.change_point(step, labels, what="revenue", bucket="day")
    assert cp["significant"] and abs(cp["details"]["position"] - 35) <= 2 and cp["estimate"] == pytest.approx(-3.0, abs=0.5)
    assert cp["details"]["shift_after"] == labels[cp["details"]["position"] - 1] and cp["details"]["p_value_is_approximate"]
    quiet = sel.change_point(rng.normal(10, 1, 80), labels, what="revenue", bucket="day")
    assert not quiet["significant"]
    assert sel.change_point(step[:8], labels[:8], what="r", bucket="day")["status"] == "unavailable"
    k = ts.pettitt(np.r_[np.zeros(10), np.ones(10)])
    assert k["index"] == 9 and k["statistic"] == pytest.approx(100.0), "U at the true split is n1 * n2"


def test_regression_separates_an_adjusted_association_from_a_raw_one():
    """The outcome depends only on region; the later period simply has more
    West rows. Unadjusted, the periods differ. Adjusted for region, they do not."""
    rng, n = RNG(31), 800
    later = rng.integers(0, 2, n)
    region = np.where(rng.random(n) < 0.25 + 0.5 * later, "West", "East")
    y = 10 + 5 * (region == "West") + rng.normal(0, 2, n)
    r = sel.adjusted_difference(y, later, {"region": region}, what="units", units="units")
    d = r["details"]
    assert d["unadjusted"]["estimate"] == pytest.approx(y[later == 1].mean() - y[later == 0].mean())
    assert d["unadjusted"]["p_value"] < 0.001 and not r["significant"] and abs(r["estimate"]) < 0.5
    assert d["attenuation"] > 0.8 and d["controls"] == ["region"] and d["parameters"] == 3
    west = next(c for c in d["coefficients"] if c["term"].startswith("region = "))
    assert abs(west["estimate"]) == pytest.approx(5.0, abs=0.4) and west["low"] < west["estimate"] < west["high"]
    assert "not a causal estimate" in r["limitations"][0] and "omitted-variable bias" in r["limitations"][0]
    assert {a["name"]: a["status"] for a in r["assumptions"]}["no omitted confounders"] == "not_checked"

    real = y + 3.0 * later                                   # now the period itself matters as well
    kept = sel.adjusted_difference(real, later, {"region": region}, what="units")
    assert kept["significant"] and kept["estimate"] == pytest.approx(3.0, abs=0.4)
    assert kept["interval"]["low"] < 3.0 < kept["interval"]["high"]


def test_regression_refuses_what_the_data_cannot_support():
    rng = RNG(32)
    later = np.r_[np.zeros(12), np.ones(12)]
    y = rng.normal(0, 1, 24)
    small = sel.adjusted_difference(y, later, {"g": list("abcdef") * 4}, what="y")
    assert small["status"] == "unavailable" and "per coefficient" in small["reason"]
    n = 300
    later = np.r_[np.zeros(150), np.ones(150)]
    same = sel.adjusted_difference(rng.normal(0, 1, n), later, {"copy": later}, what="y")
    assert same["status"] == "unavailable" and "collinear" in same["reason"], "a control that is the period cannot be held fixed"
    ids = [f"id{i}" for i in range(n)]
    wide = sel.adjusted_difference(rng.normal(0, 1, n), later, {"customer": ids, "one": ["x"] * n}, what="y")
    assert wide["status"] == "unavailable" and wide["reason"] == "no usable control variable"
    assert {d["control"] for d in wide["sample"]["excluded"]["controls_dropped"]} == {"customer", "one"}
    missing = pd.Series(rng.normal(0, 1, n))
    missing[:40] = np.nan
    ok = sel.adjusted_difference(missing, later, {"x": rng.normal(0, 1, n)}, what="y")
    assert ok["sample"]["n"] == 260 and ok["sample"]["excluded"]["rows_with_missing_values"] == 40
    assert "complete cases" in ok["sample"]["exclusion_rule"]


# ============================================== operations on the fixture ==

@pytest.fixture(scope="module")
def sales():
    return load_and_profile_csv(str(FIXTURES / "sales.csv"))


def context(df, profile, formula="sum(revenue)", a=("2025-01-01", SPLIT), b=(SPLIT, "2025-04-29")) -> ops.AnalysisContext:
    ctx = ops.AnalysisContext(formula=me.parse_formula(formula), label="Revenue", base_filters=[], time_column="order_date",
                              period_a=list(a), period_b=list(b), profile=profile)
    base = ops.execute(df, ctx, {"op": "metric_change", "params": {}})
    ctx.total_a, ctx.total_b = base["facts"]["a"], base["facts"]["b"]
    return ctx


def run(df, ctx, name, params, expect):
    op = ops.normalize({"op": name, "params": params}, ctx)
    out = ops.execute(df, ctx, op)
    return out, ops.evaluate(op, out, {"expect": expect})


def test_the_daily_series_matches_an_independent_count_and_handles_gaps(sales):
    df, profile = sales
    ctx = context(df, profile)
    series = ops.bucket_series(df, ctx, ctx.formula)
    by_day: dict[str, float] = {}
    for r in ROWS:
        by_day[r["order_date"]] = by_day.get(r["order_date"], 0.0) + float(r["revenue"])
    earlier = [by_day[d] for d in sorted(by_day) if d < SPLIT]
    assert series["bucket"] == "day" and list(series["a"]) == pytest.approx(earlier) and series["labels_b"][0] == FIRST_LATER
    assert float(series["a"].sum()) == ctx.total_a and float(series["b"].sum()) == ctx.total_b
    assert any("Less than two years" in l for l in series["context"]["limitations"])
    assert series["context"]["assumptions"][0]["name"] == "comparable calendar windows"

    out, verdict = run(df, ctx, "level_shift", {}, "decrease")
    r = out["facts"]["stat"]
    assert r["estimate"] == pytest.approx(np.mean(series["b"]) - np.mean(series["a"])) and r["sample"]["bucket"] == "day"
    assert r["interval"]["high"] < 0 and r["test"]["df"] == len(series["a"]) + len(series["b"]) - 2
    assert verdict["status"] == ops.SUPPORTED and "Newey-West" in verdict["reasons"][0], "on its own calendar the drop is clear"

    # The fixture has rows every second day: those days are its calendar, not a string of zero-sales days.
    assert len(series["a"]) == 30 and 0.0 not in series["a"] and series["context"]["excluded"]["buckets_without_rows"]["handling"] == "left out"
    assert any("uses only days with data" in w for w in series["context"]["warnings"])

    days = pd.date_range("2025-01-01", "2025-04-30", freq="D")           # a dense calendar with three empty days
    dense = pd.DataFrame({"order_date": np.repeat(days, 2), "revenue": 50.0, "units": 2})
    gaps = dense[~dense["order_date"].isin(pd.to_datetime(["2025-03-10", "2025-03-11", "2025-03-12"]))]
    gctx = context(gaps, profile_dataframe(gaps), b=(SPLIT, "2025-04-30"))
    holed = ops.bucket_series(gaps, gctx, gctx.formula)
    assert len(holed["b"]) == 61 and list(holed["b"]).count(0.0) == 3, "an occasional empty day is a zero for a sum"
    assert holed["context"]["excluded"]["buckets_without_rows"] == {"earlier": 0, "later": 3, "handling": "counted as 0"}
    assert any("counted as 0" in w and "missing for those days" in w for w in holed["context"]["warnings"])
    avg = ops.bucket_series(gaps, gctx, me.parse_formula("avg(units)"))
    assert len(avg["b"]) == 58 and avg["context"]["excluded"]["buckets_without_rows"]["handling"] == "left out", "an average has no value on an empty day"

    cut = df[df["order_date"] < "2025-03-05"]                # the later period has two days with data
    short, verdict = run(cut, context(cut, profile), "level_shift", {}, "decrease")
    assert short["facts"]["stat"]["status"] == "unavailable" and short["facts"]["buckets_b"] == 2
    assert verdict["status"] == ops.INCONCLUSIVE, "an incomplete period yields no statistical conclusion, not a negative one"


def test_long_periods_are_analysed_by_week_and_partial_weeks_are_left_out():
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    df = pd.DataFrame({"order_date": days, "revenue": 100.0 + (days.dayofweek >= 5) * 50})
    profile = profile_dataframe(df)
    ctx = context(df, profile, a=("2023-01-01", "2024-01-01"), b=("2024-01-01", "2024-12-31"))
    series = ops.bucket_series(df, ctx, ctx.formula)
    assert series["bucket"] == "week" and len(series["a"]) == 52 and set(series["a"]) == {800.0}, "only whole Monday-Sunday weeks"
    assert pd.Timestamp(series["labels_a"][0]).weekday() == 0 and "only partly inside a period" in series["context"]["exclusion_rule"]
    assert not any("Less than two years" in l for l in series["context"]["limitations"])

    uneven = pd.DataFrame({"order_date": pd.date_range("2025-01-01", "2025-01-31", freq="D"), "revenue": 1.0})
    ctx = context(uneven, profile_dataframe(uneven), a=("2025-01-01", "2025-01-11"), b=("2025-01-11", "2025-01-31"))
    weekday = ops.bucket_series(uneven, ctx, ctx.formula)["context"]["assumptions"][0]
    assert weekday["status"] == "violated" and "weekly pattern" in weekday["detail"], "10 days against 21: the weekday mix differs"


def test_statistical_operations_on_the_sales_fixture(sales):
    df, profile = sales
    ctx = context(df, profile)
    units_a = [float(r["units"]) for r in ROWS if r["order_date"] < SPLIT]
    units_b = [float(r["units"]) for r in ROWS if r["order_date"] >= SPLIT]

    out, verdict = run(df, ctx, "distribution_shift", {"column": "units"}, "decrease")
    r = out["facts"]["stat"]
    assert r["estimate"] == pytest.approx(np.mean(units_b) - np.mean(units_a)) and r["sample"]["groups"] == {"earlier": 360, "later": 360}
    assert verdict["status"] == ops.SUPPORTED and verdict["effect"]["interval"][1] < 0

    out, verdict = run(df, ctx, "paired_change", {"unit": "region"}, "decrease")
    assert out["facts"]["units_compared"] == 4 and (out["facts"]["units_down"], out["facts"]["units_up"]) == (1, 3)
    assert out["facts"]["stat"]["method"] == "sign_flip_permutation" and out["facts"]["stat"]["interval"] is None
    assert verdict["status"] == ops.INCONCLUSIVE and "1 of 4 region values went down, 3 up" in verdict["reasons"][0]
    assert out["table"][0]["unit"] == "West", "sorted by change: the one region that fell comes first"

    out, verdict = run(df, ctx, "composition_change", {"dimension": "region"}, "changed")
    assert out["facts"]["max_share_change_points"] == pytest.approx(0.0) and verdict["status"] == ops.WEAKENED
    assert "practically unchanged" in verdict["reasons"][0]

    out, verdict = run(df, ctx, "group_difference", {"outcome": "units", "dimension": "region"}, "differs")
    assert out["facts"]["stat"]["method"] == "welch_anova" and len(out["facts"]["stat"]["comparisons"]) == 6
    assert verdict["status"] == ops.SUPPORTED and out["facts"]["period"] == "later"

    out, verdict = run(df, ctx, "change_point", {}, "abrupt")
    assert verdict["status"] == ops.SUPPORTED and f"between {LAST_EARLIER} and {FIRST_LATER}" in verdict["reasons"][0], \
        "the shift is located exactly at the planted change"
    assert out["facts"]["stat"]["details"]["later_period_starts"] == FIRST_LATER

    out, verdict = run(df, ctx, "pre_trend", {}, "pre_existing_trend")
    assert verdict["status"] == ops.INCONCLUSIVE and "A trend is not ruled out" in verdict["reasons"][0]

    out, verdict = run(df, ctx, "adjusted_change", {"outcome": "units", "controls": ["region", "product"]}, "persists")
    assert verdict["status"] == ops.SUPPORTED and "Variables not in the dataset are not ruled out" in verdict["reasons"][0]
    _, explained = run(df, ctx, "adjusted_change", {"outcome": "units", "controls": ["region", "product"]}, "explained")
    assert explained["status"] == ops.WEAKENED, "the same result weakens the opposite prediction"

    out, verdict = run(df, ctx, "correlation", {"x": "unit_price", "y": "units"}, "negative")
    assert verdict["status"] == ops.WEAKENED and "An association, not a cause" not in verdict["reasons"][0]
    _, positive = run(df, ctx, "correlation", {"y": "unit_price", "x": "units"}, "positive")
    assert positive["status"] == ops.SUPPORTED and "An association, not a cause" in positive["reasons"][0]


@pytest.mark.parametrize("op,code", [
    ({"op": "correlation", "params": {"x": "units", "y": "units"}}, "invalid_operation"),
    ({"op": "correlation", "params": {"x": "units", "y": "region"}}, "wrong_column_type"),
    ({"op": "adjusted_change", "params": {"outcome": "units", "controls": []}}, "invalid_operation"),
    ({"op": "adjusted_change", "params": {"outcome": "units", "controls": ["order_date"]}}, "wrong_column_type"),
    ({"op": "adjusted_change", "params": {"outcome": "units", "controls": ["units"]}}, "wrong_column_type"),
    ({"op": "adjusted_change", "params": {"outcome": "region", "controls": ["product"]}}, "wrong_column_type"),
    ({"op": "paired_change", "params": {"unit": "order_date"}}, "wrong_column_type"),
    ({"op": "paired_change", "params": {"unit": "store"}}, "missing_column"),
    ({"op": "group_difference", "params": {"outcome": "ghost", "dimension": "region"}}, "missing_column"),
    ({"op": "composition_change", "params": {}}, "missing_column"),
    ({"op": "level_shift", "params": {"formula": "exec('x')"}}, "invalid_formula"),
    ({"op": "level_shift", "params": {"filters": [{"column": "ghost", "op": "==", "value": "1"}]}}, "missing_column"),
])
def test_statistical_operations_are_validated_before_they_run(sales, op, code):
    df, profile = sales
    with pytest.raises(ops.OpError) as e:
        ops.normalize(op, context(df, profile))
    assert e.value.code == code


def test_a_prediction_must_fit_the_statistical_operation():
    for name, bad in (("paired_change", "abrupt"), ("change_point", "decrease"), ("adjusted_change", "caused"), ("correlation", "differs")):
        with pytest.raises(ops.OpError):
            ops.validate_prediction(name, {"expect": bad})
    assert ops.validate_prediction("adjusted_change", {"expect": "explained"}) == {"expect": "explained"}


# ============================================================ the engine ==

import test_hypothesis_engine as engine_fixtures  # noqa: E402

QUESTION = engine_fixtures.QUESTION
hyp = engine_fixtures.hyp
STAT_HYPOTHESES = {"hypotheses": [
    hyp("Orders became smaller: fewer units per order", "distribution_shift", {"column": "units", "method": "t-test"}, "decrease", 5),
    hyp("Every region lost revenue", "paired_change", {"unit": "region"}, "decrease", 4),
    hyp("The drop began abruptly rather than gradually", "change_point", {}, "abrupt", 4),
    hyp("Revenue was already declining before March", "pre_trend", {}, "pre_existing_trend", 3),
    hyp("Smaller orders are explained by a different region and product mix", "adjusted_change",
        {"outcome": "units", "controls": ["region", "product"]}, "explained", 3),
    hyp("Prices were cut", "distribution_shift", {"column": "unit_price", "method": "tarot"}, "decrease", 2),
]}
REPORT = {"executive_summary": "Revenue fell from 145,800 to 131,310. Units per order fell by 2.25 on average (95% interval "
                               "-2.80 to -1.70, p < 0.001), which the data is consistent with. Only 1 of the 4 regions declined.",
          "possible_explanations": ["Demand in one region may have weakened."], "follow_up_analyses": ["Compare West with other regions by week."]}


def stat_script(**over) -> dict:
    base = engine_fixtures.script(submit_hypotheses=[STAT_HYPOTHESES], submit_next_step=[{"decision": "stop", "rationale": "resolved"}],
                                  submit_report=[REPORT])
    base.update(over)
    return copy.deepcopy(base)


@pytest.fixture(scope="module")
def stat_run(sales):
    df, profile = sales
    eng, provider = engine_fixtures.engine_for(df, profile, stat_script())
    return eng.run(), eng, provider


def test_every_hypothesis_carries_its_statistical_result(stat_run):
    report, _, _ = stat_run
    stats = {s["analysis_id"]: s for s in report["statistics"]}
    analyses = {a["id"]: a for a in report["analyses"]}
    by = {h["statement"]: h for h in report["hypotheses"]}
    for h in report["hypotheses"]:
        aid = h["evidence"][0]["analysis_id"]
        assert h["id"] in stats[aid]["hypothesis_ids"] and stats[aid]["schema_version"] == sr.SCHEMA_VERSION
        assert stats[aid] == {**analyses[aid]["result"]["facts"]["stat"], "analysis_id": aid, "op": analyses[aid]["op"],
                              "params": analyses[aid]["params"], "requested_by": "hypothesis", "hypothesis_ids": [h["id"]],
                              "provenance": analyses[aid]["provenance"]}
        assert analyses[aid]["provenance"]["context"] and analyses[aid]["provenance"]["computed_at"].endswith("Z")
    smaller = by["Orders became smaller: fewer units per order"]
    assert smaller["status"] == "supported" and smaller["effect"]["interval"] == pytest.approx([-2.80, -1.70], abs=0.01)
    s = stats[smaller["evidence"][0]["analysis_id"]]
    assert s["method"] == "welch_t" and s["suggested_method"]["outcome"] == "accepted" and s["units"] == "units"
    assert s["test"]["null"] == "the two groups have the same mean units" and s["sample"]["n"] == 720
    assert by["Every region lost revenue"]["status"] == "inconclusive", "one region of four fell: the data cannot support 'every'"
    assert by["The drop began abruptly rather than gradually"]["status"] == "supported"
    assert by["Revenue was already declining before March"]["status"] == "inconclusive"
    mix = by["Smaller orders are explained by a different region and product mix"]
    assert mix["status"] == "weakened" and "remains after adjustment" in mix["reasons"][0]
    prices = by["Prices were cut"]
    assert prices["status"] == "inconclusive"
    assert stats[prices["evidence"][0]["analysis_id"]]["suggested_method"] == {
        "method": "tarot", "outcome": "unknown", "reason": "not a supported method; the selector chose independently"}


def test_the_headline_change_gets_its_own_uncertainty_and_is_outside_the_family(stat_run):
    report, _, _ = stat_run
    headline = next(s for s in report["statistics"] if s["requested_by"] == "baseline")
    assert headline["op"] == "level_shift" and headline["hypothesis_ids"] == [] and headline["adjusted"] is None
    assert any(line.startswith(f"{headline['analysis_id']} (headline change)") for line in report["statistical_evidence"])
    m_ = report["multiplicity"]
    assert m_["method"] == "Holm" and m_["comparisons"] == 6 and headline["analysis_id"] not in m_["analysis_ids"]
    assert m_["family"] == "statistical tests of this investigation's hypotheses"
    family = [s for s in report["statistics"] if s["analysis_id"] in m_["analysis_ids"]]
    raw = [s["test"]["p_value"] for s in family]
    assert [s["adjusted"]["p_value"] for s in family] == pytest.approx(m.holm(raw)), "raw and adjusted are both kept"
    assert all(s["adjusted"]["comparisons"] == 6 and s["adjusted"]["method"] == "Holm" for s in family)
    assert not [s for s in report["statistics"] if s["requested_by"] == "scan"], "exploratory breakdowns carry no p-values"
    assert any("6 statistical tests were run" in l and "Holm-adjusted" in l for l in report["limitations"])
    assert report["reproducibility"]["statistics"] == {
        "schema_version": "1", "methods_version": "1", "significance_criterion": 0.05, "confidence_level": 0.95,
        "resampling_seed": config.STAT_SEED, "multiplicity_method": "holm"}


def test_the_report_separates_measurement_uncertainty_importance_and_cause(stat_run):
    report, _, _ = stat_run
    assert report["executive_summary"]["source"] == "model", "a summary quoting computed statistics is accepted"
    assert any("Revenue was 145,800" in f for f in report["observed_facts"])
    practical = " ".join(report["practical_significance"])
    assert "-12.5% of the earlier level: at or above the 2% materiality threshold, and statistically distinguishable" in practical
    assert "+3.1% of the earlier level" in practical and "but not statistically distinguishable from no change" in practical
    evidence = " ".join(report["statistical_evidence"])
    assert "Welch's t-test: p < 0.001" in evidence and "not evidence that there is no difference" in evidence
    reading = " ".join(report["how_to_read_statistics"])
    assert "does not say the difference is large, important or caused" in reading and "not evidence of no difference" in reading
    limits = " ".join(report["limitations"])
    assert "does not correct for missing, duplicated or unrepresentative records" in limits
    assert "every recorded row" in limits and "not proof of cause" in limits
    assert report["determination"]["code"] == "explanations_supported" and "do not prove what caused it" in report["determination"]["text"]


@pytest.mark.parametrize("summary,why", [
    ("Revenue fell from 145,800 to 131,310; units per order fell significantly (p = 0.012).", "0.012"),
    ("Revenue fell from 145,800 to 131,310, a drop with a 95% interval of -31,000 to -4,400.", "31,000"),
    ("Revenue fell from 145,800 to 131,310 across a sample of 9,431 orders (t = 8.04).", "9,431"),
])
def test_invented_statistics_never_reach_the_report(sales, summary, why):
    df, profile = sales
    eng, _ = engine_fixtures.engine_for(df, profile, stat_script(submit_report=[{**REPORT, "executive_summary": summary}]))
    report = eng.run()
    assert report["executive_summary"]["source"] == "template" and why in report["executive_summary"]["rejected_reason"]
    assert why not in report["summary"]


def test_a_result_that_does_not_survive_correction_is_inconclusive_not_supported():
    """Five tests; one has raw p = 0.03. Alone it would support its hypothesis.
    As one of five it does not (Holm: 0.03 x 5 = 0.15)."""
    rng, n = RNG(41), 200
    days = pd.date_range("2025-01-01", periods=2 * n, freq="D")
    data = {"day": days, "v": np.r_[np.full(n, 100.0), np.full(n, 80.0)]}
    for i in range(5):
        a, b = rng.normal(100, 30, n), rng.normal(100, 30, n)
        se = math.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
        target = -2.18 if i == 0 else -0.3                   # t fixed exactly: about p = 0.03 for x0
        data[f"x{i}"] = np.r_[a, b - (b.mean() - a.mean()) + target * se]
    df = pd.DataFrame(data)
    understanding = {**engine_fixtures.UNDERSTANDING, "metric_formula": "sum(v)", "time_column": "day",
                     "period_a": ["2025-01-01", str(days[n].date())], "period_b": [str(days[n].date()), str(days[-1].date())]}
    hyps = {"hypotheses": [hyp(f"x{i} fell", "distribution_shift", {"column": f"x{i}"}, "decrease", 5 - i) for i in range(5)]}
    replies = engine_fixtures.script(submit_understanding=[understanding], submit_hypotheses=[hyps],
                                     submit_next_step=[{"decision": "stop", "rationale": "x"}], submit_report=[{"executive_summary": ""}])
    alone = engine_fixtures.script(submit_understanding=[understanding], submit_hypotheses=[{"hypotheses": hyps["hypotheses"][:1]}],
                                   submit_next_step=[{"decision": "stop", "rationale": "x"}], submit_report=[{"executive_summary": ""}])
    single = engine_fixtures.engine_for(df, profile_dataframe(df), alone, question="Why did v fall?")[0].run()
    assert single["hypotheses"][0]["status"] == "supported" and single["multiplicity"] is None, "one test: nothing to correct"
    assert single["statistics"][-1]["adjusted"] is None

    report = engine_fixtures.engine_for(df, profile_dataframe(df), replies, question="Why did v fall?")[0].run()
    h = next(x for x in report["hypotheses"] if x["statement"] == "x0 fell")
    s = next(x for x in report["statistics"] if x["hypothesis_ids"] == [h["id"]])
    assert 0.02 < s["test"]["p_value"] < 0.04 and s["significant"] is True, "the raw result is reported as it is"
    assert s["adjusted"]["p_value"] == pytest.approx(min(1.0, 5 * s["test"]["p_value"])) and s["adjusted"]["p_value"] > 0.05
    assert h["status"] == "inconclusive" and "Holm over 5 tests" in h["reasons"][0] and "adjusted p = " in h["reasons"][0]
    assert report["hypothesis_counts"]["supported"] == 0 and report["determination"]["code"] == "cause_not_determined"
    assert any("but not statistically distinguishable" in l for l in report["practical_significance"])


def test_statistics_survive_without_a_model_and_are_never_filled_in_by_one(sales):
    df, profile = sales
    provider = ScriptedProvider(stat_script())

    def model(**k):
        if k["tool_name"] in ("submit_next_step", "submit_report"):
            raise runtime.ProviderUnavailable("quota")
        return provider("t", **k)

    report = he.HypothesisEngine(question=QUESTION, df=df, profile=profile, call_model=model).run()
    assert report["executive_summary"]["source"] == "template" and len(report["statistics"]) == 7
    assert report["multiplicity"]["comparisons"] == 6 and report["hypothesis_counts"]["supported"] == 2

    bad = {"hypotheses": [hyp("Prices fell", "distribution_shift", {"column": "unit_price", "method": {"p": 0.001}, "estimand": "vibes"}, "decrease", 5),
                          hyp("Units and price move together", "correlation", {"x": "units", "y": "units"}, "positive", 4),
                          hyp("Mix explains it", "adjusted_change", {"outcome": "units", "controls": "region"}, "explained", 3)]}
    eng, _ = engine_fixtures.engine_for(df, profile, stat_script(submit_hypotheses=[bad]))
    report = eng.run()
    assert [h["statement"] for h in report["hypotheses"]] == ["Prices fell"]
    s = next(x for x in report["statistics"] if x["hypothesis_ids"])
    assert s["suggested_method"] is None and "difference in mean" in s["estimand"], "a malformed suggestion is ignored"
    assert {r["statement"] for r in report["rejected_hypotheses"]} == {"Units and price move together", "Mix explains it"}

    prompts = []
    eng, provider = engine_fixtures.engine_for(df, profile, stat_script())
    real = eng.call_model
    eng.call_model = lambda **k: (prompts.append(k), real(**k))[1]
    eng.run()
    text = "\n".join(k["user_content"] + k["system"] for k in prompts)
    assert "Never state a p-value, interval, sample size, effect size" in text and "paired_change" in text
    assert '"statistics":{"method":"Welch\'s t-test"' in text and "interpretation" not in text, "prompts carry a brief, not the full result"
    assert all(len(k["user_content"]) < 40_000 for k in prompts)


def test_too_little_data_gives_an_inconclusive_investigation_not_a_guess(sales):
    df, profile = sales
    few = pd.concat([df[df["order_date"] < SPLIT].head(3), df[df["order_date"] >= SPLIT].head(3)])
    few = few.assign(revenue=[100.0, 110.0, 120.0, 60.0, 70.0, 65.0], units=[5, 6, 7, 3, 4, 3])
    hyps = {"hypotheses": [hyp("Orders became smaller", "distribution_shift", {"column": "units"}, "decrease", 5),
                           hyp("The level shifted", "level_shift", {}, "decrease", 4)]}
    report = engine_fixtures.engine_for(few, profile_dataframe(few), stat_script(submit_hypotheses=[hyps]))[0].run()
    by = {h["statement"]: h for h in report["hypotheses"]}
    small = next(s for s in report["statistics"] if s["op"] == "distribution_shift")
    assert small["method"] == "permutation_means" and small["interval"] is None and small["test"]["p_value"] == pytest.approx(0.1)
    assert by["Orders became smaller"]["status"] == "inconclusive"
    assert by["The level shifted"]["status"] == "inconclusive" and "no statistical conclusion" in by["The level shifted"]["reasons"][0]
    assert report["hypothesis_counts"]["supported"] == 0 and report["status"] == "inconclusive"
    assert report["determination"]["code"] in ("cause_not_determined", "data_cannot_support_conclusion")


# ============================================================ integration ==

@pytest.fixture
def ws(client):
    return workspace(client)


def test_statistics_are_persisted_and_served_with_tenancy(client, ws):
    created = ask(client, ws, QUESTION)
    assert run_job(created["job_id"], stat_script()) == "succeeded"
    q = question_row(created["question_id"])
    assert (q.status, q.route) == ("verified", "hypothesis")
    inv = client.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
    out = client.get(f"/api/investigations/{inv['id']}/statistics", headers=ws["h"])
    assert out.status_code == 200
    body = out.json()
    assert len(body["statistics"]) == 7 and body["multiplicity"]["comparisons"] == 6 and body["settings"]["significance_criterion"] == 0.05
    welch = next(s for s in body["statistics"] if s["op"] == "distribution_shift" and s["params"]["column"] == "units")
    assert welch["test"]["p_value"] < 0.001 and welch["adjusted"]["p_value"] >= welch["test"]["p_value"]
    assert welch["interval"]["level"] == 0.95 and welch["hypothesis_ids"] == ["H1"]
    paired = next(s for s in body["statistics"] if s["op"] == "paired_change")
    assert paired["interval"] is None and paired["test"]["p_value"] == 1.0, "unavailable is null in the API, never 0"
    detail = client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()
    assert len(detail["report"]["statistics"]) == 7 and detail["report"]["practical_significance"]
    stored = next(a for a in detail["analyses"] if a["op"] == "paired_change")
    assert stored["result"]["facts"]["stat"]["adjusted"]["method"] == "Holm", "the adjusted value is persisted with the analysis"
    assert "rows" not in str(stored["result"].get("table", [{}])[0].keys()), "aggregates per unit, not raw rows"

    ev = client.get(f"/api/questions/{q.id}/evidence", headers=ws["h"]).json()["records"]
    narrative = next(r for r in ev if r["claim_type"] == "narrative")
    checks = {c["check"]: c["outcome"] for c in narrative["checks"]}
    assert checks["narrative_numbers_supported"] == "pass" and checks["model_summary_grounded"] == "pass"
    assert narrative["validity"]["causal"] == "not_established"

    other = workspace(client)
    assert client.get(f"/api/investigations/{inv['id']}/statistics", headers=other["h"]).status_code == 404
    assert client.get(f"/api/investigations/{inv['id']}/statistics").status_code == 401
    assert client.get("/api/investigations/999999/statistics", headers=ws["h"]).status_code == 404


def test_the_deterministic_investigator_and_fast_path_are_unchanged(client, ws):
    """Regression: questions that do not go through the hypothesis engine get
    no statistical output bolted on."""
    created = ask(client, ws, "Why did revenue change?")
    assert run_job(created["job_id"], {}) == "succeeded"
    assert question_row(created["question_id"]).route == "root_cause"
    inv = client.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
    body = client.get(f"/api/investigations/{inv['id']}/statistics", headers=ws["h"]).json()
    assert body["statistics"] == [] and body["multiplicity"] is None
    report = client.get(f"/api/investigations/{inv['id']}", headers=ws["h"]).json()["report"]
    assert "statistics" not in report and report["ranked_contributors"][0]["segment"] == "West"
    fast = ask(client, ws, "total revenue")
    assert run_job(fast["job_id"], {}) == "succeeded" and question_row(fast["question_id"]).route == "fast"
