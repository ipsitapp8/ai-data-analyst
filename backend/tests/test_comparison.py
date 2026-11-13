"""The comparison workflow: a validated plan, carried out by the statistics
engine, published as an association with its uncertainty.

Reference values come from backend/evals/build_comparison_fixtures.py, which
uses the standard library only. Calibration figures come from seeded
simulation (app/evals/calibration.py). Acceptance gates are asserted as they
were set; a gate that is not met fails here and is not adjusted.
"""
from __future__ import annotations

import hashlib
import io
import json
import math

import numpy as np
import pandas as pd
import pytest
from platform_helpers import FIXTURES, ask, question_row, run_job, workspace

from app import comparison, config, runtime, verification
from app.agents import router
from app.evals import calibration, runner
from app.evals.scripted import ScriptedProvider
from app.profiling import load_and_profile_csv, profile_dataframe
from app.stats import methods, selector

NO_SEMANTICS = {"metrics": [], "ambiguous": [], "dimensions": []}
CASES = [c for c in runner.load_suite()["cases"] if c["component"] == "comparison"]
REPORT_PATH = config.BACKEND_DIR / "evals" / "reports" / "comparison_calibration_v4.json"


@pytest.fixture(scope="module")
def sales():
    return load_and_profile_csv(str(FIXTURES / "sales.csv"))


def plan_for(question: str, profile: dict) -> dict:
    out = comparison.parse(question, profile)
    assert out is not None and out["kind"] == "plan", question
    return out["plan"]


# ================================================== plans are validated first ==

@pytest.mark.parametrize("plan, code", [
    ("compare revenue by region", "invalid_plan"),
    ({"outcome": "revenue"}, "invalid_plan"),
    ({"outcome": "revenue", "explanatory": "regoin"}, "unknown_column"),
    ({"outcome": "profit", "explanatory": "region"}, "unknown_column"),
    ({"outcome": "revenue", "explanatory": "revenue"}, "same_column"),
    ({"outcome": "revenue", "explanatory": "order_date"}, "incompatible_type"),
    ({"outcome": "order_date", "explanatory": "region"}, "incompatible_type"),
    ({"outcome": "revenue", "explanatory": "region", "method": "bayesian_magic"}, "unsupported_method"),
    ({"outcome": "revenue", "explanatory": "region", "design": "crossover"}, "unsupported_design"),
    ({"outcome": "revenue", "explanatory": "region", "estimand": "mode"}, "unsupported_estimand"),
    ({"outcome": "revenue", "explanatory": "region", "levels": ["West"]}, "invalid_levels"),
    ({"outcome": "revenue", "explanatory": "region", "levels": ["West", "West"]}, "invalid_levels"),
    ({"outcome": "revenue", "explanatory": "units", "levels": ["1", "2"]}, "invalid_plan"),
    ({"outcome": "revenue", "explanatory": "region", "design": "paired"}, "invalid_plan"),
    ({"outcome": "revenue", "explanatory": "region", "design": "paired", "unit": "product"}, "invalid_levels"),
    ({"outcome": "channel", "explanatory": "region", "design": "paired", "unit": "product"}, "unsupported_design"),
    ({"outcome": "revenue", "explanatory": "region", "unit": "product"}, "invalid_plan"),
    ({"outcome": "revenue", "explanatory": "region", "outcome_level": "yes"}, "invalid_plan"),
    ({"outcome": "revenue", "explanatory": "region", "filters": [{"column": "nope", "op": "==", "value": "x"}]}, "invalid_filter"),
    ({"outcome": "revenue", "explanatory": "region", "filters": [{"column": "region", "op": "LIKE", "value": "x"}]}, "invalid_filter"),
    ({"outcome": "revenue", "explanatory": "region", "code": "import os"}, "invalid_plan"),
])
def test_an_invalid_plan_is_rejected_before_anything_runs(sales, monkeypatch, plan, code):
    _, profile = sales
    for name in ("two_groups", "k_groups", "paired", "proportions", "contingency", "correlation"):
        monkeypatch.setattr(selector, name, lambda *a, **k: pytest.fail("the statistics engine was called for an invalid plan"))
    with pytest.raises(comparison.PlanError) as e:
        comparison.validate(plan, profile)
    assert e.value.code == code


def test_a_grouping_column_with_too_many_groups_is_refused_unless_two_are_named():
    df = pd.DataFrame({"customer": [f"c{i % 40}" for i in range(400)], "spend": range(400)})
    profile = profile_dataframe(df)
    with pytest.raises(comparison.PlanError) as e:
        comparison.validate({"outcome": "spend", "explanatory": "customer"}, profile)
    assert e.value.code == "too_many_groups" and "name two of them" in str(e.value)
    assert comparison.validate({"outcome": "spend", "explanatory": "customer", "levels": ["c1", "c2"]}, profile)["levels"] == ["c1", "c2"]


@pytest.mark.parametrize("plan, code, says", [
    ({"outcome": "revenue", "explanatory": "channel", "levels": ["Online", "Wholesale"]}, "level_not_found", "Wholesale"),
    ({"outcome": "revenue", "explanatory": "region", "filters": [{"column": "region", "op": "==", "value": "Mars"}]}, "no_rows", "no rows"),
    ({"outcome": "revenue", "explanatory": "channel", "design": "paired", "unit": "product"}, "invalid_pairing", "more than once"),
])
def test_a_plan_the_data_contradicts_is_refused_without_computing_a_statistic(sales, monkeypatch, plan, code, says):
    df, profile = sales
    clean = comparison.validate(plan, profile)
    for name in ("two_groups", "k_groups", "paired", "proportions", "contingency", "correlation"):
        monkeypatch.setattr(selector, name, lambda *a, **k: pytest.fail("a statistic was computed for a refused plan"))
    with pytest.raises(comparison.PlanError) as e:
        comparison.run(df, clean)
    assert e.value.code == code and says in str(e.value)


def test_the_kind_of_comparison_follows_from_the_column_types_not_from_the_planner(sales):
    _, profile = sales
    family = lambda o, x: comparison.validate({"outcome": o, "explanatory": x}, profile)
    assert family("revenue", "region")["family"] == "groups"
    assert family("revenue", "units")["family"] == "correlation"
    assert family("channel", "region")["family"] == "table"
    swapped = family("region", "revenue")
    assert (swapped["family"], swapped["outcome"], swapped["explanatory"], swapped["roles_swapped"]) == ("groups", "revenue", "region", True)
    flag = profile_dataframe(pd.DataFrame({"returned": [0, 1] * 20, "amount": range(40)}))
    assert comparison.validate({"outcome": "amount", "explanatory": "returned"}, flag)["family"] == "groups", "a 0/1 column is a grouping"


# ======================================================= deterministic planner ==

@pytest.mark.parametrize("question, outcome, explanatory, levels, cause", [
    ("Compare revenue across channels", "revenue", "channel", None, False),
    ("Is average revenue different between regions?", "revenue", "region", None, False),
    ("How does revenue differ by region?", "revenue", "region", None, False),
    ("Do online orders have higher unit price than retail?", "unit_price", "channel", ["Online", "Retail"], False),
    ("Is revenue correlated with units?", "revenue", "units", None, False),
    ("Is there a correlation between units and unit price?", "units", "unit_price", None, False),
    ("Is channel related to region?", "channel", "region", None, False),
    ("Does region affect revenue?", "revenue", "region", None, True),
    ("Does unit price cause lower units?", "units", "unit_price", None, True),
    ("Is the typical revenue different between channels?", "revenue", "channel", None, False),
])
def test_a_comparison_question_is_read_into_a_plan(sales, question, outcome, explanatory, levels, cause):
    plan = plan_for(question, sales[1])
    assert (plan["outcome"], plan["explanatory"], plan["levels"], plan["claims_cause"]) == (outcome, explanatory, levels, cause)
    assert plan["source"] == "deterministic" and plan["plan_version"] == comparison.PLAN_VERSION


@pytest.mark.parametrize("question", [
    "What is the total revenue?",                               # no comparison asked
    "Average unit price per product",
    "Which region has the highest revenue?",                    # asks for an extreme
    "List the different regions and channels",                  # "different" is not "differ"
    "What is the maximum revenue for different regions?",
    "Is revenue higher in the last 2 months?",                  # a period the planner cannot honour
    "Is revenue different between regions excluding West?",     # a constraint it would have to ignore
    "Is revenue above 500 more common in West?",                # a threshold
    "Why is revenue different between regions?",                # a why question: another route
    "Is revenue different?",                                    # one column only
    "Is revenue related to units and cost?",                    # three columns
    "Show the trend of revenue over time",
    "",
])
def test_the_planner_declines_what_it_cannot_read_rather_than_guessing(sales, question):
    assert comparison.parse(question, sales[1]) is None


def test_a_word_that_could_mean_two_columns_is_asked_about_not_guessed():
    df = pd.DataFrame({"list_price": [10.0, 12, 14, 9, 11] * 8, "sale_price": [9.0, 11, 12, 9, 10] * 8,
                       "units": list(range(40)), "region": ["N", "S"] * 20})
    profile = profile_dataframe(df)
    out = comparison.parse("Is units related to price?", profile)
    assert out["kind"] == "ambiguous" and {o["label"] for o in out["options"]} == {"units and list_price", "units and sale_price"}
    decision = router.decide("Is units related to price?", profile, NO_SEMANTICS)
    assert decision["route"] == "clarify" and len(decision["clarification"]["options"]) == 2
    assert "Which one do you mean" in decision["clarification"]["reason"]
    assert comparison.parse("Is units related to list price?", profile)["plan"]["explanatory"] == "list_price"


@pytest.mark.parametrize("question, route", [
    ("Compare revenue across channels", "comparison"),
    ("Is revenue correlated with units?", "comparison"),
    ("Does region affect revenue?", "comparison"),
    ("What is the total revenue?", "fast"),
    ("Average unit price per product", "fast"),
    ("Why did revenue drop?", "root_cause"),
    ("Show the trend of revenue over time", "statistical"),
    ("Are there outliers in unit price?", "statistical"),
    ("Summarise this dataset for the board", "standard"),
])
def test_routing_sends_only_comparisons_to_the_comparison_route(sales, question, route):
    decision = router.decide(question, sales[1], NO_SEMANTICS)
    assert decision["route"] == route
    if route == "comparison":
        assert decision["spec"]["source"] == "deterministic" and any("statistics engine" in r for r in decision["reasons"])
        assert router.ROUTE_BUDGETS["comparison"] == {"llm_calls": 1, "sandbox_runs": 0, "seconds": 120}, "no code runs"


def test_the_route_can_be_switched_off(sales, monkeypatch):
    monkeypatch.setattr(config, "COMPARISON_ROUTE_ENABLED", False)
    assert router.decide("Compare revenue across channels", sales[1], NO_SEMANTICS)["route"] == "statistical"


def test_single_table_spider_questions_keep_their_route():
    """Regression against real questions: none of the verified Spider lookups
    may be taken for a comparison. Skipped where the dataset is not present
    (it is not committed)."""
    root = config.BACKEND_DIR.parent
    cases_path = root / "training" / "data" / "eval" / "spider_dev_single_table" / "cases.jsonl"
    if not cases_path.exists():
        pytest.skip("the Spider reference set is not on this machine")
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    valid = [c for c in cases if c.get("valid_for_scoring")]
    profiles: dict[str, dict] = {}
    taken = []
    for c in valid:
        profile = profiles.setdefault(c["csv"], load_and_profile_csv(str(root / c["csv"]))[1])
        if comparison.parse(c["question"], profile) is not None:
            taken.append(c["question"])
    assert len(valid) == 490 and taken == []


# ============================================ independent reference fixtures ==

@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_the_expected_method_is_selected(case):
    profile, path = runner._profile(case["fixture"])
    from app import metric_engine as me

    report = comparison.run(me.load_frame(path), comparison.validate(case["plan"], profile))
    stat = report["statistic"]
    assert stat["status"] == case["expect"].get("status", "ok") and stat["method"] == case["expect"]["method"], case["why"]
    assert stat["selection"] and stat["sample"] and stat["alpha"] == 0.05
    assert stat["method"] not in [r["method"] for r in stat["rejected_methods"]]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_estimate_interval_and_p_value_match_the_independent_reference(case):
    """Gate: agreement to 1e-6 with values computed by the standard library alone."""
    result = runner.run_case(case)
    failed = [(c["check"], c["detail"]) for c in result["detail"]["checks"] if not c["ok"]]
    assert result["passed"], failed


def test_the_reference_shares_no_code_with_the_application():
    source = (config.BACKEND_DIR / "evals" / "build_comparison_fixtures.py").read_text()
    imports = {line.split()[1].split(".")[0] for line in source.splitlines() if line.startswith(("import ", "from "))}
    assert imports <= {"__future__", "csv", "itertools", "json", "math", "random", "statistics", "pathlib"}, imports
    assert len(CASES) == 19 and len({c["expect"]["method"] for c in CASES}) == 12
    assert sum(c["expect"].get("status") == "unavailable" for c in CASES) == 3, "three fixtures are measured and not tested"
    assert all(isinstance(c["seed"], int) and c["n"] for c in CASES), "every fixture records its seed and sample size"


# ============================================================== what is said ==

def test_a_result_states_what_a_reader_needs_to_judge_it(sales):
    df, profile = sales
    report = comparison.run(df, plan_for("Is average revenue different between regions?", profile))
    stat = report["statistic"]
    assert stat["method"] == "welch_anova" and stat["test"]["p_value"] < 0.05 and stat["sample"]["n"] == 720
    assert stat["sample"]["groups"] == {"East": 180, "North": 180, "South": 180, "West": 180}
    assert {r["method"] for r in stat["rejected_methods"]} == {"kruskal_wallis"} and all(r["reason"] for r in stat["rejected_methods"])
    assert any(a["name"] == "independent observations" and a["status"] == "not_checked" for a in stat["assumptions"])
    assert stat["effects"][0]["name"].startswith("eta squared") and 0 < stat["effects"][0]["value"] < 1
    assert len(stat["comparisons"]) == 6 and all(c["p_adjusted"] >= c["p_value"] for c in stat["comparisons"])
    assert any("Holm-adjusted" in l for l in report["limitations"]) and any("observational" in l for l in report["limitations"])
    assert [g["level"] for g in report["groups"]] == ["East", "North", "South", "West"]

    two = comparison.run(df, plan_for("Do online orders have higher unit price than retail?", profile))["statistic"]
    assert two["interval"]["low"] < two["estimate"] < two["interval"]["high"] and two["interval"]["level"] == 0.95
    assert [e["name"] for e in two["effects"]] == ["mean difference", "relative difference, percent", "Hedges' g (standardised mean difference)"]


def test_too_little_data_gives_no_conclusion_and_says_what_is_missing():
    df = pd.DataFrame({"arm": ["a", "b", "b"], "score": [1.0, 2.0, 4.0]})
    report = comparison.run(df, comparison.validate({"outcome": "score", "explanatory": "arm"}, profile_dataframe(df)))
    stat = report["statistic"]
    assert report["status"] == "unavailable" and stat["test"] is None and stat["interval"] is None
    assert "at least 2 values" in stat["reason"]
    text = comparison.narrative(report)
    assert text.startswith("No statistical comparison") and "at least 2 values" in text


# =========================================== intervals beside permutation tests ==
# Reliability fixes after the first calibration run: the bootstrap interval held the true value in
# 86.0% and 87.2% of small samples. It is replaced by the interval the permutation test implies,
# or by no interval at all.

def _small(seed: int, shift: float = 8.0, n: int = 12):
    """Small and not visibly skewed: the permutation test is still used."""
    rng = np.random.default_rng(seed)
    return rng.normal(50, 8, n), rng.normal(50, 8, n) + shift


def test_the_interval_beside_a_permutation_test_is_the_one_that_test_implies():
    a, b = _small(42)
    r = selector.two_groups(a, b, what="spend", labels=("a", "b"))
    assert r["method"] == "permutation_means" and r["interval"]["method"] == methods.SHIFT_INTERVAL
    assert r["interval"]["low"] < r["estimate"] < r["interval"]["high"]
    assert any(a["name"] == "the groups have the same shape" and a["status"] == "not_checked" for a in r["assumptions"])
    assert any("81% to 88%" in l and "94% to 95%" in l for l in r["limitations"]), "the measured coverage, good and bad, is stated with the result"
    assert r["test"]["statistic_name"] == "Welch's t"
    rejected = {x["method"]: x["reason"] for x in r["rejected_methods"]}
    assert "84% to 89%" in rejected["bootstrap_interval"], "why the earlier interval method is not used"
    assert "bootstrap" not in r["interval"]["method"]

    d = np.random.default_rng(10).normal(4, 4, 8)
    p = selector.paired(np.zeros(8), d, what="sales", unit_name="store")
    assert p["method"] == "sign_flip_permutation" and p["interval"]["method"] == methods.SYMMETRY_INTERVAL
    assert any(a["name"] == "differences symmetric about their centre" for a in p["assumptions"])
    assert any("10% to 14%" in l and "86% to 90%" in l for l in p["limitations"])


@pytest.mark.parametrize("kind", ["two groups", "paired"])
def test_the_interval_excludes_no_difference_exactly_when_the_exact_test_rejects(kind):
    """One assumption, one statement: an interval that disagreed with its own test would be two."""
    agree = total = 0
    for i in range(300):
        rng = np.random.default_rng(40_000 + i)
        if kind == "two groups":
            r = selector.two_groups(rng.normal(30, 9, 6), rng.normal(30, 9, 6) + rng.choice([0.0, 20.0]), what="x")
        else:
            r = selector.paired(np.zeros(9), rng.normal(rng.choice([0.0, 3.0]), 4, 9), what="x", unit_name="u")
        if r.get("inference_unavailable"):          # visibly skewed samples are measured and not tested
            assert r["test"] is None and r["interval"] is None
            continue
        assert r["test"]["exact"] is True and r["interval"] is not None
        total += 1
        agree += (not (r["interval"]["low"] <= 0 <= r["interval"]["high"])) == (r["test"]["p_value"] < 0.05)
    assert agree == total and total >= 60, (agree, total)


@pytest.mark.parametrize("build, reason", [
    (lambda rng: selector.paired(np.zeros(5), np.array([3.0, 5.5, 4.0, 6.5, 2.0]), what="x", unit_name="u"), "would be unbounded"),
    (lambda rng: selector.two_groups([0.0, 1.0, 2.0], [2.5, 3.5, 4.5], what="x"), "would be unbounded"),
    (lambda rng: selector.paired(np.zeros(5), rng.normal(4, 4, 5), what="x", unit_name="u", estimand="typical"), "too few"),
])
def test_when_no_defensible_interval_exists_none_is_reported_and_the_reason_is_given(build, reason):
    r = build(np.random.default_rng(11))
    assert r["status"] == "ok" and r["estimate"] is not None and r["test"]["p_value"] is not None, "the estimate and test still stand"
    assert r["interval"] is None, "no other interval is put in its place"
    assert reason in r["interval_unavailable"], "the reason is a field of the result, not only a sentence in a list"
    assert not any("interval" in e and e["interval"] for e in r["effects"] if isinstance(e, dict) and e.get("name") == "mean difference")


# ------------------------------------------------ audit of the permutation paths --

def test_every_result_has_an_interval_or_says_why_it_has_none():
    """Schema rule: an "ok" result carries an interval or a reason, never neither and never both."""
    rng = np.random.default_rng(21)
    results = [
        selector.two_groups(rng.normal(0, 1, 60), rng.normal(0, 1, 60), what="x"),
        selector.two_groups(rng.lognormal(0, 1, 9), rng.lognormal(0, 1, 9), what="x"),
        selector.two_groups(rng.normal(0, 1, 3), rng.normal(0, 1, 3), what="x"),
        selector.two_groups(rng.lognormal(0, 1, 9), rng.lognormal(0, 1, 9), what="x", estimand="typical"),
        selector.paired(np.zeros(60), rng.normal(1, 2, 60), what="x", unit_name="u"),
        selector.paired(np.zeros(8), rng.normal(1, 2, 8), what="x", unit_name="u"),
        selector.paired(np.zeros(5), rng.normal(1, 2, 5), what="x", unit_name="u"),
        selector.paired(np.zeros(5), rng.normal(1, 2, 5), what="x", unit_name="u", estimand="typical"),
        selector.paired(np.zeros(12), rng.normal(1, 2, 12), what="x", unit_name="u", estimand="typical"),
        selector.k_groups({k: rng.normal(0, 1, 40) for k in "abc"}, what="x", group_name="g"),
        selector.k_groups({k: rng.normal(0, 1, 5) for k in "abc"}, what="x", group_name="g"),
        selector.correlation(rng.normal(0, 1, 30), rng.normal(0, 1, 30), x_name="x", y_name="y"),
        selector.proportions(12, 40, 20, 40, what="share"),
        selector.contingency(rng.choice(list("abc"), 120), rng.choice(list("xyz"), 120), row_name="r", col_name="c"),
    ]
    assert {r["status"] for r in results} == {"ok", "unavailable"}
    for r in results:
        if r["status"] == "ok":
            assert (r["interval"] is None) != (r["interval_unavailable"] is None), r["method"]
        else:
            # measured, not tested: nothing inferential, and a structured reason in place of all of it
            assert r["inference_unavailable"]["code"] and r["sample"]["n"] and r["descriptives"]
            assert r["estimate"] is not None or "between" in r["estimand"], "several groups have no single difference to keep"
            assert r["test"] is None and r["interval"] is None and r["significant"] is None
        if r["status"] == "ok" and r["interval"] is None:
            assert len(r["interval_unavailable"]) > 15, r["method"]
    from app.stats.result import StatResultError, build
    with pytest.raises(StatResultError, match="never both"):
        build(**{**results[0], "interpretation": [], "interval_unavailable": "but there is one"})


def test_the_two_sided_rule_is_at_least_as_extreme_in_absolute_value():
    """With 24 and 8 values and unequal spread, "twice the smaller tail" rejected a true null in
    13.0% of simulations and this rule in 7.5%. Checked here on one exact case by enumeration."""
    import itertools
    a, b = np.array([10.0, 11, 9, 12, 10.5, 9.5, 10.2]), np.array([30.0, -5, 22])
    r = methods.permutation_means(a, b, 0.95, 1)
    pooled, t = np.r_[a, b], methods.welch_statistic(a, b)
    splits = [(np.array([pooled[i] for i in range(10) if i not in c]), pooled[list(c)]) for c in itertools.combinations(range(10), 3)]
    ts = np.array([float(methods.welch_statistic(x, y)) for x, y in splits])
    assert r["test"]["exact"] is True and r["test"]["resamples"] == 120 and r["test"]["statistic_name"] == "Welch's t"
    assert r["test"]["p_value"] == pytest.approx((np.abs(ts) >= abs(t) * (1 - 1e-9)).mean())
    twice_smaller_tail = min(1.0, 2 * min((ts <= t).mean(), (ts >= t).mean()))
    assert r["test"]["p_value"] != pytest.approx(twice_smaller_tail), "the two rules differ on this lopsided sample"


def test_the_permutation_tests_permute_welch_statistics_not_raw_differences():
    rng = np.random.default_rng(22)
    groups = [rng.normal(100, 5, 20), rng.normal(100, 5, 20), rng.normal(100, 20, 6)]
    assert float(methods.welch_f_statistic(*groups)) == pytest.approx(methods.welch_anova(groups)["statistic"], rel=1e-12)
    batch = np.stack([rng.permutation(np.concatenate(groups)) for _ in range(4)])
    vectorised = methods.welch_f_statistic(batch[:, :20], batch[:, 20:40], batch[:, 40:])
    one_by_one = [methods.welch_anova([row[:20], row[20:40], row[40:]])["statistic"] for row in batch]
    assert vectorised == pytest.approx(one_by_one, rel=1e-10)
    r = selector.k_groups({"a": np.arange(20.0), "b": np.arange(20.0) + 2, "c": np.arange(6.0) * 6}, what="x", group_name="g")
    assert r["method"] == "permutation_anova" and r["test"]["statistic_name"] == "Welch's F"
    assert methods.welch_statistic(np.array([1.0, 1, 1]), np.array([2.0, 2, 2])) == np.inf, "no variation: as far out as can be"
    assert methods.welch_statistic(np.array([1.0, 1, 1]), np.array([1.0, 1, 1])) == 0


def test_the_median_interval_is_distribution_free_and_exact():
    """Order statistics: the chance that the k-th smallest and k-th largest values bracket the
    median is a binomial sum, whatever the distribution. No resampling, no symmetry assumed."""
    from scipy import stats as st

    assert methods.median_interval(np.arange(5.0), 0.95) is None, "5 values: 1 - 2/32 = 93.75%, short of 95%"
    assert methods.median_interval(np.arange(6.0), 0.95) == (0.0, 5.0), "6 values: 1 - 2/64 = 96.9%"
    for n in (6, 9, 15, 30, 101):
        low, high = methods.median_interval(np.arange(float(n)), 0.95)
        k = int(low) + 1
        assert high == n - k and 1 - 2 * st.binom.cdf(k - 1, n, 0.5) >= 0.95 > 1 - 2 * st.binom.cdf(k, n, 0.5), n
    rng = np.random.default_rng(23)
    covered = sum(lo <= 0.0 <= hi for lo, hi in (methods.median_interval(rng.lognormal(0, 1.2, 12) - 1.0, 0.95) for _ in range(2000)))
    assert covered / 2000 >= 0.95, "skewed data with a true median of zero: at least the stated level"
    r = selector.paired(np.zeros(12), rng.lognormal(0, 1.2, 12) - 1.0, what="x", unit_name="u", estimand="typical")
    assert r["method"] == "wilcoxon_signed_rank" and r["interval"]["method"].startswith("order-statistic interval")
    assert r["test"]["null"] == "the changes are symmetric about zero"
    assert "bootstrap_interval" in [x["method"] for x in r["rejected_methods"]]
    assert any("symmetric about zero" in l and "14%" in l for l in r["limitations"])


def test_rank_test_results_state_their_measured_limits():
    rng = np.random.default_rng(24)
    r = selector.two_groups(rng.lognormal(0, 1, 10), rng.lognormal(0, 1, 10), what="x", estimand="typical")
    assert r["method"] == "brunner_munzel" and r["interval"]["method"].startswith("bootstrap")
    assert not any(a["name"] == "the groups have the same shape" for a in r["assumptions"]), "the test does not assume it"
    assert any("does not test whether the medians differ" in l and "3% to 8%" in l for l in r["limitations"])
    assert any("93% to 99%" in l for l in r["limitations"]), "the median difference is a description with a resampled interval"
    assert "14%" in next(x["reason"] for x in r["rejected_methods"] if x["method"] == "mann_whitney")
    assert r["test"]["null"] == "values in the two groups are equally likely to be higher", "the question asked is the one it was"


def test_tiny_samples_are_consistent_about_what_can_be_rejected():
    rng = np.random.default_rng(25)
    for na, nb, can_reject in ((3, 3, False), (2, 2, False), (4, 4, True), (2, 5, True), (3, 4, True)):
        r = selector.two_groups(np.arange(float(na)), 6 + np.arange(float(nb)), what="x")     # evenly spaced: no skewness
        assert (r["interval"] is not None) is can_reject, (na, nb)
        assert any("cannot detect an effect of any size" in w for w in r["warnings"]) is (not can_reject), (na, nb)
        assert r["test"]["p_value"] >= (2 if na == nb else 1) / math.comb(na + nb, nb) - 1e-12


# ------------------------------------------------------ method decisions (v3) --

def test_brunner_munzel_matches_an_independent_implementation_and_keeps_the_question():
    from scipy import stats as st

    rng = np.random.default_rng(31)
    a, b = rng.lognormal(0, 0.5, 60), rng.lognormal(0.3, 1.2, 45)
    r = selector.two_groups(a, b, what="x", estimand="typical")
    ref = st.brunnermunzel(a, b, alternative="two-sided", distribution="t")
    assert r["method"] == "brunner_munzel" and r["test"]["name"] == "Brunner-Munzel test" and not r["test"]["resamples"]
    assert abs(r["test"]["statistic"]) == pytest.approx(abs(ref.statistic), rel=1e-10) and r["test"]["p_value"] == pytest.approx(ref.pvalue, rel=1e-10)
    superiority = r["effects"][0]["value"]
    assert superiority == pytest.approx(np.mean([(y > x) + 0.5 * (y == x) for x in a for y in b]), rel=1e-12), "P(later > earlier), ties half"
    assert r["estimate"] == pytest.approx(np.median(b) - np.median(a)) and "median" in r["estimand"], "the reported estimate is what it was"
    assert r["test"]["null"] == "values in the two groups are equally likely to be higher"

    # small groups: the same statistic, referred to every split instead of a t distribution
    x, y = np.array([3.0, 5, 5, 6, 8, 9, 9]), np.array([1.0, 2, 2, 4, 5, 7])
    small = selector.two_groups(x, y, what="x", estimand="typical")
    assert small["test"]["exact"] is True and small["test"]["resamples"] == math.comb(13, 7) and "permutation" in small["method_label"]
    assert abs(small["test"]["statistic"]) == pytest.approx(abs(st.brunnermunzel(x, y).statistic), rel=1e-10), "ties by midranks"
    assert small["test"]["p_value"] * math.comb(13, 7) == pytest.approx(round(small["test"]["p_value"] * math.comb(13, 7))), "a count of splits"

    # no overlap at all: the standard error is zero; the permutation p-value is the smallest the sizes allow
    apart = selector.two_groups(np.arange(1.0, 6), np.arange(11.0, 16), what="x", estimand="typical")
    assert apart["test"]["p_value"] == pytest.approx(2 / math.comb(10, 5)) and apart["test"]["statistic"] is None
    assert apart["effects"][0]["value"] == 1.0
    # a suggestion of Mann-Whitney is recorded as overridden, with the reason
    sm = selector.two_groups(a, b, what="x", estimand="typical", suggested="mann_whitney")["suggested_method"]
    assert sm["outcome"] == "overridden" and "without assuming the groups have the same shape" in sm["reason"]


def test_mann_whitney_rejects_equal_tendency_under_unequal_spread_and_brunner_munzel_does_not():
    """The reason for the change, reproduced small: 24 and 8 values, same centre, the smaller group four times as variable."""
    from scipy import stats as st

    mw = bm = 0
    for i in range(400):
        rng = np.random.default_rng(70_000 + i)
        a, b = rng.normal(0, 1, 24), rng.normal(0, 4, 8)
        mw += st.mannwhitneyu(a, b, alternative="two-sided").pvalue < 0.05
        bm += selector.two_groups(a, b, what="x", estimand="typical")["test"]["p_value"] < 0.05
    # Brunner-Munzel is not exact here either (6% to 7% in the stored study); Mann-Whitney is at 14%.
    assert bm / 400 < 0.10 < mw / 400, (mw, bm)


def test_the_stored_method_decision_report_says_what_was_measured():
    report = json.loads((config.BACKEND_DIR / "evals" / "reports" / "method_decisions_v3.json").read_text())
    assert report["report_version"] == 3 and report["base_seed"] == 20_270_101
    rank = {d["design"]: d for d in report["rank_tests"]["designs"]}
    nulls = [d for d in rank.values() if d["null_is_true"]]
    assert len(nulls) >= 15 and all(d["simulations"] == 2000 for d in nulls)
    assert max(d["rejections"]["mann_whitney"]["rate"] for d in nulls) > 0.13, "Mann-Whitney's failure is on record"
    assert max(d["rejections"]["brunner_munzel_permutation"]["rate"] for d in nulls) < 0.075
    cells = report["workflow"]["mean"]["cells"]
    null_cells = [c for c in cells if c["null_is_true"]]
    assert len(null_cells) == 7 * 12 + 6 * 7 and all(c["simulations"] == 2000 and c["seed_first"] for c in cells)
    worst_off = max(c["rule_switched_off"]["type_one_error_of_all_samples"]["rate"] for c in null_cells)
    worst = max(c["engine"]["type_one_error_of_all_samples"]["rate"] for c in null_cells)
    assert worst_off > 0.20 and worst < 0.10, (worst_off, worst)
    assert all("interval" in c["engine"]["type_one_error_of_all_samples"] for c in null_cells), "every rate carries its uncertainty"
    assert any(not c["null_is_true"] for c in cells), "power is measured too"
    edges = {e["case"]: e for e in report["workflow"]["mean"]["edge_cases"]}
    assert edges["both groups constant and equal"]["status"] == "unavailable" and edges["one extreme value among 16"]["inference_unavailable"]


def test_measured_but_not_tested_is_a_state_of_its_own_everywhere(client):
    """Result, verdict, model brief, investigation summary, narrative, evidence and dashboard: the
    measurement and the sample sizes are kept, and nothing reads as 'no difference' or 'not significant'."""
    from app import analysis_ops as ops
    from app import investigator
    from app.stats.result import StatResultError, build

    a, b = [10.0, 11, 12, 9, 10, 11, 12, 10], [10.0, 11, 12, 9, 10, 11, 12, 95]
    r = selector.two_groups(a, b, what="spend", labels=("A", "B"))
    assert (r["status"], r["method"], r["test"], r["interval"], r["significant"]) == ("unavailable", "none", None, None, None)
    assert r["estimate"] == pytest.approx(np.mean(b) - np.mean(a)) and r["sample"]["groups"] == {"A": 8, "B": 8}
    assert r["descriptives"]["B"]["mean"] == pytest.approx(np.mean(b)) and r["effects"][0]["value"] == r["estimate"]
    w = r["inference_unavailable"]
    assert set(w) == {"code", "summary", "reason", "supported_instead", "supported_analysis"} and w["code"] == "too_skewed_for_sample_size"
    assert not any(ch.isdigit() for ch in w["summary"]), "the published sentence carries no figure that was not computed for this data"
    text = " ".join(r["interpretation"])
    assert "No test or interval is reported" in text and "not a finding of no difference" in text and "not a non-significant result" in text
    for wrong in ({"status": "ok"}, {"significant": False}, {"interval": {"low": 0, "high": 1, "level": 0.95, "method": "x"}}):
        with pytest.raises(StatResultError):
            build(**{**r, "interpretation": [], **wrong})

    gate = ops._stat_gate(r, "distribution shift")
    assert gate[0] == ops.INCONCLUSIVE and "but it was not tested" in gate[1][0] and gate[2] == {"estimate": r["estimate"], "tested": False}
    brief = ops.stat_brief(r)
    assert brief["status"] == "unavailable" and brief["estimate"] == r["estimate"] and brief["n"] == 16 and "not available" in brief["inference"]
    compact = investigator._compact_stat(r)
    assert compact["status"] == "unavailable" and compact["estimate"] == r["estimate"] and compact["inference_unavailable"]["code"] == w["code"]
    assert "significant" not in compact and "p_value" not in compact

    paired = selector.paired(np.zeros(8), np.array([-2.0, -1.5, -2.5, -1.0, -2.2, -1.8, -2.1, 14.0]), what="sales", unit_name="store")
    assert paired["inference_unavailable"]["code"] == "too_skewed_for_sample_size" and paired["estimate"] == pytest.approx(0.1125)
    assert paired["sample"]["pairs"] == 8 and any(x["status"] == "violated" and "symmetric" in x["name"] for x in paired["assumptions"])
    assert "median change" in paired["inference_unavailable"]["supported_instead"]

    csv = ("segment,spend\n" + "".join(f"A,{v}\n" for v in a) + "".join(f"B,{v}\n" for v in b)).encode()
    ws = workspace(client, csv=csv, filename="skewed_groups.csv")
    created = ask(client, ws, "Is spend different between segment groups?")
    from app import jobs
    assert jobs.run_inline(created["job_id"]) == "succeeded"
    qid = created["question_id"]
    assert question_row(qid).route == "comparison"
    dash = client.get(f"/api/questions/{qid}/dashboard", headers=ws["h"]).json()
    said = dash["narrative"]
    assert "No test or interval is reported for it" in said and "not a finding of no difference and not a non-significant result" in said
    # no p-value or interval answers the question asked; a p-value may appear only for the different question,
    # named as one and said not to test the difference in means
    asked, _, different = said.partition("A different question the data can answer")
    assert "p =" not in asked and "% interval" not in said and "does not distinguish" not in asked
    assert not different or "it is not a test of the difference in means" in different
    assert "Brunner-Munzel" in said, "the question the data does support is named"
    assert len(dash["kpis"]) >= 2, "the group means are still published"
    records = client.get(f"/api/questions/{qid}/evidence", headers=ws["h"]).json()["records"]
    narrative = next(x for x in records if x["claim_type"] == "narrative")
    assert "inference_not_available" in narrative["limitations"] and "no_statistical_conclusion" not in narrative["limitations"]
    assert not [c["check"] for x in records for c in x["checks"] if c["outcome"] == "fail"]
    assert any(c["check"] == "narrative_numbers_supported" and c["outcome"] == "pass" for c in narrative["checks"])


# ----------------------------------------------------- inference policy (v4) --

HISTORICAL_REPORTS = {'comparison_calibration.json': '9d56f72ed6676e687b9eed2d66b6612bd4a54171e234cc187ccffcf30e214a4f', 'comparison_calibration_large_null.json': '7f68d1e2a915468c4166221594cbbf4cef18ae6f6ae4e31f6abf3398c66682e2', 'comparison_calibration_v1.json': '691a3385c9ddb8065d177d10d6aaddd860457a3c97fde20c1de6844ecd866dc5', 'comparison_calibration_v2.json': 'cfdf1ecb1356cd8ad2fd19ad360e0724cdf9400ba22e38c2671b958ded0371b2', 'method_decisions_v3.json': '4b981c8f0bb6f86f823df471c6ed22778629e29a60401d4b68ae6ce13f9fea02', 'paired_interval_by_sample_size.json': 'dc5bc0dca0f3761ed7fcaf6b432974eb6c3d199f49d2eeff2407769b95a3e637', 'permutation_audit.json': '7a6d7847bb6490b944a85b763850cb9f8b211af498f54bb419ce5792373fc20e', 'permutation_audit_before.json': 'abe9a9020d2dec6fc7c4c84298e98b6200c2738bd7fa9519cd48db55f8a646ab', 'rank_test_alternative.json': 'ac5790939d5f54d9a5f3737dd4a868417dc60ed4939c02425142f6041682c0e2', 'small_sample_interval_study.json': '173dccd442d05c168e0cb9a491c9732a474165c0b3215d084e21131c55a9f309', 'welch_interval_stress.json': '26098745f59c9800a9d37adbf8d303eca5ff848d039eba18f1980648f66ff21e'}


def test_every_earlier_report_is_byte_for_byte_what_it_was():
    for name, digest in HISTORICAL_REPORTS.items():
        assert hashlib.sha256(REPORT_PATH.with_name(name).read_bytes()).hexdigest() == digest, name


def test_the_policy_report_records_its_criteria_and_the_failures_with_them():
    report = json.loads(REPORT_PATH.with_name("inference_policy_v4.json").read_text())
    assert report["report_version"] == 4 and report["base_seed"] == 20_270_201
    assert report["acceptance_limits"] == {"C1": 0.075, "C2": 0.10, "C3": 0.925, "C4": 0.80, "C5": 0.90}
    assert "written before any of this was run" in report["acceptance_policy"]
    two = report["two_groups_policies"]
    assert two["simulations_per_cell"] == 4000 and len(two["cells"]) == 14 * 12
    judged = two["judgement"]
    assert judged["chosen_by_the_declared_rule"] == "v3", "no candidate that tests more samples kept the error criterion"
    fails = {p: judged["by_policy"][p]["C1"]["fail"] for p in judged["candidates"]}
    assert fails["v3"] < fails["diff"] < fails["shape"] < fails["off"] and fails["v3"] == 5
    assert judged["by_policy"]["v3"]["C4"]["fail"] > 0 and judged["by_policy"]["v3"]["C5"]["fail"] > 0, "what v3 costs is on record as failed criteria"
    assert all("outliers" in c for c in judged["by_policy"]["v3"]["C1"]["failed_cells"]), "the population the rule does not protect"
    groups = report["several_groups_policies"]["cells"]
    worst_off = max(c["policies"]["off"]["type_one_error_of_all_samples"]["rate"] for c in groups if c["null_is_true"])
    worst_v3 = max(c["policies"]["v3"]["type_one_error_of_all_samples"]["rate"] for c in groups if c["null_is_true"])
    assert worst_off > 0.20 and worst_v3 < 0.06
    audit = report["audit"]
    assert audit["simulations"] == 20_000
    bm = audit["brunner_munzel_unequal_spread"][0]
    assert 0.07 < bm["brunner_munzel_permutation"]["rate"] < 0.085 < bm["mann_whitney"]["rate"], "liberal there, and said to be"
    engine = report["engine_workflow"]
    assert engine["simulations_per_cell"] == 2000 and set(engine["judgement"]) == {"C1", "C2", "C3", "C4", "D1"}
    assert {c["design"] for c in engine["cells"]} == {"two", "paired", "groups"}
    assert all(c["seed_first"] >= 20_270_201 + 600_000_000 for c in engine["cells"]), "seeds no policy was chosen on"


def test_the_sign_test_is_exact_and_the_different_question_is_never_passed_off_as_the_one_asked():
    from scipy import stats as st

    d = np.array([-2.0, -1.5, -2.5, -1.0, -2.2, -1.8, -2.1, 14.0, 0.0])
    s = methods.sign_test(d, 0.95)
    assert (s["increases"], s["decreases"], s["unchanged"]) == (1, 7, 1) and s["p_value"] == pytest.approx(2 * st.binom.cdf(1, 8, 0.5))
    assert s["estimate"] == np.median(d) and s["interval"]["method"].startswith("order-statistic")
    assert methods.sign_test(np.zeros(4), 0.95)["p_value"] is None, "no unit changed: nothing to test"

    r = selector.paired(np.zeros(9), d, what="sales", unit_name="store")
    other = r["inference_unavailable"]["supported_analysis"]
    assert r["status"] == "unavailable" and r["test"] is None and r["significant"] is None, "the question asked stays unanswered"
    assert other["method"] == "sign_test" and other["p_value"] == pytest.approx(s["p_value"]) and other["exact"] is True
    assert "not a test of the mean" in other["note"] and "median change" in other["estimand"]
    assert any("A different question the data can answer" in line for line in r["interpretation"])

    a, b = np.array([10.0, 11, 12, 9, 10, 11, 12, 10]), np.array([10.5, 11.5, 12.5, 9.5, 10.5, 11.5, 12.5, 95.0])
    r = selector.two_groups(a, b, what="spend", labels=("A", "B"))
    other = r["inference_unavailable"]["supported_analysis"]
    typical = selector.two_groups(a, b, what="spend", labels=("A", "B"), estimand="typical")
    assert other["method"] == "brunner_munzel" and other["p_value"] == typical["test"]["p_value"], "the same test a typical-values question gets"
    assert other["estimate"] == typical["effects"][0]["value"] and other["null"] == typical["test"]["null"]
    assert r["test"] is None and r["significant"] is None and r["interval"] is None
    assert not any(ch.isdigit() for ch in other["estimand"]), "its wording carries no figure of its own"

    three = selector.k_groups({"a": a, "b": a + 1, "c": b}, what="spend", group_name="tier")
    w = three["inference_unavailable"]
    assert three["status"] == "unavailable" and w["code"] == "too_skewed_for_sample_size" and w["supported_analysis"] is None
    assert three["estimate"] is None and three["sample"]["groups"] == {"a": 8, "b": 8, "c": 8} and three["descriptives"]["c"]["mean"]
    assert [c["p_value"] for c in three["comparisons"]] == [None] * 3 and all(c["estimate"] is not None for c in three["comparisons"])
    assert "not a finding of no difference" in " ".join(three["interpretation"])


def test_an_unavailable_interval_is_consistent_from_result_to_dashboard(client, monkeypatch):
    """Schema, step result, narrative, evidence record and dashboard all say the same thing,
    and nowhere does another interval appear in its place."""
    csv = ("store,period,sales\n" + "".join(f"s{i},before,{v}\n" for i, v in enumerate([50, 61, 47, 55, 52]))
           + "".join(f"s{i},after,{v}\n" for i, v in enumerate([58, 66, 55, 60, 61]))).encode()
    ws = workspace(client, csv=csv, filename="five_pairs.csv")
    monkeypatch.setattr(config, "COMPARISON_MODEL_PLANNER", True)
    question = "Did sales differ after compared with before, store by store?"
    assert comparison.parse(question, load_and_profile_csv(io.BytesIO(csv))[1]) is None, "paired designs are the model planner's to propose"
    created = ask(client, ws, question)
    plan = {"outcome": "sales", "explanatory": "period", "levels": ["before", "after"], "design": "paired", "unit": "store"}
    with runtime.provider_override(model_plan(plan)):
        from app import jobs
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    qid = created["question_id"]
    from app.database import SessionLocal
    from app.models import ExecutionLog

    db = SessionLocal()
    try:
        stat = db.query(ExecutionLog).filter_by(question_id=qid).order_by(ExecutionLog.id.desc()).first().result_json["statistic"]
    finally:
        db.close()
    assert stat["status"] == "ok" and stat["interval"] is None and "unbounded" in stat["interval_unavailable"]
    assert all(e.get("interval") is None for e in stat["effects"])
    dash = client.get(f"/api/questions/{qid}/dashboard", headers=ws["h"]).json()
    assert "No interval is reported: the sign-flip test cannot reject any value with this few pairs" in dash["narrative"]
    assert "% interval" not in dash["narrative"]
    records = client.get(f"/api/questions/{qid}/evidence", headers=ws["h"]).json()["records"]
    narrative = next(r for r in records if r["claim_type"] == "narrative")
    assert "interval_not_available" in narrative["limitations"] and "association_not_causal" in narrative["limitations"]
    assert not [c["check"] for r in records for c in r["checks"] if c["outcome"] == "fail"]
    assert any(c["check"] == "narrative_numbers_supported" and c["outcome"] == "pass" for c in narrative["checks"])
    assert not any("interval" in k["label"].lower() for k in dash["kpis"]), "no figure on the dashboard stands in for one"


def _audit(name: str) -> dict:
    return json.loads(REPORT_PATH.with_name(name).read_text())


def _main(design: dict) -> dict:
    return max(design["by_method"].values(), key=lambda v: v["runs"])


def test_the_audit_measured_the_defect_and_the_repair():
    """3,000 seeded null simulations per design, on the engine before and after the correction."""
    before, after = _audit("permutation_audit_before.json")["engine"], _audit("permutation_audit.json")["engine"]
    assert _audit("permutation_audit.json")["base_seed"] == 20_261_201 and after["simulations_per_design"] == 3000
    b = {d["design"]: _main(d) for d in before["two_groups"]}
    a = {d["design"]: _main(d) for d in after["two_groups"]}
    assert len(a) == 13 and all(v["runs"] == 3000 and v["no_interval"] == 0 for v in a.values())
    smaller_more_variable = [k for k in a if k.startswith("normal") and "smaller group has spread 20" in k]
    assert len(smaller_more_variable) == 4
    for k in smaller_more_variable:
        assert b[k]["type_one_error"]["rate"] > 0.10, k               # the defect: up to 22% of true nulls rejected
        assert a[k]["type_one_error"]["rate"] < 0.09 and a[k]["coverage"]["rate"] > 0.91, k
    for k in ("normal 12 + 12, equal spread", "normal 6 + 18, equal spread", "lognormal 12 + 12, same shape"):
        low, high = a[k]["type_one_error"]["interval"]
        assert low <= 0.05 <= high and a[k]["coverage"]["interval"][1] >= 0.95, k       # exact where exchangeable
    k3 = {d["design"]: (_main(d0)["type_one_error"]["rate"], _main(d)["type_one_error"]["rate"])
          for d0, d in zip(before["k_groups"], after["k_groups"])}
    was, now = k3["3 groups 20 + 20 + 6, smallest has spread 20 (others 5)"]
    assert was > 0.30 and now < 0.09


def test_the_audit_records_what_is_still_unreliable():
    """Not fixed, and not hidden: these designs stay outside any reasonable tolerance, and the
    results that can meet them say so (see the limitation tests above)."""
    after = _audit("permutation_audit.json")
    two = {d["design"]: _main(d) for d in after["engine"]["two_groups"]}
    assert two["lognormal 20 + 8, equal means, smaller group more skewed"]["type_one_error"]["rate"] > 0.15
    assert two["exponential 8 vs normal 20, equal means"]["type_one_error"]["rate"] > 0.10
    alternatives = {d["design"]: d["type_one_error"] for d in after["alternatives"]["designs"]}
    worst = alternatives["lognormal 20 + 8, equal means, smaller group more skewed"]
    assert all(v["rate"] > 0.15 for v in worst.values()), "no alternative tested does better there"
    paired = {d["design"]: _main(d) for d in after["engine"]["paired"]}
    assert all(paired[k]["type_one_error"]["rate"] > 0.09 for k in paired if "skewed" in k or "losses" in k)
    assert all(paired[k]["type_one_error"]["interval"][0] <= 0.05 for k in ("8 pairs, normal", "12 pairs, heavy-tailed symmetric"))
    mw = {d["design"]: _main(d) for d in after["engine"]["mann_whitney"]}
    assert mw["normal 24 + 8, same median, smaller group spread 4"]["type_one_error"]["rate"] > 0.12
    assert 0.92 < min(v["coverage"]["rate"] for v in mw.values()) < 0.95, "its bootstrap interval: 93% at worst"
    bm = {r["design"]: r for r in _audit("rank_test_alternative.json")["rows"]}
    assert bm["normal 24 + 8, same median, smaller group spread 4"]["brunner_munzel"]["rate"] < 0.07, "the recommended replacement"


def test_the_wilcoxon_interval_was_short_and_is_now_at_least_its_stated_level():
    before, after = _audit("permutation_audit_before.json")["engine"]["wilcoxon"], _audit("permutation_audit.json")["engine"]["wilcoxon"]
    assert len(after) == 9
    normal = [i for i, d in enumerate(after) if "normal" in d["design"]]
    assert all(_main(before[i])["coverage"]["interval"][1] < 0.95 for i in normal), "the bootstrap fell short at every size"
    assert all(_main(d)["coverage"]["rate"] >= 0.95 and _main(d)["no_interval"] == 0 for d in after)


def test_a_missing_interval_is_said_in_the_published_summary_not_papered_over():
    df = pd.DataFrame({"store": [f"s{i}" for i in range(5)] * 2, "period": ["before"] * 5 + ["after"] * 5,
                       "sales": [50.0, 61, 47, 55, 52, 58, 66, 55, 60, 61]})
    plan = comparison.validate({"outcome": "sales", "explanatory": "period", "levels": ["before", "after"], "design": "paired",
                                "unit": "store"}, profile_dataframe(df))
    report = comparison.run(df, plan)
    assert report["statistic"]["interval"] is None
    text = comparison.narrative(report)
    assert "% interval" not in text, "no interval is quoted"
    assert "No interval is reported: the sign-flip test cannot reject any value with this few pairs" in text


def test_a_large_skewed_sample_no_longer_overflows():
    """The count of arrangements was converted to a float, which fails above about 1,030 observations."""
    rng = np.random.default_rng(12)
    a, b, d = rng.lognormal(0, 2.5, 600), rng.lognormal(0.2, 2.5, 600), rng.lognormal(0, 2.5, 1100) - 5
    r = methods.permutation_means(a, b, 0.95, 1)
    assert r["test"]["p_value"] is not None and r["interval"] is not None
    assert methods._smallest_p(math.comb(4000, 2000)) == 0.0 and methods._smallest_p(2 ** 5) == 0.0625
    assert methods.sign_flip(d, 0.95, 1)["test"]["p_value"] is not None
    # Through the selector these samples are too skewed for their size: measured, and not tested.
    for result in (selector.two_groups(a, b, what="x"), selector.paired(np.zeros(1100), d, what="x", unit_name="u")):
        assert result["inference_unavailable"]["code"] == "too_skewed_for_sample_size" and result["estimate"] is not None
    big_a, big_b = rng.normal(0, 1, 3000), rng.normal(0.1, 1, 3000)
    over = methods.permutation_means(big_a, big_b, 0.95, 1)
    assert over["test"]["p_value"] is not None and over["interval"] is None and "not computed at this size" in over["interval_unavailable"]


def test_spearman_interval_uses_fiellers_variance():
    """Regression: the half-width on the Fisher z scale is z * sqrt(1.06 / (n - 3)), not z * 1.03 / sqrt(n - 3)."""
    case = next(c for c in CASES if c["id"] == "cmp-spearman-extreme-values")
    result = runner.run_case(case)
    checks = {c["check"]: c["ok"] for c in result["detail"]["checks"]}
    assert checks["interval_low"] and checks["interval_high"] and checks["estimate"] and checks["p_value"]
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 40)
    out = methods.spearman(x, 0.5 * x + rng.normal(0, 1, 40), 0.95)
    half = (math.atanh(out["interval"]["high"]) - math.atanh(out["interval"]["low"])) / 2
    assert half == pytest.approx(1.959963984540054 * math.sqrt(1.06 / 37), rel=1e-12)
    assert half != pytest.approx(1.959963984540054 * 1.03 / math.sqrt(37), rel=1e-6), "the rounded constant is gone"
    assert "Fieller" in out["interval"]["method"]


def test_the_methods_version_was_raised_and_stored_results_are_not_rewritten():
    """Results stored under the earlier methods keep their own version label. They are set aside,
    kept on record, only when that investigation is run again (hypothesis.DbStore.begin)."""
    from app import hypothesis_engine as he
    from app.stats import result as stat_result

    assert stat_result.METHODS_VERSION == "2"
    store = he.MemoryStore()
    store.evidence = {"context": "made-under-version-1", "methods_version": "1"}
    assert store.evidence["methods_version"] == "1", "nothing reads an old record and relabels it"


def test_duplicate_rows_are_reported_as_a_limit_on_the_result():
    df = pd.DataFrame({"arm": ["a"] * 20 + ["b"] * 20, "score": [float(i % 7) for i in range(40)]})
    report = comparison.run(df, comparison.validate({"outcome": "score", "explanatory": "arm"}, profile_dataframe(df)))
    assert report["data_checks"]["duplicate_rows"] > 0 and any("exact duplicates" in l for l in report["limitations"])


# ====================================================== association, not cause ==

CAUSAL_QUESTIONS = [
    "Does region affect revenue?", "Does unit price cause lower units?", "Does channel impact revenue?",
    "Is revenue different between regions because of region?", "Does unit price influence units?",
    "Does a higher unit price lead to lower units?", "Do channels improve revenue?", "Is cost related to revenue due to units?",
]


@pytest.mark.parametrize("question", CAUSAL_QUESTIONS)
def test_a_causal_question_gets_the_association_and_what_cause_would_need(sales, question):
    df, profile = sales
    out = comparison.parse(question, profile)
    if out is None:
        pytest.skip("this wording is left to another route")           # declining is also not overclaiming
    plan = out["plan"]
    assert plan["claims_cause"] is True
    report = comparison.run(df, plan)
    causal = report["causal"]
    assert causal == {**causal, "asked": True, "status": "not_established"} and len(causal["evidence_needed"]) == 3
    assert any("randomised experiment" in e for e in causal["evidence_needed"])
    text = comparison.narrative(report)
    assert "does not show that" in text and "A causal answer would need evidence this dataset does not contain." in text
    assert not verification._CAUSAL.search(text), "the summary itself uses no causal wording"


def test_no_published_summary_makes_a_causal_claim_or_states_an_uncomputed_number(sales):
    df, profile = sales
    from app import metric_engine as me

    reports = [comparison.run(df, plan_for(q, profile)) for q in (
        "Compare revenue across channels", "Is average revenue different between regions?", "Is revenue correlated with units?",
        "Is channel related to region?", "Do online orders have higher unit price than retail?")]
    for case in CASES:
        case_profile, path = runner._profile(case["fixture"])
        reports.append(comparison.run(me.load_frame(path), comparison.validate(case["plan"], case_profile)))
    claims = unsupported = 0
    for report in reports:
        text = comparison.narrative(report)
        claims += bool(verification._CAUSAL.search(text))
        pool = verification.extract_numbers([report])
        unsupported += sum(1 for token in verification.narrative_numbers(text) if not verification.number_matches(token, pool)[0])
        assert report["causal"]["status"] == "not_established" and "association" in text
    assert len(reports) == 24 and claims == 0 and unsupported == 0


# ============================================================= end to end ==

@pytest.fixture
def ws(client):
    return workspace(client)


def test_a_comparison_question_is_answered_end_to_end_without_a_model_or_generated_code(client, ws):
    created = ask(client, ws, "Is average revenue different between regions?")
    assert run_job(created["job_id"]) == "succeeded"
    q = question_row(created["question_id"])
    assert q.route == "comparison" and q.status in ("verified", "unverified"), q.error
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=ws["h"]).json()
    assert trace["usage"]["llm_calls"] == 0 and trace["usage"]["sandbox_runs"] == 0
    assert [e["name"] for e in trace["events"] if e["kind"] == "node"] == ["router", "compare"]
    dash = client.get(f"/api/questions/{created['question_id']}/dashboard", headers=ws["h"]).json()
    assert dash["verdict_state"] == "VERIFIED_WITH_CAVEATS", "the figures are verified; the association carries its caveat"
    kpis = {k["label"]: k for k in dash["kpis"]}
    assert set(kpis) == {"Mean revenue: South", "Mean revenue: East", "Number of region groups"}
    assert kpis["Mean revenue: South"]["value"] == "432" and kpis["Mean revenue: East"]["value"] == "344"
    assert all(k["evidence_status"] == "verified" for k in dash["kpis"]), "each figure was recomputed independently"
    assert "Welch's one-way test: p = 0.004" in dash["narrative"] and "does not show that region changes revenue" in dash["narrative"]
    records = client.get(f"/api/questions/{created['question_id']}/evidence", headers=ws["h"]).json()["records"]
    for r in records:
        assert not [(c["check"], c["detail"]) for c in r["checks"] if c["outcome"] == "fail"], r["claim_text"]
        assert r["validity"]["statistical"] == "assessed"
    narrative = next(r for r in records if r["claim_type"] == "narrative")
    assert narrative["validity"]["causal"] == "not_established" and "association_not_causal" in narrative["limitations"]
    assert any(c["check"] == "narrative_numbers_supported" and c["outcome"] == "pass" for c in narrative["checks"])
    recomputed = next(r for r in records if r["claim_text"].startswith("Mean revenue: South"))
    assert any(c["check"] == "independent_recomputation" and c["outcome"] == "pass" for c in recomputed["checks"])


def test_a_correlation_question_end_to_end(client, ws):
    created = ask(client, ws, "Is revenue correlated with units?")
    assert run_job(created["job_id"]) == "succeeded"
    dash = client.get(f"/api/questions/{created['question_id']}/dashboard", headers=ws["h"]).json()
    kpis = {k["label"]: k["value"] for k in dash["kpis"]}
    assert kpis == {"Correlation: units and revenue": "0.65", "Rows used": "720"}
    assert all(k["evidence_status"] == "verified" for k in dash["kpis"])
    assert "95% interval 0.609 to 0.693" in dash["narrative"] and "p < 0.001" in dash["narrative"]


# ===================================================== a model as the planner ==

def model_plan(plan: dict) -> ScriptedProvider:
    return ScriptedProvider({comparison.PLAN_TOOL_NAME: [plan]})


QUESTION_ONLY_A_MODEL_CAN_READ = "Do web orders earn more per sale than store orders?"


def test_a_model_may_propose_the_plan_and_only_a_valid_one_is_used(client, ws, monkeypatch):
    """The slot a fine-tuned model will fill: it proposes, `validate` decides,
    the statistics engine computes. Exercised here with the scripted provider."""
    assert comparison.parse(QUESTION_ONLY_A_MODEL_CAN_READ, load_and_profile_csv(str(FIXTURES / "sales.csv"))[1]) is None
    monkeypatch.setattr(config, "COMPARISON_MODEL_PLANNER", True)
    created = ask(client, ws, QUESTION_ONLY_A_MODEL_CAN_READ)
    provider = model_plan({"outcome": "revenue", "explanatory": "channel", "levels": ["Online", "Retail"], "method": "t_test",
                           "claims_cause": False, "assumptions": ["web orders are the Online channel"]})
    with runtime.provider_override(provider):
        from app import jobs
        assert jobs.run_inline(created["job_id"]) == "succeeded"
    q = question_row(created["question_id"])
    assert q.route == "comparison" and provider.count() == 1, "one model call: the plan. Every number came from the engine."
    trace = client.get(f"/api/questions/{created['question_id']}/trace", headers=ws["h"]).json()
    assert any("a model proposed a comparison plan" in r for r in trace["route_reasons"]) and trace["usage"]["sandbox_runs"] == 0
    dash = client.get(f"/api/questions/{created['question_id']}/dashboard", headers=ws["h"]).json()
    assert {k["label"] for k in dash["kpis"]} == {"Mean revenue: Online", "Mean revenue: Retail", "Difference (Retail − Online)"}


@pytest.mark.parametrize("bad_plan, code", [
    ({"outcome": "profit_margin", "explanatory": "channel"}, "unknown_column"),
    ({"outcome": "revenue", "explanatory": "order_date"}, "incompatible_type"),
    ({"outcome": "revenue", "explanatory": "channel", "method": "neural_test"}, "unsupported_method"),
    ({"outcome": "revenue", "explanatory": "channel", "sql": "DROP TABLE"}, "invalid_plan"),
])
def test_a_models_invalid_plan_is_discarded_and_recorded(sales, monkeypatch, bad_plan, code):
    recorded = []
    monkeypatch.setattr(runtime, "record_event", lambda kind, name="", detail=None, *a, **k: recorded.append((kind, name, detail)))
    with runtime.provider_override(model_plan(bad_plan)):
        assert comparison.propose_with_model(QUESTION_ONLY_A_MODEL_CAN_READ, sales[1]) is None
    assert [(k, n, d["code"]) for k, n, d in recorded if k == "comparison_plan"] == [("comparison_plan", "rejected", code)]


def test_the_model_planner_is_off_by_default_and_never_overrides_the_deterministic_one(client, ws, monkeypatch):
    assert config.COMPARISON_MODEL_PLANNER is False
    created = ask(client, ws, QUESTION_ONLY_A_MODEL_CAN_READ)
    provider = model_plan({"outcome": "revenue", "explanatory": "channel"})
    with runtime.provider_override(provider):
        from app.agents.router import router_node
        out = router_node({"question_id": created["question_id"], "question_text": QUESTION_ONLY_A_MODEL_CAN_READ,
                           "profile": load_and_profile_csv(str(FIXTURES / "sales.csv"))[1]})
    assert out["route"] != "comparison" and provider.count(comparison.PLAN_TOOL_NAME) == 0

    monkeypatch.setattr(config, "COMPARISON_MODEL_PLANNER", True)
    readable = ask(client, ws, "Compare revenue across channels")
    provider = model_plan({"outcome": "cost", "explanatory": "region"})
    with runtime.provider_override(provider):
        from app.agents.router import router_node
        out = router_node({"question_id": readable["question_id"], "question_text": "Compare revenue across channels",
                           "profile": load_and_profile_csv(str(FIXTURES / "sales.csv"))[1]})
    assert out["route"] == "comparison" and out["route_spec"]["source"] == "deterministic" and provider.count() == 0


def test_a_plan_whose_group_does_not_exist_fails_with_the_reason(client, ws, monkeypatch):
    monkeypatch.setattr(config, "COMPARISON_MODEL_PLANNER", True)
    created = ask(client, ws, QUESTION_ONLY_A_MODEL_CAN_READ)
    with runtime.provider_override(model_plan({"outcome": "revenue", "explanatory": "channel", "levels": ["Online", "Wholesale"]})):
        from app import jobs
        assert jobs.run_inline(created["job_id"]) == "failed"
    q = question_row(created["question_id"])
    assert "has no value 'Wholesale'" in q.error and "Online, Retail" in q.error and "internal error" not in q.error.lower()


# ================================================================ calibration ==
# Criteria are method-appropriate (see app/evals/calibration.py). The first run's report is kept
# unchanged, and the original fixed bands are still evaluated and recorded beside the new criteria.

V1_PATH = REPORT_PATH.with_name("comparison_calibration_v1.json")
STUDY_PATH = REPORT_PATH.with_name("small_sample_interval_study.json")
ALL_METHODS = ["brunner_munzel", "chi_square", "fisher_exact", "paired_t", "pearson", "permutation_anova",
               "permutation_means", "sign_flip_permutation", "spearman", "welch_anova", "welch_t"]


@pytest.fixture(scope="module")
def stored_calibration():
    return json.loads(REPORT_PATH.read_text())


def _row(report: dict, method: str) -> dict:
    return next(r for r in report["methods"] if r["method"] == method)


def test_the_first_calibration_report_is_kept_unchanged():
    v1 = json.loads(V1_PATH.read_text())
    got = {r["method"]: (round(r["false_positive_rate"], 4), None if r["coverage"] is None else round(r["coverage"], 4)) for r in v1["methods"]}
    assert got["fisher_exact"] == (0.0087, 0.9622) and got["paired_t"] == (0.0725, 0.9413)
    assert got["permutation_means"] == (0.0425, 0.86) and got["sign_flip_permutation"] == (0.045, 0.8725)
    failed = sorted((r["method"], g) for r in v1["methods"] for g in ("false_positive_gate", "coverage_gate") if r[g] is False)
    assert failed == [("fisher_exact", "false_positive_gate"), ("paired_t", "false_positive_gate"),
                      ("permutation_means", "coverage_gate"), ("sign_flip_permutation", "coverage_gate")]
    assert hashlib.sha256(V1_PATH.read_bytes()).hexdigest() == "691a3385c9ddb8065d177d10d6aaddd860457a3c97fde20c1de6844ecd866dc5"
    # the report as it stood before Brunner-Munzel replaced Mann-Whitney and skewed small samples stopped being tested
    v2 = REPORT_PATH.with_name("comparison_calibration_v2.json")
    assert hashlib.sha256(v2.read_bytes()).hexdigest() == "cfdf1ecb1356cd8ad2fd19ad360e0724cdf9400ba22e38c2671b958ded0371b2"
    assert "mann_whitney" in [r["method"] for r in json.loads(v2.read_text())["methods"]]
    # and the one from before several-group comparisons got the same rule (report 3)
    v3 = REPORT_PATH.with_name("comparison_calibration.json")
    assert hashlib.sha256(v3.read_bytes()).hexdigest() == "9d56f72ed6676e687b9eed2d66b6612bd4a54171e234cc187ccffcf30e214a4f"


def test_the_stored_primary_run_is_what_the_code_produces(stored_calibration):
    """The 400-simulation run is reproduced here in full from its seeds."""
    fresh = json.loads(json.dumps(calibration.run(calibration.SIMS, confirmation_sims=0)))
    keys = ("false_positive_rate", "coverage", "power", "intervals_not_reported", "original_gates", "designs", "unavailable")
    assert [{k: r[k] for k in keys} for r in fresh["methods"]] == [{k: r[k] for k in keys} for r in stored_calibration["methods"]]
    assert stored_calibration["base_seed"] == 20_261_013 and stored_calibration["simulations_per_design_and_condition"] == 400
    assert stored_calibration["confirmation_null_simulations_per_design"] == 4000


def test_part_of_the_stored_confirmation_run_is_reproduced(stored_calibration):
    """Two of the twelve designs, 4,000 simulations each, recomputed from their seeds. The full
    confirmation run takes several minutes and is regenerated by `python -m app.evals.calibration`."""
    chosen = [(d, calibration.DESIGNS[d]) for d in (0, 3)]
    fresh = calibration.simulate(4000, chosen, conditions=("null",), offset=calibration.CONFIRMATION_OFFSET)["__by_design__"]
    for name in ("two groups, normal", "paired, normal differences"):
        assert fresh[name] == stored_calibration["confirmation_by_design"][name], name


def test_every_method_was_simulated_often_enough_to_be_judged(stored_calibration):
    assert sorted(r["method"] for r in stored_calibration["methods"]) == sorted(ALL_METHODS + ["none"])
    # "none" is not a method: it counts the samples that were measured and reported without a test.
    not_tested = _row(stored_calibration, "none")
    assert not_tested["false_positive_rate"] is None and not_tested["coverage"] is None and not_tested["unavailable"] > 0
    assert set(not_tested["designs"]) == {"two groups, skewed, small", "two groups, small", "paired, few pairs",
                                          "three groups, small", "three groups, moderate"}
    assert stored_calibration["designs_simulated_more_than_once"] == {"paired, few pairs": 2, "two groups, small": 2,
                                                                      "three groups, moderate": 2}
    for r in stored_calibration["methods"]:
        if r["method"] == "none":
            continue
        primary, confirmed = r["false_positive_rate"], r["false_positive_rate_confirmation"]
        assert primary["simulations"] >= 340 and confirmed["simulations"] >= 3400 and r["unavailable"] == 0, r["method"]
        low, high = confirmed["interval"]
        assert high - low < 0.02, "4,000 simulations measure a rate to within a point"
        assert r["claim"] == ("exact: at most alpha" if r["method"] in calibration.EXACT else "approximate: alpha")


@pytest.mark.parametrize("method", sorted(calibration.APPROXIMATE))
def test_an_approximate_test_rejects_a_true_null_at_its_stated_rate(stored_calibration, method):
    """Criterion: alpha lies inside the 95% interval of the rate observed in 4,000 or more null datasets."""
    r = _row(stored_calibration, method)
    seen = r["false_positive_rate_confirmation"]
    assert r["criteria"]["consistent_with_alpha"], (
        f"{method}: {seen['rate']:.2%} of {seen['simulations']} (95% interval {seen['interval'][0]:.2%} to {seen['interval'][1]:.2%})")


@pytest.mark.parametrize("method", sorted(calibration.EXACT))
def test_an_exact_test_does_not_reject_a_true_null_more_often_than_alpha(stored_calibration, method):
    """Criterion: no evidence of a rate above alpha. Below alpha is what an exact test on discrete data does."""
    r = _row(stored_calibration, method)
    seen = r["false_positive_rate_confirmation"]
    assert r["criteria"]["no_excess_false_positives"], (
        f"{method}: {seen['rate']:.2%} of {seen['simulations']} (95% interval {seen['interval'][0]:.2%} to {seen['interval'][1]:.2%})")


@pytest.mark.parametrize("method", ["fisher_exact", "sign_flip_permutation"])
def test_a_discrete_exact_test_rejects_as_often_as_its_exact_size_says(stored_calibration, method):
    """The conservatism is the test's own: the observed rate agrees with the size worked out
    from the design without simulating."""
    r = _row(stored_calibration, method)
    size, seen = r["exact_size"], r["false_positive_rate_confirmation"]
    assert size < 0.05 and r["criteria"]["agrees_with_exact_size"], (size, seen)
    assert calibration.exact_sizes()[method]["size"] == pytest.approx(size)


def test_the_exact_sizes_are_what_the_arithmetic_gives():
    sizes = calibration.exact_sizes()
    assert sizes["sign_flip_permutation"]["size"] == 12 / 256, "p = 2k/256 is below 0.05 for k <= 6"
    assert 0.010 < sizes["fisher_exact"]["size"] < 0.020, "a 12 + 12 table cannot reject 5% of the time"
    assert 0.80 < sizes["fisher_exact"]["share_of_datasets_tested_this_way"] < 0.95


@pytest.mark.parametrize("method", ["chi_square", "fisher_exact", "brunner_munzel", "paired_t", "pearson", "permutation_means",
                                    "sign_flip_permutation", "spearman", "welch_t"])
def test_an_interval_holds_the_true_value_as_often_as_it_says(stored_calibration, method):
    """Criterion: no evidence of covering less often than the stated 95%."""
    r = _row(stored_calibration, method)
    seen = r["coverage"]
    assert r["criteria"]["no_undercoverage"], (
        f"{method}: {seen['rate']:.1%} of {seen['simulations']} (95% interval {seen['interval'][0]:.1%} to {seen['interval'][1]:.1%})")
    assert r["intervals_not_reported"]["count"] == 0, "in these designs an interval exists and was reported every time"


def test_the_original_bands_are_still_evaluated_and_recorded(stored_calibration):
    """Not asserted to pass: an exact test on a 12 + 12 table cannot land between 3% and 7%. Recorded, so
    that the change of criterion is visible next to what the old one would have said."""
    assert stored_calibration["original_gates"] == {"false_positive_rate": [0.03, 0.07], "coverage": [0.93, 0.97]}
    fpr = {r["method"]: r["original_gates"]["false_positive_rate_3_to_7_percent"] for r in stored_calibration["methods"]}
    coverage = {r["method"]: r["original_gates"]["coverage_93_to_97_percent"] for r in stored_calibration["methods"]}
    assert set(fpr) == set(ALL_METHODS) | {"none"} and all(v in (True, False) for m_, v in fpr.items() if m_ != "none")
    assert fpr["fisher_exact"] is False, "still outside the old band, as it must be"
    assert coverage["permutation_means"] is True and coverage["sign_flip_permutation"] is True, "the two that failed on coverage"


def test_planted_effects_are_detected_more_often_than_chance(stored_calibration):
    for r in stored_calibration["methods"]:
        if r["power"] and r["power"]["simulations"] >= 100:
            assert r["power"]["rate"] > 0.25 and r["power"]["interval"][0] > r["false_positive_rate"]["rate"], r["method"]


def test_the_paired_t_result_was_sampling_variation_not_a_defect():
    """The first run saw 29 rejections in 400 (7.2%). 40,000 null datasets per design settle it."""
    large = json.loads(REPORT_PATH.with_name("comparison_calibration_large_null.json").read_text())
    assert large["simulations_per_design"] == 40_000 and large["seed_offset"] == 200_000_000
    rates = {m: v for d in large["designs"].values() for m, v in d.items() if v["null_n"] > 30_000}
    assert set(rates) == {"welch_t", "paired_t", "welch_anova", "pearson"}
    for method in ("paired_t", "welch_t", "pearson"):
        low, high = rates[method]["interval"]
        assert low <= 0.05 <= high and high - low < 0.005, method
    anova = rates["welch_anova"]
    assert 0.045 < anova["rate"] < 0.05 and anova["interval"][1] < 0.0505, \
        "measured slightly conservative (about 4.8%); recorded as found, on the safe side of alpha"


def test_the_interval_study_supports_the_choice_and_shows_its_limits():
    """The independent study the interval choice rests on (backend/evals/study_small_sample_intervals.py)."""
    study = json.loads(STUDY_PATH.read_text())
    assert study["simulations_per_design"] == 2000 and study["base_seed"] == 20_261_101
    by = {d["design"]: d for d in study["designs"]}
    holds = [d for d in study["designs"] if "holds" in d["assumption"]]
    violated = [d for d in study["designs"] if "VIOLATED" in d["assumption"]]
    assert len(holds) == 8 and len(violated) == 5
    for d in holds:                       # exact under its assumption: never short of 93%, in every such design
        assert d["methods"]["inversion"]["coverage"] >= 0.93, d["design"]
    assert all(d["methods"]["bca"]["coverage"] < 0.90 for d in study["designs"]), "the replaced method fell short everywhere"
    assert min(d["methods"]["inversion"]["coverage"] for d in violated) < 0.80, "and the new one is not safe when its assumption fails"
    guarded = by["10 pairs, skewed differences"]["methods"]["inversion_guarded"]
    assert guarded["coverage"] < by["10 pairs, skewed differences"]["methods"]["inversion"]["coverage"], \
        "declining when the sample looks skewed made things worse, so it is not done"
