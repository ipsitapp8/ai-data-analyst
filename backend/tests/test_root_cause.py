"""Root-cause evidence layer: every scenario has a planted, known structure (rc_fixtures.py) and the engine's
numbers are checked against plain-pandas ground truth computed here, independently of the engine."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import rc_fixtures as F  # noqa: E402

from app import investigator as inv  # noqa: E402
from app import root_cause as rc  # noqa: E402
from app.profiling import profile_dataframe  # noqa: E402


def run(df, formula, question="Why did it change?", label="Outcome", **extra):
    spec = {"formula": formula, "label": label, "time_column": "order_date", **F.SPEC_PERIODS, "question": question, **extra}
    return inv.investigate(df, profile_dataframe(df), spec)


def tiers(r):
    return {f["factor"]: f["tier"] for f in r["root_cause"]["factors"]}


def plausible(r):
    return {k for k, t in tiers(r).items() if t == "plausible_contributor"}


def pct(df, col, mask_fn, agg="sum"):
    a, b = df[~F.in_b(df)], df[F.in_b(df)]
    va, vb = getattr(a[mask_fn(a)][col], agg)(), getattr(b[mask_fn(b)][col], agg)()
    return vb / va - 1


# ----------------------------------------------------------------- known drivers ----

def test_sales_decline_names_the_collapsed_region_with_correct_evidence():
    df = F.sales_decline_segment()
    r = run(df, "sum(revenue)", "Why did revenue fall?", "Revenue")
    assert plausible(r) == {"region = North"}
    f = next(x for x in r["root_cause"]["factors"] if x["factor"] == "region = North")
    assert math.isclose(f["evidence"]["change_pct"], pct(df, "revenue", lambda d: d.region == "North"), abs_tol=1e-6)
    assert math.isclose(f["evidence"]["rest_change_pct"], pct(df, "revenue", lambda d: d.region != "North"), abs_tol=1e-6)
    assert f["evidence"]["change_pct"] < -0.3 and f["evidence"]["rest_change_pct"] > -0.1
    assert r["root_cause"]["diagnosis"]["code"] == "driver_located"
    assert r["root_cause"]["problem"]["direction"] == "decrease"


def test_cost_increase_finds_both_independent_drivers():
    df = F.cost_competing_drivers()
    r = run(df, "sum(cost)", "Why did cost rise?", "Cost")
    assert {"category = Freight", "vendor = Acme"} <= plausible(r)
    assert r["root_cause"]["problem"]["direction"] == "increase"
    assert "more than one source" in r["root_cause"]["diagnosis"]["headline"]


def test_productivity_decline_finds_night_shift_and_reports_overtime_as_association_only():
    df = F.productivity_decline()
    r = run(df, "sum(output)", "Why did output fall?", "Output")
    assert "shift = Night" in plausible(r)
    nums = {n["column"]: n for n in r["root_cause"]["numeric_factors"]}
    assert "overtime_hours" in nums
    # no numeric band is promoted to a plausible driver without a row-level relationship in both periods
    assert not any("overtime" in k for k in plausible(r))
    assert r["root_cause"]["causal_status"]["level"] == "observational"


def test_conversion_decline_uses_a_ratio_and_finds_mobile():
    df = F.conversion_decline()
    r = run(df, "avg(converted)", "Why did conversion fall?", "Conversion")
    assert "device = Mobile" in plausible(r)
    f = next(x for x in r["root_cause"]["factors"] if x["factor"] == "device = Mobile")
    assert math.isclose(f["evidence"]["change_pct"], pct(df, "converted", lambda d: d.device == "Mobile", "mean"), abs_tol=1e-6)
    # page load time also rose, but only as a flagged association
    assert not any("page_load" in k for k in plausible(r))


# ------------------------------------------------------- misleading simple evidence ----

def test_confounded_region_is_not_credited_when_channel_explains_it():
    df = F.confounded_proxy()
    r = run(df, "sum(revenue)", "Why did revenue fall?", "Revenue")
    assert plausible(r) == {"channel = Paid"}
    assert tiers(r).get("region = West") != "plausible_contributor"
    test = next(t for t in r["root_cause"]["adjustment_tests"] if {t["a"], t["b"]} == {"channel = Paid", "region = West"})
    paid_first = test["a_given_b"] if test["a"] == "channel = Paid" else test["b_given_a"]
    west_first = test["b_given_a"] if test["a"] == "channel = Paid" else test["a_given_b"]
    assert paid_first["result"] == "holds" and west_first["result"] == "vanishes"
    # independent check: within each channel the West gap is ~0
    for ch in ("Paid", "Organic"):
        d = df[df.channel == ch]
        w, o = pct(d, "revenue", lambda x: x.region == "West"), pct(d, "revenue", lambda x: x.region != "West")
        assert abs(w - o) < 0.12


def test_drifting_unrelated_numeric_is_not_a_driver():
    df = F.wrong_factor()
    r = run(df, "sum(revenue)", "Why did revenue fall?", "Revenue")
    assert plausible(r) == {"region = North"}
    assert not any(k.startswith("discount") for k in plausible(r))
    assert any(n["column"] == "discount" for n in r["root_cause"]["numeric_factors"])   # reported, not promoted


# ----------------------------------------------------------------- honest refusals ----

def test_diffuse_decline_names_no_single_factor():
    r = run(F.diffuse_decline(), "sum(revenue)", "Why did revenue fall?", "Revenue")
    assert plausible(r) == set()
    assert r["root_cause"]["diagnosis"]["code"] != "driver_located"
    assert "spread across" in r["root_cause"]["diagnosis"]["headline"]


def test_small_noisy_data_is_inconclusive_and_names_nothing():
    r = run(F.noisy_small(), "sum(revenue)", "Why did revenue change?", "Revenue")
    assert r["status"] == "inconclusive"
    assert plausible(r) == set()
    assert "noise" in r["root_cause"]["diagnosis"]["headline"] or "ordinary variation" in r["root_cause"]["diagnosis"]["headline"]


def test_missing_inputs_are_flagged_before_any_behavioural_cause():
    df = F.missing_inputs()
    r = run(df, "sum(revenue)", "Why did revenue fall?", "Revenue")
    head = r["root_cause"]["diagnosis"]["headline"]
    assert "missing" in head and "recorded" in head
    assert plausible(r) == set() or r["root_cause"]["diagnosis"]["code"] != "driver_located"
    a, b = df[~F.in_b(df)], df[F.in_b(df)]
    assert a.revenue.isna().mean() == 0 and 0.15 < b.revenue.isna().mean() < 0.2    # ground truth for the message


def test_outlier_artefact_blocks_a_driver_headline():
    r = run(F.outlier_artefact(), "sum(revenue)", "Why did revenue fall?", "Revenue")
    assert plausible(r) == set()
    assert r["root_cause"]["outlier_check"] is not None
    assert "extreme" in r["root_cause"]["diagnosis"]["headline"]


# ----------------------------------------------------------------- output contract ----

def test_no_confidence_percentages_and_causal_language_is_bounded():
    r = run(F.sales_decline_segment(), "sum(revenue)", "Why did revenue fall?", "Revenue")
    blob = repr(r["root_cause"]).lower()
    assert "confidence_pct" not in blob and "probability" not in blob
    assert "caused by" not in r["root_cause"]["diagnosis"]["headline"].lower()
    assert "not an experiment" in r["root_cause"]["causal_status"]["text"]
    assert r["root_cause"]["next_steps"]


def test_charts_are_real_plotly_figures_with_data():
    r = run(F.sales_decline_segment(), "sum(revenue)", "Why did revenue fall?", "Revenue")
    charts = r["root_cause"]["charts"]
    assert charts
    for c in charts:
        fig = c["plotly_json"]
        assert fig["data"] and all(len(t.get("x", t.get("y", []))) > 0 for t in fig["data"])


def test_scope_in_the_question_is_isolated_and_not_propagated_as_a_driver():
    df = F.sales_decline_segment()
    profile = profile_dataframe(df)
    spec = inv.infer_spec(profile, "Why did revenue fall in the North?")
    assert spec is not None
    assert {"column": "region", "op": "==", "value": "North"}.items() <= next(
        (f for f in spec["filters"] if f["column"] == "region"), {}).items()
    r = inv.investigate(df, profile, {**spec, **F.SPEC_PERIODS})
    assert r["root_cause"]["problem"]["scope"] != "all rows"
    assert not any(f["dimension"] == "region" for f in r["root_cause"]["factors"])   # the filtered column is never a "cause"
    west = df[df.region == "North"]
    assert r["root_cause"]["problem"]["rows_baseline"] == int((~F.in_b(west)).sum())


def test_deterministic():
    df = F.cost_competing_drivers()
    a, b = run(df, "sum(cost)"), run(df, "sum(cost)")
    assert a["root_cause"]["diagnosis"] == b["root_cause"]["diagnosis"]
    assert [f["factor"] for f in a["root_cause"]["factors"]] == [f["factor"] for f in b["root_cause"]["factors"]]


def test_evidence_layer_failure_does_not_break_the_decomposition(monkeypatch):
    def boom(*a, **k):
        raise ValueError("synthetic failure")
    monkeypatch.setattr(rc, "diagnose", boom)
    r = run(F.sales_decline_segment(), "sum(revenue)", "Why did revenue fall?", "Revenue")
    assert r["status"] == "complete" and "synthetic failure" in r["root_cause"]["error"]
    assert r["ranked_contributors"]


def test_regression_unstable_strata_never_report_an_absurd_adjusted_gap():
    """Holding a shifting numeric band fixed left near-empty baseline cells, which produced 'gap shrinks to +463 points'."""
    r = run(F.productivity_decline(), "sum(output)", "Why did output fall?", "Output")
    for t in r["root_cause"]["adjustment_tests"]:
        for side in ("a_given_b", "b_given_a"):
            s = t[side]
            if s["adjusted_gap"] is not None and s["marginal_gap"] and s["result"] != "no_gap":   # no_gap: nothing is claimed
                assert s["adjusted_gap"] * s["marginal_gap"] >= 0 or s["result"] == "unclear"
                assert abs(s["adjusted_gap"]) <= max(1.0, 3 * abs(s["marginal_gap"])) or s["result"] == "unclear"
    night = next(f for f in r["root_cause"]["factors"] if f["factor"] == "shift = Night")
    assert not any("463" in x for x in night["limitations"])


def test_regression_segment_that_exists_in_one_period_only_is_found():
    """A new vendor has no baseline, so it has no % change; it used to be dropped and the result read 'diffuse'."""
    df = F.new_vendor_slows_resolution()
    r = run(df, "sum(hours)", "Why did hours increase?", "Hours")
    assert plausible(r) == {"vendor = Brightline"}
    f = next(x for x in r["root_cause"]["factors"] if x["factor"] == "vendor = Brightline")
    b = df[F.in_b(df)]
    assert f["presence"] == "new" and f["evidence"]["rows_a"] == 0
    assert f["evidence"]["rows_b"] == int((b.vendor == "Brightline").sum())
    assert math.isclose(f["evidence"]["value_b"], b.loc[b.vendor == "Brightline", "hours"].sum(), rel_tol=1e-9)
    text = r["root_cause"]["diagnosis"]["headline"]
    assert r["root_cause"]["diagnosis"]["code"] == "driver_located" and "n/a" not in text and "no rows in the baseline" in text
