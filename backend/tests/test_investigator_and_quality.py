"""Root-cause investigator (decomposition, bounds, honesty about causation) and
data observability (snapshots, drift metrics, incidents, configuration)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from platform_helpers import DRIFTED_CSV, FIXTURES, add_member, ask_and_run, question_row, run_job, workspace

from app import data_quality as dq
from app import investigator as inv
from app import metric_engine as me
from app.database import SessionLocal
from app.models import DQIncident, DQSnapshot
from app.profiling import load_and_profile_csv

SPLIT = "2025-03-01"
PERIODS = {"period_a": ["2025-01-01", SPLIT], "period_b": [SPLIT, "2025-04-29"]}


@pytest.fixture(scope="module")
def sales():
    df, profile = load_and_profile_csv(str(FIXTURES / "sales.csv"))
    return df, profile


def spec(formula="sum(revenue)", **kw):
    return {"formula": formula, "label": "Metric", "time_column": "order_date", **PERIODS, **kw}


# ========================================================= investigator ====

def test_additive_decomposition_reconciles_and_finds_the_planted_change(sales):
    df, profile = sales
    r = inv.investigate(df, profile, spec())
    assert r["status"] == "complete" and r["decomposition_method"] == "additive"
    change = r["totals"]["change"]
    assert math.isclose(change, r["totals"]["b"] - r["totals"]["a"])
    for dim in r["dimensions"]:
        assert dim["reconciles"], dim["dimension"]
        assert math.isclose(sum(s["contribution"] for s in dim["segments"]), change, rel_tol=1e-9)
    top = r["ranked_contributors"][0]
    assert (top["dimension"], top["segment"]) == ("region", "West")
    assert top["share_of_change"] > 1.0, "West fell by more than the total (others rose)"
    assert top["evidence_quality"] == "high" and top["consistent_across_halves"] is True
    assert "West" in r["summary"] and "117%" in r["summary"]


def test_ratio_metric_uses_rate_and_mix_and_still_reconciles(sales):
    df, profile = sales
    for formula in ("sum(revenue) / sum(units)", "avg(unit_price)"):
        r = inv.investigate(df, profile, spec(formula))
        assert r["decomposition_method"] == "rate_and_mix"
        change = r["totals"]["change"]
        for dim in r["dimensions"]:
            assert dim["reconciles"], (formula, dim["dimension"])
            assert math.isclose(dim["rate_effect_total"] + dim["mix_effect_total"], change, rel_tol=1e-9, abs_tol=1e-12)
            for s in dim["segments"]:
                assert math.isclose(s["rate_effect"] + s["mix_effect"], s["contribution"], rel_tol=1e-9, abs_tol=1e-12)


def test_price_volume_split_sums_exactly_to_the_change(sales):
    df, profile = sales
    pvm = inv.investigate(df, profile, spec())["price_volume"]
    assert pvm["reconciles"] and pvm["amount_column"] == "revenue" and pvm["quantity_column"] == "units"
    assert "confirmed on a sample" in pvm["basis"]
    assert pvm["price_effect"] > 0, "Gadget's price rose 10%"
    assert pvm["volume_effect"] < 0, "West's units fell"
    assert "not necessarily a list-price change" in pvm["note"]
    assert inv.investigate(df, profile, spec("sum(units)"))["price_volume"] is None, "not offered when it does not apply"


def test_entry_and_exit_of_segments_is_accounted_for():
    rows = [{"d": "2025-01-10", "seg": "old", "amt": 100.0}, {"d": "2025-01-20", "seg": "stay", "amt": 50.0},
            {"d": "2025-02-10", "seg": "stay", "amt": 80.0}, {"d": "2025-02-20", "seg": "new", "amt": 40.0}]
    df = pd.DataFrame(rows)
    df["d"] = pd.to_datetime(df["d"])
    profile = {"row_count": 4, "columns": [{"name": "d", "kind": "datetime"}, {"name": "seg", "kind": "categorical", "unique_count": 3},
                                           {"name": "amt", "kind": "numeric"}]}
    r = inv.investigate(df, profile, {"formula": "sum(amt)", "time_column": "d", "dimensions": ["seg"],
                                      "period_a": ["2025-01-01", "2025-02-01"], "period_b": ["2025-02-01", "2025-02-28"]})
    seg = {s["segment"]: s for s in r["dimensions"][0]["segments"]}
    assert seg["old"]["presence"] == "lost" and seg["new"]["presence"] == "new" and seg["stay"]["presence"] == "both"
    assert r["dimensions"][0]["reconciles"] and r["totals"]["change"] == -30.0
    assert all(c["evidence_quality"] == "low" for c in r["ranked_contributors"]), "two rows per period is thin evidence"
    assert r["status"] == "inconclusive"
    assert any("few rows" in u for u in r["unresolved"])


def test_inconclusive_when_nothing_changed_or_no_time_range(sales):
    df, profile = sales
    flat = inv.investigate(df, profile, spec("count_distinct(product)"))
    assert flat["status"] == "inconclusive" and "no material change" in flat["summary"]
    assert flat["ranked_contributors"] == [] and flat["hypotheses"] == []
    one_day = df[df["order_date"] == df["order_date"].min()]
    auto = inv.investigate(one_day, profile, {"formula": "sum(revenue)", "time_column": "order_date"})
    assert auto["status"] == "inconclusive" and "fewer than two distinct moments" in auto["summary"]


def test_report_never_claims_causation_and_lists_what_it_could_not_test(sales):
    df, profile = sales
    r = inv.investigate(df, profile, spec())
    assert any("not what caused it" in c for c in r["caveats"])
    assert any("data-quality" in a for a in r["alternative_explanations"])
    text = (r["summary"] + " ".join(h["hypothesis"] for h in r["hypotheses"])).lower()
    for word in ("because", "caused by", "due to", "led to"):
        assert word not in text
    assert all(h["status"] in ("supported", "weak", "not_supported") and h["test"] and h["result"] for h in r["hypotheses"])
    assert r["hypotheses"][0]["status"] == "supported"
    assert r["drilldowns"] and r["drilldowns"][0]["within"] == {"dimension": "region", "segment": "West"}
    assert r["drilldowns"][0]["reconciles"]


def test_investigation_is_bounded_and_says_when_it_stopped(sales):
    df, profile = sales
    tight = inv.Budget(max_engine_calls=5, max_dimensions=6)
    r = inv.investigate(df, profile, spec(), tight)
    assert r["budget"]["engine_calls"] <= 5 and r["budget"]["exhausted"] == "engine_calls"
    assert any("limit" in u for u in r["unresolved"]) and any("Not examined" in u for u in r["unresolved"])
    one_dim = inv.investigate(df, profile, spec(), inv.Budget(max_dimensions=1))
    assert len(one_dim["dimensions"]) == 1 and any("Not examined" in u for u in one_dim["unresolved"])
    timed = inv.Budget(max_seconds=0.0)
    assert inv.investigate(df, profile, spec(), timed)["budget"]["exhausted"] == "time"
    full = inv.investigate(df, profile, spec())
    assert full["budget"]["exhausted"] == "" and full["budget"]["engine_calls"] <= full["budget"]["limits"]["engine_calls"]


def test_invalid_specs_are_rejected(sales):
    df, profile = sales
    for bad in (spec("sum(nope)"), spec("os.system('x')"), {**spec(), "time_column": "nope"},
                {**spec(), "period_a": ["not a date", SPLIT]}, {**spec(), "filters": [{"column": "nope", "op": "==", "value": 1}]}):
        with pytest.raises(me.FormulaError):
            inv.investigate(df, profile, bad)


def test_question_routed_to_the_investigator_publishes_a_report_and_dashboard(client):
    ws = workspace(client)
    out = ask_and_run(client, ws, "Why did revenue change?")
    assert out["state"] == "succeeded"
    q = question_row(out["question_id"])
    assert q.route == "root_cause" and q.status == "verified"
    dash = client.get(f"/api/questions/{out['question_id']}/dashboard", headers=ws["h"]).json()
    assert dash["verdict_state"] == "VERIFIED_WITH_CAVEATS", "arithmetic verified; the explanation is descriptive"
    assert [k["evidence_status"] for k in dash["kpis"]] == ["verified"] * 3 and len(dash["charts"]) == 2
    assert "does not establish" in dash["narrative"]
    ev = client.get(f"/api/questions/{out['question_id']}/evidence", headers=ws["h"]).json()["records"]
    narrative = next(r for r in ev if r["claim_type"] == "narrative")
    assert narrative["validity"]["causal"] == "not_established" and "descriptive_not_causal" in narrative["limitations"]
    listing = client.get("/api/investigations", headers=ws["h"]).json()
    assert listing["total"] == 1 and listing["investigations"][0]["question_id"] == out["question_id"]
    report = client.get(f"/api/investigations/{listing['investigations'][0]['id']}", headers=ws["h"]).json()["report"]
    assert report["ranked_contributors"][0]["segment"] == "West" and report["provenance"]["dataset_fingerprint"]
    trace = client.get(f"/api/questions/{out['question_id']}/trace", headers=ws["h"]).json()
    assert trace["usage"]["llm_calls"] == 0 and [e["name"] for e in trace["events"] if e["kind"] == "node"] == ["router", "investigate"]


# ---- asking in words: periods read from the question, statistics, evidence table

def _frame(days: int, start: str = "2025-01-01", value=lambda i, seg: 100.0) -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    return pd.DataFrame([{"day": d, "region": seg, "revenue": float(value(i, seg))}
                         for i, d in enumerate(dates) for seg in ("North", "South")])


def _profile(df: pd.DataFrame) -> dict:
    return {"row_count": len(df), "columns": [
        {"name": "day", "kind": "datetime"}, {"name": "region", "kind": "categorical", "unique_count": df["region"].nunique()},
        {"name": "revenue", "kind": "numeric"}]}


@pytest.mark.parametrize("question, a, b", [
    ("Why did revenue decrease last month?", ("2025-02-01", "2025-03-01"), ("2025-03-01", "2025-03-31")),
    ("why did revenue drop in the last 14 days", ("2025-04-02", "2025-04-16"), ("2025-04-16", "2025-04-29")),
    ("What explains revenue in the past 2 weeks?", ("2025-04-02", "2025-04-16"), ("2025-04-16", "2025-04-29")),
    ("why did revenue fall in February?", ("2025-01-01", "2025-02-01"), ("2025-02-01", "2025-02-28")),
    ("why is revenue down this month", ("2025-03-01", "2025-03-30"), ("2025-04-01", "2025-04-29")),
    ("why did revenue fall last quarter", None, None),
])
def test_periods_are_read_from_the_question_against_the_datasets_own_calendar(sales, question, a, b):
    df, _ = sales                                   # covers 2025-01-01 .. 2025-04-29
    read = inv.question_periods(question, df, "order_date")
    if a is None:
        assert "no earlier period" in read["error"], "Q1 is the first quarter in the data"
        return
    assert (read["period_a"][0][:10], read["period_a"][1][:10]) == a
    assert (read["period_b"][0][:10], read["period_b"][1][:10]) == b
    assert "read from the question" in read["source"] and "2025-04-29" in read["source"]


def test_a_month_the_data_ends_in_counts_as_last_month_only_when_complete():
    full = _frame(59)                               # Jan 1 .. Feb 28: February is complete
    assert inv.question_periods("why last month", full, "day")["period_b"][0][:10] == "2025-02-01"
    partial = _frame(90)                            # .. Mar 31 is day 90; stop a day short
    partial = partial[partial["day"] < "2025-03-31"]
    assert inv.question_periods("why last month", partial, "day")["period_b"][0][:10] == "2025-02-01", "March is incomplete"
    assert inv.question_periods("why did it change?", full, "day") is None, "no period named: the caller falls back"
    assert "only covers" in inv.question_periods("why in 2019 last year", _frame(40), "day")["error"]


def test_a_question_sets_the_periods_of_an_investigation_and_an_unanswerable_one_says_why(sales):
    df, profile = sales
    spec = inv.infer_spec(profile, "Why did revenue decrease last month?")
    r = inv.investigate(df, profile, spec)
    assert r["periods"]["a"][0][:10] == "2025-02-01" and r["periods"]["b"][0][:10] == "2025-03-01"
    assert "last month" in r["periods"]["source"]
    assert r["ranked_contributors"][0]["segment"] == "West"
    explicit = inv.investigate(df, profile, {**spec, **PERIODS})
    assert explicit["periods"]["source"] == "specified", "periods given outright win over the question"
    r = inv.investigate(df, profile, inv.infer_spec(profile, "Why did revenue fall last year?"))
    assert r["status"] == "inconclusive" and "only covers" in r["summary"] and r["evidence_table"] == []


def test_statistical_evidence_is_per_bucket_adjusted_and_separates_measured_from_hypothesis(sales):
    df, profile = sales
    r = inv.investigate(df, profile, spec())
    overall = r["change_evidence"]["overall"]
    assert overall["status"] == "ok" and overall["significant"] and overall["bucket"] == "day"
    assert overall["low"] < overall["estimate"] < overall["high"] < 0
    assert "interval" in r["summary"] and "p < 0.001" in r["summary"]
    tested = [c for c in r["ranked_contributors"] if c["statistic"]["status"] == "ok"]
    assert 1 <= len(tested) <= inv.STAT_CONTRIBUTORS and r["change_evidence"]["family"]["comparisons"] == len(tested)
    assert all(c["statistic"]["p_adjusted"] >= c["statistic"]["p_value"] for c in tested), "Holm never lowers a p-value"
    table = r["evidence_table"]
    west = next(row for row in table if row["candidate"] == "region = West")
    assert west["kind"] == "measured" and west["support"] == "strong" and west["limitations"] == []
    assert west["value_a"] > west["value_b"] and west["contribution"] == west["value_b"] - west["value_a"]
    hypotheses = [row for row in table if row["kind"] == "hypothesis"]
    assert len(hypotheses) == 2 and all(h["claim"] == "not tested" and "statistic" not in h for h in hypotheses)
    assert any("not testable" in h["support"] for h in hypotheses)
    assert all("caused" not in row["claim"] for row in table), "a location is never called a cause"
    assert any("where the change is, not what brought it about" in f for f in r["follow_ups"])


def test_a_change_within_ordinary_variation_is_inconclusive_not_a_finding():
    rng = np.random.default_rng(7)
    noise = {(i, s): v for (i, s), v in zip(((i, s) for i in range(60) for s in ("North", "South")), rng.normal(100, 30, 120))}
    df = _frame(60, value=lambda i, seg: noise[(i, seg)])
    r = inv.investigate(df, _profile(df), {"formula": "sum(revenue)", "label": "Revenue", "time_column": "day"})
    if r["decomposition_method"] == "none":
        assert "less than" in r["summary"]          # the totals happened to agree within 1%
        return
    overall = r["change_evidence"]["overall"]
    assert overall["status"] == "ok" and not overall["significant"]
    assert r["status"] == "inconclusive" and "within its ordinary variation" in r["summary"]
    assert any("may be describing noise" in u for u in r["unresolved"])
    assert all(row["support"] != "strong" for row in r["evidence_table"])


def test_short_periods_zero_baselines_and_unequal_coverage_are_reported_not_hidden():
    short = _frame(6, value=lambda i, seg: 100.0 if i < 3 else (40.0 if seg == "North" else 100.0))
    r = inv.investigate(short, _profile(short), {"formula": "sum(revenue)", "label": "Revenue", "time_column": "day"})
    overall = r["change_evidence"]["overall"]
    assert overall["status"] == "unavailable" and "needed" in overall["reason"], "three days a side is no series"
    north = next(row for row in r["evidence_table"] if row["candidate"] == "region = North")
    assert north["support"] != "strong" and any("no statistical test" in x for x in north["limitations"])
    assert any("longer periods" in f for f in r["follow_ups"])

    zero = _frame(40, value=lambda i, seg: 0.0 if i < 20 else 50.0)
    r = inv.investigate(zero, _profile(zero), {"formula": "sum(revenue)", "label": "Revenue", "time_column": "day",
                                               "period_a": ["2025-01-01", "2025-01-21"], "period_b": ["2025-01-21", "2025-02-09"]})
    assert r["totals"]["a"] == 0 and r["totals"]["change_pct"] is None
    assert any("no percentage change is defined" in u for u in r["unresolved"])

    empty = inv.investigate(zero, _profile(zero), {"formula": "sum(revenue)", "label": "Revenue", "time_column": "day",
                                                   "period_a": ["2024-11-01", "2024-12-01"], "period_b": ["2025-01-01", "2025-01-31"]})
    assert empty["status"] == "inconclusive" and "earlier period has no rows" in empty["summary"]


# ---- regressions: each of these was reported as a finding, or not flagged, before the fix

JAN, FEB, MAR = range(0, 31), range(31, 59), range(59, 90)          # day indices from 2025-01-01
REVENUE = {"formula": "sum(revenue)", "label": "Revenue", "time_column": "day"}


def _ask(df: pd.DataFrame, question: str) -> dict:
    return inv.investigate(df, _profile(df), inv.infer_spec(_profile(df), question))


def test_a_longer_month_is_not_reported_as_a_change():
    """February against March with identical daily revenue: the totals differ
    by 10.7% and nothing changed. Was: "complete", revenue rose 10.7%."""
    r = _ask(_frame(90), "Why did revenue rise last month?")
    t = r["totals"]
    assert (t["days_a"], t["days_b"], t["length_adjusted"]) == (28, 31, True)
    assert t["a"] == 5600 and t["b"] == 6200 and math.isclose(t["length_effect"], 600) and abs(t["like_for_like_change"]) < 1e-9
    assert t["per_day_a"] == t["per_day_b"] == 200
    assert r["status"] == "inconclusive" and "difference in length" in r["summary"] and r["ranked_contributors"] == []
    _numbers_are_backed(r["summary"], r["totals"])


def _numbers_are_backed(text: str, *computed) -> None:
    """Every number stated in `text` is one of the computed values, to the precision shown."""
    from app import verification

    pool = verification.extract_numbers(list(computed))
    stated = verification.narrative_numbers(text)
    assert stated and all(verification.number_matches(token, pool)[0] for token in stated), (stated, text)


def test_report_figures_match_an_independent_recomputation_from_the_rows():
    df = _frame(90, value=lambda i, seg: 60.0 + i % 3 if (i in MAR and seg == "North") else 100.0 + i % 3)
    r = _ask(df, "Why did revenue fall in March?")
    feb, mar = df[df["day"].dt.month == 2], df[df["day"].dt.month == 3]
    t = r["totals"]
    assert math.isclose(t["a"], feb["revenue"].sum()) and math.isclose(t["b"], mar["revenue"].sum())
    assert math.isclose(t["per_day_a"], feb["revenue"].sum() / 28) and math.isclose(t["per_day_b"], mar["revenue"].sum() / 31)
    assert math.isclose(t["like_for_like_change"], mar["revenue"].sum() - feb["revenue"].sum() * 31 / 28)
    for row in (x for x in r["evidence_table"] if x["candidate"].startswith("region = ")):
        seg = row["candidate"].split(" = ")[1]
        a, b = feb.loc[feb["region"] == seg, "revenue"].sum(), mar.loc[mar["region"] == seg, "revenue"].sum()
        assert math.isclose(row["value_a"], a) and math.isclose(row["value_b"], b)
        assert math.isclose(row["contribution"], b - a * 31 / 28)
        assert math.isclose(row["share_of_change"], row["contribution"] / t["like_for_like_change"])
    # the stated level difference is the difference of the two daily means
    daily = df.groupby("day")["revenue"].sum()
    overall = r["change_evidence"]["overall"]
    assert math.isclose(overall["estimate"], daily[daily.index.month == 3].mean() - daily[daily.index.month == 2].mean())
    assert overall["low"] < overall["estimate"] < overall["high"]
    _numbers_are_backed(r["summary"], r["totals"], overall, r["ranked_contributors"], r["price_volume"], {"p_floor": 0.001})


def test_unequal_months_are_compared_like_for_like_and_still_reconcile():
    df = _frame(90, value=lambda i, seg: 60.0 + i % 3 if (i in MAR and seg == "North") else 100.0 + i % 3)
    r = _ask(df, "Why did revenue fall in March?")
    t = r["totals"]
    assert t["length_adjusted"] and math.isclose(t["length_effect"] + t["like_for_like_change"], t["change"])
    assert math.isclose(t["length_effect"], t["a"] * (31 / 28 - 1))
    for dim in r["dimensions"]:
        assert dim["reconciles"] and math.isclose(sum(x["contribution"] for x in dim["segments"]), t["like_for_like_change"])
    north = next(row for row in r["evidence_table"] if row["candidate"] == "region = North")
    assert north["support"] == "strong" and 0.95 < north["share_of_change"] < 1.05, "all of the like-for-like fall is in North"
    length = r["evidence_table"][0]
    assert length["candidate"] == "Length of the periods (31 days against 28)" and length["support"] == "exact"
    assert math.isclose(length["contribution"], t["length_effect"])
    assert "like for like" in r["summary"] and "of the like-for-like change" in r["summary"]
    assert r["comparability"]["issues"] == [] and r["status"] == "complete"
    ratio = inv.investigate(df, _profile(df), {**REVENUE, "formula": "avg(revenue)", "question": "why in March"})
    assert ratio["totals"]["length_adjusted"] is False, "an average does not grow with the length of the period"


def test_a_segment_that_moved_against_the_change_is_an_offset_not_a_location():
    """Was: South, which rose while the total fell, labelled "the change is located in this segment" with strong support."""
    df = _frame(90, value=lambda i, seg: (60.0 if i in MAR else 100.0) + i % 3 if seg == "North"
                else (125.0 if i in MAR else 100.0) + i % 2)
    r = _ask(df, "Why did revenue fall in March?")
    rows = {row["candidate"]: row for row in r["evidence_table"]}
    assert rows["region = South"]["support"] == "offset" and "moved against" in rows["region = South"]["claim"]
    assert rows["region = South"]["share_of_change"] < 0
    assert rows["region = North"]["support"] == "strong" and "located" in rows["region = North"]["claim"]
    assert "region = North" in r["summary"] and "other segments moved the other way" in r["summary"]


def test_a_month_the_data_only_partly_covers_is_compared_over_the_same_days():
    """Was: 20 days of March against all 28 of February, a 28.6% "fall" in flat data."""
    df = _frame(79)                                                 # .. 2025-03-20
    read = inv.question_periods("why did revenue fall in March?", df, "day")
    assert (read["period_a"][0][:10], read["period_a"][1][:10]) == ("2025-02-01", "2025-02-21")
    assert (read["period_b"][0][:10], read["period_b"][1][:10]) == ("2025-03-01", "2025-03-20")
    assert "cut to their first 20 day(s)" in read["source"]
    r = _ask(df, "why did revenue fall in March?")
    assert r["totals"]["a"] == r["totals"]["b"] == 4000 and r["status"] == "inconclusive" and "less than" in r["summary"]


def test_an_earlier_period_the_data_only_partly_covers_is_flagged():
    """Was: "complete", revenue rose 133%, no warning: the dataset starts on 20 January."""
    df = _frame(40, start="2025-01-20")
    r = _ask(df, "Why did revenue change in February?")
    issues = r["comparability"]["issues"]
    assert any("earlier period's data starts on 2025-01-20" in i for i in issues), issues
    assert r["status"] == "inconclusive" and "not fully comparable" in r["summary"]
    assert all(row["support"] != "strong" for row in r["evidence_table"]), "nothing is strong evidence on uncomparable periods"
    assert any("covers completely" in f for f in r["follow_ups"])
    recording = next(row for row in r["evidence_table"] if row["candidate"].startswith("A change in how the data was recorded"))
    assert "not ruled out" in recording["support"]


def test_missing_days_in_a_period_make_the_result_inconclusive():
    """Was: flagged, but still "complete": March recorded every other day read as a 43% fall."""
    df = _frame(90)
    df = df[~((df["day"].dt.month == 3) & (df["day"].dt.day % 2 == 0))]
    r = _ask(df, "Why did revenue fall last month?")
    assert r["status"] == "inconclusive" and any("16 days against 28" in i for i in r["comparability"]["issues"])


@pytest.mark.parametrize("question, a, b", [
    ("why was revenue down in sept?", None, "only covers"),
    ("why did revenue fall in Q2?", ("2025-01-01", "2025-01-31"), ("2025-04-01", "2025-04-30")),
    ("why did revenue fall in Q1 2025", None, "no earlier period"),
    ("why was revenue lower yesterday", ("2025-04-29", "2025-04-30"), ("2025-04-30", "2025-04-30")),
])
def test_more_period_phrases_are_read_or_refused_with_a_reason(question, a, b):
    df = _frame(120)                                                # 2025-01-01 .. 2025-04-30
    read = inv.question_periods(question, df, "day")
    if a is None:
        assert b in read["error"]
        return
    assert (read["period_a"][0][:10], read["period_a"][1][:10]) == a
    assert (read["period_b"][0][:10], read["period_b"][1][:10]) == b


def test_a_question_whose_period_is_not_understood_says_so():
    df = _frame(60, value=lambda i, seg: 100.0 + i % 4 if i < 30 or seg == "South" else 50.0 + i % 4)
    r = _ask(df, "Why has revenue been down since the relaunch?")
    assert r["periods"]["source"].startswith("automatic") and "names no period" in r["periods"]["source"]


def test_a_trend_that_predates_the_later_period_is_measured_and_reported():
    df = _frame(60, value=lambda i, seg: 300.0 - 2.0 * i + (i * 7 % 5))          # falling steadily throughout
    r = inv.investigate(df, _profile(df), REVENUE)
    pre = r["change_evidence"]["pre_trend"]
    assert pre["status"] == "ok" and pre["significant"] and pre["same_direction"] and pre["slope"] < 0 and pre["share"] > 0.4
    assert "already moving this way before" in r["alternative_explanations"][0]
    row = next(x for x in r["evidence_table"] if x["candidate"].startswith("A trend already under way"))
    assert row["kind"] == "measured" and "extrapolation" in row["limitations"][0]
    assert any("when the trend began" in f for f in r["follow_ups"])
    step = _frame(60, value=lambda i, seg: (100.0 if i < 30 else 60.0) + (i * 7 % 5))
    flat = inv.investigate(step, _profile(step), REVENUE)
    assert not any(x["candidate"].startswith("A trend already under way") for x in flat["evidence_table"]), "a step is not a trend"


def test_ask_endpoint_investigates_a_question_without_a_model_and_explains_a_dataset_it_cannot_use(client):
    ws = workspace(client)
    out = client.post("/api/investigations/ask", headers=ws["h"],
                      json={"dataset_id": ws["ds"], "question": "Why did revenue decrease last month?"})
    assert out.status_code == 201, out.text
    body = out.json()
    assert body["read_as"] == {"formula": 'sum("revenue")', "label": "Total revenue", "time_column": "order_date"}
    assert run_job(body["job_id"]) == "succeeded"
    q = question_row(body["question_id"])
    assert q.route == "root_cause" and q.status == "verified", "the stated statistics are all backed by the recorded step"
    got = client.get(f"/api/investigations/{body['investigation_id']}", headers=ws["h"]).json()
    report = got["report"]
    assert got["engine"] == "deterministic" and report["periods"]["b"][0][:10] == "2025-03-01"
    assert report["totals"]["length_adjusted"] and "like for like" in report["summary"], "February has 28 days, March 31"
    assert [row["candidate"] for row in report["evidence_table"][:2]] == ["Length of the periods (31 days against 28)", "region = West"]
    assert report["follow_ups"] and "Price effect" not in report["summary"], "the raw price/volume split is not mixed into a like-for-like summary"
    volume = next(row for row in report["evidence_table"] if row["candidate"] == "Volume effect")
    assert any("includes the difference in the periods' lengths" in x for x in volume["limitations"])
    trace = client.get(f"/api/questions/{body['question_id']}/trace", headers=ws["h"]).json()
    assert trace["usage"]["llm_calls"] == 0

    vague = client.post("/api/investigations/ask", headers=ws["h"], json={"dataset_id": ws["ds"], "question": "Why did things get worse?"})
    assert vague.status_code == 422 and "revenue" in vague.json()["detail"] and "Name one of" in vague.json()["detail"]
    other = workspace(client)
    assert client.post("/api/investigations/ask", headers=other["h"],
                       json={"dataset_id": ws["ds"], "question": "Why did revenue fall?"}).status_code == 404


def test_investigation_api_with_explicit_spec_validation_and_tenancy(client):
    ws = workspace(client)
    body = {"dataset_id": ws["ds"], "formula": "sum(revenue) / sum(units)", "label": "Average selling price",
            "time_column": "order_date", "dimensions": ["product"], **PERIODS}
    r = client.post("/api/investigations", json=body, headers={**ws["h"], "Idempotency-Key": "inv-1"})
    assert r.status_code == 201, r.text
    created = r.json()
    again = client.post("/api/investigations", json=body, headers={**ws["h"], "Idempotency-Key": "inv-1"}).json()
    assert again["investigation_id"] == created["investigation_id"] and again["created"] is False
    assert run_job(created["job_id"], {}) == "succeeded"
    got = client.get(f"/api/investigations/{created['investigation_id']}", headers=ws["h"]).json()
    assert got["status"] == "complete" and got["report"]["decomposition_method"] == "rate_and_mix"
    assert [d["dimension"] for d in got["report"]["dimensions"]] == ["product"]
    assert got["report"]["ranked_contributors"][0]["segment"] == "Gadget"

    for bad, code in (({**body, "formula": "sum(nope)"}, 400), ({**body, "formula": "import os"}, 400),
                      ({**body, "time_column": "region"}, 400), ({**body, "dimensions": ["ghost"]}, 400),
                      ({**body, "period_b": None}, 400), ({**body, "dataset_id": 10**9}, 404),
                      ({**body, "filters": [{"column": "region", "op": "nope", "value": 1}]}, 400)):
        assert client.post("/api/investigations", json=bad, headers=ws["h"]).status_code == code
    other = workspace(client)
    assert client.get(f"/api/investigations/{created['investigation_id']}", headers=other["h"]).status_code == 404
    assert client.get("/api/investigations", headers=other["h"]).json()["total"] == 0
    assert client.post("/api/investigations", json=body, headers=other["h"]).status_code == 404
    assert client.get("/api/investigations").status_code == 401


# ========================================================= data quality ====

def test_psi_and_ks_behave_like_drift_metrics():
    rng = np.random.default_rng(7)
    base = list(rng.normal(100, 10, 400))
    same = list(rng.normal(100, 10, 400))
    shifted = list(rng.normal(130, 10, 400))
    assert dq.numeric_psi(base, same) < 0.1 and dq.ks_statistic(base, same)[1] > 0.01
    assert dq.numeric_psi(base, shifted) > 0.25
    d, p = dq.ks_statistic(base, shifted)
    assert d > 0.5 and p < 1e-6
    assert dq.ks_statistic(base, base) == (0.0, 1.0)
    assert dq.psi([0.5, 0.5], [0.5, 0.5]) == 0.0 and dq.psi([0.9, 0.1], [0.1, 0.9]) > 1.0
    assert dq.numeric_psi([5.0] * 200, [5.0] * 200) == 0.0, "a constant column does not crash the binning"
    assert dq.numeric_psi([5.0] * 200, [6.0] * 200) > 1.0
    value, appeared, vanished = dq.categorical_psi({"categories": {"a": 50, "b": 50}, "other": 0},
                                                   {"categories": {"a": 50, "c": 50}, "other": 0})
    assert value > 0.25 and appeared == ["c"] and vanished == ["b"]


def test_comparison_separates_schema_quality_and_distribution(sales):
    df, _ = sales
    base = dq.build_snapshot(df)
    assert base["rows"] == 720 and base["duplicate_rows"] == 0 and base["columns"]["region"]["kind"] == "categorical"
    assert len(base["columns"]["revenue"]["sample"]) <= dq.SAMPLE_POINTS

    assert dq.compare(base, base, dq.default_thresholds()) == {"findings": [], "skipped": []}

    broken = df.copy()
    broken["units"] = broken["units"].astype(object)
    broken.loc[broken.index[:50], "units"] = "n/a"                       # numeric -> text
    broken.loc[broken.index[100:400], "cost"] = None                     # 42% missing
    broken = pd.concat([broken, broken.head(200)], ignore_index=True)    # duplicates
    broken = broken.drop(columns=["channel"])
    found = {(f["category"], f["check"], f["column"]): f for f in dq.compare(dq.build_snapshot(broken), base, dq.default_thresholds())["findings"]}
    assert found[("schema", "column_removed", "channel")]["severity"] == "high"
    assert found[("schema", "type_changed", "units")]["severity"] == "high"
    assert found[("quality", "missing_increase", "cost")]["severity"] == "high"
    assert ("quality", "duplicate_rows", None) not in found, "duplicates are a single-version check"
    dup = dq.single_version_findings(dq.build_snapshot(broken), dq.default_thresholds())
    assert dup and dup[0]["check"] == "duplicate_rows" and dup[0]["severity"] == "high"
    assert all(f["message"] for f in found.values()), "every finding explains itself"


def test_small_samples_are_skipped_not_alarmed(sales):
    df, _ = sales
    base = dq.build_snapshot(df)
    tiny = df.head(40).copy()
    tiny["unit_price"] = tiny["unit_price"] * 50
    result = dq.compare(dq.build_snapshot(tiny), base, dq.default_thresholds())
    assert not [f for f in result["findings"] if f["category"] == "distribution"]
    assert {s["column"] for s in result["skipped"]} >= {"unit_price", "region"}
    assert "fewer than" in result["skipped"][0]["reason"]
    low_bar = dq.merged_thresholds({"min_sample": 10})
    assert [f for f in dq.compare(dq.build_snapshot(tiny), base, low_bar)["findings"] if f["check"] == "numeric_drift"]


def test_numeric_drift_needs_both_signals_and_range_and_staleness_are_reported(sales):
    df, _ = sales
    base = dq.build_snapshot(df)
    scaled = df.copy()
    scaled["revenue"] = scaled["revenue"] * 100
    f = {(x["check"], x["column"]) for x in dq.compare(dq.build_snapshot(scaled), base, dq.default_thresholds())["findings"]}
    assert ("numeric_drift", "revenue") in f and ("range_violation", "revenue") in f
    older = df[df["order_date"] < "2025-04-01"]
    f = {(x["check"], x["column"]) for x in dq.compare(dq.build_snapshot(older), base, dq.default_thresholds())["findings"]}
    assert ("stale_data", "order_date") in f
    half = df.iloc[:200]
    f = [x for x in dq.compare(dq.build_snapshot(half), base, dq.default_thresholds())["findings"] if x["check"] == "row_count_change"]
    assert f and f[0]["severity"] == "high" and "partial load" in f[0]["message"]


def test_freshness_check_is_off_unless_configured(sales):
    import datetime as dt

    df, _ = sales
    snap = dq.build_snapshot(df)
    now = dt.datetime(2025, 6, 1)
    assert dq.single_version_findings(snap, dq.default_thresholds(), now) == []
    stale = dq.single_version_findings(snap, dq.merged_thresholds({"freshness_max_age_days": 7}), now)
    assert stale[0]["check"] == "freshness" and stale[0]["severity"] == "high" and stale[0]["column"] == "order_date"
    assert dq.single_version_findings(snap, dq.merged_thresholds({"freshness_max_age_days": 60}), now) == []


@pytest.mark.parametrize("bad", [{"nope": 1}, {"psi_warn": -1}, {"psi_warn": "x"}, {"psi_warn": 0.5, "psi_high": 0.2},
                                 {"ks_alpha": 0}, {"ks_alpha": 1.5}, {"min_sample": None}, {"psi_warn": float("nan")},
                                 {"duplicate_pct_warn": True}])
def test_threshold_validation(bad):
    with pytest.raises(ValueError):
        dq.validate_thresholds(bad)


def test_upload_creates_a_snapshot_and_a_new_version_raises_incidents(client):
    ws = workspace(client)
    first = client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()
    assert first["latest_version"] == 1 and first["incidents"] == [] and first["history"][0]["has_snapshot"]
    assert first["baseline"]["version"] is None and first["thresholds"]["psi_warn"] == dq.default_thresholds()["psi_warn"]

    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    body = client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()
    assert body["latest_version"] == 2 and body["baseline"] == {"version_id": body["history"][0]["version_id"], "version": 1,
                                                                 "source": "previous version", "pinned": False}
    kinds = {(i["category"], i["check"], i["column"]) for i in body["incidents"]}
    assert {("schema", "column_removed", "channel"), ("schema", "column_added", "discount"),
            ("distribution", "numeric_drift", "unit_price"), ("distribution", "category_drift", "region")} <= kinds
    assert body["open_incidents"] == len(body["incidents"]) and body["history"][1]["incidents"]["schema"] == 2
    assert body["history"][1]["max_drift_psi"] > 0.25 and body["history"][1]["rows"] == 540
    assert all(i["baseline_version_id"] == body["history"][0]["version_id"] for i in body["incidents"])

    before = len(body["incidents"])
    rerun = client.post(f"/api/datasets/{ws['ds']}/quality/run", headers=ws["h"])
    assert rerun.status_code == 200 and rerun.json()["baseline_version"] == 1
    assert len(client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()["incidents"]) == before, "re-running does not duplicate incidents"
    db = SessionLocal()
    try:
        assert db.query(DQSnapshot).filter_by(dataset_id=ws["ds"]).count() == 2
    finally:
        db.close()


def test_incident_lifecycle_configuration_permissions_and_tenancy(client):
    ws = workspace(client)
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    incidents = client.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()["incidents"]
    iid = incidents[0]["id"]
    assert client.post(f"/api/quality/incidents/{iid}/acknowledge", headers=ws["h"]).json()["status"] == "acknowledged"
    resolved = client.post(f"/api/quality/incidents/{iid}/resolve", headers=ws["h"]).json()
    assert resolved["status"] == "resolved" and resolved["resolved_at"]
    assert client.post(f"/api/quality/incidents/{iid}/acknowledge", headers=ws["h"]).json()["status"] == "resolved"

    member = add_member(client, ws, "member")
    cfg = {"baseline_version_id": None, "thresholds": {"psi_warn": 0.05, "min_sample": 50}}
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json=cfg, headers=member).status_code == 403
    assert client.get(f"/api/datasets/{ws['ds']}/quality", headers=member).status_code == 200, "members can read"
    ok = client.put(f"/api/datasets/{ws['ds']}/quality/config", json=cfg, headers=ws["h"])
    assert ok.status_code == 200 and ok.json()["thresholds"]["psi_warn"] == 0.05 and ok.json()["threshold_overrides"] == cfg["thresholds"]
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json={"thresholds": {"bogus": 1}}, headers=ws["h"]).status_code == 400

    versions = client.get(f"/api/datasets/{ws['ds']}/versions", headers=ws["h"]).json()
    pinned = client.put(f"/api/datasets/{ws['ds']}/quality/config", json={"baseline_version_id": versions[0]["id"], "thresholds": {}}, headers=ws["h"]).json()
    assert pinned["baseline"]["pinned"] is True and pinned["baseline"]["version"] == 1
    other = workspace(client)
    foreign = client.get(f"/api/datasets/{other['ds']}/versions", headers=other["h"]).json()[0]["id"]
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json={"baseline_version_id": foreign, "thresholds": {}}, headers=ws["h"]).status_code == 404
    for path, method in ((f"/api/datasets/{ws['ds']}/quality", "get"), (f"/api/datasets/{ws['ds']}/quality/run", "post"),
                         (f"/api/quality/incidents/{iid}/resolve", "post"), (f"/api/quality/incidents/{iid}/acknowledge", "post")):
        assert getattr(client, method)(path, headers=other["h"]).status_code == 404, path
    assert client.put(f"/api/datasets/{ws['ds']}/quality/config", json=cfg, headers=other["h"]).status_code == 404
    assert client.get(f"/api/datasets/{ws['ds']}/quality").status_code == 401


def test_a_broken_quality_check_never_fails_an_upload(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("profiler exploded")

    monkeypatch.setattr(dq, "build_snapshot", boom)
    ws = workspace(client)  # asserts the upload returned 200
    db = SessionLocal()
    try:
        assert db.query(DQSnapshot).filter_by(dataset_id=ws["ds"]).count() == 0
        assert db.query(DQIncident).filter_by(dataset_id=ws["ds"]).count() == 0
    finally:
        db.close()
