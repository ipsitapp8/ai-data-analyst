"""Fixes from the pre-demo audit: group rankings are not answered with each group's single extreme record, and a short
filter follow-up ("Only Rahul.") refines a plain verified total or average, not only a calculator answer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import intent, orchestrator, planner


@pytest.fixture(autouse=True)
def model_off(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)


def orders(seed=42, n=300):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"department": rng.choice(["Kitchen", "Bar", "Bakery"], n),
                         "employee": rng.choice(["Rahul", "Priya", "Arjun"], n),
                         "outlet": rng.choice(["Andheri", "Bandra", "Powai"], n),
                         "revenue": rng.gamma(4, 450, n).round(2)})


EXTREMES = [{"id": "a", "op": "aggregate", "from": "data", "fn": "max", "column": "revenue", "by": "outlet"},
            {"id": "b", "op": "aggregate", "from": "data", "fn": "min", "column": "revenue", "by": "outlet"}]


def test_ranking_groups_by_their_single_extreme_record_is_rejected():
    df = orders()
    q = "Which outlet has the highest revenue and which has the lowest?"
    assert any("compare the groups" in p for p in intent.check(EXTREMES, intent.read(q, df), df, q))
    total = [{"id": "a", "op": "aggregate", "from": "data", "fn": "sum", "column": "revenue", "by": "outlet"}]
    assert intent.check(total, intent.read(q, df), df, q) == []


@pytest.mark.parametrize("q", ["Which outlet has the highest single order revenue?",
                               "Which outlet has the highest maximum revenue?"])
def test_an_explicit_single_record_question_may_use_max(q):
    df = orders()
    assert not any("compare the groups" in p for p in intent.check(EXTREMES, intent.read(q, df), df, q))


def test_only_rahul_refines_a_verified_total_by_group():
    df, mem = orders(), {}
    first = orchestrator.ask("What is the total revenue by department?", df, mem, use_model=False)
    assert first["verify_status"] == "verified"
    out = orchestrator.ask("Only Rahul.", df, mem, use_model=False)
    assert out["plan_source"] == "refine"
    want = df[df.employee == "Rahul"].groupby("department").revenue.sum()
    assert all(f"{v:,.2f}" in out["answer"] for v in want)
    out = orchestrator.ask("What is the average revenue?", df, mem, use_model=False)
    out = orchestrator.ask("What about only Rahul?", df, mem, use_model=False)
    assert out["plan_source"] == "refine" and f"{df[df.employee == 'Rahul'].revenue.mean():,.2f}" in out["answer"]


def test_the_not_verified_reason_keeps_its_closing_parenthesis():
    df, mem = orders(), {}
    orchestrator.ask("What is the average revenue?", df, mem, use_model=False)
    out = orchestrator.ask("Only Rahul.", df, mem, use_model=False)
    line = next(x for x in out["answer"].split("\n") if x.startswith("_Verification"))
    assert line.count("(") == line.count(")"), line


def test_a_refined_result_compares_with_the_original_as_the_same_measure():
    df, mem = orders(), {}
    orchestrator.ask("What is the total revenue by department?", df, mem, use_model=False)
    orchestrator.ask("Only Rahul.", df, mem, use_model=False)
    out = orchestrator.ask("Compare that with the previous result.", df, mem, use_model=False)
    r = out["calls"][0]["result"]
    assert r["same_measure"] and "change" in out["answer"]
    rahul = df[df.employee == "Rahul"].groupby("department").revenue.sum()
    assert r["figures"]["Bar"]["b"] == pytest.approx(rahul["Bar"])


def test_a_closure_with_one_measure_in_the_data_needs_no_measure_named():
    df, mem = orders(), {}
    orchestrator.ask("What is the total revenue by department?", df, mem, use_model=False)
    out = orchestrator.ask("What if the Powai outlet closes?", df, mem, use_model=False)
    assert out["plan"]["objective"] == "closure"
    assert f"{df[df.outlet == 'Powai'].revenue.sum():,.2f}" in out["answer"]


def test_missing_and_negative_values_of_a_plain_total_are_stated():
    df = orders()
    df.loc[[3, 8], "revenue"] = np.nan
    df.loc[5, "revenue"] = -100.0
    out = orchestrator.ask("What is the total revenue?", df, {}, use_model=False)
    assert out["verify_status"] == "verified"
    assert "2 record(s) with no revenue (lines 5, 10) were not counted" in out["answer"]
    assert "1 record(s) with a negative revenue (line 7) were counted as recorded" in out["answer"]
    assert f"{df.revenue.sum():,.2f}" in out["answer"]


def test_a_bar_chart_request_is_not_read_as_a_category_named_bar():
    df, mem = orders(), {}                    # "Bar" is a department here
    orchestrator.ask("What is the average revenue by department?", df, mem, use_model=False)
    out = orchestrator.ask("Show that as a bar chart.", df, mem, use_model=False)
    assert out["calls"] and out["calls"][0]["tool"].startswith("chart_of_previous") and out["charts"]
    out = orchestrator.ask("What is the total revenue for Bar?", df, {}, use_model=False)
    assert f"{df[df.department == 'Bar'].revenue.sum():,.2f}" in out["answer"]


def dated_orders():
    df = orders()
    rng = np.random.default_rng(1)
    return df.assign(sold_on=pd.Timestamp.today().normalize() - pd.to_timedelta(rng.integers(0, 90, len(df)), "D"))


def test_comparing_scenarios_skips_an_analysis_of_another_kind():
    df, mem = dated_orders(), {}
    for q in ("Can Andheri reach 40000 revenue this month?", "What if the target is 45000?",
              "What if the Powai outlet closes?"):
        orchestrator.ask(q, df, mem, use_model=False)
    out = orchestrator.ask("Compare these scenarios.", df, mem, use_model=False)
    r = out["calls"][0]["result"]
    assert (r["previous"]["target"], r["latest"]["target"]) == (40000, 45000)
    assert "two most recent target analyses" in out["answer"]


def test_scenarios_of_different_kinds_are_listed_not_an_empty_table():
    df, mem = dated_orders(), {}
    orchestrator.ask("Can Andheri reach 40000 revenue this month?", df, mem, use_model=False)
    orchestrator.ask("What if the Powai outlet closes?", df, mem, use_model=False)
    out = orchestrator.ask("Compare these scenarios.", df, mem, use_model=False)
    assert "different kinds" in out["answer"]
    out = orchestrator.ask("Compare the second result with the first one.", df, mem, use_model=False)
    assert "different kinds" in out["answer"] and "| | previous | latest |" not in out["answer"]
