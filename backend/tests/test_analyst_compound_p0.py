"""P0: compound requests, deliverable tracking, period-aware targets and real forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import compound, deliverables as D, forecasting as F, orchestrator
from app.analyst.data import load_table

EXAMPLE = ("Show me the summary first, then the average and variance of the dataset. My company has a target of 300 "
           "working hours per month. Will Rahul make it to the target? If yes, how many hours do you think he can "
           "reach upto and show me in a graph.")


@pytest.fixture()
def timesheet(tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    months = pd.date_range("2025-01-01", periods=12, freq="MS")
    for name, base, trend in (("Rahul", 262, 3.2), ("Priya", 298, 0.5), ("Arjun", 240, -1.0), ("Meera", 310, 1.5)):
        for i, m in enumerate(months):
            rows.append({"employee": name, "department": "Ops" if name in ("Rahul", "Arjun") else "Sales",
                         "month": m.strftime("%Y-%m-%d"), "hours_worked": round(base + trend * i + rng.normal(0, 6), 1),
                         "overtime_hours": round(max(0, rng.normal(12, 5)), 1)})
    p = tmp_path / "timesheet.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return load_table(str(p))


# ------------------------------------------------------------------ forecasting --
def test_grain_monthly_totals_are_not_days(timesheet):
    g = F.infer_grain(timesheet, "month", "employee")
    assert g["freq"] == "M" and g["one_per_period"] is True


def test_grain_daily_events():
    d = pd.DataFrame({"when": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03"] * 5),
                      "v": range(20)})
    g = F.infer_grain(d, "when")
    assert g["freq"] == "D" and g["one_per_period"] is False


def test_forecast_refuses_short_history():
    s = pd.Series([10.0, 11, 12, 13], index=pd.period_range("2025-01", periods=4, freq="M"))
    fc = F.forecast(s, 2, "M")
    assert fc["status"] == "insufficient" and "forecasts" not in fc


def test_forecast_beats_naive_on_a_trend_and_picks_the_lowest_backtest_error():
    rng = np.random.default_rng(1)
    s = pd.Series(100 + 5 * np.arange(24) + rng.normal(0, 1, 24), index=pd.period_range("2023-01", periods=24, freq="M"))
    fc = F.forecast(s, 3, "M", target=230)
    assert fc["status"] == "ok" and fc["method"] in ("trend", "drift")
    assert fc["backtest_mae"] == min(r["mae"] for r in fc["backtest"]) and fc["skill_vs_naive"] > 0.5
    assert len(fc["forecasts"]) == 3 and all(p["lower"] <= p["forecast"] <= p["upper"] for p in fc["forecasts"])
    assert fc["forecasts"][0]["target_verdict"] in ("likely", "unlikely", "uncertain")


def test_backtest_uses_only_the_past():
    """Changing the LAST value cannot change any backtest error measured before it (no leakage)."""
    y = np.array([5.0, 7, 6, 8, 9, 8, 10, 11, 10, 12])
    a = F.backtest(y, 1, F._candidates(len(y), "M"))
    y2 = y.copy(); y2[-1] = 500.0
    b = F.backtest(y2, 1, F._candidates(len(y2), "M"))
    for k in a:
        assert a[k]["errors"][1][:-1] == b[k]["errors"][1][:-1]


def test_no_range_without_enough_backtest_errors():
    s = pd.Series([10.0, 12, 11, 13, 12, 14, 13], index=pd.period_range("2025-01", periods=7, freq="M"))
    fc = F.forecast(s, 3, "M")
    assert fc["status"] == "ok" and fc["forecasts"][-1]["lower"] is None


# ------------------------------------------------------------------ deliverables --
def test_example_is_decomposed_into_every_deliverable(timesheet):
    ds = D.extract(EXAMPLE, timesheet)
    kinds = [d.kind for d in ds]
    assert kinds == ["summary", "stat", "stat", "target_check", "forecast", "chart"]
    assert [d.stat for d in ds if d.kind == "stat"] == ["avg", "var"]
    assert all(d.entities == {} for d in ds[:3])                       # the summary and statistics are dataset-wide
    tc, fc, ch = ds[3], ds[4], ds[5]
    assert tc.entities == {"employee": "Rahul"} and tc.target == 300 and tc.period == "M"
    assert fc.entities == {"employee": "Rahul"} and fc.entity_from_context and fc.condition == "yes"
    assert ch.chart_for == fc.id and D.is_compound(ds)


def test_entities_do_not_leak_between_clauses(timesheet):
    ds = D.extract("What is Rahul's total hours worked and Priya's average overtime hours?", timesheet)
    assert [(d.stat, d.measures, d.entities) for d in ds] == [
        ("sum", ["hours_worked"], {"employee": "Rahul"}), ("avg", ["overtime_hours"], {"employee": "Priya"})]
    ds = D.extract("Give me the total overtime hours, the top 3 employees by hours worked, and plot monthly hours "
                   "worked for Priya.", timesheet)
    assert ds[0].kind == "stat" and ds[0].entities == {} and ds[1].kind == "ranking" and ds[1].top == 3
    assert ds[2].kind == "series" and ds[2].entities == {"employee": "Priya"}


def test_unknown_measure_is_not_replaced(timesheet):
    ds = D.extract("Give me the total revenue, the average hours worked and plot overtime hours for Arjun.", timesheet)
    assert ds[0].measures == [] and "revenue" in ds[0].detail


# ------------------------------------------------------------------ end to end --
def ask(q, df, mem=None):
    return orchestrator.ask(q, df, {} if mem is None else mem, use_model=False)


def test_example_end_to_end(timesheet):
    out = ask(EXAMPLE, timesheet)
    assert out["plan_source"] == "compound" and not out.get("incomplete")
    a = out["answer"]
    for sec in ("Summary of the dataset", "Average of", "Variance of", "Target check", "Forecast of"):
        assert sec in a
    assert a.index("Summary of the dataset") < a.index("Average of") < a.index("Variance of") < a.index("Target check")
    rahul = timesheet[timesheet.employee == "Rahul"].sort_values("month")
    assert f"{rahul.hours_worked.iloc[-1]:,.2f}" in a                       # latest actual
    assert f"{timesheet.hours_worked.var():,.2f}" in a                      # dataset variance, not Rahul's
    chart = out["charts"][0]
    assert chart["type"] == "forecast" and chart["target"] == 300
    assert [x["value"] for x in chart["data"] if x["kind"] == "actual"] == list(rahul.hours_worked)
    assert any(x["kind"] == "forecast" for x in chart["data"])
    assert "per day" not in a and "All 6 requested items are answered" in a and out["verify_status"] == "verified"


def test_monthly_target_is_never_projected_from_a_daily_pace(timesheet):
    out = ask("Will Rahul reach 300 hours worked this month?", timesheet)
    assert out["plan_source"] == "compound" and "per day" not in out["answer"]
    assert "forecast" in out["answer"].lower()


def test_condition_uses_the_question_polarity(timesheet):
    out = ask("Is Rahul below the 300 hours per month target? If so, what would he need to average per week to "
              "reach it?", timesheet)
    assert "only if" not in out["answer"]                                   # he is below, so the condition holds
    assert f"{300 * 7 / (365.25 / 12):,.2f} per week" in out["answer"]


def test_one_failed_item_keeps_the_others_and_is_reported(timesheet):
    out = ask("Give me the total revenue, the average hours worked and plot overtime hours for Arjun.", timesheet)
    a = out["answer"]
    assert "no column for “revenue”" in a and f"{timesheet.hours_worked.mean():,.2f}" in a
    assert out["incomplete"] and "Not complete: 1 of 4" in a and out["charts"]


def test_forecast_without_dates_is_refused_and_the_rest_delivered(tmp_path):
    p = tmp_path / "items.csv"
    pd.DataFrame({"product": [f"P{i}" for i in range(12)], "category": ["A", "B", "C"] * 4,
                  "units_sold": range(10, 130, 10)}).to_csv(p, index=False)
    df = load_table(str(p))
    out = ask("Show the total units sold, forecast units sold for next month and rank the top 2 categories by units sold.", df)
    a = out["answer"]
    assert "780" in a and "no date column" in a and not any(c.get("type") == "forecast" for c in out["charts"])
    assert out["incomplete"]


def test_chart_validation_catches_wrong_data(timesheet):
    ds = D.extract(EXAMPLE, timesheet)
    ctx = compound.execute(ds, timesheet)
    fc = next(d for d in ds if d.kind == "forecast")
    good = ctx["results"][fc.id]["chart"]
    assert compound.validate_chart(good, ctx["results"][fc.id], fc) == []
    bad = {**good, "data": [dict(x, value=x["value"] + 1) if x["kind"] == "actual" else x for x in good["data"]]}
    assert compound.validate_chart(bad, ctx["results"][fc.id], fc)


# ------------------------------------------- regressions found by the held-out compound evaluation --
@pytest.fixture()
def shops(tmp_path):
    rng = np.random.default_rng(2)
    rows = [{"outlet_name": n, "zone": z, "week": w.strftime("%Y-%m-%d"), "takings_eur": round(float(rng.normal(9000, 500)), 2),
             "footfall": int(rng.normal(400, 30))}
            for n, z in (("Harbour Lane", "North"), ("Kings Row", "South"), ("Mill Street", "North"))
            for w in pd.date_range("2025-01-06", periods=20, freq="W-MON")]
    p = tmp_path / "shops.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return load_table(str(p))


def test_a_leading_scope_applies_to_every_listed_item(timesheet):
    ds = D.extract("What's the overall mean overtime hours? Then for Priya only: the total hours worked and the "
                   "maximum overtime hours.", timesheet)
    assert [(d.stat, d.entities) for d in ds] == [("avg", {}), ("sum", {"employee": "Priya"}), ("max", {"employee": "Priya"})]


def test_a_condition_never_swallows_its_clause(timesheet):
    ds = D.extract("Is Meera below the 330 hours per month target, and if so what would she need to average per week?", timesheet)
    assert [d.kind for d in ds] == ["target_check", "pace"] and ds[1].condition == "yes"


@pytest.mark.parametrize("text,value,unit,period", [
    ("Target: 290 hours a month", 290, "hours", "M"), ("we expect 9,500 euros per week from it", 9500, "euros", "W"),
    ("is she short of 270 hours a month", 270, "hours", "M"), ("check it against 8,800 euros per week", 8800, "euros", "W"),
])
def test_targets_stated_in_many_ways(text, value, unit, period):
    assert D._target(text) == (value, unit, period)


def test_money_targets_map_to_money_columns_and_unit_words_are_not_measures(shops):
    ds = D.extract("We want 9,800 euros per week from Kings Row. Will Kings Row get there? How many euros could it "
                   "reach in the next 2 weeks? Chart it.", shops)
    kinds = {d.kind: d for d in ds}
    assert kinds["target_check"].measures == ["takings_eur"] and kinds["forecast"].measures == ["takings_eur"]
    assert kinds["forecast"].entities == {"outlet_name": "Kings Row"} and kinds["forecast"].horizon == 2


def test_a_value_of_a_column_is_the_subject_not_a_requested_measure(shops):
    ds = D.extract("Our target is 9,000 euros per week. Forecast Mill Street for the next week and plot it.", shops)
    fc = next(d for d in ds if d.kind == "forecast")
    assert fc.measures == ["takings_eur"] and fc.entities == {"outlet_name": "Mill Street"}


def test_no_measure_named_is_never_the_first_column(shops):
    ds = D.extract("Give me a summary, and forecast Harbour Lane for next week.", shops)
    fc = next(d for d in ds if d.kind == "forecast")
    assert fc.measures == []                      # two measures, none named, no target: not silently the first one
