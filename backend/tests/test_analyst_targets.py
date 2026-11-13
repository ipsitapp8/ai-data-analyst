"""The requested prediction target survives the whole pipeline: it is read from the target phrase only (inputs named in
"using ..." are never the target), a column is used when the phrase names one, several matching columns are asked
about, a metric that is not a column is derived per record (by the local model through the validated calculator --
scripted here, no LLM is called) and checked against the requested name, a unit the user names justifies its
conversion factor, and the answer carries the requested name. Datasets: timesheet, retail, clinic. Expected values
are computed from the fixture rows in plain pandas/Python."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import calc, followup, orchestrator, planner, scope

TODAY = pd.Timestamp("2026-11-12")


@pytest.fixture(autouse=True)
def fixed(monkeypatch):
    monkeypatch.setattr(scope, "today", lambda: TODAY)
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "The estimate is shown above.")


def scripted(monkeypatch, sections):
    """The local model's plan for the derived metric, section by section (what the v1 model writes)."""
    it = iter(sections)
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(calc, "_complete", lambda prompt, max_tokens: next(it))


def timesheet() -> pd.DataFrame:
    rows = []
    for d in range(1, 13):
        for name, rate, start in (("Rahul Sharma", 25, 9), ("Priya Singh", 22, 8), ("Aarav Mehta", 30, 10)):
            rows.append({"employee_name": name, "date": pd.Timestamp(f"2026-09-{d:02d}"),
                         "clock_in": f"{start:02d}:00", "clock_out": f"{start + 7 + d % 3:02d}:{'30' if d % 2 else '00'}",
                         "hourly_rate": rate})
    return pd.DataFrame(rows)


def retail() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for m in range(14):
        for region in ("North", "South"):
            for store in range(3):
                rows.append({"month": pd.Timestamp("2025-01-01") + pd.DateOffset(months=m), "region": region,
                             "store": f"{region[0]}{store}", "units_sold": int(rng.integers(20, 120)),
                             "unit_price": float(rng.choice([9.5, 12.0, 15.25]))})
    return pd.DataFrame(rows)


def costs() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame({"week": pd.date_range("2025-01-06", periods=20, freq="W-MON"),
                         "labor_cost": rng.uniform(500, 900, 20).round(2), "material_cost": rng.uniform(100, 300, 20).round(2)})


def _hours(a, b):
    h1, m1 = map(int, a.split(":"))
    h2, m2 = map(int, b.split(":"))
    return (h2 * 60 + m2 - h1 * 60 - m1) / 60


LABOR_PLAN = ['calc", "derive": {}}', '"labor_cost": "hourly_rate * hours_between(clock_in, clock_out)"}}', ']}',
              'null}', '"labor_cost": "sum(labor_cost)"}}']


# ------------------------------------------------------------- reading the target --
@pytest.mark.parametrize("question,phrase", [
    ("Using Rahul Sharma's historical hourly rate, predict his labor cost for the next day.", "labor cost"),
    ("Predict Rahul Sharma's next labor cost based on his hourly rate and hours worked.", "labor cost"),
    ("Predict revenue for the North region based on units sold and unit price.", "revenue"),
    ("What will unit price be next month?", "unit price"),
    ("Forecast the fee for Clinic A next month.", "fee"),
    ("Predict the next value using historical data.", ""),
    ("Forecast Rahul Sharma's hours worked in minutes for the next day.", "hours worked"),
    ("Predict the next day's total labor cost across all employees.", "labor cost"),
    ("Forecast tomorrow's total delivery cost across all drivers.", "delivery cost"),
    ("Predict next month's revenue for all stores.", "revenue"),
    ("Estimate company-wide labor cost for the coming week.", "labor cost"),
])
def test_the_target_phrase_excludes_inputs_names_units_and_generic_words(question, phrase):
    df = timesheet()
    assert followup.target_phrase(question, scope.establish(question, df, {})) == phrase


def test_an_input_column_named_in_the_question_is_never_taken_as_the_target():
    df = timesheet()
    q = "Using Rahul Sharma's historical hourly rate, predict his labor cost for the next day."
    t = followup.forecast_target(q, df, {}, scope.establish(q, df, {}))
    assert t == {"kind": "derived", "phrase": "labor cost"}


def _today(monkeypatch, ts):
    monkeypatch.setattr(scope, "today", lambda: pd.Timestamp(ts))


def test_a_column_named_as_the_target_is_used_directly(monkeypatch):
    _today(monkeypatch, "2026-03-10")
    df = retail()
    q = "Forecast unit price for the North region next month."
    out = orchestrator.ask(q, df, {}, use_model=False)
    f = out["calls"][0]["result"]
    north = df[df["region"] == "North"]
    truth = north.groupby(north["month"].dt.to_period("M"))["unit_price"].mean()       # a price is averaged
    assert f["label"] == "unit price" and [p["value"] for p in f["points"]] == pytest.approx(list(truth.values))
    assert out["verification"][0]["ok"] and out["verification"][0]["check"] == "forecast target is the one asked for"


def test_several_matching_columns_are_asked_about_not_picked(monkeypatch):
    _today(monkeypatch, "2025-05-20")
    out = orchestrator.ask("Forecast the cost for next week.", costs(), {}, use_model=False)
    assert out["plan"]["type"] == "clarify" and "`labor_cost`" in out["answer"] and "`material_cost`" in out["answer"]
    out = orchestrator.ask("Forecast the material cost for next week.", costs(), {}, use_model=False)
    f = out["calls"][0]["result"]
    assert f["label"] == "material cost"


# ------------------------------------------------------------- derived targets --
def test_a_derived_target_is_derived_not_replaced_by_an_input(monkeypatch):
    _today(monkeypatch, "2026-09-13")
    df = timesheet()
    scripted(monkeypatch, LABOR_PLAN)
    q = "Using Rahul Sharma's historical hourly rate, predict his labor cost for the next day."
    out = orchestrator.ask(q, df, {})
    f = out["calls"][0]["result"]
    rahul = df[df["employee_name"] == "Rahul Sharma"]
    per_day = [_hours(a, b) * r for a, b, r in zip(rahul["clock_in"], rahul["clock_out"], rahul["hourly_rate"])]
    assert f["label"] == "labor cost" and f["frequency"] == "day"
    assert [p["value"] for p in f["points"]] == pytest.approx(per_day)
    assert "hourly rate × hours from clock in to clock out" in f["definition"]
    assert out["answer"].count("hourly rate") <= 2 and "Estimated labor cost" in out["answer"] or \
        "Predicted labor cost" in out["answer"]
    assert out["verification"][0]["ok"] and "labor cost" in out["verification"][0]["detail"]


def test_a_derived_target_on_another_schema(monkeypatch):
    _today(monkeypatch, "2026-03-10")
    df = retail()
    scripted(monkeypatch, ['calc", "derive": {}}', '"revenue": "units_sold * unit_price"}}', ']}', 'null}',
                           '"revenue": "sum(revenue)"}}'])
    out = orchestrator.ask("Predict revenue for the North region for next month.", df, {})
    f = out["calls"][0]["result"]
    north = df[df["region"] == "North"].assign(rev=lambda d: d["units_sold"] * d["unit_price"])
    truth = north.groupby(north["month"].dt.to_period("M"))["rev"].sum()
    assert f["label"] == "revenue" and [p["value"] for p in f["points"]] == pytest.approx(list(truth.values))
    assert f["records_used"] == len(north)
    # 14 months: enough to fit and validate a model; it is called a prediction only if it beat the baseline
    assert (f["kind"] == "ml_prediction") == (f["validation"]["model_mae"] < f["validation"]["naive_mae"])


def test_a_definition_of_a_different_metric_is_refused(monkeypatch):
    df = timesheet()
    scripted(monkeypatch, ['calc", "derive": {}}', '}', ']}', 'null}', '"hourly_rate": "sum(hourly_rate)"}}'])
    out = orchestrator.ask("Predict Rahul Sharma's labor cost for the next day.", df, {})
    assert out["plan"]["type"] == "clarify" and not out["calls"]
    assert "couldn't define labor cost" in out["answer"] and "won't forecast a different quantity" in out["answer"]


def test_a_derived_target_without_the_model_is_not_substituted():
    out = orchestrator.ask("Predict Rahul Sharma's labor cost for the next day.", timesheet(), {}, use_model=False)
    assert out["plan"]["type"] == "clarify" and not out["calls"]
    assert "not a column" in out["answer"] and "won't forecast a different quantity" in out["answer"]


# ------------------------------------------------------------------ unit conversion --
def test_a_named_unit_justifies_its_conversion_factor(monkeypatch):
    _today(monkeypatch, "2026-09-13")
    df = timesheet()
    scripted(monkeypatch, ['calc", "derive": {}}', '"minutes_worked": "hours_between(clock_in, clock_out) * 60"}}', ']}',
                           'null}', '"hours_worked": "sum(minutes_worked)"}}'])
    out = orchestrator.ask("Forecast Rahul Sharma's hours worked in minutes for the next day.", df, {})
    f = out["calls"][0]["result"]
    rahul = df[df["employee_name"] == "Rahul Sharma"]
    assert [p["value"] for p in f["points"]] == pytest.approx(
        [_hours(a, b) * 60 for a, b in zip(rahul["clock_in"], rahul["clock_out"])])
    assert f["label"] == "hours worked in minutes"


def test_an_unnamed_unit_factor_is_still_rejected():
    df = timesheet()
    frame = calc.run_derive(df, {})
    probs = calc._check_section("derive", {"minutes": "hours_between(clock_in, clock_out) * 60"},  # noqa: SLF001
                                {"derive": {}, "exclude": []}, frame, "Total hours worked", "Total hours worked")
    assert any("60" in p for p in probs)
    assert calc.unit_factors("labor time in minutes") == {60.0} and calc.unit_factors("in thousands") == {1000.0}


# ------------------------------------------------------------------ period & method --
def test_a_date_in_a_forecast_question_is_the_target_not_a_filter_on_history():
    sc = scope.establish("Forecast the fee for Clinic A for March 3, 2026.", pd.DataFrame({
        "visit_date": pd.date_range("2025-01-01", periods=10, freq="MS"), "clinic": ["Clinic A"] * 10,
        "fee": np.arange(10.0)}), {})
    assert sc["time"] is None and sc["target_period"]["month"] == 3


def test_the_estimate_states_its_exact_method_and_is_not_called_ml(monkeypatch):
    _today(monkeypatch, "2025-02-04")
    df = costs()
    out = orchestrator.ask("Forecast the material cost for next week.", df.head(5), {}, use_model=False)
    f = out["calls"][0]["result"]
    assert f["kind"] == "historical_estimate" and f["method"] == "the average of the 5 past weeks"
    assert f["estimate"] == pytest.approx(df.head(5)["material_cost"].mean())
    assert "**Estimated material cost for" in out["answer"] and "not a machine-learning prediction" in out["answer"]


def test_a_named_date_is_the_period_forecast(monkeypatch):
    df = timesheet()
    scripted(monkeypatch, LABOR_PLAN)
    out = orchestrator.ask("Forecast Rahul Sharma's labor cost for September 20, 2026.", df, {})
    f = out["calls"][0]["result"]
    assert f["next_period"] == "2026-09-20" and f["horizon"] == 8 and f["label"] == "labor cost"
    assert followup.requested_date("on 4 September 2026") == pd.Timestamp("2026-09-04")
    assert followup.requested_date("by 2026-09-04") == pd.Timestamp("2026-09-04")



# ------------------------------------------------- dataset-wide vs individual derived targets --
def shifts() -> pd.DataFrame:
    """4 employees x 10 days. Day 5: two employees did not record. Day 7: one invalid record (negative rate) -- 3 of 4
    valid, still a comparable day. Day 8: two invalid records (end before start) -- 2 of 4 valid, not comparable."""
    rows = []
    staff = [("A. Okafor", 24), ("B. Lindqvist", 28), ("C. Haddad", 31), ("D. Novak", 26)]
    for d in range(1, 11):
        for k, (name, rate) in enumerate(staff):
            if d == 5 and k >= 2:
                continue
            start, end = 8 + k % 2, 16 + (d + k) % 3
            r = rate
            if d == 7 and k == 0:
                r = -rate
            if d == 8 and k >= 2:
                start, end = end, start
            rows.append({"worker": name, "day": pd.Timestamp(f"2026-05-{d:02d}"), "start": f"{start:02d}:00",
                         "end": f"{end:02d}:30", "rate": r})
    return pd.DataFrame(rows)


SHIFT_PLAN = ['calc", "derive": {}}', '"labor_cost": "rate * hours_between(start, end)"}}', ']}', 'null}',
              '"labor_cost": "sum(labor_cost)"}}']


def _shift_truth(df, name=None):
    """Per-day total over valid records (no negative rate, end not before start); and, for all workers, only days
    where at least 75% of the usual number of workers recording per day have a valid record."""
    valid, recorded = {}, {}
    for r in df.to_dict("records"):
        if name and r["worker"] != name:
            continue
        day = r["day"].strftime("%Y-%m-%d")
        recorded.setdefault(day, set()).add(r["worker"])
        h = _hours(r["start"], r["end"])
        if r["rate"] < 0 or h < 0:
            continue
        valid.setdefault(day, [set(), 0.0])
        valid[day][0].add(r["worker"])
        valid[day][1] += h * r["rate"]
    if name:
        return {d: v[1] for d, v in valid.items()}, set()
    usual = float(np.median([len(v) for v in recorded.values()]))
    keep = {d: v[1] for d, v in valid.items() if len(v[0]) >= 0.75 * usual}
    return keep, set(recorded) - set(keep)


def test_dataset_wide_derived_daily_total_leaves_out_incomplete_days(monkeypatch):
    _today(monkeypatch, "2026-05-11")
    df = shifts()
    scripted(monkeypatch, SHIFT_PLAN)
    out = orchestrator.ask("Predict the next day's total labor cost across all workers.", df, {})
    f = out["calls"][0]["result"]
    truth, dropped = _shift_truth(df)
    assert f["label"] == "labor cost" and f["frequency"] == "day"
    assert {p["period"]: p["value"] for p in f["points"]} == pytest.approx(truth)
    assert {x["period"] for x in f["coverage"]["incomplete"]} == dropped == {"2026-05-05", "2026-05-08"}
    assert "2026-05-07" in {p["period"] for p in f["points"]}, "3 of 4 valid is still a comparable day"
    assert f["coverage"]["unit"] == "worker" and "Left out as incomplete periods" in out["answer"]
    assert f["next_period"] == "2026-05-11", "the day after the last RECORDED day, even though some days were left out"
    if f["kind"] == "historical_estimate":
        assert f["estimate"] == pytest.approx(sum(truth.values()) / len(truth)) or f["method"].startswith("the most recent")
    assert out["verification"][0]["ok"]


def test_individual_level_named_through_another_column_has_no_coverage_rule(monkeypatch):
    _today(monkeypatch, "2026-05-11")
    df = shifts().assign(worker_id=lambda d: d["worker"].map({"A. Okafor": "W1", "B. Lindqvist": "W2",
                                                              "C. Haddad": "W3", "D. Novak": "W4"}))
    scripted(monkeypatch, SHIFT_PLAN)
    out = orchestrator.ask("Predict B. Lindqvist's labor cost for the next day.", df, {})
    f = out["calls"][0]["result"]
    truth, _ = _shift_truth(df, "B. Lindqvist")
    assert f["coverage"] is None and {p["period"]: p["value"] for p in f["points"]} == pytest.approx(truth)


def test_individual_level_uses_one_workers_valid_history_with_no_coverage_rule(monkeypatch):
    _today(monkeypatch, "2026-05-11")
    df = shifts()
    scripted(monkeypatch, SHIFT_PLAN)
    out = orchestrator.ask("Predict A. Okafor's labor cost for the next day.", df, {})
    f = out["calls"][0]["result"]
    truth, _ = _shift_truth(df, "A. Okafor")
    assert {p["period"]: p["value"] for p in f["points"]} == pytest.approx(truth)
    assert f["coverage"] is None and "2026-05-07" not in {p["period"] for p in f["points"]}   # her invalid day


def deliveries() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    for d in range(1, 15):
        for drv in ("Ana", "Ben", "Chen"):
            if d == 9 and drv != "Ana":
                continue                                  # a day with only one of three drivers recorded
            rows.append({"driver": drv, "date": pd.Timestamp(f"2026-03-{d:02d}"),
                         "distance_km": float(rng.uniform(20, 120)), "rate_per_km": float(rng.choice([0.4, 0.55]))})
    return pd.DataFrame(rows)


def test_another_schema_and_metric_dataset_wide(monkeypatch):
    _today(monkeypatch, "2026-03-15")
    df = deliveries()
    scripted(monkeypatch, ['calc", "derive": {}}', '"delivery_cost": "distance_km * rate_per_km"}}', ']}', 'null}',
                           '"delivery_cost": "sum(delivery_cost)"}}'])
    out = orchestrator.ask("Forecast tomorrow's total delivery cost across all drivers.", df, {})
    f = out["calls"][0]["result"]
    d = df.assign(cost=df["distance_km"] * df["rate_per_km"])
    per_day = d.groupby(d["date"].dt.strftime("%Y-%m-%d"))["cost"].sum()
    truth = {k: v for k, v in per_day.items() if k != "2026-03-09"}
    assert f["label"] == "delivery cost" and {p["period"]: p["value"] for p in f["points"]} == pytest.approx(truth)
    assert [x["period"] for x in f["coverage"]["incomplete"]] == ["2026-03-09"]
    # 13 comparable days: enough to fit and test a trend; ML is claimed only if it beat the naive baseline
    assert (f["kind"] == "ml_prediction") == (f["validation"]["model_mae"] < f["validation"]["naive_mae"])


# ------------------------------------------------------------------- horizon --
def _trending():
    days = pd.date_range("2026-01-01", periods=30, freq="D")
    return pd.DataFrame({"date": days, "store": "S1", "orders": [100.0 + 4 * i for i in range(30)]})


def test_an_ambiguous_horizon_that_changes_a_trend_forecast_is_asked(monkeypatch):
    _today(monkeypatch, "2026-06-01")                     # records end in January
    out = orchestrator.ask("Forecast orders for the next day.", _trending(), {}, use_model=False)
    assert out["plan"]["type"] == "clarify" and "2026-01-31" in out["answer"] and "2026-06-02" in out["answer"]
    out = orchestrator.ask("Forecast orders for February 1, 2026.", _trending(), {}, use_model=False)
    f = out["calls"][0]["result"]
    assert f["kind"] == "ml_prediction" and f["prediction"] == pytest.approx(100 + 4 * 31)


def test_a_horizon_that_does_not_change_the_answer_is_stated_not_asked(monkeypatch):
    _today(monkeypatch, "2026-06-01")
    flat = _trending().assign(orders=50.0)
    out = orchestrator.ask("Forecast orders for the next day.", flat, {}, use_model=False)
    f = out["calls"][0]["result"]
    assert out["plan"]["type"] != "clarify" and "records end" in f["horizon_note"]
