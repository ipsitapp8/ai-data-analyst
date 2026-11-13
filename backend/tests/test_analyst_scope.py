"""Analytical scope: the people, categories and periods a question names are established once and every path --
calculation, trend, forecast, scenario, ranking, routed tools, the trained planner -- works on exactly those records.
Record validity for historical analysis is judged from the columns' meaning. Answers are checked against the scope.

Generalisation: four unrelated schemas (timesheet, retail sales, clinic visits, HR), none special-cased in the code.
Expected values are computed from the fixture rows in plain Python/pandas, never copied from an answer.
Metamorphic: changing an unrelated record must not change a scoped answer; adding a duplicate must not inflate a
deduplicated history; changing the requested target must change the computation. No LLM is called (stubbed or off).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import calc, followup, orchestrator, planner, scope

TODAY = pd.Timestamp("2026-11-12")


@pytest.fixture(autouse=True)
def fixed_today_and_no_model(monkeypatch):
    monkeypatch.setattr(scope, "today", lambda: TODAY)
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))


# ------------------------------------------------------------------- fixtures --
def timesheet() -> pd.DataFrame:
    rows = []
    people = [("E1", "Rahul Sharma", "Engineering", 25), ("E2", "Aarav Mehta", "Engineering", 30),
              ("E3", "Priya Singh", "HR", 22), ("E4", "Neha Kapoor", "Finance", 28)]
    for d in range(1, 11):
        for k, (eid, name, dept, rate) in enumerate(people):
            start = 8 + (d + k) % 3
            rows.append({"employee_id": eid, "employee_name": name, "department": dept,
                         "date": pd.Timestamp(f"2026-09-{d:02d}"), "clock_in": f"{start:02d}:00",
                         "clock_out": f"{start + 7 + (d * k) % 3:02d}:30", "hourly_rate": rate})
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)                    # an exact duplicate (Rahul, day 1)
    df.loc[len(df)] = ["E4", "Neha Kapoor", "Finance", pd.Timestamp("2026-09-05"), "09:00", "17:00", -10]   # invalid rate
    df.loc[len(df)] = ["E1", "Rahul Sharma", "Engineering", pd.Timestamp("2026-12-01"), "09:00", "17:00", 25]  # future
    df.loc[len(df)] = ["E1", "Rahul Sharma", "Engineering", pd.Timestamp("2026-09-11"), "18:00", "09:00", 25]  # end<start
    return df


def sales() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for m in range(12):
        for region in ("North", "South", "East", "West"):
            for product in ("Outdoor", "Kitchen", "Garden"):
                rows.append({"month": pd.Timestamp("2025-01-01") + pd.DateOffset(months=m), "region": region,
                             "product_line": product, "units_sold": int(rng.integers(50, 150)),
                             "sales": round(float(rng.uniform(1000, 5000)), 2)})
    return pd.DataFrame(rows)


def clinic() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for i in range(90):
        rows.append({"visit_date": pd.Timestamp("2025-01-06") + pd.Timedelta(days=int(i * 4)),
                     "clinic": ["Clinic A", "Clinic B", "Clinic C"][i % 3],
                     "doctor": ["Dr Osei", "Dr Lindqvist", "Dr Haddad", "Dr Novak"][i % 4],
                     "fee": round(float(rng.uniform(40, 160)), 2), "minutes": int(rng.integers(10, 60))})
    return pd.DataFrame(rows)


def _hours(a: str, b: str) -> float:
    h1, m1 = map(int, a.split(":"))
    h2, m2 = map(int, b.split(":"))
    return (h2 * 60 + m2 - h1 * 60 - m1) / 60


LABOR = {"type": "calc", "derive": {"labor_cost": "hourly_rate * hours_between(clock_in, clock_out)"}, "exclude": [],
         "group_by": "department", "aggregate": {"labor_cost": "sum(labor_cost)"}}


def _mem_with_labor(df):
    return {"last_calc": {"question": "labor cost by department", "spec": LABOR, "result": calc.run(LABOR, df)}}


def _valid_history(df, name=None):
    """Per-day labor cost over valid history, independently: no exact duplicate, no negative rate, clock-out not
    before clock-in, not after today; optionally for one person."""
    seen, per_day = set(), {}
    for r in df.to_dict("records"):
        key = tuple(map(str, r.values()))
        dup = key in seen
        seen.add(key)
        h = _hours(r["clock_in"], r["clock_out"])
        if dup or r["hourly_rate"] < 0 or h < 0 or r["date"] > TODAY or (name and r["employee_name"] != name):
            continue
        k = r["date"].strftime("%Y-%m-%d")
        per_day[k] = per_day.get(k, 0.0) + h * r["hourly_rate"]
    return per_day


# ------------------------------------------------------- establishing the scope --
@pytest.mark.parametrize("make,question,expect", [
    (timesheet, "Show the trend of labor cost for Rahul Sharma.", {"employee_name": ["Rahul Sharma"]}),
    (timesheet, "How did Priya do?", {"employee_name": ["Priya Singh"]}),                 # first name only
    (sales, "Monthly sales in the West region", {"region": ["West"]}),
    (sales, "Compare Kitchen and Garden in North", {"product_line": ["Kitchen", "Garden"], "region": ["North"]}),
    (clinic, "Average fee at Clinic B for Dr Novak", {"clinic": ["Clinic B"], "doctor": ["Dr Novak"]}),
    (clinic, "Which clinic has the highest average fee?", {}),
])
def test_named_entities_are_read_from_the_data_on_any_schema(make, question, expect):
    assert scope.establish(question, make(), {})["entities"] == expect


def test_a_named_period_becomes_a_time_filter_only_when_there_is_a_date_column():
    sc = scope.establish("Total fee in March 2025", clinic(), {})
    assert sc["time"] == {"column": "visit_date", "year": 2025, "month": 3}
    assert scope.establish("sales by month", sales(), {})["time"] is None
    assert scope.establish("Total in March 2025", pd.DataFrame({"a": [1.0]}), {})["time"] is None


def test_a_value_that_names_two_columns_is_asked_about_not_guessed():
    df = pd.DataFrame({"overtime": ["Yes", "No"] * 10, "remote": ["No", "Yes"] * 10, "pay": np.arange(20.0)})
    sc = scope.establish("average pay for Yes", df, {})
    assert sc["clarify"] and "overtime" in sc["clarify"] and "remote" in sc["clarify"]
    assert scope.establish("average pay for overtime Yes", df, {})["entities"] == {"overtime": ["Yes"]}
    # in a share/rate question a binary value is the event measured, not a filter
    assert scope.establish("What share of employees have overtime Yes?", df, {})["entities"] == {}


def test_follow_ups_inherit_the_scope_until_the_question_widens_it():
    df = timesheet()
    prev = {"scope": scope.establish("Rahul Sharma's labor cost", df, {})}
    assert scope.establish("And excluding duplicates, what is it?", df, prev)["inherited"] is True
    assert scope.establish("What about everyone?", df, prev)["entities"] == {}
    assert scope.establish("And for Priya?", df, prev)["entities"] == {"employee_name": ["Priya Singh"]}


# ------------------------------------------------------ every path keeps the scope --
def test_trend_for_a_named_person_uses_only_their_valid_history():
    df = timesheet()
    out = orchestrator.ask("Show the trend of labor cost for Rahul Sharma.", df, _mem_with_labor(df), use_model=False)
    pts = out["calls"][0]["result"]["points"]
    assert {p["period"]: p["value"] for p in pts} == pytest.approx(_valid_history(df, "Rahul Sharma"))
    assert out["answer"].startswith("**Scope:** employee name = Rahul Sharma")


def test_forecast_for_a_named_person_uses_only_their_valid_history():
    df = timesheet()
    out = orchestrator.ask("Predict Rahul Sharma's labor cost for next week.", df, _mem_with_labor(df), use_model=False)
    f = out["calls"][0]["result"]
    hist = _valid_history(df, "Rahul Sharma")
    weekly = {}                                                    # "next week": weekly totals, computed here
    for day, v in hist.items():
        wk = str(pd.Period(day, freq="W"))
        weekly[wk] = weekly.get(wk, 0.0) + v
    assert f["frequency"] == "week"
    assert [p["value"] for p in f["points"]] == pytest.approx([weekly[k] for k in sorted(weekly)])
    assert f["ml_used"] is True or f["kind"] == "historical_estimate"
    assert "Rahul Sharma" in f["scope"]


def test_scenario_for_a_named_person_changes_only_their_records():
    df = timesheet()
    out = orchestrator.ask("What if Rahul Sharma's hourly rate increases by 10%?", df, _mem_with_labor(df),
                           use_model=False)
    s = out["calls"][0]["result"]
    rahul = df[df["employee_name"] == "Rahul Sharma"]
    base = sum(_hours(a, b) * r for a, b, r in zip(rahul["clock_in"], rahul["clock_out"], rahul["hourly_rate"]))
    assert s["results"][0]["baseline"] == pytest.approx(base)
    assert s["results"][0]["scenario"] == pytest.approx(base * 1.1)


@pytest.mark.parametrize("make,question,col,val,measure,date", [
    (sales, "Show the monthly trend of sales in the West region.", "region", "West", "sales", "month"),
    (clinic, "Show the monthly trend of fee for Clinic B.", "clinic", "Clinic B", "fee", "visit_date"),
])
def test_routed_trend_uses_only_the_named_group(make, question, col, val, measure, date):
    df = make()
    out = orchestrator.ask(question, df, {}, use_model=False)
    sub = df[df[col] == val]
    truth = sub.groupby(sub[date].dt.to_period("M"))[measure].sum()
    got = {d["period"]: d[measure] for d in out["charts"][0]["data"]}
    assert got == pytest.approx({str(k): v for k, v in truth.items()})


def test_ranking_within_a_named_group_and_period():
    df = clinic()
    out = orchestrator.ask("Show the top 2 doctors by fee in Clinic A.", df, {}, use_model=False)
    sub = df[df["clinic"] == "Clinic A"]
    truth = sub.groupby("doctor")["fee"].sum().nlargest(2)
    assert [(r["group"], r["value"]) for r in out["calls"][0]["result"]["rows"]] == \
        [(k, pytest.approx(v)) for k, v in truth.items()]
    out = orchestrator.ask("Show the top 2 doctors by fee in March 2025.", df, {}, use_model=False)
    sub = df[(df["visit_date"].dt.year == 2025) & (df["visit_date"].dt.month == 3)]
    truth = sub.groupby("doctor")["fee"].sum().nlargest(2)
    assert [r["group"] for r in out["calls"][0]["result"]["rows"]] == list(truth.index)


def test_the_trained_planner_path_also_works_on_the_scoped_records():
    df = clinic()
    out = orchestrator.ask("What is the average fee in Clinic C?", df, {}, use_model=False)
    vals = [c["result"].get("value") for c in out["calls"] if isinstance(c.get("result"), dict) and "value" in c["result"]]
    assert vals and vals[0] == pytest.approx(df.loc[df["clinic"] == "Clinic C", "fee"].mean())


def test_a_calculation_follow_up_keeps_a_scope_set_by_a_different_path(monkeypatch):
    """Rahul's scope comes from a trend answer (not the calculator). The follow-up names nobody, so the calculator's
    own named-value check has nothing to enforce; the model's plan totals everyone. The shared scope adds the filter."""
    df = timesheet()
    mem = _mem_with_labor(df)
    orchestrator.ask("Show the trend of labor cost for Rahul Sharma.", df, mem, use_model=False)
    assert mem["scope"]["entities"] == {"employee_name": ["Rahul Sharma"]}
    monkeypatch.setattr(planner, "model_available", lambda: True)
    plan = iter(['calc", "derive": {}}', '"labor_cost": "hourly_rate * hours_between(clock_in, clock_out)"}}', ']}',
                 'null}', '"total": "sum(labor_cost)"}}'])
    monkeypatch.setattr(calc, "_complete", lambda p, m: next(plan))
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "The total labor cost is shown above.")
    out = orchestrator.ask("And what is his total labor cost?", df, mem)
    res = next(c["result"] for c in out["calls"] if c["tool"] == "calculator")
    rahul = df[df["employee_name"] == "Rahul Sharma"]
    # labor cost is a standard metric, computed deterministically: held to Rahul's scope, with impossible records
    # (clock-out before clock-in, a negative rate) left out and listed -- recomputed here independently
    truth = sum(_hours(a, b) * r for a, b, r in zip(rahul["clock_in"], rahul["clock_out"], rahul["hourly_rate"])
                if _hours(a, b) >= 0 and r >= 0)
    assert out["plan_source"] == "metric"
    assert res["totals"]["labor_cost"] == pytest.approx(truth)
    assert any("kept the question's scope" in n for n in out["notes"])
    assert out["verification"][0]["ok"] and "carried over" in out["verification"][0]["detail"]


def test_holding_a_plan_to_its_scope_adds_only_the_questions_own_filters():
    df = clinic()
    sc = scope.establish("Total fee for Clinic B in March 2025", df, {})
    spec = {"type": "calc", "derive": {}, "exclude": [], "group_by": None, "aggregate": {"total": "sum(fee)"}}
    notes: list[str] = []
    assert orchestrator._hold_to_scope(spec, df, sc, notes) == []  # noqa: SLF001
    sub = df[(df["clinic"] == "Clinic B") & (df["visit_date"].dt.month == 3) & (df["visit_date"].dt.year == 2025)]
    assert calc.run(spec, df)["totals"]["total"] == pytest.approx(sub["fee"].sum())
    assert notes and len(spec["exclude"]) == 2


def test_an_empty_scope_is_reported_never_widened_to_all_data():
    df = clinic()
    out = orchestrator.ask("Show the top 2 doctors by fee in March 2024.", df, {}, use_model=False)
    assert "No records match" in out["answer"] and not out["calls"]


def test_predicting_one_case_names_inputs_not_filters():
    from app.analyst import tools
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"department": rng.choice(["Sales", "Support"], 200), "overtime": rng.choice(["Yes", "No"], 200),
                       "age": rng.integers(21, 60, 200)})
    df["attrition"] = np.where((df["overtime"] == "Yes") & (rng.random(200) < 0.6), "Yes", "No")
    mem = {}
    tools.run({"tool": "random_forest", "args": {"target": "attrition"}}, df, mem)
    orchestrator.ask("Predict attrition for someone in Sales, overtime Yes, age 30.", df, mem, use_model=False)
    assert mem.get("scope") is None, "the inputs of a single prediction were not turned into a record filter"


# ------------------------------------------------------------ record validity --
def test_validity_follows_the_meaning_of_the_columns_used():
    df = timesheet()
    keep, why = scope.eligibility(df, ["hourly_rate", "clock_in", "clock_out", "date"], historical=True)
    reasons = {w["reason"].split(" (")[0] for w in why}
    assert reasons == {"exact duplicate of an earlier record", "negative hourly rate", "clock out earlier than clock in",
                       "dated after today"}
    assert int((~keep).sum()) == 4
    # the same records are fine for an analysis that does not use those columns or is not historical
    keep2, why2 = scope.eligibility(df, ["employee_name"], historical=False)
    assert [w["reason"] for w in why2] == ["exact duplicate of an earlier record"]


def test_zero_is_a_value_missing_is_not_and_outliers_stay():
    df = pd.DataFrame({"units": [0, 5, None, 7, 9000], "price": [1.0, 2.0, 3.0, 4.0, 5.0]})
    keep, why = scope.eligibility(df, ["units"], historical=False)
    assert list(keep) == [True, True, False, True, True]
    assert why == [{"reason": "missing units", "records": 1, "lines": [4]}]


def test_future_dated_records_are_flagged_in_a_calculation():
    df = timesheet()
    r = calc.run(LABOR, df)
    future = [c for c in r["data_checks"] if "after today" in c["issue"]]
    assert future and future[0]["lines"] == [int(i) + 2 for i in df.index[df["date"] > TODAY]]


# --------------------------------------------------------------- metamorphic --
def _perturb_others(df, name):
    out = df.copy()
    others = out["employee_name"] != name
    out.loc[others, "hourly_rate"] = out.loc[others, "hourly_rate"] * 3 + 7
    return out


@pytest.mark.parametrize("question", ["Show the trend of labor cost for Rahul Sharma.",
                                      "Predict Rahul Sharma's labor cost for next week.",
                                      "What if Rahul Sharma's hourly rate increases by 10%?"])
def test_changing_other_peoples_records_does_not_change_a_person_specific_answer(question):
    df = timesheet()
    a = orchestrator.ask(question, df, _mem_with_labor(df), use_model=False)["calls"][0]["result"]
    df2 = _perturb_others(df, "Rahul Sharma")
    b = orchestrator.ask(question, df2, _mem_with_labor(df2), use_model=False)["calls"][0]["result"]
    key = "points" if "points" in a else "results"
    assert a[key] == b[key]


def test_changing_other_groups_does_not_change_a_group_scoped_ranking():
    df = clinic()
    q = "Show the top 2 doctors by fee in Clinic A."
    a = orchestrator.ask(q, df, {}, use_model=False)["calls"][0]["result"]["rows"]
    df2 = df.copy()
    df2.loc[df2["clinic"] != "Clinic A", "fee"] *= 10
    b = orchestrator.ask(q, df2, {}, use_model=False)["calls"][0]["result"]["rows"]
    assert a == b


def test_adding_a_duplicate_does_not_inflate_deduplicated_history():
    df = timesheet()
    q = "Show the trend of labor cost for Rahul Sharma."
    a = orchestrator.ask(q, df, _mem_with_labor(df), use_model=False)["calls"][0]["result"]["points"]
    df2 = pd.concat([df, df[df["employee_name"] == "Rahul Sharma"].iloc[[3]]], ignore_index=True)
    b = orchestrator.ask(q, df2, _mem_with_labor(df2), use_model=False)["calls"][0]["result"]["points"]
    assert [p["value"] for p in a] == pytest.approx([p["value"] for p in b])


def test_changing_the_requested_target_changes_the_computation():
    df = clinic()
    fee = orchestrator.ask("Show the top 2 doctors by fee in Clinic A.", df, {}, use_model=False)["calls"][0]["result"]
    mins = orchestrator.ask("Show the top 2 doctors by minutes in Clinic A.", df, {}, use_model=False)["calls"][0]["result"]
    assert fee["measure"] == "fee" and mins["measure"] == "minutes"
    sub = df[df["clinic"] == "Clinic A"]
    assert [r["value"] for r in mins["rows"]] == pytest.approx(list(sub.groupby("doctor")["minutes"].sum().nlargest(2)))


def test_a_forecast_of_an_unknown_target_is_not_replaced_by_another_measure():
    df = timesheet()
    out = orchestrator.ask("Forecast the overtime bonus for next week.", df, _mem_with_labor(df), use_model=False)
    assert out["plan"]["type"] == "clarify" and "overtime bonus" in out["answer"] and not out["calls"]
    assert "won't forecast a different quantity" in out["answer"]
    assert followup.forecast_target("Predict the next value.", df, _mem_with_labor(df), None)["kind"] == "measure"
    assert followup.forecast_target("Forecast minutes next month", clinic(), {}, None) == \
        {"kind": "column", "column": "minutes", "phrase": "minutes"}


def test_the_answer_records_its_scope_checks():
    df = clinic()
    out = orchestrator.ask("Show the top 2 doctors by fee in Clinic A.", df, {}, use_model=False)
    v = out["verification"][0]
    assert v["ok"] and v["detail"].startswith("clinic = Clinic A — 30 of 90 records")
    assert "Checks before answering" in out["details"]


# ---------------------------------------------- defects found by the real-model probe --
def _person_calc_mem(df, name):
    spec = {**LABOR, "group_by": None, "exclude": [{"reason": f"only {name}", "where": f"employee_name != '{name}'"}]}
    return {"last_calc": {"question": f"{name}'s labor cost", "spec": spec, "result": calc.run(spec, df)},
            "scope": scope.establish(f"{name}'s labor cost", df, {})}


def test_a_new_person_replaces_the_previous_persons_filter_in_a_scenario():
    df = timesheet()
    mem = _person_calc_mem(df, "Rahul Sharma")
    out = orchestrator.ask("What if Aarav Mehta's hourly rate increases by 10%?", df, mem, use_model=False)
    s = out["calls"][0]["result"]
    aarav = df[df["employee_name"] == "Aarav Mehta"]
    base = sum(_hours(a, b) * r for a, b, r in zip(aarav["clock_in"], aarav["clock_out"], aarav["hourly_rate"]))
    assert s["results"][0]["baseline"] == pytest.approx(base) and s["rows_used"] == len(aarav)


def test_a_trend_for_another_person_is_a_trend_of_that_person():
    df = timesheet()
    mem = _person_calc_mem(df, "Priya Singh")
    out = orchestrator.ask("Show the trend of labor cost for Rahul Sharma.", df, mem, use_model=False)
    pts = out["calls"][0]["result"]["points"]
    assert out["calls"][0]["tool"] == "trend_of_measure"
    assert {p["period"]: p["value"] for p in pts} == pytest.approx(_valid_history(df, "Rahul Sharma"))


def test_a_result_over_no_records_in_scope_is_not_shown_as_an_answer():
    df = timesheet()
    df.loc[len(df)] = ["E9", "Ghost Writer", "Finance", pd.Timestamp("2026-09-03"), "09:00", "17:00", -5]
    spec = {**LABOR, "exclude": [{"reason": "negative rate", "where": "hourly_rate < 0"}]}
    mem = {"last_calc": {"question": "labor cost without negative rates", "spec": spec, "result": calc.run(spec, df)}}
    out = orchestrator.ask("What if Ghost Writer's hourly rate increases by 10%?", df, mem, use_model=False)
    assert "couldn't answer this within the records you asked about" in out["answer"]
    assert out["verification"][0]["ok"] is False and not out["charts"]


@pytest.mark.parametrize("question,args,expect", [
    ("What is the attrition rate in Sales?", {"aggregation": "count", "column": None}, "rate or share"),
    ("What is the average fee per visit?", {"aggregation": "sum", "column": "fee"}, "average"),
    ("What is the total fee?", {"aggregation": "avg", "column": "fee"}, "total"),
    ("What is the total minutes?", {"aggregation": "sum", "column": "fee"}, "names minutes"),
    ("What is the total fee?", {"aggregation": "sum", "column": "fee"}, None),
    ("How many visits were there?", {"aggregation": "count", "column": None}, None),
])
def test_a_computed_metric_must_be_the_one_asked_for(question, args, expect):
    from app.analyst import plan_checks
    got = plan_checks.metric_mismatch(question, [{"tool": "compute_metric", "status": "ok", "args": args}], clinic())
    assert (got is None) if expect is None else (expect in got)


def test_a_mismatched_metric_is_not_shown_when_it_cannot_be_recomputed(monkeypatch):
    from app.analyst import tools
    df = clinic()
    monkeypatch.setattr(planner, "rule_plan", lambda m, d, mem: {"type": "tools", "calls": [
        {"tool": "compute_metric", "args": {"aggregation": "count"}}]})
    monkeypatch.setattr(calc, "wanted", lambda *a, **k: None)
    out = orchestrator.ask("What share of visits were at Clinic B?", df, {}, use_model=False)
    assert "couldn't compute what you asked for" in out["answer"] and "rate or share" in out["answer"]


def _hr_people(n=200, seed=5):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"department": rng.choice(["Sales", "Support", "Finance"], n),
                       "overtime": rng.choice(["Yes", "No"], n), "leads": rng.integers(10, 50, n)})
    df["attrition"] = np.where(rng.random(n) < np.where(df["overtime"] == "Yes", 0.4, 0.1), "Yes", "No")
    df["conversions"] = (df["leads"] * 0.2).astype(int)
    return df


def test_a_rate_made_only_of_counts_is_repaired_to_count_the_event(monkeypatch):
    df = _hr_people()
    monkeypatch.setattr(planner, "model_available", lambda: True)
    plan = iter(['calc", "derive": {}}', '}', '{"reason": "not Sales", "where": "department != \'Sales\'"}]}', 'null}',
                 '"attrition_rate": "count(attrition) / count(*)"}}'])
    monkeypatch.setattr(calc, "_complete", lambda p, m: next(plan))
    spec, trace = calc.model_plan("What is the attrition rate in the Sales department?", df, {})
    assert spec["aggregate"]["attrition_rate"] == "mean(attrition == 'Yes')"
    r = calc.run(spec, df)
    sales = df[df["department"] == "Sales"]
    assert r["totals"]["attrition_rate"] == pytest.approx((sales["attrition"] == "Yes").mean())
    assert 0 < r["totals"]["attrition_rate"] < 1


def test_a_ratio_of_two_different_sums_is_a_legitimate_rate():
    df = _hr_people()
    frame = calc.run_derive(df, {})
    probs = calc._check_section("aggregate", {"conversion_rate": "sum(conversions) / sum(leads)"},  # noqa: SLF001
                                {"derive": {}, "exclude": [], "group_by": "department"}, frame,
                                "What is the conversion rate by department?", "What is the conversion rate by department?")
    assert not any("always 100%" in p for p in probs)
