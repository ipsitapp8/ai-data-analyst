"""Derived metrics are understood and computed without a language model, on any schema.

Every expected value is computed here in plain Python from the fixture rows, never by Silt. The fixtures use
different column names for the same concepts, so nothing can depend on one dataset. The model is made "available"
but any call fails the test: these answers must come from the deterministic metric layer.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from app.analyst import calc, metrics, ml, orchestrator, planner


# ------------------------------------------------------------------ fixtures --
def _hhmm(t: str) -> float:
    h, m = t.split(":")
    return int(h) + int(m) / 60


def shifts_a(seed: int = 7) -> pd.DataFrame:
    """Clock times as text; one negative rate, one end-before-start, one exact duplicate."""
    rnd = random.Random(seed)
    names = ["Mira Kovacs", "Tobias Lind", "Sana Iqbal", "Owen Pryce"]
    depts = {"Mira Kovacs": "Kitchen", "Tobias Lind": "Kitchen", "Sana Iqbal": "Front", "Owen Pryce": "Front"}
    rows = []
    for d in range(1, 11):
        for n in names:
            start = rnd.choice([7, 8, 9])
            end = start + rnd.choice([6, 7, 8, 9, 10])
            rows.append({"staff_name": n, "dept": depts[n], "shift_date": f"2026-03-{d:02d}",
                         "clock_in": f"{start:02d}:00", "clock_out": f"{end:02d}:{rnd.choice(['00', '30'])}",
                         "hourly_rate": {"Mira Kovacs": 21, "Tobias Lind": 19, "Sana Iqbal": 24, "Owen Pryce": 17}[n]})
    rows[5]["hourly_rate"] = -19                     # impossible
    rows[9]["clock_in"], rows[9]["clock_out"] = "18:00", "09:00"   # end before start
    rows.append(dict(rows[2]))                       # exact duplicate
    df = pd.DataFrame(rows)
    df["shift_date"] = pd.to_datetime(df["shift_date"])
    return df


def shifts_b() -> pd.DataFrame:
    """Same concept, different vocabulary: worker / team / start_time / end_time / pay_rate."""
    a = shifts_a()
    return a.rename(columns={"staff_name": "worker", "dept": "team", "shift_date": "day", "clock_in": "start_time",
                             "clock_out": "end_time", "hourly_rate": "pay_rate"})


def marketing() -> pd.DataFrame:
    rnd = random.Random(3)
    rows = []
    for m in range(1, 13):
        for ch in ["Search", "Social", "Email"]:
            v = rnd.randint(800, 2000)
            leads = rnd.randint(50, 200)
            conv = rnd.randint(5, 40)
            spend = round(rnd.uniform(500, 3000), 2)
            rows.append({"month": pd.Timestamp(2025, m, 1), "channel": ch, "visitors": v, "leads": leads,
                         "conversions": conv, "marketing_spend": spend, "revenue": round(conv * rnd.uniform(60, 140), 2)})
    return pd.DataFrame(rows)


def _labor_truth(df, name, start, end, rate, group=None, drop_dupes=False):
    """Valid records only: no negative rate, no end before start. Duplicates kept unless asked."""
    seen, out = set(), {}
    for rec in df.astype({c: str for c in df.columns if str(df[c].dtype).startswith("datetime")}).to_dict("records"):
        key = tuple(rec.values())
        if drop_dupes and key in seen:
            continue
        seen.add(key)
        h = _hhmm(rec[end]) - _hhmm(rec[start])
        if h < 0 or rec[rate] < 0:
            continue
        g = rec[group] if group else "all"
        out[g] = out.get(g, 0.0) + h * rec[rate]
    return out


@pytest.fixture
def no_model(monkeypatch):
    """The model is reachable, but calling it fails: answers must come from the deterministic layer."""
    def boom(*a, **k):
        raise AssertionError("the language model was called for a deterministic calculation")
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "See the table above.")   # narration only
    monkeypatch.setattr(calc, "_complete", boom)


def _ask(q, df, mem=None, use_model=False):
    return orchestrator.ask(q, df, mem if mem is not None else {}, use_model=use_model)


def _res(out):
    return next(c["result"] for c in out["calls"] if c["tool"] == "calculator")


# ------------------------------------------------------------- recognition --
@pytest.mark.parametrize("q,name", [
    ("What is the total labor cost by department?", "labor_cost"), ("labour costs per team", "labor_cost"),
    ("How much did we spend on payroll?", "labor_cost"), ("total hours worked by each employee", "shift_hours"),
    ("average shift length", "shift_hours"), ("overtime beyond 8 hours per worker", "overtime_hours"),
    ("profit by region", "profit"), ("what's the profit margin for each channel", "profit_margin"),
    ("Which channel has the highest ROI?", "roi"), ("return on ad spend by channel", "roas"),
    ("conversion rate per channel", "conversion_rate"), ("cost per acquisition", "cost_per_conversion"),
    ("CPL by channel", "cost_per_lead"), ("average order value by region", "average_order_value"),
    ("AOV", "average_order_value")])
def test_metric_wordings_are_recognised(q, name):
    assert metrics.recognise(q).name == name


@pytest.mark.parametrize("q", ["Show monthly sales as a line chart.", "Which region has the most units?",
                               "What is the average hourly rate?", "Explain this dataset", "Total revenue by channel"])
def test_ordinary_questions_are_not_taken_for_derived_metrics(q):
    assert metrics.recognise(q) is None


# ----------------------------------------------------- correctness on schemas --
@pytest.mark.parametrize("make,cols", [
    (shifts_a, ("staff_name", "dept", "clock_in", "clock_out", "hourly_rate")),
    (shifts_b, ("worker", "team", "start_time", "end_time", "pay_rate"))])
def test_labor_cost_by_group_on_different_schemas(make, cols, no_model):
    df = make()
    name, group, s, e, r = cols
    truth = _labor_truth(df, name, s, e, r, group=group)
    word = "department" if group == "dept" else "team"
    out = _ask(f"What is the total labor cost by {word}?", df, use_model=True)
    assert out["plan_source"] == "metric"
    res = _res(out)
    assert res["group_by"] == group
    assert {g["group"]: g["labor_cost"] for g in res["groups"]} == pytest.approx(truth)
    assert res["rows_excluded"] == 2, "the negative rate and the end-before-start record are left out and listed"
    assert "Data checks" in out["answer"] and "duplicate" in out["answer"], "the kept duplicate is flagged"


def test_an_individual_total_is_scoped_to_that_person(no_model):
    df = shifts_a()
    truth = _labor_truth(df[df["staff_name"] == "Sana Iqbal"], None, "clock_in", "clock_out", "hourly_rate")["all"]
    out = _ask("How much labor cost did Sana Iqbal generate?", df, use_model=True)
    assert _res(out)["totals"]["labor_cost"] == pytest.approx(truth)
    assert out["answer"].startswith("**Scope:** staff name = Sana Iqbal")


def test_excluding_a_person_leaves_them_out_rather_than_selecting_them(no_model):
    df = shifts_a()
    truth = _labor_truth(df[df["staff_name"] != "Owen Pryce"], None, "clock_in", "clock_out", "hourly_rate")["all"]
    out = _ask("Total labor cost excluding Owen Pryce", df, use_model=True)
    assert _res(out)["totals"]["labor_cost"] == pytest.approx(truth)
    assert "not Owen Pryce" in out["answer"]
    count = _ask("How many records are there for everyone except Owen Pryce?", df)
    assert f"**{int((df['staff_name'] != 'Owen Pryce').sum())}" in count["answer"]


def test_overtime_needs_a_stated_threshold_and_uses_it(no_model):
    df = shifts_a()
    ask = _ask("How much overtime did each employee work?", df, use_model=True)
    assert ask["plan"]["type"] == "clarify" and not ask["calls"] and "threshold" in ask["answer"]
    out = _ask("Overtime beyond 8 hours by staff name", df, use_model=True)
    truth = {}
    for rec in df.to_dict("records"):
        h = _hhmm(rec["clock_out"]) - _hhmm(rec["clock_in"])
        if h >= 0:
            truth[rec["staff_name"]] = truth.get(rec["staff_name"], 0.0) + max(h - 8, 0)
    assert {g["group"]: g["overtime_hours"] for g in _res(out)["groups"]} == pytest.approx(truth)


def test_overnight_shifts_only_when_asked():
    df = pd.DataFrame({"nurse": ["A", "A", "B"], "start": ["22:00", "08:00", "09:00"], "end": ["06:00", "16:00", "17:30"]})
    plain = _ask("Total hours worked by nurse", df)
    assert {g["group"]: g["shift_hours"] for g in _res(plain)["groups"]} == pytest.approx({"A": 8.0, "B": 8.5})
    assert "overnight" in plain["details"] or "earlier than" in plain["details"]
    night = _ask("Total hours worked by nurse, counting overnight shifts as the next day", df)
    assert {g["group"]: g["shift_hours"] for g in _res(night)["groups"]} == pytest.approx({"A": 16.0, "B": 8.5})


# ------------------------------------------------------------------ ratios --
def _sums(df, by):
    return df.groupby(by)[["visitors", "leads", "conversions", "marketing_spend", "revenue"]].sum()


@pytest.mark.parametrize("q,key,fn,pct", [
    ("conversion rate by channel", "conversion_rate", lambda s: s.conversions / s.visitors, True),
    ("profit margin by channel", "profit_margin", lambda s: (s.revenue - s.marketing_spend) / s.revenue, True),
    ("ROI by channel", "roi", lambda s: (s.revenue - s.marketing_spend) / s.marketing_spend, True),
    ("ROAS by channel", "roas", lambda s: s.revenue / s.marketing_spend, False),
    ("cost per conversion by channel", "cost_per_conversion", lambda s: s.marketing_spend / s.conversions, False),
    ("cost per lead by channel", "cost_per_lead", lambda s: s.marketing_spend / s.leads, False)])
def test_ratio_metrics_are_pooled_ratios_of_totals(q, key, fn, pct, no_model):
    df = marketing()
    s = _sums(df, "channel")
    out = _ask(q, df, use_model=True)
    res = _res(out)
    assert {g["group"]: g[key] for g in res["groups"]} == pytest.approx(fn(s).to_dict())
    assert res["primary"] == key
    assert ("%" in out["answer"].split("\n")[0]) == pct
    chart = out["charts"][0]
    assert chart["y"] == key.replace("_", " ") and chart["additive"] is False, "a ratio is not parts of a whole"


def test_a_zero_denominator_has_no_value_and_missing_inputs_are_not_zero():
    df = pd.DataFrame({"site": ["A", "A", "B", "B", "C"], "sessions": [100, 50, 0, 0, 40],
                       "signups": [5, 5, 0, 0, np.nan]})
    out = _ask("What is the conversion rate by site?", df)
    res = _res(out)
    groups = {g["group"]: g["conversion_rate"] for g in res["groups"]}
    assert groups["A"] == pytest.approx(10 / 150) and groups["B"] is None and "C" not in groups
    assert res["totals"]["conversion_rate"] == pytest.approx(10 / 150), "C's sessions are not counted without signups"
    assert "—" in out["answer"]


# --------------------------------------------------- asks instead of guessing --
def test_an_ambiguous_input_is_asked_about_and_a_named_one_is_used():
    df = pd.DataFrame({"team": ["x", "y"], "gross_sales": [100, 200], "net_sales": [90, 180], "cost": [50, 60]})
    ask = _ask("What is the profit by team?", df)
    assert ask["plan"]["type"] == "clarify" and "gross_sales" in ask["answer"] and "net_sales" in ask["answer"]
    out = _ask("What is the profit by team using net sales?", df)
    assert {g["group"]: g["profit"] for g in _res(out)["groups"]} == {"x": 40, "y": 120}


def test_a_missing_input_is_named_and_nothing_else_is_substituted():
    df = marketing().drop(columns=["visitors", "leads"])
    out = _ask("What is the conversion rate by channel?", df)
    assert out["plan"]["type"] == "clarify" and not out["calls"]
    assert "won't substitute" in out["answer"]


def test_average_order_value_needs_orders():
    rev = pd.DataFrame({"region": ["N", "S"], "sales": [100.0, 50.0]})
    out = _ask("Average order value by region", rev)
    assert out["plan"]["type"] == "clarify" and "order" in out["answer"] and not out["calls"]
    ids = pd.DataFrame({"order_id": ["o1", "o1", "o2", "o3"], "region": ["N", "N", "N", "S"],
                        "line_revenue": [10.0, 5.0, 30.0, 8.0]})
    res = _res(_ask("Average order value by region", ids))
    assert {g["group"]: g["average_order_value"] for g in res["groups"]} == pytest.approx({"N": 45 / 2, "S": 8.0})


def test_a_recorded_column_is_used_as_it_is_not_re_derived():
    df = pd.DataFrame({"region": ["N", "S"], "revenue": [10.0, 20.0], "cost": [4.0, 5.0], "profit": [6.5, 15.5]})
    assert metrics.plan("profit by region", df) is None


def test_pay_in_an_employee_table_is_salary_not_labor_cost():
    df = pd.DataFrame({"name": ["a", "b"], "salary": [100.0, 200.0], "dept": ["x", "y"]})
    assert metrics.plan("average pay by dept", df) is None


# ------------------------------------------------ conditions are never dropped --
def test_an_unknown_condition_is_not_silently_ignored(monkeypatch):
    df = shifts_a()
    off = _ask("Labor cost by dept excluding training shifts", df)
    assert off["plan"]["type"] == "clarify" and not off["calls"] and "training shifts" in off["answer"]
    called = []
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(calc, "model_plan", lambda *a, **k: (called.append(1), ({"type": "decline", "reason": "x"}, []))[1])
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: '{"type":"decline","reason":"no such column"}')
    on = _ask("Labor cost by dept excluding training shifts", df, use_model=True)
    assert called, "with the model available the validated calculator planner gets the question"
    assert on["plan"]["type"] == "clarify" and not on["calls"] and "couldn't apply" in on["answer"], \
        "when the model cannot apply it either, nothing else is answered instead"


def test_a_custom_formula_is_not_replaced_by_the_standard_one():
    out = _ask("Labor cost as hours times rate plus 10% overhead", shifts_a())
    assert out["plan"]["type"] == "clarify" and "overhead" in out["answer"]


# ------------------------------------------------------------- metamorphic --
def test_row_order_does_not_change_results():
    df = shifts_a()
    a = _res(_ask("Total labor cost by dept", df))
    b = _res(_ask("Total labor cost by dept", df.sample(frac=1, random_state=4).reset_index(drop=True)))
    assert {g["group"]: g["labor_cost"] for g in a["groups"]} == pytest.approx({g["group"]: g["labor_cost"] for g in b["groups"]})


def test_renaming_an_entity_moves_its_result_with_it():
    df = shifts_a()
    a = _res(_ask("Total hours worked by staff name", df))
    renamed = df.replace({"staff_name": {"Tobias Lind": "Zed Quill"}})
    b = _res(_ask("Total hours worked by staff name", renamed))
    ga = {g["group"]: g["shift_hours"] for g in a["groups"]}
    gb = {g["group"]: g["shift_hours"] for g in b["groups"]}
    assert gb["Zed Quill"] == pytest.approx(ga["Tobias Lind"]) and "Tobias Lind" not in gb


def test_an_added_exact_duplicate_is_flagged_and_excluded_on_request():
    df = shifts_a()
    base = _res(_ask("Total labor cost excluding duplicates", df))["totals"]["labor_cost"]
    more = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert _res(_ask("Total labor cost excluding duplicates", more))["totals"]["labor_cost"] == pytest.approx(base)
    kept = _res(_ask("Total labor cost", more))
    row = df.iloc[0]
    assert kept["totals"]["labor_cost"] == pytest.approx(
        _res(_ask("Total labor cost", df))["totals"]["labor_cost"] + (_hhmm(row.clock_out) - _hhmm(row.clock_in)) * row.hourly_rate)


def test_scaling_money_does_not_change_ratios_and_hours_columns_match_clock_times():
    df = marketing()
    a = _res(_ask("profit margin by channel", df))
    scaled = df.assign(revenue=df.revenue * 1000, marketing_spend=df.marketing_spend * 1000)
    b = _res(_ask("profit margin by channel", scaled))
    assert {g["group"]: g["profit_margin"] for g in a["groups"]} == pytest.approx({g["group"]: g["profit_margin"] for g in b["groups"]})
    s = shifts_a()
    s = s[[(_hhmm(o) - _hhmm(i)) >= 0 and r >= 0 for i, o, r in zip(s.clock_in, s.clock_out, s.hourly_rate)]]
    hrs = s.assign(hours_worked=[_hhmm(o) - _hhmm(i) for i, o in zip(s.clock_in, s.clock_out)]).drop(columns=["clock_in", "clock_out"])
    t1 = _res(_ask("Total labor cost", s))["totals"]["labor_cost"]
    t2 = _res(_ask("Total labor cost", hrs))["totals"]["labor_cost"]
    assert t1 == pytest.approx(t2), "a recorded hours column and the clock times it came from give the same cost"


# ------------------------------------------------------------- follow-ups --
def test_follow_ups_refine_the_previous_calculation_without_a_model(no_model):
    df = shifts_a()
    mem = {}
    _ask("What is the total labor cost by department?", df, mem, use_model=True)
    only = _ask("Only Mira Kovacs.", df, mem, use_model=True)
    assert only["plan_source"] == "refine"
    mira = df[df["staff_name"] == "Mira Kovacs"]
    assert _res(only)["totals"]["labor_cost"] == pytest.approx(_labor_truth(mira, None, "clock_in", "clock_out", "hourly_rate")["all"])
    dedup = _ask("Exclude duplicates.", df, mem, use_model=True)
    assert _res(dedup)["totals"]["labor_cost"] == pytest.approx(
        _labor_truth(mira, None, "clock_in", "clock_out", "hourly_rate", drop_dupes=True)["all"])
    pie = _ask("Show the same result as a pie chart.", df, mem, use_model=True)
    # one person's total has no parts by group: the chart shows the records it was computed from
    per = {d["shift_date"]: d["labor cost"] for d in pie["charts"][0]["data"]}
    mira_rows = mira[(mira.clock_out.map(_hhmm) >= mira.clock_in.map(_hhmm)) & (mira.hourly_rate >= 0)]
    assert sorted(per.values()) == pytest.approx(sorted(
        (_hhmm(o) - _hhmm(i)) * r for i, o, r in zip(mira_rows.clock_in, mira_rows.clock_out, mira_rows.hourly_rate)))
    assert "each of the" in pie["answer"]
    what_if = _ask("What if the rate increases by 10%?", df, mem, use_model=True)
    assert "hourly rate" in what_if["answer"] and "+10%" in what_if["answer"]


def test_a_grouped_result_is_drawn_as_a_pie_from_the_same_numbers(no_model):
    df = shifts_a()
    mem = {}
    first = _ask("Total labor cost by dept", df, mem, use_model=True)
    pie = _ask("Show that as a pie chart.", df, mem, use_model=True)
    assert pie["charts"][0]["type"] == "pie"
    # drawing the result again does not replace it: a follow-up still refines the calculation
    only = _ask("Only Mira Kovacs.", df, mem, use_model=True)
    assert only["plan_source"] == "refine"
    assert {d["dept"]: d["labor cost"] for d in pie["charts"][0]["data"]} == pytest.approx(
        {g["group"]: g["labor_cost"] for g in _res(first)["groups"]})


def test_a_refinement_does_not_apply_after_a_different_analysis():
    df = shifts_a()
    mem = {}
    _ask("Total labor cost by dept", df, mem)
    rank = _ask("Show the top 2 staff names by hourly rate.", df, mem)        # a ranking, not a calculation
    assert rank["calls"] and rank["calls"][0]["tool"] == "rank_groups"
    assert mem["last_analysis"] == "other"
    out = _ask("Exclude duplicates.", df, mem)
    assert out["plan_source"] != "refine"


# --------------------------------------------------------- forecasts and ML --
def test_a_derived_total_is_forecast_without_the_model(monkeypatch):
    from app.analyst import scope
    monkeypatch.setattr(scope, "today", lambda: pd.Timestamp("2026-03-11"))
    df = shifts_a()
    out = _ask("Predict the next day's total labor cost across all employees.", df)
    f = out["calls"][0]["result"]
    assert f["label"] == "labor cost" and "hourly rate" in f["definition"]
    assert "machine-learning" in out["answer"] or "historical estimate" in out["answer"] or "Predicted" in out["answer"]


def test_a_ratio_is_not_forecast_as_if_it_added_up(monkeypatch):
    out = _ask("Forecast next month's conversion rate.", marketing())
    assert out["plan"]["type"] == "clarify" and "ratio" in out["answer"] and not out["calls"]


def test_ml_is_judged_by_cross_validation_against_a_baseline():
    rnd = np.random.default_rng(1)
    x = rnd.normal(size=200)
    # a real but imperfect relation (correlation about 0.9: below the leakage screen's 0.98)
    signal = pd.DataFrame({"x": x, "z": rnd.normal(size=200), "y": 3 * x + rnd.normal(scale=1.5, size=200)})
    r = ml.train(signal, "y", "regression")
    cv = r["metrics"]["cross_validation"]
    assert cv["folds"] == 5 and cv["folds_better"] == 5 and r["beats_baseline"]
    noise = signal.assign(y=rnd.normal(size=200))
    r2 = ml.train(noise, "y", "regression")
    assert not r2["beats_baseline"] and "does not clearly beat" in r2["limitations"][0]


def test_records_are_counted_per_group_without_a_model(no_model):
    df = shifts_a()
    out = _ask("Which dept has the most shifts?", df, use_model=True)
    res = _res(out)
    assert {g["group"]: g["number_of_shifts"] for g in res["groups"]} == df["dept"].value_counts().to_dict()
    assert "Each record (row) is counted as one shift" in " ".join(out["notes"])
    people = _res(_ask("Which dept has the most staff?", df, use_model=True))
    assert {g["group"]: g["number_of_staff"] for g in people["groups"]} == df.groupby("dept")["staff_name"].nunique().to_dict()
