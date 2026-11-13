"""Silt Copilot: plan repair, follow-up context, conversational answers, trends, forecasts, scenarios, rankings and
chart data. Expected values are computed independently (plain Python over the CSV text); no real LLM is called --
the model is either switched off or replaced by a scripted stub."""
from __future__ import annotations

import csv
import io
import json

import numpy as np
import pandas as pd
import pytest

from app.analyst import calc, followup, orchestrator, overview, planner, present

TIMESHEET = """employee_id,employee_name,department,date,clock_in,clock_out,hourly_rate
EMP-0101,Rahul Sharma,Engineering,2026-09-01,09:00,17:00,25
EMP-0102,Aarav Mehta,Engineering,2026-09-01,09:00,18:30,30
EMP-0103,Priya Singh,HR,2026-09-01,09:30,17:30,22
EMP-0101,Rahul Sharma,Engineering,2026-09-02,09:00,19:00,25
EMP-0102,Aarav Mehta,Engineering,2026-09-02,10:00,18:00,30
EMP-0103,Priya Singh,HR,2026-09-02,09:00,17:00,22
EMP-0104,Vikram Rao,Finance,2026-09-03,08:00,20:00,28
EMP-0105,Neha Kapoor,Finance,2026-09-03,09:00,17:00,-10
EMP-0106,Ananya Das,Marketing,2026-12-01,09:00,17:00,20
EMP-0107,Karan Verma,Marketing,2026-09-03,10:00,09:00,24
EMP-0101,Rahul Sharma,Engineering,2026-09-01,09:00,17:00,25
"""


def _df() -> pd.DataFrame:
    from app.analyst.data import load_table
    p = __import__("pathlib").Path(__import__("tempfile").mkdtemp()) / "timesheet.csv"
    p.write_text(TIMESHEET)
    return load_table(str(p))


def _rows():
    return list(csv.DictReader(io.StringIO(TIMESHEET)))


def _hours(r) -> float:
    h1, m1 = map(int, r["clock_in"].split(":"))
    h2, m2 = map(int, r["clock_out"].split(":"))
    return (h2 * 60 + m2 - h1 * 60 - m1) / 60


def _truth(name: str | None = None, dedupe=False):
    """Labor cost per department and total, in plain Python, optionally for one employee / without duplicates."""
    by, seen = {}, set()
    for r in _rows():
        key = tuple(r.values())
        if dedupe and key in seen:
            continue
        seen.add(key)
        if name and r["employee_name"] != name:
            continue
        by[r["department"]] = by.get(r["department"], 0.0) + _hours(r) * float(r["hourly_rate"])
    return by, sum(by.values())


@pytest.fixture(autouse=True)
def model_off(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))


LABOR = {"type": "calc", "derive": {"labor_cost": "hourly_rate * hours_between(clock_in, clock_out)"}, "exclude": [],
         "group_by": "department", "aggregate": {"labor_cost": "sum(labor_cost)"}}


def _with_last_calc(q="Which department has the highest labor cost and why?", spec=LABOR):
    df = _df()
    mem = {"last_calc": {"question": q, "spec": spec, "result": calc.run(spec, df, question=q)}}
    mem["last_answer"] = {"question": q, "answer": present.calc(mem["last_calc"]["result"], q), "evidence": []}
    return df, mem


def _scripted(monkeypatch, sections):
    it = iter(sections)
    monkeypatch.setattr(calc, "_complete", lambda prompt, max_tokens: next(it))


# ------------------------------------------------------------ plan repair --

def test_an_inverted_selection_of_the_named_person_is_repaired_not_rejected(monkeypatch):
    df = _df()
    _scripted(monkeypatch, ['calc", "derive": {}}', '"hours": "hours_between(clock_in, clock_out)"}}',
                            '{"reason": "Priya", "where": "employee_name == \'Priya Singh\'"}]}', 'null}',
                            '"avg_hours": "mean(hours)"}}'])
    spec, trace = calc.model_plan("How many hours did Priya work on average?", df, {})
    assert spec["type"] == "calc"
    assert spec["exclude"] == [{"reason": "only Priya Singh", "where": "employee_name != 'Priya Singh'"}]
    assert any(t.get("repaired") for t in trace)
    r = calc.run(spec, df, question="How many hours did Priya work on average?")
    priya = [_hours(x) for x in _rows() if x["employee_name"] == "Priya Singh"]
    assert r["totals"]["avg_hours"] == pytest.approx(sum(priya) / len(priya))


def test_excluding_a_named_person_on_request_is_kept():
    df = _df()
    assert calc.inverted_selection("employee_name == 'Rahul Sharma'", df, "Total cost excluding Rahul") is None
    assert calc.inverted_selection("employee_name == 'Rahul Sharma'", df, "Rahul's total cost") == ("employee_name", "Rahul Sharma")


def test_unrequested_exclusion_and_invented_break_are_removed_and_reported(monkeypatch):
    df = _df()
    _scripted(monkeypatch, ['calc", "derive": {}}', '}',
                            '{"reason": "duplicate row", "where": "duplicate()"}]}', '"department"}',
                            '"labor_cost": "sum(hourly_rate * (hours_between(clock_in, clock_out) - 0.5))"}}'])
    spec, trace = calc.model_plan("Which department has the highest labor cost?", df, {})
    assert spec["exclude"] == [] and spec["aggregate"]["labor_cost"] == "sum(hourly_rate * hours_between(clock_in, clock_out))"
    assert sum(bool(t.get("repaired")) for t in trace) == 2
    by, total = _truth()
    assert calc.run(spec, df)["totals"]["labor_cost"] == pytest.approx(total)


def test_max_for_a_ranking_question_becomes_sum():
    fixed = calc.repair_section("aggregate", {"c": "max(labor_cost)"},
                                ["'c' uses max(), the single largest/smallest record in each department; ... use sum() "
                                 "unless the user asks about a single record"], {}, _df(), "highest labor cost")
    assert fixed[0] == {"c": "sum(labor_cost)"}


def test_a_named_person_is_selected_instead_of_grouping_everyone(monkeypatch):
    df = _df()
    _scripted(monkeypatch, ['calc", "derive": {}}', '"labor_cost": "hourly_rate * hours_between(clock_in, clock_out)"}}',
                            ']}', '"employee_name"}', '"total": "sum(labor_cost)"}}'])
    spec, trace = calc.model_plan("What is Rahul Sharma's total labor cost?", df, {})
    assert spec["group_by"] is None
    assert {"reason": "only Rahul Sharma", "where": "employee_name != 'Rahul Sharma'"} in spec["exclude"]
    _, rahul = _truth("Rahul Sharma")
    r = calc.run(spec, df, question="What is Rahul Sharma's total labor cost?")
    assert r["totals"]["total"] == pytest.approx(rahul) == 650
    assert present.calc_headline(r).startswith("For Rahul Sharma")


def test_a_follow_up_keeps_the_person_selected_before(monkeypatch):
    df = _df()
    first = {"type": "calc", "derive": {"labor_cost": "hourly_rate * hours_between(clock_in, clock_out)"},
             "exclude": [{"reason": "only Rahul Sharma", "where": "employee_name != 'Rahul Sharma'"}],
             "group_by": None, "aggregate": {"total": "sum(labor_cost)"}}
    mem = {"calc_history": [("What is Rahul Sharma's total labor cost?", first)]}
    _scripted(monkeypatch, ['calc", "derive": {}}', '}', '{"reason": "dup", "where": "duplicate()"}]}', 'null}',
                            '"total": "sum(labor_cost)"}}'])
    spec, trace = calc.model_plan("Excluding duplicates, what is it?", df, mem)
    assert any(t.get("carried_over") for t in trace)
    _, truth = _truth("Rahul Sharma", dedupe=True)
    assert calc.run(spec, df)["totals"]["total"] == pytest.approx(truth) == 450
    # widening the scope does not carry the person over
    assert calc.carried_selections(mem["calc_history"], [], df, "What is the total for all employees?") == []


# ------------------------------------------------------- data checks & times --

def test_suspicious_records_included_in_a_result_are_reported_with_their_lines():
    df = _df()
    r = calc.run(LABOR, df)
    checks = {c["issue"].split(" (")[0]: c["lines"] for c in r["data_checks"]}
    rows = _rows()
    assert checks["exact duplicate of an earlier record"] == [12]
    assert checks["clock out earlier than clock in"] == [i + 2 for i, x in enumerate(rows) if _hours(x) < 0]
    assert checks["negative hourly rate"] == [i + 2 for i, x in enumerate(rows) if float(x["hourly_rate"]) < 0]
    assert "Data checks" in present.calc(r)


def test_clock_time_durations_in_a_scenario_are_stated_as_assumptions():
    df = _df()
    spec = {"type": "calc", "derive": {}, "group_by": None,
            "exclude": [{"reason": "only Priya Singh", "where": "employee_name != 'Priya Singh'"}],
            "aggregate": {"cost": "mean(hourly_rate) * hours_between('09:00', '17:30')"}}
    r = calc.run(spec, df, question="What would Priya's cost be for 09:00 to 17:30?")
    assert {"name": "hours from 09:00 to 17:30", "formula": "hours_between('09:00', '17:30')", "value": 8.5} in r["assumptions"]
    assert r["totals"]["cost"] == pytest.approx(8.5 * 22) and r["answer_type"] == "scenario"


def test_overview_recognises_times_of_day_and_their_data_problems():
    ov = overview.understand(_df(), "timesheet.csv")
    kinds = {c["name"]: c.get("time_of_day") for c in ov["column_details"]}
    assert kinds["clock_in"] and kinds["clock_out"]
    assert any("clock_out' is earlier than 'clock_in' in 1 row (line 11)" in i for i in ov["quality_issues"])
    assert any("'hourly_rate' is negative in 1 row (line 9)" in i for i in ov["quality_issues"])
    assert not any(t["column"] in ("clock_in", "employee_name", "employee_id") for t in ov["possible_targets"])
    assert present.understand(ov).startswith("**In simple terms:**")


# ------------------------------------------------------------ conversation --

def test_explain_in_simple_terms_gives_an_explanation_not_a_promise():
    df, mem = _with_last_calc()
    out = orchestrator.ask("Explain this in simple terms.", df, mem, use_model=False)
    by, total = _truth()
    assert out["plan_source"] == "conversation_rules"
    a = out["answer"]
    assert "Engineering" in a and f"{by['Engineering']:,.0f}" in a and "Marketing" in a
    assert "not a machine-learning prediction" in a and "line" in a          # caution about included bad records
    assert "will explain" not in a.lower()
    out2 = orchestrator.ask("What does this result mean?", df, mem, use_model=False)
    assert out2["plan_source"] == "conversation_rules" and "Engineering" in out2["answer"]


def test_model_explanation_is_used_only_when_grounded_and_retried_once(monkeypatch):
    df, mem = _with_last_calc()
    by, _ = _truth()
    drafts = iter(["Sure, I will explain.", f"Engineering spends the most: {by['Engineering']:,.0f}. That is because it "
                                           "has the most records."])
    monkeypatch.setattr(planner, "model_chat", lambda msgs, max_tokens=700: next(drafts))
    out = orchestrator.ask("Explain this in simple terms.", df, mem, use_model=True)
    assert out["plan_source"] == "conversation" and out["answer"].startswith("Engineering spends the most")
    assert any("first draft was not used" in n for n in out["notes"])
    monkeypatch.setattr(planner, "model_chat", lambda msgs, max_tokens=700: "Engineering spends 99,999 on staff.")
    out = orchestrator.ask("Explain this in simple terms.", df, mem, use_model=True)
    assert out["plan_source"] == "conversation_rules" and "99,999" not in out["answer"]


def test_recommendations_are_built_from_the_evidence():
    df, mem = _with_last_calc()
    out = orchestrator.ask("What should the company do to improve performance?", df, mem, use_model=False)
    a = out["answer"]
    assert "Fix the data errors" in a and "Review Engineering first" in a and "not proven causes" in a
    assert "line 12" in a or "lines 12" in a


def test_why_is_a_group_doing_badly_uses_its_records_and_data_problems():
    df, mem = _with_last_calc()
    out = orchestrator.ask("Why is Marketing performing badly?", df, mem, use_model=False)
    by, _ = _truth()
    assert f"ranks 4 of 4 on labor cost with {by['Marketing']:,.0f}" in out["answer"]
    assert "line 11" in out["answer"] and "not proven causes" in out["answer"]


def test_a_vague_question_gets_dataset_specific_suggestions_not_a_placeholder(monkeypatch):
    df = _df()
    monkeypatch.setattr(planner, "model_available", lambda: False)
    out = orchestrator.ask("Hmm, ok?", df, {}, use_model=False)
    assert out["answer"].strip() and "not sure what to compute" not in out["answer"]


# ------------------------------------------------- trends, forecasts, scenarios --

def _complete_days(today="2026-11-12"):
    """Days whose total is comparable: at least 75% of the usual number of employees recording per day (median over
    days up to today) have a valid record that day. Plain Python, independent of the code under test."""
    seen, recorded, valid = set(), {}, {}
    for r in _rows():
        key = tuple(r.values())
        dup = key in seen
        seen.add(key)
        if r["date"] > today:
            continue
        recorded.setdefault(r["date"], set()).add(r["employee_id"])
        if not (dup or float(r["hourly_rate"]) < 0 or _hours(r) < 0):
            valid.setdefault(r["date"], set()).add(r["employee_id"])
    counts = sorted(len(v) for v in recorded.values())
    mid = len(counts) // 2
    usual = counts[mid] if len(counts) % 2 else (counts[mid - 1] + counts[mid]) / 2
    return {d for d, v in valid.items() if len(v) >= 0.75 * usual}


def _eligible_history_by_day(today="2026-11-12"):
    """Labor cost per day over the records that are valid history, in plain Python: not an exact duplicate of an
    earlier record, no negative rate, clock-out not before clock-in, not dated after today."""
    seen, per_day = set(), {}
    for r in _rows():
        key = tuple(r.values())
        dup = key in seen
        seen.add(key)
        if dup or float(r["hourly_rate"]) < 0 or _hours(r) < 0 or r["date"] > today:
            continue
        per_day[r["date"]] = per_day.get(r["date"], 0.0) + _hours(r) * float(r["hourly_rate"])
    return per_day


@pytest.fixture
def fixed_today(monkeypatch):
    from app.analyst import scope
    monkeypatch.setattr(scope, "today", lambda: pd.Timestamp("2026-11-12"))


def test_trend_of_the_previous_measure_uses_only_valid_history(fixed_today):
    df, mem = _with_last_calc()
    out = orchestrator.ask("What trends do you notice in this dataset?", df, mem, use_model=False)
    pts = out["calls"][0]["result"]["points"]
    assert {p["period"]: p["value"] for p in pts} == pytest.approx(_eligible_history_by_day())
    excluded = {e["reason"].split(" (")[0]: e["lines"] for e in out["calls"][0]["result"]["excluded"]}
    assert excluded["exact duplicate of an earlier record"] == [12]
    assert excluded["negative hourly rate"] == [9] and excluded["clock out earlier than clock in"] == [11]
    assert excluded["dated after today"] == [10]
    assert out["charts"][0]["type"] == "line" and "Not used as history" in out["answer"]


def test_too_little_history_gives_a_labelled_estimate_not_an_ml_prediction(fixed_today):
    df, mem = _with_last_calc()
    out = orchestrator.ask("Predict the next value using historical data.", df, mem, use_model=False)
    f = out["calls"][0]["result"]
    assert f["kind"] == "historical_estimate" and f["ml_used"] is False
    per_day = {d: v for d, v in _eligible_history_by_day().items() if d in _complete_days()}
    assert f["estimate"] == pytest.approx(sum(per_day.values()) / len(per_day))
    assert [x["period"] for x in f["coverage"]["incomplete"]] == sorted(set(_eligible_history_by_day()) - _complete_days())
    assert "not a machine-learning prediction" in out["answer"]


def test_enough_history_gives_a_validated_ml_forecast(monkeypatch):
    from app.analyst import scope
    monkeypatch.setattr(scope, "today", lambda: pd.Timestamp("2026-01-15"))      # the records end last month
    dates = pd.date_range("2024-01-01", periods=24, freq="MS")
    df = pd.DataFrame({"month": dates, "revenue": [100 + 5 * i for i in range(24)]})
    f = followup.forecast(df, {}, "revenue")
    assert f["kind"] == "ml_prediction" and f["ml_used"]
    assert f["prediction"] == pytest.approx(100 + 5 * 24)
    assert f["validation"]["model_mae"] < f["validation"]["naive_mae"]
    assert f["next_period"] == "2026-01"
    assert "machine-learning prediction" in followup.forecast_text(f)


def test_a_forecast_without_a_known_value_asks_which():
    df = _df()
    out = orchestrator.ask("Predict the next value using historical data.", df, {}, use_model=False)
    assert out["plan"]["type"] == "clarify" and "hourly_rate" in out["answer"]


def test_percentage_change_reruns_the_previous_calculation():
    df, mem = _with_last_calc()
    out = orchestrator.ask("What if hourly rates increase by 10%?", df, mem, use_model=False)
    by, total = _truth()
    s = out["calls"][0]["result"]
    assert s["results"][0]["baseline"] == pytest.approx(total)
    assert s["results"][0]["scenario"] == pytest.approx(total * 1.1)
    assert {g["group"]: g["scenario"] for g in s["groups"]} == pytest.approx({k: v * 1.1 for k, v in by.items()})
    assert "not a forecast and not a machine-learning prediction" in out["answer"]
    out = orchestrator.ask("What happens if salaries increase by 10%?", df, mem, use_model=False)
    assert "no salary column" in out["answer"] and out["calls"][0]["result"]["column"] == "hourly_rate"


# ------------------------------------------------------------------- charts --

@pytest.mark.parametrize("q,kind", [("Show monthly revenue as a line chart.", "line"),
                                    ("Compare labor costs by department as a bar chart", "bar"),
                                    ("Show it as a pie chart", "pie"),
                                    ("Plot actual versus predicted sales.", "avp"),
                                    ("Show the correlation between advertising spend and revenue.", "scatter"),
                                    ("Show the distribution of hourly rate.", "histogram"),
                                    ("Show me a chart of labor cost by department.", "auto"),
                                    ("What is the total labor cost?", None)])
def test_chart_requests_are_recognised(q, kind):
    assert followup.chart_request(q) == kind


def test_chart_of_the_previous_result_uses_its_exact_numbers():
    df, mem = _with_last_calc()
    out = orchestrator.ask("Show me a chart of labor cost by department.", df, mem, use_model=False)
    by, _ = _truth()
    c = out["charts"][0]
    assert out["chart_requested"] and c["type"] == "bar"
    assert {d["department"]: d["labor cost"] for d in c["data"]} == pytest.approx(by)


def test_a_pie_is_refused_for_negative_parts_and_top_n_keeps_the_rest_as_one_slice():
    charts = [{"type": "bar", "x": "g", "y": "v", "title": "v by g",
               "data": [{"g": "a", "v": 5.0}, {"g": "b", "v": 3.0}, {"g": "c", "v": 2.0}]}]
    out, notes = followup.restyle(charts, "pie", "top 2 as a pie")
    assert out[0]["type"] == "pie" and out[0]["data"][-1] == {"g": "all other (1)", "v": 2.0} and not notes
    neg = [{**charts[0], "data": [{"g": "a", "v": 5.0}, {"g": "b", "v": -1.0}]}]
    out, notes = followup.restyle(neg, "pie", "as a pie")
    assert out[0]["type"] == "bar" and "negative" in notes[0]


def test_scatter_and_histogram_data_are_computed_from_the_records():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"spend": rng.normal(10, 2, 200)})
    df["revenue"] = 3 * df["spend"] + rng.normal(0, 1, 200)
    s = followup.scatter(df, "spend", "revenue")
    assert s["r"] == pytest.approx(np.corrcoef(df["spend"], df["revenue"])[0, 1]) and len(s["data"]) == 200
    h = followup.histogram(df, "spend")
    assert sum(b["count"] for b in h["data"]) == 200


def test_top_n_ranks_groups_by_the_main_measure():
    df = pd.DataFrame({"product": list("AABBBCD"), "revenue": [5, 5, 1, 1, 1, 20, 2.0]})
    rk = followup.rank_request("Show the top two products", df)
    assert rk == {"group": "product", "measure": "revenue", "n": 2, "low": False, "assumed_measure": True}
    r = followup.rank_groups(df, "product", "revenue", 2)
    assert [(x["group"], x["value"]) for x in r["rows"]] == [("C", 20.0), ("A", 10.0)]


def test_actual_vs_predicted_comes_from_held_out_rows_of_a_trained_model():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 120), "g": rng.choice(["a", "b"], 120)})
    df["y"] = 4 * df["x"] + (df["g"] == "a") * 2 + rng.normal(0, 0.3, 120)
    mem = {}
    from app.analyst import tools
    tools.run({"tool": "random_forest", "args": {"target": "y"}}, df, mem)
    out = orchestrator.ask("Plot actual versus predicted y.", df, mem, use_model=False)
    c = out["charts"][0]
    assert c["type"] == "avp" and len(c["data"]) == mem["model"]["test_rows"]
    assert all(set(d) == {"actual", "predicted"} for d in c["data"])
    assert set(round(d["actual"], 9) for d in c["data"]) <= set(round(v, 9) for v in df["y"])


def test_actual_vs_predicted_without_a_model_says_so():
    out = orchestrator.ask("Plot actual versus predicted sales.", _df(), {}, use_model=False)
    assert "no prediction in this conversation" in out["answer"] and not out["charts"]


# ------------------------------------------------------------ narration checks --

def test_a_significance_claim_that_contradicts_the_test_is_removed():
    notes = []
    t = orchestrator._check_significance(  # noqa: SLF001
        "The Pearson correlation is 0.653. The data do not show a reliable difference between groups.",
        {"p_value": 0.0001}, notes)
    assert t == "The Pearson correlation is 0.653." and notes
    t = orchestrator._check_significance("The difference is statistically significant.", {"p_value": 0.4}, [])  # noqa: SLF001
    assert "significant" not in t


def test_answers_put_the_computed_result_first_and_details_aside():
    df = _df()
    r = calc.run(LABOR, df, question="Which department has the highest labor cost?")
    main, details = present.calc(r, "Which department has the highest labor cost?"), present.calc_details(r)
    by, total = _truth()
    assert main.startswith(f"**Engineering** has the highest labor cost: **{by['Engineering']:,.0f}**")
    assert "```" not in main and "{" not in main
    assert "hourly rate × hours from clock in to clock out" in details
    assert json.dumps(r, default=str)  # result stays serialisable


# ------------------------------------------- defects found in the browser rehearsal --

def test_a_rate_copied_from_the_data_is_read_from_the_records_instead(monkeypatch):
    """The model wrote Priya's rate 22 as a constant; it is replaced by her records' rate, and she is selected."""
    df = _df()
    _scripted(monkeypatch, ['calc", "derive": {}}', '}', ']}', 'null}',
                            '"labor_cost": "hours_between(\'09:00\', \'17:30\') * 22.0"}}'])
    q = "What would Priya Singh's labor cost be for a shift from 09:00 to 17:30?"
    spec, trace = calc.model_plan(q, df, {})
    assert spec["type"] == "calc", spec
    assert spec["aggregate"]["labor_cost"] == "hours_between('09:00', '17:30') * mean(hourly_rate)"
    assert {"reason": "only Priya Singh", "where": "employee_name != 'Priya Singh'"} in spec["exclude"]
    r = calc.run(spec, df, question=q)
    rates = [float(x["hourly_rate"]) for x in _rows() if x["employee_name"] == "Priya Singh"]
    assert r["totals"]["labor_cost"] == pytest.approx(8.5 * sum(rates) / len(rates)) == 187


def test_a_failed_calculation_does_not_fall_back_to_an_answer_over_no_records():
    failed = {"plan_source": "model_calc_failed", "first_pass": {"error": "x"}, "notes": [], "plan": {"type": "calc"}}
    empty = {"calls": [{"status": "ok", "result": {"value": None, "rows": 0}}], "plan": {"type": "tools"}, "notes": []}
    assert orchestrator._after_failed(empty, failed) is failed  # noqa: SLF001
    real = {"calls": [{"status": "ok", "result": {"value": 3.0, "rows": 5}}], "plan": {"type": "tools"}, "notes": []}
    assert orchestrator._after_failed(real, failed) is real  # noqa: SLF001


def test_model_text_naming_internal_fields_is_not_shown(monkeypatch):
    df, mem = _with_last_calc()
    monkeypatch.setattr(planner, "model_chat", lambda msgs, max_tokens=700: "Review Engineering, as the suggested_steps say.")
    out = orchestrator.ask("What should the company do to improve performance?", df, mem, use_model=True)
    assert "suggested_steps" not in out["answer"] and "Review Engineering first" in out["answer"]
    assert out["answer"].count("line 12") + out["answer"].count("lines 12") == 1      # each problem listed once


# ------------------------------------------------ resumed scope: defects found in the real-model probe --

def _campaigns(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for area, eff in (("North", 4.0), ("South", 1.5), ("East", 3.0), ("West", 2.5)):
        for m in range(12):
            spend = float(rng.uniform(800, 1200))
            leads = int(rng.integers(300, 500))
            rows.append({"month": pd.Timestamp("2025-01-01") + pd.DateOffset(months=m), "area": area,
                         "ad_spend": round(spend, 2), "leads": leads, "conversions": int(leads * 0.03),
                         "revenue": round(spend * eff, 2)})
    return pd.DataFrame(rows)


def test_monthly_wording_is_not_eaten_by_a_column_called_month():
    from app.analyst import router
    df = pd.DataFrame({"month": pd.date_range("2025-01-01", periods=6, freq="MS"), "sales": [1.0, 2, 3, 4, 5, 6]})
    plan = router.route("Show monthly sales as a line chart.", df, {})
    assert plan["tool"] == "trend" and plan["args"]["value"] == "sales" and plan["args"]["freq"] == "M"


def test_conversion_improving_by_a_percentage_asks_which_kind_of_change():
    from app.analyst import router
    df = _campaigns()
    assert router.route("What if conversion improves by 5%?", df, {})["type"] == "clarify"
    plan = router.route("What if conversion improves from 3% to 5%?", df, {})
    assert plan["tool"] == "conversion_change" and plan["args"] == {"to_rate": 0.05, "from_rate": 0.03}


def test_budget_cut_in_the_lowest_performing_group_targets_that_group_only():
    df = _campaigns()
    roi = {a: (g["revenue"].sum() - g["ad_spend"].sum()) / g["ad_spend"].sum() for a, g in df.groupby("area")}
    worst = min(roi, key=roi.get)
    out = orchestrator.ask("What if we reduce marketing spend in the lowest-performing area by 10%?", df, {}, use_model=False)
    r = out["calls"][0]["result"]
    assert r["scope"] == worst == "South"
    assert r["baseline_spend"] == pytest.approx(df.loc[df["area"] == worst, "ad_spend"].sum())
    assert r["new_spend"] == pytest.approx(0.9 * r["baseline_spend"])
    assert "lowest-performing area" in out["answer"] and worst in out["answer"]


def test_a_clarification_that_ignores_the_named_target_is_a_contradiction():
    from app.analyst import plan_checks
    df = pd.DataFrame({"region": ["a", "b"] * 30, "sales": np.arange(60.0)})
    plan = {"type": "clarify", "question": "Which column should I predict?", "options": ["region", "sales"]}
    assert "'sales'" in plan_checks.contradiction(plan, df, "Predict sales.")
    assert plan_checks.contradiction(plan, df, "Predict something.") is None


def _hr(n=300, seed=4):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"department": rng.choice(["Sales", "Engineering", "Support"], n),
                       "job_level": rng.choice(["Junior", "Mid", "Senior"], n),
                       "age": rng.integers(21, 60, n), "overtime": rng.choice(["Yes", "No"], n),
                       "satisfaction": rng.integers(1, 6, n)})
    risk = (df["overtime"] == "Yes") * 0.5 + (df["satisfaction"] <= 2) * 0.3
    df["attrition"] = np.where(rng.random(n) < 0.1 + risk, "Yes", "No")
    return df


def test_missing_or_unknown_prediction_inputs_are_asked_for_never_filled_in():
    from app.analyst import tools
    df, mem = _hr(), {}
    tools.run({"tool": "random_forest", "args": {"target": "attrition"}}, df, mem)
    out = orchestrator.ask("Predict attrition for an employee who works overtime.", df, mem, use_model=False)
    assert out["plan"]["type"] == "clarify" and "Still needed" in out["answer"]
    for f in ("department", "job level", "age", "satisfaction"):
        assert f"**{f}**" in out["answer"]
    assert not out["calls"], "nothing was predicted"
    out = orchestrator.ask("Predict attrition for an employee in Sales, job level 2, age 30, overtime Yes, satisfaction 2.",
                           df, mem, use_model=False)
    assert "Not a value the data has" in out["answer"] and "**job level**" in out["answer"]
    assert followup.missing_prediction_inputs(
        "Predict attrition for a Mid employee in Sales, age 30, overtime Yes, satisfaction 2.", df, mem) is None


def test_prioritise_next_month_is_advice_not_a_forecast_and_uses_the_model_evidence():
    from app.analyst import tools
    df, mem = _hr(), {}
    tools.run({"tool": "random_forest", "args": {"target": "attrition"}}, df, mem)
    out = orchestrator.ask("What should management prioritize next month?", df, mem, use_model=False)
    assert out["plan"]["kind"] == "advice"
    top = [i["feature"] for i in mem["model"]["importance"] if i["importance"] > 0][:1]
    assert top and top[0].replace("_", " ") in out["answer"] and "not proven causes" in out["answer"]


def test_show_that_as_a_pie_reuses_any_previous_chart_unchanged():
    df = pd.DataFrame({"product": list("AABBBCD"), "revenue": [5, 5, 1, 1, 1, 20, 2.0]})
    mem = {}
    first = orchestrator.ask("Show the top three products.", df, mem, use_model=False)
    out = orchestrator.ask("Show that as a donut chart.", df, mem, use_model=False)
    c = out["charts"][0]
    assert c["type"] == "pie" and c["data"] == first["charts"][0]["data"]
    truth = df.groupby("product")["revenue"].sum().nlargest(3).to_dict()
    assert {d["product"]: d["revenue"] for d in c["data"]} == truth == {"C": 20.0, "A": 10.0, "B": 3.0}


def test_scatter_axes_follow_the_order_the_user_named_them():
    assert followup.mentioned_order(["sales", "marketing_spend"], "correlation between marketing spend and sales") == \
        ["marketing_spend", "sales"]


def test_a_share_of_records_is_shown_as_a_percentage():
    df = _hr()
    spec = {"type": "calc", "derive": {}, "exclude": [], "group_by": "department",
            "aggregate": {"attrition_rate": "mean(attrition == 'Yes')"}}
    r = calc.run(spec, df, question="Which department has the highest attrition rate?")
    rate = df.assign(y=df["attrition"] == "Yes").groupby("department")["y"].mean()
    top = rate.idxmax()
    assert present.calc(r, "Which department has the highest attrition rate?").startswith(
        f"**{top}** has the highest attrition rate: **{rate.max() * 100:.1f}%**")


def test_a_target_that_is_a_formula_of_the_inputs_is_flagged():
    from app.analyst import ml
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"units": rng.integers(1, 50, 200), "price": rng.choice([10.0, 20.0, 35.0], 200)})
    df["sales"] = df["units"] * df["price"]
    rep = ml.train(df, "sales")
    assert rep["metrics"]["r2"] > 0.99 and "formula of the inputs" in rep["limitations"][0]


def test_next_month_without_a_forecasting_verb_is_not_a_forecast():
    assert not followup.wants_forecast("What should management prioritize next month?")


def test_advice_does_not_call_an_identifier_or_an_unusual_value_a_data_error():
    df = _hr().reset_index().rename(columns={"index": "employee_id"})
    df.loc[0, "age"] = 95                                             # one statistically unusual value
    out = orchestrator.ask("What should management prioritize next month?", df, {}, use_model=False)
    a = out["answer"]
    assert "Fix the data errors" not in a and "Check the unusual values" in a
    assert "Track it over time" not in a, "no date column, so no trend to track"


def test_a_pie_is_refused_for_averages_and_rates_but_kept_for_totals():
    df = _hr()
    spec = {"type": "calc", "derive": {}, "exclude": [], "group_by": "department",
            "aggregate": {"attrition_rate": "mean(attrition == 'Yes')"}}
    rate_chart = present.calc_charts(calc.run(spec, df))
    out, notes = followup.restyle(rate_chart, "pie", "show that as a pie chart")
    assert out[0]["type"] == "bar" and "do not add up" in notes[0]
    total = present.calc_charts(calc.run(LABOR, _df()))          # department totals: all positive, they add up
    by, _ = _truth()
    assert total[0]["additive"] is True and all(v > 0 for v in by.values())
    assert followup.restyle(total, "pie", "as a pie")[0][0]["type"] == "pie"
