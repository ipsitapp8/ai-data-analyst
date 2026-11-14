"""Silt calculator (app/analyst/calc.py): model-planned free-form calculations.

Expected values are computed independently in each test with plain Python over
the rows, never by calling the code under test.
"""
from __future__ import annotations

import json
import re
import random

import pandas as pd
import pytest

from app.analyst import calc, orchestrator


def _timesheet(seed: int = 3) -> pd.DataFrame:
    rnd = random.Random(seed)
    depts = ["Engineering", "HR", "Finance", "Marketing"]
    rows = []
    for i in range(40):
        start = rnd.choice([8, 9, 10])
        end = start + rnd.choice([6, 7, 8, 9])
        rows.append({"employee_id": f"E{i % 9:03d}", "department": rnd.choice(depts),
                     "date": f"2026-{rnd.choice(['08', '09', '09', '09', '10'])}-{rnd.randint(1, 28):02d}",
                     "clock_in": f"{start:02d}:{rnd.choice(['00', '30'])}", "clock_out": f"{end:02d}:00",
                     "hourly_rate": rnd.choice([20, 22, 25, 30, -5])})
    rows[5]["clock_out"], rows[5]["clock_in"] = "08:00", "09:00"      # clock-out before clock-in
    rows.append(dict(rows[2]))                                          # an exact duplicate
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _truth(df: pd.DataFrame):
    """Hours, cost and per-department cost with the four exclusions, in plain Python."""
    seen, total_h, total_c, by = set(), 0.0, 0.0, {}
    for rec in df.astype({"date": str}).to_dict("records"):
        key = tuple(rec.values())
        dup = key in seen
        seen.add(key)
        h = lambda t: int(t[:2]) + int(t[3:5]) / 60  # noqa: E731
        hours = h(rec["clock_out"]) - h(rec["clock_in"])
        if dup or rec["hourly_rate"] < 0 or hours < 0 or not rec["date"].startswith("2026-09"):
            continue
        total_h += hours
        total_c += hours * rec["hourly_rate"]
        by[rec["department"]] = by.get(rec["department"], 0.0) + hours * rec["hourly_rate"]
    return total_h, total_c, by


SPEC = {"type": "calc", "derive": {"working_hours": "hours_between(clock_in, clock_out)",
                                   "labor_cost": "working_hours * hourly_rate"},
        "exclude": [{"reason": "duplicate", "where": "duplicate()"},
                    {"reason": "negative rate", "where": "hourly_rate < 0"},
                    {"reason": "clock-out before clock-in", "where": "clock_out < clock_in"},
                    {"reason": "outside September 2026", "where": "year(date) != 2026 or month(date) != 9"}],
        "group_by": "department", "aggregate": {"hours": "sum(working_hours)", "labor_cost": "sum(labor_cost)"},
        "sort": "-labor_cost", "explain": ""}
QUESTION = ("Calculate total working hours and total labor cost as hours times hourly rate, excluding duplicates, "
            "negative rates, clock-out before clock-in and records outside September 2026, by department.")


def test_calculator_matches_an_independent_computation():
    df = _timesheet()
    hours, cost, by = _truth(df)
    r = calc.run(calc.parse(json.dumps(SPEC)), df, question=QUESTION)
    assert r["totals"]["hours"] == pytest.approx(hours) and r["totals"]["labor_cost"] == pytest.approx(cost)
    assert {g["group"]: g["labor_cost"] for g in r["groups"]} == pytest.approx(by)
    assert [g["labor_cost"] for g in r["groups"]] == sorted((g["labor_cost"] for g in r["groups"]), reverse=True)
    assert r["rows_valid"] + r["rows_excluded"] == len(df)
    assert all(e["excluded_because"] for e in r["excluded_rows"])
    assert any("duplicate" in e["excluded_because"] for e in r["excluded_rows"])


def test_expressions_cannot_escape_the_whitelist():
    df = _timesheet()
    for bad in ("__import__('os')", "hourly_rate.__class__", "(lambda: 1)()", "hourly_rate[0]", "open('x')",
                "[c for c in hourly_rate]", "hourly_rate ** 2"):
        with pytest.raises(calc.CalcError):
            calc.evaluate(bad, df, df)


def test_unstated_constants_are_rejected_and_stated_ones_accepted():
    df = _timesheet()
    spec = calc.parse(json.dumps({**SPEC, "derive": {"working_hours": "hours_between(clock_in, clock_out) - 0.5",
                                                     "labor_cost": "working_hours * hourly_rate"}}))
    assert any("0.5" in p for p in calc.check(spec, df, QUESTION))
    assert not any("0.5" in p for p in calc.check(spec, df, QUESTION + " Deduct a 0.5 hour break."))
    raise10 = calc.parse(json.dumps({**SPEC, "aggregate": {"c": "sum(labor_cost * 1.10)"}}))
    assert calc.check(raise10, df, QUESTION + " What if rates rise by 10%?") == []


def test_plan_without_results_names_what_the_question_asks_for():
    df = _timesheet()
    probs = calc.check(calc.parse(json.dumps({**SPEC, "aggregate": {}})), df, QUESTION)
    assert any("working hours" in p and "labor cost" in p for p in probs)


def test_routing_sends_definitions_and_exclusions_to_the_calculator_but_not_statistics():
    df = _timesheet()
    assert calc.wanted(QUESTION, df, {})
    assert calc.wanted("Calculate total labor cost.", df, {})            # not a column: must be derived
    for q in ("Calculate correlations between hourly_rate and department", "Predict hourly_rate",
              "Which department has the highest hourly_rate?", "Find outliers in hourly_rate"):
        assert calc.wanted(q, df, {}) is None, q
    mem = {"calc_history": [("q", {"derive": {"working_hours": "x"}, "aggregate": {"labor_cost": "sum(x)"}})]}
    assert calc.wanted("How many hours did Engineering work?", df, mem)


def test_section_checks_catch_wrong_aggregation_missing_group_and_unrequested_exclusions():
    df = _timesheet()
    frame = calc.run_derive(df, SPEC["derive"])
    spec = {**SPEC, "exclude": [], "group_by": "department"}
    probs = calc._check_section("aggregate", {"labor_cost": "max(labor_cost)"}, spec, frame, "x",  # noqa: SLF001
                                "Which department has the highest labor cost?")
    assert any("sum()" in p for p in probs)
    assert not calc._check_section("aggregate", {"labor_cost": "max(labor_cost)"}, spec, frame, "x",  # noqa: SLF001
                                   "Which department has the single highest labor cost record?")
    probs = calc._check_section("group_by", None, {**spec, "group_by": None}, frame, "x",  # noqa: SLF001
                                "What is the labor cost for HR?")
    assert any("'HR'" in p for p in probs)
    probs = calc._check_section("exclude", [{"reason": "dups", "where": "duplicate()"}],  # noqa: SLF001
                                {**spec, "_history": []}, frame, "Calculate profit by region", "Calculate profit by region")
    assert any("did not ask" in p for p in probs)


def test_model_plan_is_built_in_sections_and_a_rejected_section_is_reconsidered_once(monkeypatch):
    df = _timesheet()
    replies = iter(['calc", "derive": {}}',
                    '"working_hours": "hours_between(clock_in, clock_outt)"}}',          # a column that does not exist
                    '"working_hours": "hours_between(clock_in, clock_out)"}}',
                    '{"reason": "negative rate", "where": "hourly_rate < 0"}]}',
                    '"department"}',
                    '"hours": "sum(working_hours", "labor_cost": "sum(working_hours * hourly_rate)"}}'])
    prompts = []

    def fake(prompt, max_tokens):
        prompts.append(prompt)
        return next(replies)
    monkeypatch.setattr(calc, "_complete", fake)
    spec, trace = calc.model_plan("Total working hours and labor cost by department, excluding negative rates.", df, {})
    assert spec["type"] == "calc" and spec["group_by"] == "department"
    assert spec["derive"] == {"working_hours": "hours_between(clock_in, clock_out)"}
    rejected = [t for t in trace if t.get("accepted") is False]
    assert len(rejected) == 1 and "clock_outt" in rejected[0]["problems"][0]
    assert "rejected that derive" in prompts[2]                  # the reason was shown to the model
    r = calc.run(spec, df)                                       # the unbalanced "sum(working_hours" was closed
    assert r["totals"]["hours"] > 0


def test_model_plan_passes_through_a_clarification(monkeypatch):
    monkeypatch.setattr(calc, "_complete", lambda p, m: 'clarify", "question": "Which rate column?"}')
    spec, _ = calc.model_plan("Calculate the cost", _timesheet(), {})
    assert spec == {"type": "clarify", "question": "Which rate column?"}


def test_json_repairs_are_syntax_only():
    assert calc._first_value('{"a": "sum(f(x, y)), "b": "sum(z)"}') == {"a": "sum(f(x, y))", "b": "sum(z)"}  # noqa: SLF001
    assert calc._first_value('{"h": hours_between(a, b) }') == {"h": "hours_between(a, b)"}  # noqa: SLF001
    assert calc._first_value("not json") is calc.UNPARSED  # noqa: SLF001
    assert calc._first_value("null, \"aggregate\"") is None  # noqa: SLF001


def test_narration_that_contradicts_the_computed_ranking_is_detected():
    ev = {"by_department": [{"group": "Engineering", "rows": 4, "labor_cost": 975.0,
                             "per_row": {"average labor_cost per row": 243.75}},
                            {"group": "Finance", "rows": 1, "labor_cost": 336.0,
                             "per_row": {"average labor_cost per row": 336.0}}]}
    assert orchestrator.rank_contradictions(
        "Engineering has the highest labor cost at 975.00 and the highest average labor_cost per row.", ev)
    assert not orchestrator.rank_contradictions("Engineering has the highest labor cost at 975.00.", ev)


def test_orchestrator_answers_a_free_form_calculation_with_the_model_and_labels_it(monkeypatch):
    df = _timesheet()
    hours, cost, by = _truth(df)
    sections = iter(['calc", "derive": {}}', json.dumps(SPEC["derive"])[1:] + "}",
                     json.dumps(SPEC["exclude"])[1:] + "}", '"department"}', json.dumps(SPEC["aggregate"])[1:] + "}"])
    monkeypatch.setattr(calc, "_complete", lambda p, m: next(sections))
    top = max(by, key=by.get)
    monkeypatch.setattr(orchestrator.planner, "model_chat",
                        lambda msgs, max_tokens=700: f"{top} has the highest labor cost. This compares averages only; "
                                                     "see the statistical test before treating the gap as real.")
    monkeypatch.setattr(orchestrator.planner, "model_available", lambda: True)
    mem = {}
    out = orchestrator.ask(QUESTION, df, mem)
    # labor cost is a standard metric: resolved and computed deterministically, every stated condition applied
    assert out["plan_source"] == "metric"
    from app.analyst.explain import f as fmt
    assert fmt(cost) in out["answer"] and fmt(by[top]) in out["answer"]
    # technical detail lives in the optional "How this was calculated" section, not in the answer
    assert "Records left out" in out["details"] and "Steps" in out["details"]
    body = re.sub(r"^\*\*Scope:\*\*[^\n]*\n\n", "", out["answer"])             # the scope line, then the answer
    assert body.startswith(f"**{top}** has the highest")                          # direct answer first
    assert "statistical test" not in out["answer"]                               # inapplicable boilerplate removed
    assert any("removed" in n for n in out["notes"])
    assert out["charts"] and out["charts"][0]["x"] == "department"
    assert mem["calc_history"] and mem["last_calc"]["result"]["totals"]["labor_cost"] == pytest.approx(cost)
    assert "could not turn that into a metric" not in out["answer"]


def test_quoted_column_arguments_and_unquoted_values_are_repaired_as_syntax():
    df = _timesheet()
    v = calc._first_value('{"working_hours": hours_between("clock_in", "clock_out") }, "exclude": []')  # noqa: SLF001
    assert v == {"working_hours": "hours_between('clock_in', 'clock_out')"}
    a = calc.evaluate(v["working_hours"], df, df)
    b = calc.evaluate("hours_between(clock_in, clock_out)", df, df)
    assert a.equals(b)


def test_dividing_hours_between_gets_a_specific_reason():
    probs = calc._constant_problems(["sum(hours_between(clock_in, clock_out) / 3600 * hourly_rate)"], "labor cost")  # noqa: SLF001
    assert probs and "already returns hours" in probs[0]


def test_follow_up_reuses_columns_derived_earlier_in_the_conversation(monkeypatch):
    df = _timesheet()
    history = [("first", {"type": "calc", "derive": {"working_hours": "hours_between(clock_in, clock_out)"},
                          "exclude": [], "group_by": None, "aggregate": {"h": "sum(working_hours)"}})]
    replies = iter(['calc", "derive": {}}', '}}', ']}', '"department"}',
                    '"labor_cost": "sum(working_hours * hourly_rate)"}}'])
    monkeypatch.setattr(calc, "_complete", lambda p, m: next(replies))
    spec, trace = calc.model_plan("Which department has the highest labor cost?", df, {"calc_history": history})
    assert spec["type"] == "calc" and spec["derive"] == {"working_hours": "hours_between(clock_in, clock_out)"}
    assert any(t.get("reused_from_conversation") for t in trace)


# ------------------------------------------------------------------ clock times and scenarios --

@pytest.mark.parametrize("text,hours", [("09:00", 9), ("9:30", 9.5), ("17:45:30", 17 + 45 / 60 + 30 / 3600), ("8 AM", 8),
                                        ("5 pm", 17), ("12 AM", 0), ("12 PM", 12), ("24:00", 24), ("noon", 12)])
def test_clock_times_are_parsed_as_times_of_day(text, hours):
    assert calc.parse_time(text) == pytest.approx(hours)


@pytest.mark.parametrize("text", ["25:00", "9:75", "13 PM", "9", "9.5", "", "abc", None, 9])
def test_invalid_or_numeric_values_are_never_times(text):
    assert calc.parse_time(text) is None


def test_durations_from_literal_times_columns_and_overnight():
    df = _timesheet()
    assert calc.evaluate("hours_between(09:00, 18:00)", df, df) == 9          # unquoted times are read as times
    assert calc.evaluate("hours_between('8 AM', '5 PM')", df, df) == 9
    assert calc.evaluate("overnight_hours('10 PM', '6 AM')", df, df) == 8
    assert calc.evaluate("hours_between('22:00', '06:00')", df, df) == -16    # a negative duration is visible, not wrapped
    per_row = calc.evaluate("hours_between(clock_in, clock_out)", df, df)
    expected = [(int(o[:2]) + int(o[3:]) / 60) - (int(i[:2]) + int(i[3:]) / 60) for i, o in zip(df.clock_in, df.clock_out)]
    assert list(per_row) == pytest.approx(expected)


def test_missing_and_invalid_clock_values_become_gaps_not_numbers():
    df = _timesheet().head(4).copy()
    df.loc[0, "clock_out"] = None
    df.loc[1, "clock_out"] = "25:61"
    hours = calc.evaluate("hours_between(clock_in, clock_out)", df, df)
    assert hours.isna().tolist()[:2] == [True, True] and hours.notna().tolist()[2:] == [True, True]


@pytest.mark.parametrize("expr", ["clock_in * 2", "'09:00' + 1", "hours_between(9, 18)", "hours_between('25:00', '18:00')",
                                  "month('09:00')"])
def test_time_values_are_never_silently_treated_as_numbers(expr):
    df = _timesheet()
    with pytest.raises(calc.CalcError):
        calc.evaluate(expr, df, df)


def test_times_must_be_stated_and_a_literal_night_shift_needs_overnight_hours():
    q = "Estimate the cost of an overnight shift from 10 PM to 6 AM."
    assert calc._time_problems(["mean(r) * overnight_hours('22:00', '06:00')"], q) == []  # noqa: SLF001
    assert any("overnight_hours" in p for p in calc._time_problems(["hours_between('22:00', '06:00')"], q))  # noqa: SLF001
    assert any("not stated" in p for p in calc._time_problems(["hours_between('07:00', '06:00')"], q))  # noqa: SLF001


def _employee_frame():
    rows = [("Ana Ruiz", "09:00", "17:00", 30), ("Ana Ruiz", "08:00", "12:00", 30), ("Bo Chen", "09:00", "18:00", 18),
            ("Bo Chen", "10:00", "14:00", 22), ("Cy Diaz", "07:00", "15:00", 40)]
    return pd.DataFrame(rows, columns=["employee_name", "clock_in", "clock_out", "hourly_rate"])


def test_scenario_uses_the_named_employees_historical_rate():
    df = _employee_frame()
    q = "Using Ana Ruiz's records, estimate her labor cost if she works 8 AM to 6 PM with a 45-minute unpaid break."
    spec = calc.parse(json.dumps({"type": "calc", "derive": {"shift_hours": "hours_between('8 AM', '6 PM') - 0.75"},
                                  "exclude": [{"reason": "not Ana Ruiz", "where": "employee_name != 'Ana Ruiz'"}],
                                  "aggregate": {"rate": "mean(hourly_rate)", "hours": "shift_hours",
                                                "cost": "mean(hourly_rate) * shift_hours"}}))
    assert calc.check(spec, df, q) == [] and calc.break_problems(spec, q) == []
    r = calc.run(spec, df, question=q)
    ana = [x for x in df.itertuples() if x.employee_name == "Ana Ruiz"]
    rate = sum(x.hourly_rate for x in ana) / len(ana)
    assert r["totals"] == pytest.approx({"rate": rate, "hours": 10 - 0.75, "cost": rate * (10 - 0.75)})
    assert r["answer_type"] == "scenario" and r["selected"]
    assert r["inputs_used"][0]["distinct"] == 1


def test_varying_rates_are_reported_and_an_assumption_is_not_summed_over_the_history():
    df = _employee_frame()
    q = "Estimate the labor cost for Bo Chen working a 7.5-hour shift."
    spec = calc.parse(json.dumps({"type": "calc", "derive": {"shift_hours": "7.5"},
                                  "exclude": [{"reason": "not Bo Chen", "where": "employee_name != 'Bo Chen'"}],
                                  "aggregate": {"cost": "mean(hourly_rate) * shift_hours"}}))
    r = calc.run(spec, df, question=q)
    assert r["inputs_used"][0]["distinct"] == 2 and r["totals"]["cost"] == pytest.approx((18 + 22) / 2 * 7.5)
    frame = calc.run_derive(df, spec["derive"])
    probs = calc._check_section("aggregate", {"cost": "sum(hourly_rate * shift_hours)"},  # noqa: SLF001
                                {**spec, "_history": []}, frame, q, q)
    assert any("counting it" in p for p in probs)


def test_a_stated_break_must_be_subtracted():
    q = "What if Rahul works 10 hours with a 30-minute unpaid break?"
    assert calc.stated_break_hours(q) == 0.5
    assert calc.break_problems({"derive": {"h": "10"}, "aggregate": {"c": "mean(hourly_rate) * h"}}, q)
    assert not calc.break_problems({"derive": {"h": "10 - 0.5"}, "aggregate": {"c": "mean(hourly_rate) * h"}}, q)
    assert not calc.break_problems({"derive": {}, "aggregate": {}}, "Assume there is no unpaid break.")


def test_an_earlier_questions_numbers_cannot_leak_into_a_new_formula():
    df = _employee_frame()
    spec = calc.parse(json.dumps({"type": "calc", "derive": {"h": "overnight_hours('10 PM', '6 AM') + 0.5"},
                                  "exclude": [], "aggregate": {"c": "mean(hourly_rate) * h"}}))
    earlier = "What if she works 10 hours with a 30-minute unpaid break?"
    now = "Calculate the cost of an overnight shift from 10 PM to 6 AM."
    assert any("0.5" in p for p in calc.check(spec, df, now, earlier=earlier))


def test_a_calculation_outside_the_metric_catalogue_is_planned_by_the_model(monkeypatch):
    """Not a standard metric (an hours-weighted average rate): the model plans it, the calculator runs it."""
    df = _timesheet()
    spec = {"derive": {"working_hours": "hours_between(clock_in, clock_out)", "paid": "working_hours * hourly_rate"},
            "exclude": [], "group_by": None, "aggregate": {"weighted_rate": "sum(paid) / sum(working_hours)"}}
    sections = iter(['calc", "derive": {}}', json.dumps(spec["derive"])[1:] + "}", "]}", "null}",
                     json.dumps(spec["aggregate"])[1:] + "}"])
    monkeypatch.setattr(calc, "_complete", lambda p, m: next(sections))
    monkeypatch.setattr(orchestrator.planner, "model_chat", lambda msgs, max_tokens=700: "See the result above.")
    monkeypatch.setattr(orchestrator.planner, "model_available", lambda: True)
    out = orchestrator.ask("Calculate the hours-weighted average hourly rate across all records.", df, {})
    assert out["plan_source"] == "model_calc"
    h = [int(b[:2]) + int(b[3:5]) / 60 - int(a[:2]) - int(a[3:5]) / 60 for a, b in zip(df["clock_in"], df["clock_out"])]
    # impossible records (negative hours or rate) are left out by the same validity rule as every other answer
    ok = [(x, r) for x, r in zip(h, df["hourly_rate"]) if x >= 0 and r >= 0]
    truth = sum(x * r for x, r in ok) / sum(x for x, _ in ok)
    res = out["calls"][0]["result"]
    assert res["totals"]["weighted_rate"] == pytest.approx(truth)
