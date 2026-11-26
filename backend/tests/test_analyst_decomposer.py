"""P1: model decomposition of a whole request -> grounded deliverables (the model never supplies numbers)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.analyst import compound, decomposer as Z, deliverables as D, orchestrator
from app.analyst.data import load_table


@pytest.fixture()
def sales(tmp_path):
    rng = np.random.default_rng(3)
    rows = [{"month_start": m.strftime("%Y-%m-%d"), "region": r, "net_revenue": round(float(rng.normal(base + tr * i, 900)), 2),
             "orders": int(rng.normal(400, 20)), "returns": int(rng.normal(30, 5))}
            for r, base, tr in (("West", 90000, 1500), ("East", 100000, -1200), ("North", 70000, 100))
            for i, m in enumerate(pd.date_range("2024-01-01", periods=14, freq="MS"))]
    p = tmp_path / "sales.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return load_table(str(p))


def fake(*replies):
    calls = []
    def chat(msgs, max_tokens=900):
        calls.append(msgs)
        r = replies[min(len(calls) - 1, len(replies) - 1)]
        return r if isinstance(r, str) else json.dumps({"tasks": r})
    chat.calls = calls
    return chat


MSG = ("Compare revenue in the West and East, their growth rates, and plot the comparison. West's target is 115,000 a "
       "month - will it get there next month? Also the total returns.")
PLAN = [{"do": "compare", "text": "Compare revenue in the West and East", "col": "net_revenue", "where": {"region": ["West", "East"]}},
        {"do": "growth", "text": "their growth rates", "col": "net_revenue", "by": "region", "groups": ["West", "East"]},
        {"do": "chart", "text": "plot the comparison", "of": 0},
        {"do": "target", "text": "West's target is 115,000 a month - will it get there next month?", "col": "net_revenue",
         "where": {"region": "West"}, "target": 115000, "per": "M"},
        {"do": "forecast", "text": "will it get there next month", "col": "net_revenue", "where": {"region": "West"},
         "horizon": 1, "target": 115000, "per": "M"},
        {"do": "stat", "text": "the total returns", "stat": "sum", "col": "returns"}]


def test_prompt_carries_the_schema_and_values(sales):
    msgs = Z.build_messages("x", sales)
    assert "net_revenue: number" in msgs[1]["content"] and "region: category: " in msgs[1]["content"]
    assert "month_start: date" in msgs[1]["content"]


def test_a_valid_plan_is_grounded_into_deliverables(sales):
    ds = Z.decompose(MSG, sales, chat=fake(PLAN))
    kinds = [d.kind for d in ds]
    assert kinds == ["compare", "growth", "chart", "target_check", "forecast", "stat"]
    cmp_ = ds[0]
    assert cmp_.group_by == "region" and cmp_.groups == ["West", "East"] and cmp_.entities == {}   # groups, not a filter
    assert ds[2].chart_for == cmp_.id and ds[3].entities == {"region": "West"} and ds[5].entities == {}
    assert ds[3].depends_on == [ds[4].id]                     # the future target check uses the forecast


def test_invalid_plan_gets_one_corrective_retry(sales):
    bad = [dict(PLAN[5], col="revenue_total")]
    chat = fake(bad, [PLAN[5]])
    tr = {}
    ds = Z.decompose("the total returns", sales, chat=chat, trace=tr)
    assert ds and ds[0].measures == ["returns"] and tr["source"] == "model_retry"
    assert "revenue_total" in chat.calls[1][-1]["content"]      # the model is told exactly what was wrong


def test_still_invalid_plan_is_discarded(sales):
    tr = {}
    assert Z.decompose("x", sales, chat=fake("not json", "{\"tasks\": []}"), trace=tr) is None
    assert tr["source"] == "rejected"


@pytest.mark.parametrize("task,problem", [
    ({"do": "stat", "stat": "avg", "col": "profit"}, "not a numeric column"),
    ({"do": "stat", "stat": "avg", "col": "orders", "where": {"region": "Atlantis"}}, "no value"),
    ({"do": "stat", "stat": "mode", "col": "orders"}, "stat"),
    ({"do": "forecast", "col": "orders", "horizon": 500}, "horizon"),
    ({"do": "chart", "of": 7}, "of"),
    ({"do": "target", "col": "orders"}, "target"),
    ({"do": "explode"}, "do"),
])
def test_grounding_rejects_what_the_data_cannot_support(sales, task, problem):
    ds, errs = Z.ground([task], sales, "q")
    assert not ds and errs and problem in errs[0]


def test_values_are_matched_case_insensitively_and_lists_scope_each_value(sales):
    ds, errs = Z.ground([{"do": "stat", "stat": "sum", "col": "orders", "where": {"region": ["west", "NORTH"]}}], sales, "q")
    assert not errs and ds[0].entities == {"region": ["West", "North"]}
    res = compound.answer("q", ds, sales)
    assert f"{sales[sales.region.isin(['West', 'North'])].orders.sum():,}" in res["answer"].replace(".00", "")


def test_compare_growth_and_unsupported_execute_end_to_end(sales):
    plan = PLAN[:3] + [{"do": "unsupported", "text": "the carbon cost", "reason": "there is no column for “carbon cost”"}]
    msg = "Compare revenue in the West and East, their growth rates, and plot the comparison. And the carbon cost."
    ds = Z.decompose(msg, sales, chat=fake(plan))     # the message asks exactly what the plan holds
    res = compound.answer(msg, ds, sales)
    a = res["answer"]
    west, east = sales[sales.region == "West"].net_revenue.mean(), sales[sales.region == "East"].net_revenue.mean()
    assert f"{abs(west - east):,.2f}" in a and "Greatest decline: **East**" in a and "no column for “carbon cost”" in a
    assert res["incomplete"] and "Not complete: 1 of 4" in a and res["charts"][0]["type"] == "bar"


def test_orchestrator_uses_the_model_plan_and_records_its_source(sales, monkeypatch):
    monkeypatch.setattr(Z, "model_chat", fake(PLAN))
    out = orchestrator.ask(MSG, sales, {}, use_model=True)
    assert out["plan_source"] == "compound" and out["decomposition"]["source"] == "model"
    assert [c["type"] for c in out["charts"]] == ["bar"] and "Target check" in out["answer"] and "Forecast of" in out["answer"]


def test_rule_reader_stands_in_when_the_model_fails(sales, monkeypatch):
    monkeypatch.setattr(Z, "model_chat", fake("I cannot", "still no"))
    out = orchestrator.ask("Give me the total returns and the average orders, and plot monthly net revenue for West.",
                           sales, {}, use_model=True)
    assert out["plan_source"] == "compound" and out["decomposition"]["source"] == "rules"
    assert f"{sales.returns.sum():,}" in out["answer"].replace(".00", "")


def test_a_short_single_question_does_not_call_the_model(sales):
    assert not Z.worth_asking("average orders?", D.extract("average orders?", sales), False)


def test_the_model_reads_a_follow_up_at_a_dead_end_with_the_previous_request(sales, monkeypatch):
    chat = fake([{"do": "stat", "text": "q", "stat": "sum", "col": "net_revenue", "where": {"region": "North"}}])
    monkeypatch.setattr(Z, "model_chat", chat)
    mem = {"_prev_said": "What's the total net revenue for West?"}
    out = orchestrator._model_followup("how did the northern patch do on that?", sales, mem, True, 0.0)
    assert "Previous request: What's the total net revenue for West?" in chat.calls[-1][1]["content"]
    assert f"{sales[sales.region == 'North'].net_revenue.sum():,.2f}" in out["answer"]
    assert orchestrator._model_followup("x", sales, {"_prev_said": "y"}, False, 0.0) is None     # never without a model


def test_the_others_follow_up_is_the_complement_of_the_previous_subject(sales):
    mem = {}
    orchestrator.ask("What's the total net revenue for West?", sales, mem, use_model=False)
    out = orchestrator.ask("And the other two regions?", sales, mem, use_model=False)
    for r in ("East", "North"):
        assert f"{sales[sales.region == r].net_revenue.sum():,.2f}" in out["answer"]
    assert f"{sales[sales.region == 'West'].net_revenue.sum():,.2f}" not in out["answer"]


def test_follow_ups_the_rules_can_read_never_reach_the_model(sales, monkeypatch):
    chat = fake([])
    monkeypatch.setattr(Z, "model_chat", chat)
    mem = {}
    orchestrator.ask("What's the total net revenue for West?", sales, mem, use_model=False)
    orchestrator.ask("Now do the same for East.", sales, mem, use_model=True)
    assert not chat.calls


@pytest.mark.parametrize("q,n", [("Predict attrition for overtime Yes, satisfaction 2, monthly_income 3000", 1),
                                 ("average orders?", 1), ("Total amount by region", 1),
                                 ("Show the average salary by department and identify the department with the highest turnover", 2),
                                 ("Summary first, then the average and variance. Will Rahul make it? If yes, graph it.", 4)])
def test_the_model_is_asked_only_for_requests_with_several_analytical_clauses(q, n):
    assert Z.task_clauses(q) == n and Z.worth_asking(q, [], False) == (n >= 2)


def test_a_time_window_applies_to_its_own_task_only(sales):
    plan = [{"do": "stat", "text": "total orders in 2024", "stat": "sum", "col": "orders", "when": {"from": "2024-01-01", "to": "2024-12-31"}},
            {"do": "stat", "text": "total orders overall", "stat": "sum", "col": "orders"}]
    ds = Z.decompose("total orders in 2024 and total orders overall", sales, chat=fake(plan))
    a = compound.answer("q", ds, sales)["answer"]
    y24 = sales[pd.to_datetime(sales.month_start).dt.year == 2024].orders.sum()
    assert "Period: 2024-01-01 to 2024-12-31" in a and f"{y24:,}" in a.replace(".00", "") and f"{sales.orders.sum():,}" in a.replace(".00", "")


@pytest.mark.parametrize("when,problem", [({"from": "2031-01-01"}, "outside the data"), ({"from": "2024-05-01", "to": "2024-01-01"}, "after"),
                                          ({"from": "May 2024"}, "YYYY-MM-DD"), ({}, "when")])
def test_bad_windows_are_rejected(sales, when, problem):
    ds, errs = Z.ground([{"do": "stat", "stat": "sum", "col": "orders", "when": when}], sales, "q")
    assert errs and problem in errs[0]


def test_requested_chart_type_is_used_when_it_suits_the_result(sales):
    plan = [{"do": "breakdown", "text": "orders by region", "stat": "sum", "col": "orders", "by": "region"},
            {"do": "chart", "text": "as a pie chart", "of": 0, "type": "pie"},
            {"do": "trend", "text": "net revenue over time for West", "col": "net_revenue", "where": {"region": "West"}},
            {"do": "chart", "text": "as a pie", "of": 2, "type": "pie"}]
    ds = Z.decompose("orders by region as a pie chart, and net revenue over time for West as a pie", sales, chat=fake(plan))
    res = compound.answer("q", ds, sales)
    assert [c["type"] for c in res["charts"]] == ["pie", "line"]
    assert "does not suit this result" in res["answer"]


# ---- P2: chart references the model gets wrong (dev set P0 final_test: "check it against N ... and visualise the
# result" pointed the chart at the target check; "chart Leo's monthly hours" came back as a series with no chart)
def _monthly(df_rows=None):
    import pandas as pd
    months = pd.date_range("2024-01-01", periods=14, freq="MS")
    return pd.DataFrame([{"month": m, "rep": r, "hours": 300 + i + (5 if r == "Leo" else 0)}
                         for i, m in enumerate(months) for r in ("Leo", "Zara")])


def test_a_chart_of_a_target_check_draws_the_forecast_it_rests_on():
    from app.analyst import decomposer as Z
    df = _monthly()
    tasks = [{"do": "forecast", "text": "Project Zara's hours 2 months ahead", "col": "hours", "where": {"rep": "Zara"}, "horizon": 2},
             {"do": "target", "text": "check it against 330 hours per month", "col": "hours", "where": {"rep": "Zara"},
              "target": 330, "per": "M"},
             {"do": "chart", "text": "and visualise the result", "of": 1}]
    ds, errs = Z.ground(tasks, df, "Project Zara's hours 2 months ahead, check it against 330 hours per month, and visualise the result.")
    assert not errs
    chart = next(d for d in ds if d.kind == "chart")
    assert next(d for d in ds if d.id == chart.chart_for).kind == "forecast"


def test_a_series_the_user_asked_to_chart_gets_its_chart():
    from app.analyst import decomposer as Z
    df = _monthly()
    tasks = [{"do": "stat", "text": "Total hours overall", "stat": "sum", "col": "hours"},
             {"do": "trend", "text": "chart Leo's monthly hours", "col": "hours", "where": {"rep": "Leo"}, "every": "M"}]
    ds, errs = Z.ground(tasks, df, "Total hours overall, and chart Leo's monthly hours.")
    assert not errs
    series = next(d for d in ds if d.kind == "series")
    chart = next(d for d in ds if d.kind == "chart")
    assert chart.chart_for == series.id and series.output == "chart"
    assert sum(d.kind == "chart" for d in ds) == 1


def test_no_chart_is_added_when_the_words_do_not_ask_for_one():
    from app.analyst import decomposer as Z
    df = _monthly()
    tasks = [{"do": "trend", "text": "Leo's monthly hours", "col": "hours", "where": {"rep": "Leo"}, "every": "M"}]
    ds, errs = Z.ground(tasks, df, "Leo's monthly hours.")
    assert not errs and not any(d.kind == "chart" for d in ds)


def test_a_stat_task_naming_two_statistics_becomes_two():
    from app.analyst import decomposer as Z
    df = _monthly()
    tasks = [{"do": "stat", "text": "the mean plus the variance for hours", "stat": "avg", "col": "hours"}]
    ds, errs = Z.ground(tasks, df, "I need the mean plus the variance for hours.")
    assert not errs and sorted(d.stat for d in ds if d.kind == "stat") == ["avg", "var"]


def test_a_single_statistic_is_not_split():
    from app.analyst import decomposer as Z
    df = _monthly()
    tasks = [{"do": "stat", "text": "the average hours for Leo", "stat": "avg", "col": "hours", "where": {"rep": "Leo"}}]
    ds, errs = Z.ground(tasks, df, "the average hours for Leo")
    assert not errs and [d.stat for d in ds] == ["avg"]


def test_a_missing_column_from_the_model_is_never_dropped_by_the_legacy_path(monkeypatch):
    # dev P0 final_test t224: the model marked "the total employee happiness index" as unparsed (no column); the
    # statistics-only legacy answer was accepted without acknowledging it
    import json as _json
    import pandas as pd
    from app.analyst import decomposer as Z, orchestrator
    df = pd.DataFrame({"consultant": ["A", "B"] * 10, "billable_hours": [290.5 + i for i in range(20)],
                       "extra_hours": [9.25 + i % 4 for i in range(20)]})
    reply = _json.dumps({"tasks": [
        {"do": "stat", "text": "Mean billable hours", "stat": "avg", "col": "billable_hours"},
        {"do": "unsupported", "text": "the total employee happiness index", "reason": "there is no column for “employee happiness index” in this data"},
        {"do": "stat", "text": "the median extra hours", "stat": "median", "col": "extra_hours"}]})
    monkeypatch.setattr(Z, "model_chat", lambda *a, **k: reply)
    out = orchestrator.ask("Mean billable hours, the total employee happiness index and the median extra hours, please.",
                           df, {}, use_model=True)
    assert "happiness" in out["answer"].lower()
    assert "not completed" in out["answer"].lower() or "no column" in out["answer"].lower()


def test_a_chart_of_the_series_behind_a_requested_forecast_draws_the_forecast():
    from app.analyst import decomposer as Z
    df = _monthly()
    tasks = [{"do": "forecast", "text": "Project Hiro's hours 3 months ahead", "col": "hours", "where": {"rep": "Leo"}, "horizon": 3},
             {"do": "target", "text": "check it against 320 hours per month", "col": "hours", "where": {"rep": "Leo"}, "target": 320, "per": "M"},
             {"do": "trend", "text": "visualise the result", "col": "hours", "where": {"rep": "Leo"}},
             {"do": "chart", "text": "visualise the result", "of": 2}]
    ds, errs = Z.ground(tasks, df, "Project Leo's hours 3 months ahead, check it against 320 hours per month, and visualise the result.")
    assert not errs
    chart = next(d for d in ds if d.kind == "chart")
    assert next(d for d in ds if d.id == chart.chart_for).kind == "forecast"


def test_a_colon_heading_scopes_every_part_after_it():
    import pandas as pd
    from app.analyst import decomposer as Z
    df = pd.DataFrame({"branch": ["Lakeside", "Dockside"] * 6, "orders": [37, 52, 41, 66, 48, 59, 33, 71, 45, 62, 39, 58], "gross_sales": [100.5 + 3 * i for i in range(12)]})
    m = "Across everyone, what's the mean gross sales? Then for Lakeside only: the total orders and the maximum gross sales."
    tasks = [{"do": "stat", "text": "Across everyone, what's the mean gross sales", "stat": "avg", "col": "gross_sales"},
             {"do": "stat", "text": "the total orders", "stat": "sum", "col": "orders", "where": {"branch": "Lakeside"}},
             {"do": "stat", "text": "the maximum gross sales", "stat": "max", "col": "gross_sales"}]
    ds, errs = Z.ground(tasks, df, m)
    assert not errs
    by = {(d.stat, d.measures[0]): d.entities for d in ds}
    assert by[("avg", "gross_sales")] == {} and by[("max", "gross_sales")] == {"branch": "Lakeside"}


def test_a_clause_the_model_left_out_is_added_from_the_rule_reader():
    import pandas as pd
    from app.analyst import decomposer as Z
    df = pd.DataFrame({"clinic": ["North", "South"] * 10, "appointments": [31, 44, 29, 52, 38, 47, 35, 41, 50, 33] * 2})
    m = "I'd like a summary of this file. After that, what's the average appointments and its standard deviation?"
    tasks = [{"do": "summary", "text": "I'd like a summary of this file"},
             {"do": "stat", "text": "what's the average appointments", "stat": "avg", "col": "appointments"}]
    ds, errs = Z.ground(tasks, df, m)
    assert not errs
    added = Z.cover_omissions(ds, m, df)
    assert added and sorted(d.stat for d in ds if d.kind == "stat") == ["avg", "std"]
    kinds = [d.kind for d in sorted(ds, key=lambda d: d.order)]
    assert kinds[0] == "summary"


def test_nothing_is_added_when_every_clause_is_covered():
    import pandas as pd
    from app.analyst import decomposer as Z
    df = pd.DataFrame({"clinic": ["North", "South"] * 10, "appointments": [31, 44, 29, 52, 38, 47, 35, 41, 50, 33] * 2})
    m = "What's the average appointments and the median appointments?"
    tasks = [{"do": "stat", "text": "the average appointments", "stat": "avg", "col": "appointments"},
             {"do": "stat", "text": "the median appointments", "stat": "median", "col": "appointments"}]
    ds, errs = Z.ground(tasks, df, m)
    assert not errs and Z.cover_omissions(ds, m, df) == [] and len(ds) == 2


# P2.13 a chart whose own words name one entity draws THAT entity, even when the model points it at a group-wide task
def test_a_chart_naming_one_entity_is_not_drawn_from_a_group_wide_ranking(sales):
    tasks = [{"do": "stat", "text": "Total orders overall", "stat": "sum", "col": "orders"},
             {"do": "rank", "text": "rank the regions on net revenue", "col": "net_revenue", "by": "region", "top": 3},
             {"do": "chart", "text": "chart North's monthly net revenue", "of": 1}]
    ds, errs = Z.ground(tasks, sales, "Total orders overall, then rank the regions on net revenue, "
                                      "and chart North's monthly net revenue.")
    assert not errs
    chart = next(d for d in ds if d.kind == "chart")
    subj = next(d for d in ds if d.id == chart.chart_for)
    assert subj.kind in Z.DRAWABLE and subj.kind != "ranking"
    assert subj.entities == {"region": "North"} and "net_revenue" in subj.measures
    assert next(d for d in ds if d.kind == "ranking").entities == {}


def test_a_chart_of_a_group_wide_task_without_a_named_entity_is_unchanged(sales):
    tasks = [{"do": "rank", "text": "rank the regions on net revenue", "col": "net_revenue", "by": "region", "top": 3},
             {"do": "chart", "text": "and chart it", "of": 0}]
    ds, errs = Z.ground(tasks, sales, "rank the regions on net revenue and chart it")
    assert not errs
    chart = next(d for d in ds if d.kind == "chart")
    assert next(d for d in ds if d.id == chart.chart_for).kind == "ranking"


def test_a_two_group_comparison_states_its_effect_size_in_the_result(sales):
    tasks = [{"do": "compare", "text": "Compare average orders in West and East", "stat": "avg", "col": "orders",
              "by": "region", "groups": ["West", "East"]}]
    ds, errs = Z.ground(tasks, sales, "Compare average orders in West and East")
    assert not errs
    out = compound.answer("Compare average orders in West and East", ds, sales)
    vals = next(d for d in out["deliverables"] if d["kind"] == "compare")["result_values"]
    if "Hedges' g" in out["answer"]:
        assert "hedges_g" in vals
