"""Stage 4 of the Copilot reliability repair: one conversation record (memory["last_result"]) and follow-ups that
apply to the LATEST result only.

Multi-turn conversations on the seeded datasets of Stages 1-3. Every follow-up answer is compared with pandas; the
cases that must not be treated as follow-ups (a new question, a defined metric, a finding with no investigation) are
checked too, and so is the rule that an older result never stands in for the latest one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator
from app.analyst.data import load_table

from test_analyst_stage1_guards import TODAY, _creators, _orders, _shifts, _tickets  # noqa: F401


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = np.random.default_rng(7)
    root = tmp_path_factory.mktemp("stage4")
    out = {}
    for name, make in (("tickets", _tickets), ("shifts", _shifts), ("orders", _orders), ("creators", _creators)):
        raw = make(rng)
        path = root / f"{name}.csv"
        raw.to_csv(path, index=False)
        out[name] = (load_table(str(path)), raw)
    return out


def chat(df, *turns):
    mem, outs = {}, []
    for q in turns:
        outs.append(orchestrator.ask(q, df, mem, use_model=False))
    return outs, mem


def value(out):
    for c in out["calls"]:
        if c.get("status") != "ok":
            continue
        r = c.get("result") or {}
        if c["tool"] == "analysis_program":
            res = r["results"][""]
            return (res.get("per_unit") or res.get("result"))["value"]
        if c["tool"] == "compute_metric":
            return {str(g["group"]): g["value"] for g in r["groups"]} if "groups" in r else r["value"]
    return None


def chart_values(out):
    ch = out["charts"][0]
    return {d[ch["x"]]: d[ch["y"]] for d in ch["data"]}


def no_dead_end(out):
    return "I couldn't turn" not in out["answer"]


# ------------------------------------------------------------- the named follow-ups --
def test_what_about_only_completed_orders_keeps_the_breakdown(data):
    df, raw = data["orders"]
    (first, only), mem = chat(df, "Total amount by region", "What about only completed orders?")
    want = raw[raw.status == "Completed"].groupby("region").amount.sum().to_dict()
    assert value(only) == pytest.approx(want) and only["verify_status"] == "verified"
    assert "changed: status = Completed" in only["answer"] and "by region" in only["answer"]
    assert mem["last_result"]["turn"] == 2


def test_what_about_another_person_replaces_the_person(data):
    df, raw = data["tickets"]
    (_, ben), _ = chat(df, "Average resolution hours for Asha Rao", "What about Ben Cole?")
    assert value(ben) == pytest.approx(raw[raw.agent == "Ben Cole"].resolution_hours.mean())
    assert ben["verify_status"] == "verified" and "Asha Rao" not in ben["answer"].split("_Follow-up")[1].split("\n")[1]


def test_why_explains_the_latest_result_and_does_not_replace_it(data):
    df, raw = data["tickets"]
    (_, _, why, cmp_), mem = chat(df, "Average resolution hours for Asha Rao", "What about Ben Cole?", "Why?",
                                  "Compare it with Chen Wei")
    assert why["calls"][0]["tool"] == "drivers" and "not proven causes" in why["answer"]
    assert why["verify_status"] == "not independently verified"
    # "compare it" still refers to Ben Cole's figure -- the explanation is not a new result
    want = raw[raw.agent.isin(["Ben Cole", "Chen Wei"])].groupby("agent").resolution_hours.mean().to_dict()
    assert value(cmp_) == pytest.approx(want) and cmp_["verify_status"] == "verified"


def test_compare_it_with_on_another_dataset(data):
    df, raw = data["shifts"]
    (_, cmp_), _ = chat(df, "average hourly rate in Engineering", "Compare it with Finance")
    want = raw[raw.department.isin(["Engineering", "Finance"])].groupby("department").hourly_rate.mean().to_dict()
    assert value(cmp_) == pytest.approx(want)


def test_compare_with_one_value_on_a_full_breakdown_asks_which(data):
    df = data["shifts"][0]
    (_, ask_), _ = chat(df, "average hourly rate by department", "Compare it with Support")
    assert ask_["plan"]["type"] == "clarify" and not any(c.get("status") == "ok" for c in ask_["calls"])
    assert set(ask_["plan"]["options"]) == {"Engineering", "Finance"}


def test_only_a_yes_no_column_filters_the_latest_result(data):
    df, raw = data["shifts"]
    (_, ot), _ = chat(df, "average hourly rate by department", "What about only overtime shifts?")
    want = raw[raw.overtime == 1].groupby("department").hourly_rate.mean().to_dict()
    assert value(ot) == pytest.approx(want) and ot["verify_status"] == "verified"


def test_why_over_a_filtered_result_uses_the_same_records(data):
    df, raw = data["shifts"]
    (_, _, why), _ = chat(df, "average hourly rate by department", "What about only overtime shifts?", "Why?")
    assert f"over the same {int((raw.overtime == 1).sum()):,} records" in why["answer"]


def test_a_period_follow_up_keeps_everything_else(data):
    df, raw = data["tickets"]
    (_, chat_, recent), _ = chat(df, "How many tickets were escalated?", "What about for Chat?",
                                 "What about the last 30 days?")
    base = raw[(raw.escalated == "Yes") & (raw.channel == "Chat")]
    assert value(chat_) == pytest.approx(len(base))
    d = base.opened.dt.normalize()
    assert value(recent) == pytest.approx(int(((d >= TODAY - pd.Timedelta(days=29)) & (d <= TODAY)).sum()))


# ------------------------------------------------------------------ charts and staleness --
def test_a_chart_after_average_instead_draws_the_averages_not_the_older_totals(data):
    df, raw = data["orders"]
    outs, _ = chat(df, "Total amount by region", "What about only completed orders?", "Now show the average instead",
                   "Show that as a chart")
    avg = raw[raw.status == "Completed"].groupby("region").amount.mean().to_dict()
    assert value(outs[2]) == pytest.approx(avg) and outs[2]["verify_status"] == "verified"
    assert chart_values(outs[3]) == pytest.approx(avg)
    assert outs[3]["verify_status"] == "not independently verified"     # a redraw re-presents; it is not re-verified


def test_a_chart_of_a_single_value_says_so_about_that_value(data):
    df = data["orders"][0]
    outs, _ = chat(df, "Total amount by region", "How many customers are there?", "Show that as a chart")
    assert not outs[2]["charts"] and "How many customers are there?" in outs[2]["answer"]
    assert "single value" in outs[2]["answer"]


def test_a_new_question_is_not_a_follow_up_and_inherits_no_filter(data):
    df, raw = data["orders"]
    outs, _ = chat(df, "Total amount by region", "What about only completed orders?", "How many customers are there?")
    assert value(outs[2]) == pytest.approx(raw.customer_id.nunique())          # all customers, not completed only
    df, raw = data["shifts"]
    outs, _ = chat(df, "average hourly rate for Mei Lin", "What about Tom Baker?", "Total labor cost by department")
    assert "couldn't answer this within the records" not in outs[2]["answer"]
    groups = {g["group"] for g in next(c for c in outs[2]["calls"] if c["tool"] == "calculator")["result"]["groups"]}
    assert groups == set(raw.department.unique())


def test_the_latest_result_record_is_written_once_per_computed_answer(data):
    df = data["tickets"][0]
    outs, mem = chat(df, "Average resolution hours for Asha Rao", "Hello", "What about Ben Cole?", "Why?")
    lr = mem["last_result"]
    assert lr["question"] == "What about Ben Cole?" and lr["turn"] == 3             # not the greeting, not the "why"
    assert lr["request"]["measures"] == ["resolution_hours"] and ["agent", "Ben Cole"] in [list(e) for e in lr["request"]["entities"]]


def test_older_records_are_not_refined_once_superseded(data):
    """The older follow-up handlers keep their keys (compatibility), but act only while those keys hold the latest
    result."""
    df, raw = data["orders"]
    outs, mem = chat(df, "Total amount by region", "What about only completed orders?", "Only East.")
    assert mem["last_calc"]["turn"] < mem["last_result"]["turn"]
    want = raw[(raw.status == "Completed") & (raw.region == "East")].amount.sum()
    got = value(outs[2])
    assert got == pytest.approx(want) or got == pytest.approx({"East": want})


# -------------------------------------------------------------------------- findings --
def test_a_finding_with_no_investigation_starts_nothing(data):
    for ds in ("orders", "tickets"):
        (out,), _ = chat(data[ds][0], "Tell me more about finding 2")
        assert not out["calls"] and "no numbered findings" in out["answer"]


def test_a_finding_after_an_investigation_is_that_finding(data):
    df = data["orders"][0]
    (inv, f2), _ = chat(df, "Analyse this dataset", "Tell me more about finding 2")
    findings = next(c for c in inv["calls"] if c["tool"] == "analysis_program")["result"]["findings"]
    assert f2["calls"][0]["tool"] == "investigation_finding"
    assert f2["calls"][0]["result"]["title"] == findings[1]["title"]


# ----------------------------------------------------------- creators (text numbers) --
def test_follow_ups_on_the_messy_file(data):
    df, raw = data["creators"]
    outs, _ = chat(df, "Average engagement rate by niche", "What about Food?", "And for Travel?")
    assert all(no_dead_end(o) for o in outs)
    assert f"{raw[raw.niche == 'Travel'].engagement_rate.mean():.2f}" in outs[2]["answer"]
