"""Multi-turn conversations: a clarification, a confirmation, a correction or a follow-up continues the task the
user started; it is not answered as a new, unrelated question.

Each test checks the number shown against an independent computation here AND the request that was actually run
(`understood_as`, the calculator plan), so a reply that sounds conversational but does not execute the task fails.
The model is unavailable unless a test says otherwise: conversation state must not depend on it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.analyst import calc, orchestrator, planner


def _hhmm(t: str) -> float:
    h, m = t.split(":")
    return int(h) + int(m) / 60


def timesheet() -> pd.DataFrame:
    """The shape of the reported dataset: names, departments, clock times, an hourly rate; one exact duplicate, one
    negative rate and one end-before-start record."""
    rows = [("E1", "Rahul Sharma", "Engineering", "2026-09-01", "09:00", "17:00", 25),
            ("E2", "Aarav Mehta", "Engineering", "2026-09-01", "09:00", "18:30", 30),
            ("E3", "Priya Singh", "HR", "2026-09-01", "09:30", "17:30", 22),
            ("E1", "Rahul Sharma", "Engineering", "2026-09-02", "09:00", "19:00", 25),
            ("E2", "Aarav Mehta", "Engineering", "2026-09-02", "10:00", "18:00", 30),
            ("E3", "Priya Singh", "HR", "2026-09-02", "09:00", "17:00", 22),
            ("E4", "Vikram Rao", "Finance", "2026-09-03", "08:00", "20:00", 28),
            ("E5", "Neha Kapoor", "Finance", "2026-09-03", "09:00", "17:00", -10),
            ("E7", "Karan Verma", "Marketing", "2026-09-03", "10:00", "09:00", 24),
            ("E1", "Rahul Sharma", "Engineering", "2026-09-01", "09:00", "17:00", 25)]
    df = pd.DataFrame(rows, columns=["employee_id", "employee_name", "department", "date", "clock_in", "clock_out",
                                     "hourly_rate"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def _hours(df, who=None, dedup=False):
    d = df.drop_duplicates() if dedup else df
    d = d[d.employee_name == who] if who else d
    h = [_hhmm(o) - _hhmm(i) for i, o in zip(d.clock_in, d.clock_out)]
    return [x for x in h if x >= 0]


@pytest.fixture(autouse=True)
def model_offline(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)


def say(mem, df, *messages, **kw):
    out = None
    for m in messages:
        out = orchestrator.ask(m, df, mem, **kw)
    return out


def calc_result(out):
    return next(c["result"] for c in out["calls"] if c["tool"] == "calculator")


GENERIC = "I can answer that best with a concrete calculation"


# ------------------------------------------------------------- the reported one --
def test_the_reported_conversation_is_answered_not_turned_into_generic_suggestions():
    df, mem = timesheet(), {}
    first = say(mem, df, "What is the average time Rahul works?")
    assert "Which numeric column" not in first["answer"] and "hourly_rate" not in first["answer"].split("\n")[0]
    rahul = _hours(df, "Rahul Sharma")
    assert calc_result(first)["totals"]["shift_hours"] == pytest.approx(sum(rahul) / len(rahul))
    assert calc_result(first)["aggregates"]["shift_hours"].startswith("mean(")
    for reply in ("yes", "okay do it"):
        out = say(mem, df, reply)
        assert GENERIC not in out["answer"] and "For example:" not in out["answer"]


# ------------------------------------------------- clarification -> answer -> run --
def test_a_clarification_answer_is_folded_into_the_original_request_and_run():
    df, mem = timesheet(), {}
    ask = say(mem, df, "How much overtime did each employee work?")
    assert ask["plan"]["type"] == "clarify" and mem["task"]["status"] == "awaiting"
    assert mem["task"]["pending"]["expects"] == "number"
    out = say(mem, df, "8 hours")
    assert out["understood_as"].startswith("How much overtime did each employee work") and "8" in out["understood_as"]
    res = calc_result(out)
    assert "8" in res["formulas"][-1]["formula"], "the threshold the user gave is the one applied"
    truth = {}
    for n, i, o in zip(df.employee_name, df.clock_in, df.clock_out):
        h = _hhmm(o) - _hhmm(i)
        if h >= 0:
            truth[n] = truth.get(n, 0) + max(h - 8, 0)
    assert {g["group"]: g["overtime_hours"] for g in res["groups"]} == pytest.approx(truth)
    assert mem["task"]["status"] == "done"


@pytest.mark.parametrize("reply", ["yes", "Okay, do it", "go ahead", "sure", "Proceed.", "calculate it", "please continue"])
def test_a_confirmation_runs_the_pending_request_with_its_only_option(reply):
    df = pd.DataFrame({"team": ["red", "red", "blue", "blue"], "score": [3.0, 5.0, 4.0, 8.0]})
    mem = {}
    ask = say(mem, df, "What is the average for each team?")
    assert ask["plan"]["type"] == "clarify" and ask["plan"]["options"] == ["score"]
    out = say(mem, df, reply)
    assert out["understood_as"] == "What is the average for each team using score?"
    assert out["calls"] and out["calls"][0]["status"] == "ok"
    assert "6.00" in out["answer"] and "4.00" in out["answer"]


def test_an_ambiguous_confirmation_is_asked_once_and_the_task_is_kept():
    df = pd.DataFrame({"team": ["x", "y"], "gross_sales": [100, 200], "net_sales": [90, 180], "cost": [50, 60]})
    mem = {}
    say(mem, df, "What is the profit by team?")
    pending = dict(mem["task"]["pending"])
    out = say(mem, df, "yes")
    assert out["plan"]["type"] == "clarify" and not out["calls"]
    assert "gross_sales" in out["answer"] and "net_sales" in out["answer"] and GENERIC not in out["answer"]
    assert mem["task"]["status"] == "awaiting" and mem["task"]["pending"]["options"] == pending["options"]
    done = say(mem, df, "No, use the other one")
    assert {g["group"]: g["profit"] for g in calc_result(done)["groups"]} == {"x": 40, "y": 120}, "net sales − cost"


def test_successive_clarifications_keep_the_original_request():
    df = pd.DataFrame({"team": ["x", "y", "x"], "gross_sales": [100, 200, 50], "net_sales": [90, 180, 40],
                       "cost": [50, 60, 10]})
    mem = {}
    first = say(mem, df, "Show team totals")
    assert first["plan"]["type"] == "clarify" and "profit" in first["plan"]["options"]
    second = say(mem, df, "profit")
    assert second["plan"]["type"] == "clarify" and "net_sales" in second["answer"], "which revenue column?"
    assert mem["task"]["original"].startswith("Show team totals")
    done = say(mem, df, "net sales")
    res = calc_result(done)
    assert res["group_by"] == "team" and {g["group"]: g["profit"] for g in res["groups"]} == {"x": 70, "y": 120}


# ------------------------------------------------------- follow-ups in one task --
def test_a_long_conversation_keeps_one_task():
    df, mem = timesheet(), {}
    avg = say(mem, df, "Calculate average working hours.")
    every = _hours(df)
    assert calc_result(avg)["totals"]["shift_hours"] == pytest.approx(sum(every) / len(every))
    only = say(mem, df, "Only Rahul.")
    r = _hours(df, "Rahul Sharma")
    assert only["plan_source"] == "refine" and calc_result(only)["totals"]["shift_hours"] == pytest.approx(sum(r) / len(r))
    dedup = say(mem, df, "Exclude duplicates.")
    rd = _hours(df, "Rahul Sharma", dedup=True)
    assert calc_result(dedup)["totals"]["shift_hours"] == pytest.approx(sum(rd) / len(rd))
    chart = say(mem, df, "Show it as a chart.")
    assert chart["charts"] and sorted(d["shift hours"] for d in chart["charts"][0]["data"]) == pytest.approx(sorted(rd))
    explain = say(mem, df, "Explain that in simple words.")
    assert explain["plan_source"].startswith("conversation") and not any(c.get("tool") == "calculator" for c in explain["calls"])
    nxt = say(mem, df, "Now estimate his next shift.")
    est = nxt["calls"][0]["result"]
    assert nxt["calls"][0]["tool"] == "next_record_estimate" and est["estimate"] == pytest.approx(sum(rd) / len(rd))
    assert est["records"] == len(rd) and "Rahul Sharma" in est["scope"] and "not a machine-learning" in nxt["answer"]


def test_entity_replacement_filter_removal_and_metric_correction():
    df, mem = timesheet(), {}
    say(mem, df, "What is the average time Rahul works?")
    swap = say(mem, df, "Actually, use Aarav instead")
    a = _hours(df, "Aarav Mehta")
    assert swap["plan_source"] == "refine" and calc_result(swap)["totals"]["shift_hours"] == pytest.approx(sum(a) / len(a))
    assert "Rahul" not in swap["answer"].split("\n")[0]
    rate = say(mem, df, "No, I meant hourly rate")
    assert rate["understood_as"] == "What is the average hourly rate for Aarav Mehta?"
    assert "30.00" in rate["answer"]
    everyone = say(mem, df, "Total labor cost by department", "Only Rahul.", "For everyone.")
    assert calc_result(everyone)["group_by"] == "department" and len(calc_result(everyone)["groups"]) > 1


def test_a_topic_change_drops_the_previous_filters():
    df, mem = timesheet(), {}
    say(mem, df, "What is the average time Rahul works?")
    assert mem["scope"]["entities"]
    out = say(mem, df, "Forget that, show department totals")
    assert out["plan"]["type"] == "clarify" and "labor cost" in out["plan"]["options"]
    assert not (mem.get("scope") or {}).get("entities"), "Rahul is no longer selected"
    done = say(mem, df, "labor cost")
    res = calc_result(done)
    assert res["group_by"] == "department" and sum(g["rows"] for g in res["groups"]) > 3


def test_an_offered_action_is_taken_on_yes():
    df, mem = timesheet(), {}
    first = say(mem, df, "Total labor cost by department")
    assert "exclude duplicates" in first["answer"]
    out = say(mem, df, "yes")
    assert out["understood_as"] == "exclude duplicates" and out["plan_source"] == "refine"
    assert any("duplicate()" in e["where"] for e in calc_result(out)["exclusions"])


# --------------------------------------------------------------- robustness --
def test_a_model_failure_during_a_follow_up_does_not_lose_the_task(monkeypatch):
    df, mem = timesheet(), {}
    say(mem, df, "How much overtime did each employee work?")
    monkeypatch.setattr(planner, "model_available", lambda: True)

    def down(*a, **k):
        raise ConnectionError("model server unreachable")
    monkeypatch.setattr(planner, "model_chat", down)
    monkeypatch.setattr(calc, "_complete", down)
    out = say(mem, df, "beyond 8 hours", use_model=True)
    assert out["calls"] and calc_result(out)["groups"], "the pending task was completed deterministically"


def test_state_is_separate_per_dataset_and_per_conversation():
    df = timesheet()
    other = pd.DataFrame({"team": ["a", "b"], "score": [1.0, 2.0]})
    a = orchestrator.memory_for("t:u:1:None:conv-1")
    b = orchestrator.memory_for("t:u:2:None:conv-1")
    c = orchestrator.memory_for("t:u:1:None:conv-2")
    a.clear(), b.clear(), c.clear()
    say(a, df, "How much overtime did each employee work?")
    assert a["task"]["status"] == "awaiting"
    out_b = say(b, other, "yes")
    assert not out_b["calls"] and "task" not in b or b["task"].get("status") != "awaiting"
    out_c = say(c, df, "8")
    assert not any(cl.get("tool") == "calculator" for cl in out_c["calls"]), "another conversation's question is not answered"
    assert a["task"]["status"] == "awaiting", "the first conversation still has its open question"


def test_a_new_question_while_a_clarification_is_open_is_a_new_request():
    df, mem = timesheet(), {}
    say(mem, df, "How much overtime did each employee work?")
    out = say(mem, df, "What is the total labor cost by department?")
    assert out["understood_as"] if "understood_as" in out else True
    assert "understood_as" not in out, "not folded into the overtime question"
    assert calc_result(out)["group_by"] == "department"
