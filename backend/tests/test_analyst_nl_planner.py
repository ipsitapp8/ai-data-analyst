"""Model-assisted program planning (app/analyst/nl_planner.py) with a scripted model.

The model only writes JSON; these tests check that Silt validates it, repairs only what has one meaning, rejects the
rest, falls back to the deterministic paths whenever planning fails, keeps conversation references grounded, and that
every number in a model-planned answer is the one pandas computes. The schema (clinic visits) is new to every handler.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.analyst import nl_planner, orchestrator, planner, scope

TODAY = pd.Timestamp("2025-06-20")


def visits(seed=4, n=240):
    rng = np.random.default_rng(seed)
    ward = rng.choice(["Ash", "Birch", "Cedar"], n, p=[0.5, 0.3, 0.2])
    clin = rng.choice(["Dr Amara Okafor", "Dr Lena Fischer", "Dr Tomas Silva", "Dr Priya Nair"], n)
    mins = rng.normal(30, 6, n).round(1) + np.where(ward == "Cedar", 9, 0)
    readm = np.where(rng.random(n) < 0.08 + 0.2 * (mins > 38), "Yes", "No")
    return pd.DataFrame({"visit_ref": [f"V{i:04d}" for i in range(n)], "ward": ward, "clinician": clin,
                         "visit_date": pd.Timestamp("2025-04-01") + pd.to_timedelta(rng.integers(0, 80, n), "D"),
                         "consult_minutes": mins, "fee": (mins * 2.5).round(2), "readmitted": readm})


class Model:
    """A scripted local model: returns the given replies in order, records what it was sent."""

    def __init__(self, *replies):
        self.replies = [r if isinstance(r, (str, Exception)) else json.dumps(r) for r in replies]
        self.sent = []

    def __call__(self, messages, temperature=0.0):
        self.sent.append(messages)
        if not self.replies:
            raise AssertionError("the model was asked more times than scripted")
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(scope, "today", lambda: TODAY)
    # the trained tool planner and calculator (v1) are not under test here: the tool planner returns no plan (so its
    # deterministic rules answer, as when v1 fails) and the calculator fails, so only the program planner can answer
    # what the deterministic paths cannot
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "no plan")
    from app.analyst import calc
    monkeypatch.setattr(calc, "model_plan", lambda *a, **k: ({"type": "failed", "reason": "not under test"}, []))


def use(monkeypatch, *replies):
    m = Model(*replies)
    monkeypatch.setattr(nl_planner, "model_chat", m)
    return m


def ask(q, df, mem):
    return orchestrator.ask(q, df, mem, use_model=True, filename="visits.csv")


def program(steps, answer, objective="test objective"):
    return {"action": "program", "objective": objective, "steps": steps, "answer": answer}


PER_HEAD = program([
    {"id": "mins", "op": "aggregate", "from": "data", "fn": "sum", "column": "consult_minutes", "by": "ward"},
    {"id": "staff", "op": "aggregate", "from": "data", "fn": "nunique", "column": "clinician", "by": "ward"},
    {"id": "per_head", "op": "ratio", "a": "mins", "b": "staff"},
    {"id": "low", "op": "rank", "from": "per_head", "n": 1, "order": "asc"},
    {"id": "links", "op": "associate", "from": "data", "target": "readmitted"}], "per_head")
COMPOSITE_Q = "Which ward does the least consulting for the clinicians it has, and what goes with readmission?"


# --------------------------------------------------------------- execution --
def test_a_composite_program_is_validated_run_and_numerically_right(monkeypatch):
    df, mem = visits(), {}
    use(monkeypatch, PER_HEAD)
    out = ask(COMPOSITE_Q, df, mem)
    assert out["plan_source"] == "model_program" and out["plan"]["type"] == "program"
    res = out["calls"][0]["result"]["results"][""]
    truth = df.groupby("ward").consult_minutes.sum() / df.groupby("ward").clinician.nunique()
    for w, v in truth.items():
        assert res["per_head"]["value"][w] == pytest.approx(v)
    assert list(res["low"]["value"]) == [truth.idxmin()]
    assert res["links"]["value"]["target"] == "readmitted"
    assert "not causes" in out["answer"]
    assert f"{truth.idxmin()}" in out["answer"] and "How this was calculated" in out["notes"][1]
    assert "**Analysis program**" in out["details"] and "`per_head` **ratio**" in out["details"]


def test_numbers_and_labels_in_the_answer_come_from_silt_not_the_model(monkeypatch):
    df = visits()
    plan = json.loads(json.dumps(PER_HEAD))
    plan["steps"][2]["label"] = "readmission risk 99%"         # a misleading label with an invented number
    plan["objective"] = "Ward workload, 42% higher in Cedar"     # an invented number in the objective
    use(monkeypatch, plan)
    out = ask(COMPOSITE_Q, df, {})
    assert "99%" not in out["answer"] and "readmission risk" not in out["answer"] and "42%" not in out["answer"]
    assert "(sum of consult_minutes by ward) ÷ (number of distinct clinician values by ward)" in out["answer"]
    assert "per_head" not in out["answer"], "step ids written by the model are not labels"


def test_a_feasibility_program_uses_date_tokens_and_stated_numbers(monkeypatch):
    df = visits()
    use(monkeypatch, program([
        {"id": "ash", "op": "select", "from": "data", "entities": {"ward": ["Ash"]}},
        {"id": "mtd", "op": "select", "from": "ash", "period": {"start": "$month_start", "end": "$today"}},
        {"id": "done", "op": "aggregate", "from": "mtd", "fn": "sum", "column": "fee"},
        {"id": "pace", "op": "project", "from": "ash", "column": "fee", "days": "$days_left_month",
         "history_start": "$month_start", "history_end": "$today"},
        {"id": "verdict", "op": "compare", "a": "pace", "target": 20000, "base": "done", "days": "$days_left_month"}],
        "verdict"))
    out = ask("If Ash keeps billing like this, how do its fees end up against 20000 at month end?", df, {})
    v = out["calls"][0]["result"]["results"][""]["verdict"]["value"]
    ash = df[df.ward == "Ash"]
    done = ash[(ash.visit_date >= "2025-06-01") & (ash.visit_date <= TODAY)].fee.sum()
    assert v["completed"] == pytest.approx(done) and v["days_left"] == 10
    window = pd.date_range("2025-06-01", TODAY)
    pace = ash.groupby("visit_date").fee.sum().reindex(window, fill_value=0).mean()
    assert v["current_daily"] == pytest.approx(pace) and v["expected_total"] == pytest.approx(done + pace * 10)
    assert "not a certainty" in out["answer"] and "today is 2025-06-20" in out["answer"]


def test_a_yes_no_mean_is_read_as_the_share_of_yes(monkeypatch):
    df = visits()
    monkeypatch.setattr(nl_planner, "rule_plan", lambda *a, **k: None)     # this test is about the model's plan
    use(monkeypatch, program([{"id": "rate", "op": "aggregate", "from": "data", "fn": "mean", "column": "readmitted",
                               "by": "ward"}], "rate"))
    out = ask("How often are patients readmitted in each ward, and which ward is worst?", df, {})
    got = out["calls"][0]["result"]["results"][""]["rate"]["value"]
    truth = (df.readmitted == "Yes").groupby(df.ward).mean()
    assert got == pytest.approx(truth.to_dict())


# ---------------------------------------------------------------- rejection --
@pytest.mark.parametrize("bad, why", [
    ({"action": "program", "steps": [{"id": "s", "op": "regress", "from": "data", "target": "fee"}], "answer": "s"},
     "not an operation"),
    (program([{"id": "s", "op": "aggregate", "from": "data", "fn": "mean", "column": "wait_time"}], "s"), "no column 'wait_time'"),
    (program([{"id": "a", "op": "aggregate", "from": "data", "fn": "sum", "column": "fee"},
              {"id": "b", "op": "aggregate", "from": "data", "fn": "sum", "column": "consult_minutes"},
              {"id": "c", "op": "compare", "a": "a", "b": "b"}], "c"), "cannot compare money with"),
    (program([{"id": "c", "op": "compare", "a": "later", "b": "x"},
              {"id": "later", "op": "aggregate", "from": "data", "fn": "sum", "column": "fee"}], "c"), "not an earlier step"),
    (program([{"id": "s", "op": "aggregate", "from": "data", "fn": "mean", "column": "visit_ref"}], "s"), "identifier"),
    (program([{"id": "s", "op": "aggregate", "from": "data", "fn": "count", "column": "readmitted", "by": "ward"}], "s"),
     "count counts records"),
    (program([{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "fee", "by": "ward"},
              {"id": "b", "op": "aggregate", "from": "data", "fn": "nunique", "column": "clinician", "by": "ward"},
              {"id": "c", "op": "ratio", "a": "a", "b": "b"}], "c"), "only totals and counts can be divided"),
    (program([{"id": "p", "op": "project", "from": "data", "column": "fee", "days": 30}], "p"), "not a number in the request"),
    (program([{"id": "s", "op": "select", "from": "data", "entities": {"ward": ["Ash"]}}], "s"), "computes nothing"),
    (program([{"id": "s", "op": "select", "from": "data", "entities": {"ward": ["Elm"]}},
              {"id": "t", "op": "aggregate", "from": "s", "fn": "sum", "column": "fee"}], "t"), "Elm does not occur"),
])
def test_invalid_plans_never_reach_the_executor(monkeypatch, bad, why):
    df = visits()
    m = use(monkeypatch, bad, bad)
    p = nl_planner.plan("Tell me something useful about fees for the wards this month", df, {})
    assert p["status"] == "failed" and why in p["reason"]
    assert len(m.sent) == 2, "one repair, with the reason shown to the model"
    assert why.split("'")[0] in m.sent[1][-1]["content"]


def test_a_rejected_plan_is_repaired_once_from_the_stated_reason(monkeypatch):
    df = visits()
    bad = program([{"id": "s", "op": "aggregate", "from": "data", "fn": "mean", "column": "minutes", "by": "ward"}], "s")
    good = program([{"id": "s", "op": "aggregate", "from": "data", "fn": "mean", "column": "consult_minutes", "by": "ward"}], "s")
    use(monkeypatch, bad, good)
    out = ask("How long do consultations take on each ward and which one is longest?", df, {})
    assert out["plan_source"] == "model_program"
    assert out["calls"][0]["result"]["results"][""]["s"]["value"] == pytest.approx(
        df.groupby("ward").consult_minutes.mean().to_dict())
    assert "rejected a first plan" in out["notes"][0]


def test_mechanical_slips_are_repaired_and_said(monkeypatch):
    df = visits()
    reply = ('{"action":"program","objective":"fees","steps":[{"id":"f","op":"aggregate","from":"data","fn":"avg",'
             '"column":"fee","group_by":"ward","where":{"clinician":"Dr Lena Fischer"}}}]}')   # stray brace, no answer
    use(monkeypatch, reply)
    p = nl_planner.plan("What does Dr Lena Fischer charge on average per ward, and where is it highest?", df, {})
    assert p["status"] == "ok"
    assert {"brackets balanced", "'avg' read as 'mean'"} <= set(p["repaired"])
    assert any("moved into a select step" in x for x in p["repaired"])
    r = nl_planner.execute(p, df, {}, "q")
    truth = df[df.clinician == "Dr Lena Fischer"].groupby("ward").fee.mean()
    assert r["results"]["f"]["value"] == pytest.approx(truth.to_dict())


def test_an_invented_target_or_a_hypothetical_change_must_be_stated(monkeypatch):
    df = visits()
    plan = program([{"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "fee"},
                    {"id": "v", "op": "compare", "a": "t", "target": 50000}], "v")
    use(monkeypatch, plan, plan)
    assert nl_planner.plan("Are we on track with fees and what would it take?", df, {})["status"] == "failed"
    scen = program([{"id": "s", "op": "scenario", "from": "data", "changes": [{"scale": {"column": "fee", "pct": 15}}]},
                    {"id": "t", "op": "aggregate", "from": "s", "fn": "sum", "column": "fee"}], "t")
    use(monkeypatch, scen, scen)
    assert nl_planner.plan("What if fees went up, and how would totals change?", df, {})["status"] == "failed"


def test_a_stated_number_the_plan_ignores_is_a_rejection(monkeypatch):
    df = visits()
    plan = program([{"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "fee"}], "t")
    use(monkeypatch, plan, plan)
    p = nl_planner.plan("Will total fees pass 30000 if things keep going, and by how much?", df, {})
    assert p["status"] == "failed" and "states 30000" in p["reason"]


# ------------------------------------------------- clarification and grounding --
def test_an_ambiguous_concept_is_asked_about_even_if_the_model_picks_one(monkeypatch):
    df = visits().assign(revenue=lambda d: d.fee * 1.2)          # fee and revenue: two income columns
    df = df.rename(columns={"fee": "income"})
    plan = program([{"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "income", "by": "ward"}], "t")
    use(monkeypatch, plan)
    out = ask("Where do we earn the most takings and is that driven by longer visits?", df, {})
    assert out["plan"]["type"] == "clarify" and set(out["plan"]["options"]) == {"income", "revenue"}
    assert not out["calls"]


def test_a_model_clarification_is_asked_and_its_answer_continues_the_request(monkeypatch):
    df, mem = visits(), {}
    use(monkeypatch, {"action": "clarify", "question": "Which measure shows performance: consult_minutes or fee?",
                      "options": ["consult_minutes", "fee"]},
        program([{"id": "m", "op": "aggregate", "from": "data", "fn": "sum", "column": "fee", "by": "ward"},
                 {"id": "r", "op": "rank", "from": "m", "n": 1, "order": "asc"}], "r"))
    out = ask("How are the wards getting on, and why?", df, mem)      # no evaluative word: the model asks
    assert out["plan"]["type"] == "clarify" and out["plan"]["options"] == ["consult_minutes", "fee"]
    done = ask("fee", df, mem)
    assert done["plan_source"] == "model_program"
    assert list(done["calls"][0]["result"]["results"][""]["r"]["value"]) == [df.groupby("ward").fee.sum().idxmin()]


def test_the_grounding_has_summaries_not_records(monkeypatch):
    df = visits()
    g = nl_planner.grounding("How busy is Dr Lena Fischer compared with the others and why?", df, {})
    assert "Dr Amara Okafor" not in g and "Dr Tomas Silva" not in g, "person names are not listed"
    assert "'Dr Lena Fischer' is a value of clinician" in g, "only names the message itself uses are confirmed"
    assert "V0001" not in g and "ward (category; values: Ash, Birch, Cedar" in g
    assert "$days_left_month=10" in g and len(g) < 4000


# ------------------------------------------------------------------- fallback --
@pytest.mark.parametrize("failure", [TimeoutError("timed out"), ConnectionError("refused"), "not json at all",
                                     '{"action": "program", "steps": [{"id": "x", "op": "aggregate", "fn": "sum", "column": '])
def test_planning_failure_never_replaces_a_deterministic_answer(monkeypatch, failure):
    df = visits()
    expected = orchestrator.ask("What is the average consult minutes by ward?", df, {}, use_model=False)
    use(monkeypatch, failure, failure)
    got = ask("What is the average consult minutes by ward?", df, {})
    assert got["answer"].split("\n\n")[:2] == expected["answer"].split("\n\n")[:2]
    use(monkeypatch, failure, failure)
    out = ask(COMPOSITE_Q, df, {})
    assert out["plan_source"] != "model_program" and "Something went wrong" not in out["answer"]


def test_a_simple_question_does_not_call_the_model_planner(monkeypatch):
    df = visits()
    m = use(monkeypatch)
    out = ask("What is the total fee by ward?", df, {})
    assert out["calls"] and not m.sent


# --------------------------------------------------------------- conversation --
def test_later_programs_can_use_earlier_results_and_hypotheticals_stay_marked(monkeypatch):
    df, mem = visits(), {}
    use(monkeypatch, program([{"id": "base", "op": "aggregate", "from": "data", "fn": "sum", "column": "fee"}], "base"))
    ask("Work out total fees for me and then show what drives them.", df, mem)
    assert "@r1.base" in nl_planner.grounding("x", df, mem)
    use(monkeypatch, program([
        {"id": "no_cedar", "op": "scenario", "from": "data", "changes": [{"remove": {"ward": "Cedar"}}]},
        {"id": "t", "op": "aggregate", "from": "no_cedar", "fn": "sum", "column": "fee"},
        {"id": "d", "op": "compare", "a": "t", "b": "@r1.base", "a_label": "without Cedar", "b_label": "all wards"}], "d"))
    out = ask("And if Cedar closed, how would that compare with the total you just worked out?", df, mem)
    v = out["calls"][0]["result"]["results"][""]["d"]
    assert v["value"]["b"] == pytest.approx(df.fee.sum()) and v["value"]["a"] == pytest.approx(df[df.ward != "Cedar"].fee.sum())
    assert v["hypothetical"] and "hypothetical scenario" in out["answer"]


def _feasibility_conversation(monkeypatch, df, mem):
    first = orchestrator.ask("Can Dr Lena Fischer reach 1200 consult minutes this month?", df, mem, use_model=False)
    assert first["plan"]["type"] == "program"
    return first


def test_an_assumption_change_in_new_words_reruns_the_open_objective(monkeypatch):
    df, mem = visits(), {}
    _feasibility_conversation(monkeypatch, df, mem)
    use(monkeypatch, {"action": "rerun", "change": {"target": 1500}})
    out = ask("Bump that up to 1500 — still doable?", df, mem)     # wording no deterministic pattern covers
    assert out["plan_source"] == "model_program" and mem["objective"]["params"]["target"] == 1500
    v = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"]["Dr Lena Fischer"]["verdict"]["value"]
    d = df[(df.clinician == "Dr Lena Fischer") & (df.visit_date >= "2025-06-01") & (df.visit_date <= TODAY)]
    assert v["target"] == 1500 and v["completed"] == pytest.approx(d.consult_minutes.sum())


def test_named_scenarios_are_compared_from_memory(monkeypatch):
    df, mem = visits(), {}
    _feasibility_conversation(monkeypatch, df, mem)
    orchestrator.ask("change the target to 900", df, mem, use_model=False)
    use(monkeypatch, {"action": "compare_scenarios", "a": 1, "b": 2})
    out = ask("How does the second version stack up against my original ask?", df, mem)
    assert out["calls"][0]["tool"] == "compare_scenarios"
    r = out["calls"][0]["result"]
    assert r["previous"]["target"] == 1200 and r["latest"]["target"] == 900


def test_a_fresh_conversation_has_nothing_to_refer_to(monkeypatch):
    df, old = visits(), {}
    _feasibility_conversation(monkeypatch, df, old)
    fresh = {}
    assert "analysis 1" not in nl_planner.grounding("x", df, fresh) and "analysis 1" in nl_planner.grounding("x", df, old)
    use(monkeypatch, {"action": "rerun", "change": {"target": 1500}}, {"action": "compare_scenarios", "a": 1, "b": 2})
    p = nl_planner.plan("Suppose the goal were 1500 instead — does that change things?", df, fresh)
    assert p["status"] == "failed" and "compare_scenarios is not possible" in p["reason"]


def test_findings_are_referenced_by_number(monkeypatch):
    df, mem = visits(), {}
    orchestrator.ask("Analyse this entire dataset and tell me what to investigate next.", df, mem, use_model=False)
    m = use(monkeypatch, {"action": "finding", "n": 1, "mode": "detail"})
    out = ask("Go back to the first thing you found — how sure is that and what exactly was measured?", df, mem)
    # an ordinal reference to a stored finding is resolved by Silt itself; the model is not needed for it
    assert out["answer"].startswith("**Finding 1:") and "**Calculation:**" in out["answer"] and not m.sent


def test_the_same_request_later_in_a_conversation_is_planned_again(monkeypatch):
    df, mem = visits(), {}
    m = use(monkeypatch, PER_HEAD, PER_HEAD)
    ask(COMPOSITE_Q, df, mem)
    again = ask(COMPOSITE_Q, df, mem)
    assert len(m.sent) == 2 and again["plan_source"] == "model_program"


def test_an_average_of_per_record_ratios_is_rejected_for_a_ratio_of_totals(monkeypatch):
    df = visits()
    bad = program([{"id": "d", "op": "derive", "from": "data", "name": "fee_per_min", "formula": "fee / consult_minutes"},
                   {"id": "m", "op": "aggregate", "from": "d", "fn": "mean", "column": "fee_per_min", "by": "ward"}], "m")
    good = program([{"id": "f", "op": "aggregate", "from": "data", "fn": "sum", "column": "fee", "by": "ward"},
                    {"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "consult_minutes", "by": "ward"},
                    {"id": "r", "op": "ratio", "a": "f", "b": "t"}], "r")
    m = use(monkeypatch, bad, good)
    p = nl_planner.plan("What do we bill per consulting minute in each ward, allowing for how long each one consults?", df, {})
    assert p["status"] == "ok" and "divide the totals" in m.sent[1][-1]["content"]
    got = nl_planner.execute(p, df, {}, "q")["results"]["r"]["value"]
    assert got == pytest.approx((df.groupby("ward").fee.sum() / df.groupby("ward").consult_minutes.sum()).to_dict())


def test_a_unit_word_is_a_measure_not_a_count_of_records():
    from app.analyst import metrics
    df = visits().rename(columns={"consult_minutes": "consult_mins"})
    assert metrics.count_plan("Which ward has the fewest minutes?", df) is None
    assert metrics.count_plan("Which ward has the most visits?", df) is not None


@pytest.mark.parametrize("q, yes", [
    ("Which depot gets the fewest kilometres out of each driver it employs, and is lateness linked to anything?", True),
    ("How many kilometres would we lose if the South depot were shut?", True),
    ("Are takings per person different across outlets once you account for staffing levels?", True),
    ("What is the average fee by ward?", False), ("Show monthly fees", False), ("Which ward has the most visits?", False)])
def test_the_gate_reads_structure_not_topics(q, yes):
    assert nl_planner.composite(q) is yes


def test_a_concept_two_columns_could_mean_is_asked_before_any_path_guesses(monkeypatch):
    df = visits().rename(columns={"fee": "takings"}).assign(online_sales=lambda d: d.takings * 0.3)
    m = use(monkeypatch)
    out = ask("Which ward brings in the most revenue?", df, {})
    assert out["plan"]["type"] == "clarify" and set(out["plan"]["options"]) == {"takings", "online_sales"}
    assert not out["calls"] and not m.sent
    assert orchestrator.ask("Which ward brings in the most takings?", df, {}, use_model=False)["plan"]["type"] != "clarify"


def test_compared_scenarios_can_be_charted(monkeypatch):
    df, mem = visits(), {}
    _feasibility_conversation(monkeypatch, df, mem)
    orchestrator.ask("change the target to 900", df, mem, use_model=False)
    cmp_ = orchestrator.ask("Compare this with the previous scenario.", df, mem, use_model=False)
    assert cmp_["charts"] and {d["measure"] for d in cmp_["charts"][0]["data"]} >= {"target (previous)", "target (latest)"}
    chart = orchestrator.ask("Show the difference as a chart.", df, mem, use_model=False)
    assert chart["charts"]


def test_a_follow_up_on_an_open_objective_reaches_the_planner_first(monkeypatch):
    df, mem = visits(), {}
    _feasibility_conversation(monkeypatch, df, mem)
    m = use(monkeypatch, {"action": "rerun", "change": {"target": 1300}})
    out = ask("Could she still manage it at 1300?", df, mem)
    assert out["plan_source"] == "model_program" and mem["objective"]["params"]["target"] == 1300
    assert not nl_planner.follow_up("Could she still manage it at 1300?", {})
    # the plain form needs no model at all
    ask("What if the target changes to 1250?", df, mem)
    assert mem["objective"]["params"]["target"] == 1250 and len(m.sent) == 1


def test_a_step_missing_its_opening_brace_is_restored(monkeypatch):
    df = visits()
    reply = ('{"action":"program","objective":"x","steps":[{"id":"a","op":"aggregate","from":"data","fn":"mean",'
             '"column":"fee","by":"ward"},"id":"b","op":"rank","from":"a","n":1,"order":"desc"}],"answer":"b"}')
    use(monkeypatch, reply)
    p = nl_planner.plan("What is the average fee on each ward and which ward charges most?", df, {})
    assert p["status"] == "ok" and "a missing '{' before a step restored" in p["repaired"]
    use(monkeypatch, reply.replace('},"id":"b"', '},"{id":"b"'))
    assert nl_planner.plan("What is the average fee on each ward and which ward charges most?", df, {})["status"] == "ok"


def test_the_plain_target_wording_is_answered_deterministically_with_the_same_numbers(monkeypatch):
    df = visits()
    m = use(monkeypatch)
    out = ask("If Ash keeps billing like this, does it pass 20000 in fees by month end?", df, {})
    assert out["plan_source"] == "program" and not m.sent
    v = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"]["Ash"]["verdict"]["value"]
    ash = df[df.ward == "Ash"].drop_duplicates()
    assert v["completed"] == pytest.approx(ash[(ash.visit_date >= "2025-06-01") & (ash.visit_date <= TODAY)].fee.sum())
