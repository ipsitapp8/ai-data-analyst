"""Semantic consistency (app/analyst/intent.py): a program must compute what the request asks -- the quantity, the
measure, the grouping, the denominator, the named entities and every part of a composite request. A scripted model
proposes valid-but-wrong programs; Silt must reject them, complete what has one deterministic form, and say which
parts were answered. Expected numbers come from pandas. Datasets are generated here (staff and tickets)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.analyst import intent, nl_planner, orchestrator, planner, scope, semantics

TODAY = pd.Timestamp("2025-03-20")


def staff(seed=9, n=320):
    rng = np.random.default_rng(seed)
    unit = rng.choice(["Assembly", "Dispatch", "Quality", "Tooling"], n)
    ot = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    pay = rng.normal(3200, 500, n).round(0)
    left = np.where(rng.random(n) < 0.05 + 0.2 * (ot == "Yes") + 0.1 * (unit == "Dispatch"), "Yes", "No")
    return pd.DataFrame({"badge": [f"B{i:04d}" for i in range(n)], "unit": unit, "overtime": ot, "monthly_pay": pay,
                         "attrition": left, "joined": TODAY - pd.to_timedelta(rng.integers(30, 900, n), "D")})


def tickets(seed=2, n=300):
    rng = np.random.default_rng(seed)
    agent = rng.choice(["Lea Moreau", "Tariq Aziz", "Hana Sato", "Pablo Ruiz", "Ines Kova"], n)
    team = np.vectorize({"Lea Moreau": "Billing", "Tariq Aziz": "Billing", "Hana Sato": "Tech", "Pablo Ruiz": "Tech",
                         "Ines Kova": "Tech"}.get)(agent)
    return pd.DataFrame({"agent": agent, "team": team, "opened": TODAY - pd.to_timedelta(rng.integers(1, 60, n), "D"),
                         "handle_minutes": rng.gamma(4, 9, n).round(1)})


class Model:
    def __init__(self, *replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.sent = []

    def __call__(self, messages, temperature=0.0):
        self.sent.append(messages)
        return self.replies.pop(0)


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(scope, "today", lambda: TODAY)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "no plan")
    from app.analyst import calc
    monkeypatch.setattr(calc, "model_plan", lambda *a, **k: ({"type": "failed", "reason": "not under test"}, []))


def use(monkeypatch, *replies):
    m = Model(*replies)
    monkeypatch.setattr(nl_planner, "model_chat", m)
    return m


def prog(steps, answer):
    return {"action": "program", "objective": "x", "steps": steps, "answer": answer}


def results(out):
    return out["calls"][0]["result"]["results"][""]


# ------------------------------------------------------------- reading intent --
@pytest.mark.parametrize("q, want", [
    ("Which unit has the highest attrition and why?", dict(quantity="rate", measures=["attrition"], group="unit",
                                                           extreme="max", why=True, why_target="attrition")),
    ("What is the median monthly pay for each unit?", dict(quantity="median", measures=["monthly_pay"], group="unit")),
    ("How many minutes does each team handle per agent?", dict(group="team", per=True, denominators=["agent"])),
    ("Are any handle times implausible, and how many are there?", dict(unusual=True, quantity=None)),
])
def test_requirements_are_read_from_language_and_schema(q, want):
    df = tickets() if "minutes" in q or "handle" in q else staff()
    r = intent.read(q, df)
    for k, v in want.items():
        assert getattr(r, k) == v, (k, getattr(r, k))


# ------------------------------------------------- valid but wrong programs --
WRONG = {
    "different measure": prog([{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "monthly_pay",
                                "by": "unit"}], "a"),
    "count instead of a rate": prog([{"id": "a", "op": "aggregate", "from": "data", "fn": "count", "by": "unit"}], "a"),
    "no breakdown": prog([{"id": "f", "op": "derive", "from": "data", "name": "is_left", "formula": "attrition == 'Yes'"},
                          {"id": "a", "op": "aggregate", "from": "f", "fn": "mean", "column": "is_left"}], "a"),
}


@pytest.mark.parametrize("name", list(WRONG))
def test_a_valid_program_for_a_different_quantity_is_rejected(monkeypatch, name):
    df = staff()
    use(monkeypatch, WRONG[name], WRONG[name])
    p = nl_planner.plan("What share of people leave in each unit?", df, {})
    assert p["status"] == "failed" and "does not compute what was asked" in p["reason"]


def test_the_denominator_must_be_the_one_asked_for(monkeypatch):
    df = tickets()
    wrong = prog([{"id": "m", "op": "aggregate", "from": "data", "fn": "sum", "column": "handle_minutes", "by": "team"},
                  {"id": "n", "op": "aggregate", "from": "data", "fn": "count", "by": "team"},
                  {"id": "r", "op": "ratio", "a": "m", "b": "n"}], "r")                 # per ticket, not per agent
    right = prog([{"id": "m", "op": "aggregate", "from": "data", "fn": "sum", "column": "handle_minutes", "by": "team"},
                  {"id": "n", "op": "aggregate", "from": "data", "fn": "nunique", "column": "agent", "by": "team"},
                  {"id": "r", "op": "ratio", "a": "m", "b": "n"}], "r")
    m = use(monkeypatch, wrong, right)
    p = nl_planner.plan("How many minutes does each team handle per agent?", df, {})
    assert p["status"] == "ok" and "divided by agent" in m.sent[1][-1]["content"]
    got = nl_planner.execute(p, df, {}, "q")["results"]["r"]["value"]
    assert got == pytest.approx((df.groupby("team").handle_minutes.sum() / df.groupby("team").agent.nunique()).to_dict())


def test_a_named_entity_must_be_selected(monkeypatch):
    df = tickets()
    wrong = prog([{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "handle_minutes"}], "a")
    use(monkeypatch, wrong, wrong)
    p = nl_planner.plan("On average, how long does Hana Sato take per ticket compared with the team?", df, {})
    assert p["status"] == "failed" and "Hana Sato" in p["reason"]


# ------------------------------------------------------- composite requests --
def test_every_part_of_which_and_why_is_answered(monkeypatch):
    df = staff()
    only_rate = prog([{"id": "f", "op": "derive", "from": "data", "name": "is_left", "formula": "attrition == 'Yes'"},
                      {"id": "rate", "op": "aggregate", "from": "f", "fn": "mean", "column": "is_left", "by": "unit"}],
                     "rate")
    use(monkeypatch, only_rate)
    monkeypatch.setattr(nl_planner, "rule_plan", lambda *a, **k: None)     # this test is about the model's plan
    out = orchestrator.ask("Which unit has the highest attrition and why?", df, {}, use_model=True)
    res = results(out)
    truth = (df.attrition == "Yes").groupby(df.unit).mean()
    assert res["rate"]["value"] == pytest.approx(truth.to_dict())                        # 1. rate by unit
    assert f"**Highest: {truth.idxmax()}**" in out["answer"]                             # 2. the highest
    assert res["why_links"]["value"]["target"] == "attrition"                            # 3. associations
    assert "overtime" in [x["factor"] for x in res["why_links"]["value"]["factors"]]
    assert "not causes" in out["answer"]                                                 # 4. limitations stated,
    import re
    assert not re.search(r"\b(caused by|leads to|results in|because of|is why)\b", out["answer"], re.I)  # no causal claim
    assert "What you asked, and where it is answered" in out["answer"] and "✗" not in out["answer"]


@pytest.mark.parametrize("model", ["fails", "off"])
def test_without_a_usable_model_plan_a_grounded_composite_is_built_deterministically(monkeypatch, model):
    df = staff()
    use(monkeypatch, "not json", "still not json")
    out = orchestrator.ask("Which unit has the highest attrition and why?", df, {}, use_model=model == "fails")
    assert out["plan_source"] == "rule_program"
    res = results(out)
    assert res["result"]["value"] == pytest.approx((df.attrition == "Yes").groupby(df.unit).mean().to_dict())
    assert res["why_links"]["value"]["target"] == "attrition" and "✗" not in out["answer"]


def test_a_part_that_cannot_be_answered_is_said_not_dropped(monkeypatch):
    df = tickets()
    plan = prog([{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "handle_minutes", "by": "team"}],
                "a")
    use(monkeypatch, plan)
    out = orchestrator.ask("Which team is slowest on average, and why might that be?", df, {}, use_model=True)
    # the only other columns are a name and a date: the associations are tested, and an empty result is said
    assert "What you asked" in out["answer"] and "what is associated with it" in out["answer"]


# ----------------------------------------------------------- budget, isolation --
def test_no_repair_is_attempted_beyond_the_planning_budget(monkeypatch):
    df = staff()
    monkeypatch.setattr(nl_planner, "PLAN_BUDGET", 0.0)
    m = use(monkeypatch, WRONG["different measure"], WRONG["different measure"])
    assert nl_planner.plan("What share of people leave in each unit?", df, {})["status"] == "failed"
    assert len(m.sent) == 1


def test_the_planner_endpoint_does_not_change_the_other_model_paths(monkeypatch):
    seen = []

    class FakeOpenAI:
        def __init__(self, base_url, **k):
            seen.append(base_url)
            self.chat = self

        @property
        def completions(self):
            return self

        def create(self, model, **k):
            seen.append(model)

            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": "{}"})()})()]
            return R()
    import openai
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(nl_planner, "PLANNER_URL", "http://127.0.0.1:8081/v1")
    monkeypatch.setattr(nl_planner, "PLANNER_MODEL", "base-model")
    nl_planner._chat([{"role": "user", "content": "x"}])                                   # noqa: SLF001
    planner.model_chat.__wrapped__ if hasattr(planner.model_chat, "__wrapped__") else None
    import importlib
    real = importlib.reload(importlib.import_module("app.analyst.planner"))
    real.model_chat([{"role": "user", "content": "x"}])
    assert seen[:2] == ["http://127.0.0.1:8081/v1", "base-model"]
    assert seen[2] == real.MODEL_URL and seen[3] == real.MODEL_NAME and "8081" not in seen[2]


def test_a_restricted_subset_must_be_selected_and_is_never_dropped(monkeypatch):
    df = staff()
    req = intent.read("Among people on overtime only, what is the median monthly pay for each unit?", df)
    assert req.restrict == ["overtime"]
    steps = intent.rule_program(req, df, semantics.profile(df))       # the restriction is applied, never dropped
    assert steps[0]["where"] == [{"column": "overtime", "op": "==", "value": "Yes"}]
    out = orchestrator.ask("Among people on overtime only, what is the median monthly pay for each unit?", df, {},
                           use_model=False)
    got = out["calls"][0]["result"]["results"][""]["result"]["value"]
    assert got == pytest.approx(df[df.overtime == "Yes"].groupby("unit").monthly_pay.median().to_dict())
    all_rows = prog([{"id": "a", "op": "aggregate", "from": "data", "fn": "median", "column": "monthly_pay", "by": "unit"}], "a")
    use(monkeypatch, all_rows, all_rows)
    p = nl_planner.plan("Among people on overtime only, what is the median monthly pay for each unit?", df, {})
    assert p["status"] == "failed" and "limited to records where overtime" in p["reason"]


def test_relative_to_how_many_is_a_rate_not_a_count():
    df = staff()
    req = intent.read("Which unit has the most leavers relative to how many people it has?", df)
    assert req.per and req.denominators == ["(records)"]
    df2 = staff().rename(columns={"attrition": "left_company"})
    r2 = intent.read("Which unit has the most left company records relative to how many it has, and why?", df2)
    steps = intent.rule_program(r2, df2, semantics.profile(df2))
    assert steps and any(s.get("fn") == "mean" for s in steps)


def test_a_grouping_noun_that_is_no_column_is_not_replaced(monkeypatch):
    df = tickets()
    assert intent.read("Compare the minutes each supervisor logs and say who is best.", df).unknown_group == "supervisor"
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k:
                        '{"type":"tools","intent":"metric","calls":[{"tool":"compute_metric","args":{"aggregation":"avg",'
                        '"column":"handle_minutes","group_by":"team"}}]}')
    use(monkeypatch, "not json", "not json")
    out = orchestrator.ask("Compare the minutes each supervisor logs and say who is best.", df, {}, use_model=True)
    assert "supervisor" in out["answer"] and not any(c.get("status") == "ok" for c in out["calls"])


def test_an_average_of_people_is_not_a_count_of_records():
    from app.analyst import metrics
    df = tickets()
    assert metrics.count_plan("On average how many agents does each team have?", df) is None
    assert metrics.count_plan("How many tickets does each team have?", df) is not None


def test_a_stemmed_yes_no_column_is_the_rate_measure():
    df = staff().rename(columns={"attrition": "escalated"})
    r = intent.read("What is the escalation rate for each unit, and why?", df)
    assert r.measures == ["escalated"] and r.quantity == "rate" and r.why_target == "escalated"


def test_rates_read_as_percentages_and_assumption_changes_need_no_model(monkeypatch):
    df = staff()
    use(monkeypatch, "not json", "not json")
    out = orchestrator.ask("Which unit has the highest attrition and why?", df, {}, use_model=False)
    top = (df.attrition == "Yes").groupby(df.unit).mean()
    assert f"{top.max() * 100:.1f}%" in out["answer"] and "(share of records)" in out["answer"]


def test_what_if_the_target_changes_is_a_deterministic_rerun():
    from tests.test_analyst_nl_planner import _feasibility_conversation, visits
    df, mem = visits(), {}
    scope.today = lambda: pd.Timestamp("2025-06-20")
    _feasibility_conversation(None, df, mem)
    out = orchestrator.ask("What if the target changes to 1500?", df, mem, use_model=False)
    assert out["plan_source"] == "program" and mem["objective"]["params"]["target"] == 1500
    out = orchestrator.ask("Suppose the goal were 1400 hours", df, mem, use_model=False)
    assert mem["objective"]["params"]["target"] == 1400


def test_a_grounded_composite_needs_no_model_call(monkeypatch):
    df = staff()
    m = use(monkeypatch)
    out = orchestrator.ask("Which unit has the highest attrition and why?", df, {}, use_model=True)
    assert out["plan_source"] == "rule_program" and not m.sent


def test_an_undefined_performance_measure_is_asked_and_the_answer_is_computed_in_full(monkeypatch):
    df = staff()
    m = use(monkeypatch)
    mem = {}
    out = orchestrator.ask("Which unit performs worst and why?", df, mem, use_model=True)
    assert out["plan"]["type"] == "clarify" and "attrition" in out["plan"]["options"] and not m.sent
    done = orchestrator.ask("attrition", df, mem, use_model=True)
    res = results(done)
    assert res["result"]["value"] == pytest.approx((df.attrition == "Yes").groupby(df.unit).mean().to_dict())
    assert res["why_links"]["value"]["target"] == "attrition" and "not causes" in done["answer"] and not m.sent


def test_measure_words_are_read_as_their_only_column_or_asked_about():
    from app.analyst import semantics as sem
    rng = np.random.default_rng(3)
    trips = pd.DataFrame({"driver": ["a", "b"] * 10, "km_driven": rng.gamma(5, 20, 20).round(1),
                          "fuel_litres": rng.gamma(5, 2, 20).round(1)})
    assert sem.measure_words("What is the average trip length per driver?", trips)["rewrite"] == [("length", "km_driven")]
    shops = pd.DataFrame({"shop": ["x", "y"] * 10, "food_bill": rng.gamma(5, 4, 20).round(2),
                          "drinks_bill": rng.gamma(3, 3, 20).round(2)})
    amb = sem.measure_words("What is the average spend per shop?", shops)["ambiguous"]
    assert amb and set(amb[0][1]) == {"food_bill", "drinks_bill"}
    out = orchestrator.ask("What is the average spend per shop?", shops, {}, use_model=False)
    assert out["plan"]["type"] == "clarify" and not out["calls"]
    ads = pd.DataFrame({"area": ["x", "y"] * 10, "ad_spend": rng.gamma(5, 40, 20).round(2)})
    assert sem.measure_words("Cut marketing spend in the worst area", ads) == {"rewrite": [], "ambiguous": []}


def test_entity_scoped_rates_relative_periods_and_flag_restrictions_are_applied(monkeypatch):
    df = tickets().assign(escalated=lambda d: np.where(d.handle_minutes > 40, "Yes", "No"))
    out = orchestrator.ask("What fraction of Hana Sato's tickets were escalated?", df, {}, use_model=False)
    assert results(out)["result"]["value"] == pytest.approx((df[df.agent == "Hana Sato"].escalated == "Yes").mean())
    assert "%" in out["answer"]
    out = orchestrator.ask("Total handle minutes for only the last 14 days", df, {}, use_model=False)
    start = TODAY - pd.Timedelta(days=13)
    want = df[(df.opened >= start) & (df.opened <= TODAY)].handle_minutes.sum()
    assert results(out)["result"]["value"] == pytest.approx(want)
    out = orchestrator.ask("Average handle minutes of only escalated tickets", df, {}, use_model=False)
    assert results(out)["result"]["value"] == pytest.approx(df[df.escalated == "Yes"].handle_minutes.mean())


def test_a_target_question_with_any_verb_and_a_deadline_is_feasibility():
    df = tickets()
    out = orchestrator.ask("Will Tech clock 2000 handle minutes by month end?", df, {}, use_model=False)
    assert out["plan"]["type"] == "program" and out["plan"].get("objective") == "feasibility"
    v = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"]["Tech"]["verdict"]["value"]
    d = df[(df.team == "Tech") & (df.opened >= TODAY.replace(day=1)) & (df.opened <= TODAY)]
    assert v["completed"] == pytest.approx(d.drop_duplicates().handle_minutes.sum())


def test_a_model_program_may_not_add_a_period_the_request_never_named(monkeypatch):
    df = staff()
    narrowed = prog([{"id": "p", "op": "select", "from": "data", "period": {"start": "$month_start", "end": "$today"}},
                     {"id": "a", "op": "aggregate", "from": "p", "fn": "mean", "column": "monthly_pay", "by": "unit"}], "a")
    whole = prog([{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "monthly_pay", "by": "unit"}], "a")
    m = use(monkeypatch, narrowed, whole)
    monkeypatch.setattr(nl_planner, "rule_plan", lambda *a, **k: None)
    p = nl_planner.plan("How does average monthly pay compare across units, and which is highest?", df, {})
    assert p["status"] == "ok" and "names no period" in m.sent[1][-1]["content"]


@pytest.mark.parametrize("q,allowed", [
    ("On promotion days versus ordinary days, how does average monthly pay compare?", False),
    ("Average monthly pay per week for each unit", False),
    ("Average monthly pay by unit since March", True),
    ("Average monthly pay by unit this month so far", True),
    ("Average monthly pay by unit up to the end of the year", True)])
def test_a_bare_time_unit_is_not_a_period(q, allowed):
    """'promotion days' describes records, 'per week' a breakdown: neither licenses a program to narrow the records to
    a period the user never named. A qualified period or a named date does."""
    df = staff()
    steps = [{"id": "p", "op": "select", "from": "data", "period": {"start": "$month_start", "end": "$today"}},
             {"id": "a", "op": "aggregate", "from": "p", "fn": "mean", "column": "monthly_pay", "by": "unit"}]
    probs = intent.check(steps, intent.read(q, df), df, q)
    assert any("names no period" in p for p in probs) is (not allowed)


def test_an_everyday_phrase_for_a_yes_no_outcome_is_read_as_its_only_column():
    rng = np.random.default_rng(4)
    trips = pd.DataFrame({"hub": rng.choice(["A", "B", "C"], 90), "km": rng.gamma(5, 20, 90).round(1),
                          "delayed_flag": rng.choice(["Yes", "No"], 90, p=[0.2, 0.8])})
    q = "What proportion of runs are behind schedule at each hub, and which hub fares worst?"
    out = orchestrator.ask(q, trips, {}, use_model=False)
    want = (trips.delayed_flag == "Yes").groupby(trips.hub).mean()
    got = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"][""]["result"]
    assert got["value"] == pytest.approx(want.to_dict())
    assert "read as `delayed_flag`" in out["answer"] and f"{want.max() * 100:.1f}%" in out["answer"]
    as01 = trips.assign(delayed_flag=(trips.delayed_flag == "Yes").astype(int))
    out = orchestrator.ask(q, as01, {}, use_model=False)
    assert f"{as01.groupby('hub').delayed_flag.mean().max() * 100:.1f}%" in out["answer"]
    # no column records lateness: nothing else is read in its place
    plain = trips.drop(columns=["delayed_flag"]).assign(escalated=rng.choice(["Yes", "No"], 90))
    assert semantics.measure_words(q, plain) == {"rewrite": [], "ambiguous": []}
    # two columns could record it: asked, not chosen
    two = trips.assign(late_arrival=rng.choice(["Yes", "No"], 90))
    assert semantics.measure_words(q, two)["ambiguous"]
    assert orchestrator.ask(q, two, {}, use_model=False)["plan"]["type"] == "clarify"


def test_values_set_against_each_other_are_a_breakdown_restricted_to_them():
    rng = np.random.default_rng(6)
    df = pd.DataFrame({"sitting": rng.choice(["Breakfast", "Lunch", "Dinner"], 150),
                       "cheque_total": rng.gamma(4, 6, 150).round(2), "covers": rng.integers(1, 6, 150),
                       "happy_hour": rng.choice(["Yes", "No"], 150, p=[0.3, 0.7])})
    out = orchestrator.ask("What is the typical cheque at dinner versus lunch?", df, {}, use_model=False)
    got = results(out)["result"]["value"]
    want = df[df.sitting.isin(["Dinner", "Lunch"])].groupby("sitting").cheque_total.median()
    assert got == pytest.approx(want.to_dict()), "only the two sittings named, one median each"
    out = orchestrator.ask("On happy hour days versus ordinary days, how does the average cheque compare?", df, {},
                           use_model=False)
    assert results(out)["result"]["value"] == pytest.approx(df.groupby("happy_hour").cheque_total.mean().to_dict())
    r = intent.read("Average cheque total for Dinner", df)
    assert r.group is None and not r.compared, "one named value is a filter, not a comparison"


def test_conversion_and_share_wording_divides_the_right_totals_and_tests_nothing():
    rng = np.random.default_rng(8)
    shops = pd.DataFrame({"branch": rng.choice(["N", "S", "E"], 120), "walk_ins": rng.integers(50, 400, 120),
                          "revenue": rng.gamma(6, 100, 120).round(2)})
    q = "Which branch turns walk ins into revenue least effectively?"
    r = intent.read(q, shops)
    assert r.per and r.denominators == ["walk_ins"] and r.measures == ["revenue"] and r.group == "branch"
    mean_only = [{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "revenue", "by": "branch"}]
    assert any("divide" in p for p in intent.check(mean_only, r, shops, q)), "a mean of revenue is not revenue per walk-in"
    out = orchestrator.ask(q, shops, {}, use_model=False)
    want = shops.groupby("branch").revenue.sum() / shops.groupby("branch").walk_ins.sum()
    assert results(out)["per_unit"]["value"] == pytest.approx(want.to_dict())
    assert f"Lowest: {want.idxmin()}" in out["answer"]
    bills = pd.DataFrame({"member": rng.choice(["Yes", "No"], 120), "bill": rng.gamma(5, 8, 120).round(2)})
    bills["gratuity"] = (bills.bill * rng.uniform(0.05, 0.25, 120)).round(2)
    for q in ("Do members leave a bigger slice of the bill as gratuity than non-members, and is that difference meaningful?",
              "Compare gratuity as a share of the bill by member, and is the difference significant?"):
        out = orchestrator.ask(q, bills, {}, use_model=False)
        want = bills.groupby("member").gratuity.sum() / bills.groupby("member").bill.sum()
        assert results(out)["per_unit"]["value"] == pytest.approx(want.to_dict()), q
        assert "statistically meaningful: ✗ not tested" in out["answer"] and "p =" not in out["answer"]
