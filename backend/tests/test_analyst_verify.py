"""Answer verification (app/analyst/verify.py): the figure shown must be the one the user's request asks for,
recomputed independently with pandas from the request -- never accepted because a plan or its restatement says so."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import nl_planner, orchestrator, planner, scope, verify
from app.analyst import program as P

TODAY = pd.Timestamp("2025-06-20")


def shops(seed=12, n=200):
    rng = np.random.default_rng(seed)
    branch = rng.choice(["North", "South", "East"], n)
    visitors = rng.integers(20, 500, n)
    return pd.DataFrame({"branch": branch, "sold_on": TODAY - pd.to_timedelta(rng.integers(0, 60, n), "D"),
                         "visitors": visitors, "revenue": (visitors * rng.uniform(2, 6, n)).round(2),
                         "refund": rng.choice(["Yes", "No"], n, p=[0.15, 0.85])})


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(scope, "today", lambda: TODAY)


def run(steps, df):
    res = P.run([{k: v for k, v in s.items()} for s in steps], df)
    return {k: r.summary() for k, r in res.items()}


SUM_BY = [{"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "revenue", "by": "branch"}]


def test_a_correct_figure_is_verified_by_independent_recomputation():
    df = shops()
    v = verify.check("What is the total revenue by branch?", df, SUM_BY, run(SUM_BY, df))
    assert v["status"] == "verified" and "matches step 't'" in v["checks"][-1]["detail"]


def test_a_ratio_of_averages_is_not_the_ratio_of_totals_asked_for():
    """Both programs divide revenue by visitors per branch, so the requirement check passes; only the independent
    recomputation sees that the figure is a ratio of means, not total revenue / total visitors."""
    df = shops()
    steps = [{"id": "a", "op": "aggregate", "from": "data", "fn": "mean", "column": "revenue", "by": "branch"},
             {"id": "b", "op": "aggregate", "from": "data", "fn": "median", "column": "visitors", "by": "branch"},
             {"id": "r", "op": "ratio", "a": "a", "b": "b"}]
    v = verify.check("What is the revenue per visitor at each branch?", df, steps, run(steps, df))
    assert v["checks"][0]["ok"] and v["status"] == "failed"
    want = (df.groupby("branch").revenue.sum() / df.groupby("branch").visitors.sum()).round(6).to_dict()
    assert str(want["East"]) in v["checks"][-1]["detail"]


@pytest.mark.parametrize("q,steps,why", [
    ("Total revenue for North", SUM_BY, "does not select it"),
    ("Total revenue by branch, excluding East", SUM_BY, "must exclude it"),
    ("Total revenue by branch for only refund orders", SUM_BY, "limited to records where refund"),
    ("Total revenue by branch for the last 14 days", SUM_BY, "limited to the last 14 days"),
    ("Average revenue by branch", SUM_BY, "asks for an average"),
    ("Total revenue", [{"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "visitors"}], "about revenue")])
def test_dropped_filters_and_wrong_quantities_fail(q, steps, why):
    df = shops()
    v = verify.check(q, df, steps, run(steps, df))
    assert v["status"] == "failed" and why in v["checks"][0]["detail"]


def test_filters_groupings_rates_and_counts_are_recomputed():
    df = shops()
    ok = [("What share of orders were refunds at each branch?",
           [{"id": "f", "op": "derive", "from": "data", "name": "is_refund", "formula": "refund == 'Yes'"},
            {"id": "m", "op": "aggregate", "from": "f", "fn": "mean", "column": "is_refund", "by": "branch"}]),
          ("Median revenue by branch, excluding East",
           [{"id": "s", "op": "select", "from": "data", "except": {"branch": ["East"]}},
            {"id": "m", "op": "aggregate", "from": "s", "fn": "median", "column": "revenue", "by": "branch"}]),
          ("How many orders does each branch have?",
           [{"id": "c", "op": "aggregate", "from": "data", "fn": "count", "by": "branch"}])]
    for q, steps in ok:
        assert verify.check(q, df, steps, run(steps, df))["status"] == "verified", q
    # a count over a different grouping is not the count asked for
    wrong = [{"id": "c", "op": "aggregate", "from": "data", "fn": "count", "by": "refund"}]
    assert verify.check("How many orders does each branch have?", df, wrong, run(wrong, df))["status"] == "failed"


def test_what_cannot_be_recomputed_is_reported_not_passed():
    df = shops()
    steps = [{"id": "a", "op": "associate", "from": "data", "target": "refund"}]
    v = verify.check("What is linked to refunds?", df, steps, run(steps, df))
    assert v["status"] == "not independently verified" and "not independently verified" in v["checks"][-1]["detail"]


def test_a_scenario_result_must_be_marked_hypothetical():
    df = shops()
    steps = [{"id": "w", "op": "scenario", "from": "data", "changes": [{"remove": {"branch": "East"}}]},
             {"id": "t", "op": "aggregate", "from": "w", "fn": "sum", "column": "revenue"}]
    res = run(steps, df)
    assert res["t"]["hypothetical"]
    res["t"]["hypothetical"] = False                          # a result that lost its mark
    v = verify.check("What if East were closed: total revenue?", df, steps, res)
    assert v["status"] == "failed" and "t" in v["checks"][1]["detail"]


def test_a_model_program_that_fails_verification_is_replaced_by_silts_own(monkeypatch):
    """End to end: the model's program passes every pre-execution check but quietly adds a filter nobody asked for
    (refund != Yes -- an unasked numeric threshold such as visitors >= 100 is now refused before execution, see the
    next test). Its restatement ('revenue per visitor by branch') is not evidence; the independent figure differs, so
    it is not shown and the deterministic program built from the request answers instead."""
    df = shops()
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "no plan")
    real = nl_planner.rule_plan
    calls = {"n": 0}

    def rule_plan_once_missing(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(*a, **k)    # the model is asked first, as for an ungrounded request
    monkeypatch.setattr(nl_planner, "rule_plan", rule_plan_once_missing)
    bad = {"action": "program", "objective": "revenue per visitor by branch", "answer": "r", "steps": [
        {"id": "busy", "op": "select", "from": "data", "where": [{"column": "refund", "op": "!=", "value": "Yes"}]},
        {"id": "a", "op": "aggregate", "from": "busy", "fn": "sum", "column": "revenue", "by": "branch"},
        {"id": "b", "op": "aggregate", "from": "busy", "fn": "sum", "column": "visitors", "by": "branch"},
        {"id": "r", "op": "ratio", "a": "a", "b": "b"}]}
    import json
    monkeypatch.setattr(nl_planner, "model_chat", lambda *a, **k: json.dumps(bad))
    mem = {}
    out = orchestrator.ask("Which branch earns the least revenue per visitor, and why?", df, mem, use_model=True)
    want = df.groupby("branch").revenue.sum() / df.groupby("branch").visitors.sum()
    prog = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]
    assert out["plan_source"] == "rule_program" and prog["results"][""]["per_unit"]["value"] == pytest.approx(want.to_dict())
    assert any("answer verification" in n for n in out["notes"]) and mem["_nl_last"]["status"] == "verify_failed"
    assert any(c["check"].startswith("independent recomputation") and c["ok"] for c in out["verification"])


def test_a_model_program_with_an_unasked_numeric_filter_is_refused_before_it_runs(monkeypatch):
    """The model may not invent its own exclusion rule: 'visitors >= 100' (or 'revenue >= 0') when the question sets
    no such condition is refused by the program checks, and Silt's own program for the request answers on all the
    records."""
    df = shops()
    monkeypatch.setattr(planner, "model_available", lambda: True)
    monkeypatch.setattr(planner, "model_chat", lambda *a, **k: "no plan")
    real = nl_planner.rule_plan
    calls = {"n": 0}

    def rule_plan_once_missing(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(*a, **k)
    monkeypatch.setattr(nl_planner, "rule_plan", rule_plan_once_missing)
    import json
    for cond in ({"column": "visitors", "op": ">=", "value": 100}, {"column": "revenue", "op": ">=", "value": 0}):
        bad = {"action": "program", "objective": "revenue per visitor by branch", "answer": "r", "steps": [
            {"id": "busy", "op": "select", "from": "data", "where": [cond]},
            {"id": "a", "op": "aggregate", "from": "busy", "fn": "sum", "column": "revenue", "by": "branch"},
            {"id": "b", "op": "aggregate", "from": "busy", "fn": "sum", "column": "visitors", "by": "branch"},
            {"id": "r", "op": "ratio", "a": "a", "b": "b"}]}
        monkeypatch.setattr(nl_planner, "model_chat", lambda *a, _b=bad, **k: json.dumps(_b))
        calls["n"] = 0
        mem = {}
        out = orchestrator.ask("Which branch earns the least revenue per visitor, and why?", df, mem, use_model=True)
        want = df.groupby("branch").revenue.sum() / df.groupby("branch").visitors.sum()
        prog = next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]
        assert out["plan_source"] == "rule_program"
        assert prog["results"][""]["per_unit"]["value"] == pytest.approx(want.to_dict())
        assert "must not filter records" in json.dumps(mem.get("_nl_last"), default=str)


# ------------------------------------------------------------- every execution path --
def ballots(seed=21, n=60):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"ward": rng.choice(["W1", "W2", "W3"], n), "kind": rng.choice(["close", "clear"], n),
                       "Ash": rng.integers(100, 900, n), "Birch": rng.integers(100, 900, n)})
    df["leader"] = np.where(df.Ash > df.Birch, "Ash", "Birch")
    return df


def test_a_column_named_like_a_value_is_the_measure_not_a_filter():
    df = ballots()
    out = orchestrator.ask("What is the total number of votes for Ash in each kind?", df, {}, use_model=False)
    got = next(c for c in out["calls"] if c.get("status") == "ok")["result"]
    vals = got["results"][""]["result"]["value"] if "results" in got else {g["group"]: g["value"] for g in got["groups"]}
    assert vals == pytest.approx(df.groupby("kind").Ash.sum().astype(float).to_dict())
    assert out["verify_status"] == "verified" and "Verification: ✓" in out["answer"]
    assert scope.establish("Total Birch votes where the leader is Ash", df, {})["entities"] == {"leader": ["Ash"]}, \
        "naming the value's own column keeps it a filter"
    assert scope.establish("Total votes for Ash", df, {})["entities"] == {}


def test_a_tool_answer_that_does_not_match_is_withheld_and_replaced():
    df = ballots()
    wrong = {"answer": "x", "plan": {"type": "tools"}, "plan_source": "rules", "calls": [
        {"tool": "compute_metric", "args": {"aggregation": "sum", "column": "Birch", "group_by": "kind"}, "status": "ok",
         "result": {"aggregation": "sum", "column": "Birch", "group_by": "kind",
                    "groups": [{"group": k, "value": float(v), "rows": 1} for k, v in df.groupby("kind").Birch.sum().items()]}}]}
    v = verify.check_answer("Total Ash votes by kind", df, wrong, None, {})
    assert v["status"] == "failed" and "about Ash" in v["checks"][0]["detail"]
    right = {**wrong, "calls": [{**wrong["calls"][0], "args": {"aggregation": "sum", "column": "Ash", "group_by": "kind"},
                                 "result": {"groups": [{"group": k, "value": float(v), "rows": 1}
                                                       for k, v in df.groupby("kind").Ash.sum().items()]}}]}
    assert verify.check_answer("Total Ash votes by kind", df, right, None, {})["status"] == "verified"
    # a scoped answer is recomputed over the scope it used
    sc = {"entities": {"ward": ["W1"]}}
    w1 = df[df.ward == "W1"]
    scoped = {**wrong, "calls": [{**wrong["calls"][0], "args": {"aggregation": "sum", "column": "Ash", "group_by": "kind"},
                                  "result": {"groups": [{"group": k, "value": float(v), "rows": 1}
                                                        for k, v in w1.groupby("kind").Ash.sum().items()]}}]}
    assert verify.check_answer("Total Ash votes by kind for W1", df, scoped, sc, {})["status"] == "verified"


def test_paths_with_no_independent_form_are_labelled_not_passed():
    df = ballots()
    # a correlation now has an independent form: Pearson r, Spearman rho, their p-values and n are recomputed
    out = orchestrator.ask("Is Ash correlated with Birch?", df, {}, use_model=False)
    if any(c.get("status") == "ok" for c in out["calls"]):
        assert out["verify_status"] == "verified"
        assert {c["check"] for c in out["verification"]} >= {"independent recomputation: Pearson r",
                                                            "independent recomputation: Spearman rho",
                                                            "independent recomputation: n"}
        assert all(c["ok"] for c in out["verification"] if c["check"].startswith("independent recomputation"))
    # a result that names no columns has no independent form: never called verified
    v = verify.check_answer("Total votes", df, {"plan_source": "rules", "plan": {}, "calls": [
        {"tool": "correlation", "status": "ok", "result": {"r": 0.1}}]}, None, {})
    assert v["status"] == "not independently verified"


def test_shares_read_as_percentages_and_small_rates_keep_their_digits():
    from app.analyst.explain import f
    assert f(0.000123) == "0.000123" and f(-0.00071) == "-0.00071" and f(0.1234) == "0.12" and f(2304.6) == "2,304.60"
    rng = np.random.default_rng(2)
    bills = pd.DataFrame({"party": rng.choice(["A", "B"], 80), "bill": rng.gamma(5, 8, 80).round(2)})
    bills["gratuity"] = (bills.bill * rng.uniform(0.05, 0.25, 80)).round(2)
    out = orchestrator.ask("What is gratuity as a share of the bill by party?", bills, {}, use_model=False)
    want = bills.groupby("party").gratuity.sum() / bills.groupby("party").bill.sum()
    assert "(share of bill)" in out["answer"] and all(f"{v * 100:.1f}%" in out["answer"] for v in want)
    out = orchestrator.ask("What is the bill per gratuity by party?", bills, {}, use_model=False)
    assert "(bill per gratuity)" in out["answer"] and "%" not in out["answer"].split("\n")[0], "a ratio, not a share"


def test_a_table_of_tallies_is_summed_not_counted_and_a_lone_option_is_not_a_question():
    tallies = pd.DataFrame({"club": ["Rovers", "Rovers", "City", "City", "Town"], "result": ["win", "draw"] * 2 + ["win"],
                            "count": [7, 3, 5, 6, 2]})
    out = orchestrator.ask("How many games did each club play in total?", tallies, {}, use_model=False)
    # (Stage 2: the deterministic planner now reads the grouping from the shared request and answers first; the
    # figure must be the same whichever deterministic path computed it)
    call = next(c for c in out["calls"] if c["tool"] in ("analysis_program", "compute_metric"))
    prog = (call["result"]["results"][""]["result"]["value"] if call["tool"] == "analysis_program"
            else {g["group"]: g["value"] for g in call["result"]["groups"]})
    assert prog == pytest.approx(tallies.groupby("club")["count"].sum().astype(float).to_dict())
    assert out["verify_status"] == "verified"
    out = orchestrator.ask("If Rovers were removed, how many games would remain in total?", tallies, {}, use_model=False)
    assert next(c for c in out["calls"] if c["tool"] == "analysis_program")["result"]["results"][""]["closed"]["value"] \
        == pytest.approx(13.0)


def test_a_calculation_that_cannot_be_recomputed_is_still_held_to_the_request():
    """The model calculator's plan derives a measure (so its figure is not recomputed) but sums the wrong column over
    the wrong records: Ash is a column of votes, not a value of leader to filter on. It is withheld."""
    df = ballots()
    memory = {"last_calc": {"spec": {"derive": {"votes": "Birch * 1"}, "exclude": [
        {"reason": "not Ash", "where": "leader != 'Ash'"}], "group_by": "kind", "aggregate": {"votes": "sum(votes)"}},
        "result": {}, "meta": {"primary": "votes"}}}
    out = {"plan_source": "model_calc", "plan": {"type": "calc"}, "calls": [{"tool": "calculate", "status": "ok",
                                                                            "result": {}}]}
    v = verify.check_answer("What is the total number of votes for Ash in each kind?", df, out, None, memory)
    assert v["status"] == "failed"
    assert "uses Birch" in v["checks"][0]["detail"] and "not a value of leader" in v["checks"][0]["detail"]
