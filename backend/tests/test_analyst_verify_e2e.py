"""End to end: every computed answer, whichever path produced it, carries one verification status -- verified, not
independently verified (with the reason), or withheld -- decided from the user's question, the executed calculation
and its results. Faults are injected into the execution paths; the expected figures are computed here with pandas."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from app.analyst import nl_planner, orchestrator, planner, scope, tools, verify

TODAY = pd.Timestamp("2026-05-20")


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(scope, "today", lambda: TODAY)


def shop():
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({"store": rng.choice(["Ash", "Elm", "Oak"], n), "season": rng.choice([2024, 2025], n),
                       "sold_on": TODAY - pd.to_timedelta(rng.integers(0, 90, n), "D"),
                       "revenue": rng.gamma(5, 50, n).round(2), "visitors": rng.integers(20, 300, n),
                       "refund": rng.choice(["Yes", "No"], n, p=[.2, .8])})
    return pd.concat([df, df.head(3)], ignore_index=True)          # three exact duplicates


def ask(q, df, mem=None):
    return orchestrator.ask(q, df.copy(), {} if mem is None else mem, use_model=False)


def status_line(out):
    return next((ln for ln in out["answer"].split("\n") if "Verification:" in ln), "")


def figures(out):
    return {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.\d+", out["answer"])}


def inflate(monkeypatch, factor=1.1):
    real = tools._EXEC["compute_metric"]

    def wrong(d, a, m):
        r = real(d, a, m)
        for g in r.get("groups", []):
            g["value"] = g["value"] * factor if g["value"] is not None else None
        if isinstance(r.get("value"), float):
            r["value"] *= factor
        return r
    monkeypatch.setitem(tools._EXEC, "compute_metric", wrong)


# ---------------------------------------------------------------- the three cases --
def test_case_a_a_correct_figure_is_verified():
    df = shop()
    out = ask("Total revenue by store", df)
    want = df.groupby("store").revenue.sum()
    assert {round(v, 2) for v in want} <= figures(out)
    assert out["verify_status"] == "verified" and "✓" in status_line(out)
    assert any(c["check"].startswith("independent recomputation") and c["ok"] for c in out["verification"])


def test_case_a_a_correlation_is_recomputed_and_verified():
    from scipy import stats
    df = shop()
    out = ask("Is revenue correlated with visitors?", df)
    want = stats.pearsonr(df.revenue, df.visitors).statistic
    r = next(c for c in out["calls"] if c["tool"] == "correlation")["result"]
    assert r["pearson_r"] == pytest.approx(want)
    assert out["verify_status"] == "verified" and "✓" in status_line(out)
    assert all(c["ok"] for c in out["verification"] if c["check"].startswith("independent recomputation"))


@pytest.mark.parametrize("q", ["Forecast revenue for next month"])
def test_case_b_what_cannot_be_recomputed_is_not_called_verified(q):
    out = ask(q, shop())
    assert out["verify_status"] == "not independently verified"
    line = status_line(out)
    assert "not independently verified —" in line and "✓" not in line and "results" in line, "the reason is given"
    assert out["plan"].get("type") != "withheld", "an unverifiable answer is not a failed one"


def test_case_c_a_wrong_figure_is_withheld_and_a_verified_fallback_answers(monkeypatch):
    df = shop()
    inflate(monkeypatch)
    out = ask("Total revenue by store", df)
    want = df.groupby("store").revenue.sum()
    wrong = {round(v * 1.1, 2) for v in want}
    assert out["plan_source"] == "rule_program" and out["verify_status"] == "verified"
    assert {round(v, 2) for v in want} <= figures(out) and not wrong & figures(out)
    assert not wrong & {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.\d+", out.get("details", ""))}
    assert any("did not pass answer verification" in n for n in out["notes"])


def test_case_c_with_no_safe_fallback_nothing_is_shown(monkeypatch):
    df = shop()
    inflate(monkeypatch)
    monkeypatch.setattr(nl_planner, "rule_plan", lambda *a, **k: None)
    mem = {}
    out = ask("Total revenue by store", df, mem)
    wrong = {round(v * 1.1, 2) for v in df.groupby("store").revenue.sum()}
    assert out["plan_source"] == "withheld" and out["verify_status"] == "withheld" and not out["charts"]
    assert "haven't shown it" in out["answer"] and not figures(out), "no figure at all in a withheld answer"
    assert not wrong & {float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.\d+", str(out))}
    assert not mem.get("register"), "a withheld answer is not something later turns can refer to"


# ---------------------------------------------------------------- the question's requirements --
def test_a_dropped_filter_or_year_is_caught(monkeypatch):
    df = shop()
    real = tools._EXEC["compute_metric"]
    monkeypatch.setitem(tools._EXEC, "compute_metric", lambda d, a, m: real(df, a, m))   # ignores the scope
    out = ask("Total revenue for Ash by season", df)
    ash = df[df.store == "Ash"].groupby("season").revenue.sum()
    assert out["verify_status"] == "verified" and {round(v, 2) for v in ash} <= figures(out)
    full = df.groupby("season").revenue.sum()
    assert not {round(v, 2) for v in full} & figures(out)
    df2 = shop().drop(columns=["sold_on"])                      # the year lives only in the season column
    monkeypatch.setitem(tools._EXEC, "compute_metric", lambda d, a, m: real(df2, a, m))  # ignores "in 2025"
    out = ask("Total revenue by store in 2025", df2)
    every_year = df2.groupby("store").revenue.sum()
    assert out["verify_status"] == "withheld" and not {round(v, 2) for v in every_year} & figures(out), \
        "the all-years figure is withheld; Silt's own program does not apply the year either, so nothing is shown"
    monkeypatch.setitem(tools._EXEC, "compute_metric", real)
    out = ask("Total revenue by store in 2025", df2)
    y = df2[df2.season == 2025].groupby("store").revenue.sum()
    assert out["verify_status"] == "verified" and {round(v, 2) for v in y} <= figures(out)


def test_a_breakdown_by_a_year_column_is_required():
    df = shop()
    one_total = [{"id": "s", "op": "select", "from": "data", "entities": {"store": ["Ash"]}},
                 {"id": "t", "op": "aggregate", "from": "s", "fn": "sum", "column": "revenue"}]
    import app.analyst.program as P
    res = {k: r.summary() for k, r in P.run(one_total, df).items()}
    v = verify.check("Total revenue for Ash by season", df, one_total, res)
    assert v["status"] == "failed" and "by season" in v["checks"][0]["detail"]


def test_wrong_measure_grouping_and_denominator_are_caught():
    df = shop()
    import app.analyst.program as P

    def run(steps):
        return {k: r.summary() for k, r in P.run(steps, df).items()}
    wrong_col = [{"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "visitors", "by": "store"}]
    assert "about revenue" in verify.check("Total revenue by store", df, wrong_col, run(wrong_col))["checks"][0]["detail"]
    wrong_by = [{"id": "t", "op": "aggregate", "from": "data", "fn": "sum", "column": "revenue", "by": "refund"}]
    assert "by store" in verify.check("Total revenue by store", df, wrong_by, run(wrong_by))["checks"][0]["detail"]
    wrong_den = [{"id": "a", "op": "aggregate", "from": "data", "fn": "sum", "column": "revenue", "by": "store"},
                 {"id": "b", "op": "aggregate", "from": "data", "fn": "count", "by": "store"},
                 {"id": "r", "op": "ratio", "a": "a", "b": "b"}]
    v = verify.check("Revenue per visitor by store", df, wrong_den, run(wrong_den))
    assert v["status"] == "failed" and "visitors" in v["checks"][0]["detail"]


def test_duplicate_handling_follows_the_question():
    df = shop()
    out = ask("Total revenue by store excluding duplicates", df)
    want = df.drop_duplicates().groupby("store").revenue.sum()
    assert out["verify_status"] == "verified" and {round(v, 2) for v in want} <= figures(out)
    assert "Left out, as you asked: 3 exact duplicate record(s)" in out["answer"]
    kept = df.groupby("store").revenue.sum()
    claim = {"plan_source": "rules", "plan": {}, "answer": "", "calls": [{"tool": "compute_metric", "status": "ok",
             "args": {"aggregation": "sum", "column": "revenue", "group_by": "store"},
             "result": {"groups": [{"group": k, "value": float(v)} for k, v in kept.items()]}}]}
    v = verify.check_answer("Total revenue by store excluding duplicates", df, claim, None, {})
    assert v["status"] == "failed" and "asks to exclude duplicates" in v["checks"][0]["detail"]
    plain = ask("Total revenue by store", df)
    assert "3 exact duplicate record(s)" in plain["answer"] and "counted as recorded" in plain["answer"]


def test_a_description_of_the_data_is_not_an_answer_to_a_figure():
    df = shop()
    claim = {"plan_source": "router", "plan": {}, "answer": "", "calls": [{"tool": "missing_values", "status": "ok",
                                                                          "result": {"missing": {}}}]}
    v = verify.check_answer("Total revenue by store", df, claim, None, {})
    assert v["status"] == "failed" and "missing values result" in v["checks"][0]["detail"]


def test_observed_results_are_not_hypothetical_and_scenarios_are():
    df = shop()
    import app.analyst.program as P
    steps = [{"id": "w", "op": "scenario", "from": "data", "changes": [{"remove": {"store": "Ash"}}]},
             {"id": "t", "op": "aggregate", "from": "w", "fn": "sum", "column": "revenue"},
             {"id": "o", "op": "aggregate", "from": "data", "fn": "sum", "column": "revenue"}]
    res = {k: r.summary() for k, r in P.run(steps, df).items()}
    q = "What if Ash were closed: total revenue?"
    assert verify.check(q, df, steps, res)["checks"][1]["ok"]
    res["o"]["hypothetical"] = True                            # an observed total presented as hypothetical
    v = verify.check(q, df, steps, res)
    assert v["status"] == "failed" and "observed but marked hypothetical: o" in v["checks"][1]["detail"]


def test_a_part_of_the_question_left_unanswered_is_stated():
    df = shop()
    out = {"answer": "**Elm** has the highest total revenue.", "calls": [{"tool": "compute_metric", "status": "ok"}]}
    assert verify.unanswered_parts("Which store has the highest total revenue and why?", df, out) == [
        "what is associated with it (the 'why' part)"]
    full = ask("Which store has the highest total revenue and why?", df)
    assert "Not answered here" not in full["answer"] and "associated with" in full["answer"]
