"""The conversation's ordered result register (app/analyst/register.py): earlier results are referred to reliably
("compare that with the previous result", "the original ask", "the first and third"), the comparison uses the
stored figures, and nothing leaks between conversations, users or datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator, planner, scope

TODAY = pd.Timestamp("2025-06-10")


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)
    monkeypatch.setattr(scope, "today", lambda: TODAY)


def shops(seed=5, n=160):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"outlet": rng.choice(["Quay", "Mall", "Depot"], n),
                         "sold_on": TODAY - pd.to_timedelta(rng.integers(0, 40, n), "D"),
                         "takings": rng.gamma(5, 80, n).round(2), "staff_hours": rng.gamma(4, 2, n).round(1)})


def ask(q, df, mem):
    return orchestrator.ask(q, df, mem, use_model=False)


def test_two_results_of_the_same_measure_are_compared_with_their_difference():
    df, mem = shops(), {}
    ask("Total takings for Quay", df, mem)
    ask("Total takings for Mall", df, mem)
    out = ask("Compare that with the previous result.", df, mem)
    r = out["calls"][0]["result"]
    assert out["calls"][0]["tool"] == "compare_results" and (r["a"], r["b"]) == (1, 2) and r["same_measure"]
    q, m = df[df.outlet == "Quay"].takings.sum(), df[df.outlet == "Mall"].takings.sum()
    assert r["figures"]["value"]["a"] == pytest.approx(q) and r["figures"]["value"]["b"] == pytest.approx(m)
    assert "change" in out["answer"] and "nothing was recalculated" in out["answer"]


def test_different_measures_are_set_side_by_side_but_not_subtracted():
    df, mem = shops(), {}
    ask("Total takings by outlet", df, mem)
    ask("Total staff hours by outlet", df, mem)
    out = ask("Compare the first result with the second one.", df, mem)
    r = out["calls"][0]["result"]
    assert not r["same_measure"] and "no difference is computed" in out["answer"]
    assert r["figures"]["Quay"]["a"] == pytest.approx(df[df.outlet == "Quay"].takings.sum())
    assert r["figures"]["Quay"]["b"] == pytest.approx(df[df.outlet == "Quay"].staff_hours.sum())


def test_the_original_ask_of_a_target_chain_is_the_first_scenario():
    df, mem = shops(), {}
    ask("Can Quay reach 30000 takings this month?", df, mem)
    ask("What if the target is 35000?", df, mem)
    out = ask("Set that next to the original ask and tell me what moved.", df, mem)
    r = out["calls"][0]["result"]
    assert out["calls"][0]["tool"] == "compare_scenarios"
    assert r["previous"]["target"] == 30000 and r["latest"]["target"] == 35000


def test_a_result_in_between_is_not_skipped():
    """A scenario, then a plain total: 'the previous result' is the scenario's neighbour in the register, not the
    previous scenario."""
    df, mem = shops(), {}
    ask("Can Quay reach 30000 takings this month?", df, mem)
    ask("What if the target is 35000?", df, mem)
    ask("Total takings by outlet", df, mem)
    out = ask("Compare that with the previous result.", df, mem)
    r = out["calls"][0]["result"]
    assert out["calls"][0]["tool"] == "compare_results" and (r["a"], r["b"]) == (2, 3)


def test_references_that_do_not_exist_are_asked_about():
    df, mem = shops(), {}
    ask("Total takings by outlet", df, mem)
    ask("Total staff hours by outlet", df, mem)
    out = ask("Compare the fifth result with that.", df, mem)
    assert "2 answered question(s)" in out["answer"] and not out["calls"]
    nxt = ask("Total takings for Quay", df, mem)
    assert nxt["calls"], "an unresolved reference leaves nothing pending: the next question is answered on its own"


def test_the_register_is_isolated_per_conversation_user_and_dataset():
    df = shops()
    a = orchestrator.memory_for("team1:user1:dsA:None:conv1")
    ask("Total takings for Quay", df, a)
    ask("Total takings for Mall", df, a)
    for key in ("team1:user1:dsA:None:conv2", "team1:user2:dsA:None:conv1", "team1:user1:dsB:None:conv1"):
        other = orchestrator.memory_for(key)
        out = ask("Compare that with the previous result.", df, other)
        assert "nothing earlier in this conversation" in out["answer"] and not out["calls"], key
    assert len(a["register"]) == 2


def test_compare_with_the_others_is_not_a_register_reference():
    df, mem = shops(), {}
    ask("Can Quay reach 30000 takings this month?", df, mem)
    out = ask("Compare Quay with the others.", df, mem)
    assert out["calls"][0]["tool"] == "analysis_program" and out["plan"]["objective"] == "vs_rest"


def test_the_original_target_survives_changes_and_earlier_results_can_be_recalled():
    df, mem = shops(), {}
    ask("Total takings for Quay", df, mem)
    out = ask("What was my original target?", df, mem)
    assert "No target has been set" in out["answer"] and not out["calls"]
    out = ask("What if the target is 35000?", df, mem)
    assert "no target in this conversation yet" in out["answer"] and not out["calls"]
    ask("Can Quay reach 30000 takings this month?", df, mem)
    ask("What if the target is 35000?", df, mem)
    ask("What if the target is 40000?", df, mem)
    out = ask("What was my original target?", df, mem)
    r = out["calls"][0]["result"]
    assert (r["original"], r["current"]) == (30000, 40000) and "the original is kept unchanged" in out["answer"]
    out = ask("Go back to the first result.", df, mem)
    assert out["calls"][0]["result"]["figures"]["value"] == pytest.approx(df[df.outlet == "Quay"].takings.sum())
    out = ask("Show me the previous result again.", df, mem)
    assert out["calls"][0]["result"]["n"] == 4, "the latest computed result, not the recall just shown"


def test_the_previous_result_is_a_computed_result_not_an_earlier_comparison():
    df, mem = shops(), {}
    ask("Total takings for Quay", df, mem)
    ask("Total takings for Mall", df, mem)
    ask("Compare that with the previous result.", df, mem)            # entry 3: a comparison
    ask("Total takings for Depot", df, mem)
    out = ask("Compare that with the previous result.", df, mem)
    r = out["calls"][0]["result"]
    assert (r["a"], r["b"]) == (2, 4), "Mall vs Depot; the comparison in between is skipped"
