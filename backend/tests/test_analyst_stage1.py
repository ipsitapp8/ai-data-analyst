"""Stage 1: no misleading routes, and specific clarifications instead of generic suggestions.

Each case checks that the answer is the right KIND of answer (a clarification that asks for what is missing, a
breakdown by the group named, associations computed on all records) and, where a number is shown, that it equals an
independent computation. Two schemas, model offline (as on the hosted deployment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analyst import orchestrator, planner

OLD_GENERIC = "I can answer that best"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(planner, "model_available", lambda: False)


def shifts():
    rows = []
    for d in range(1, 9):
        for name, team, rate in [("Ines Rossi", "North", 20.0), ("Omar Sato", "North", 22.0), ("Lena Weber", "South", 25.0)]:
            rows.append({"staff": name, "team": team, "work_day": f"2026-03-{d:02d}", "start": "09:00",
                         "end": "17:00" if d % 2 else "15:30", "rate": rate})
    df = pd.DataFrame(rows)
    df["work_day"] = pd.to_datetime(df["work_day"])
    return df


def shop():
    rng = np.random.default_rng(4)
    n = 120
    df = pd.DataFrame({"category": rng.choice(["Toys", "Books", "Garden"], n), "month": pd.date_range("2025-01-01", periods=n, freq="W"),
                       "revenue": rng.uniform(100, 900, n).round(2), "ad_spend": rng.uniform(10, 200, n).round(2),
                       "returned": rng.choice(["Yes", "No", "No", "No"], n), "units": rng.integers(1, 30, n)})
    return df


def ask(q, df, mem=None):
    return orchestrator.ask(q, df, mem if mem is not None else {}, use_model=False)


@pytest.mark.parametrize("make,q", [(shifts, "Which employees are at risk of missing their monthly targets?"),
                                   (shop, "Which categories are on track for their monthly goals?")])
def test_a_target_question_without_a_target_asks_for_it(make, q):
    out = ask(q, make())
    assert out["plan"]["type"] == "clarify" and out["plan"].get("expects") == "target"
    assert not out["calls"] and "What is the target?" in out["answer"]


def test_which_categories_contribute_to_profit_is_broken_down_and_honest_about_costs():
    df = shop()
    out = ask("Which product categories contribute most to profit rather than just revenue?", df)
    res = next(c["result"] for c in out["calls"] if c["tool"] == "calculator")
    assert res["group_by"] == "category"
    truth = (df.revenue - df.ad_spend).groupby(df.category).sum()
    assert {g["group"]: g["profit"] for g in res["groups"]} == pytest.approx(truth.to_dict())
    assert any("overstates true profit" in n for n in out["notes"])


@pytest.mark.parametrize("q", ["What factors appear associated with returned?", "Which variables best explain returned?",
                               "What drives returned?"])
def test_association_questions_use_all_records(q):
    df = shop()
    out = ask(q, df)
    r = next(c["result"] for c in out["calls"] if c["tool"] == "associations")
    assert r["target"] == "returned" and all(f["n"] == len(df) for f in r["factors"])
    assert "Scope:" not in out["answer"], "the outcome is not used as a filter"


def test_unanswerable_follow_ups_are_specific_not_generic():
    df, mem = shifts(), {}
    ask("Total hours worked by staff", df, mem)
    out = ask("Why is the result different?", df, mem)
    assert OLD_GENERIC not in out["answer"] and "The current result is from *Total hours worked by staff*" in out["answer"]
    out = ask("Forget that and analyse sales instead.", df, mem)
    assert "**sales**" in out["answer"] and "Dataset** selector" in out["answer"]
    out = ask("Show the zebra stripes per morale band", df, {})
    assert "nothing called" in out["answer"] and "**zebra**" in out["answer"]


def test_no_reply_uses_the_old_generic_text():
    df = shop()
    for q in ["hmm what about stuff", "Analyse this entire dataset and identify important business problems.",
              "Compare him with the others."]:
        assert OLD_GENERIC not in ask(q, df)["answer"]
