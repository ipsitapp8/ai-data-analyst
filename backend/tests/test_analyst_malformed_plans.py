"""Malformed model plans are rejected (and repaired or answered another way), never crash the turn.

Shapes taken from the P1 frozen evaluation, where 17 turns failed with AttributeError/TypeError inside
nl_planner.check: entities written as a list, a column given as an object, a where condition that is not an object.
"""
import json

import pandas as pd
import pytest

from app.analyst import nl_planner


@pytest.fixture
def df():
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=30), "region": ["North", "South", "East"] * 10,
                         "sales": range(30), "units": range(30, 60)})


BAD = [
    {"action": "program", "steps": [{"id": "s", "op": "select", "from": "data", "entities": ["region", "North"]},
                                    {"id": "t", "op": "aggregate", "from": "s", "fn": "sum", "column": "sales"}],
     "answer": "t"},
    {"action": "program", "steps": [{"id": "t", "op": "aggregate", "from": "data", "fn": "sum",
                                     "column": {"name": "sales"}}], "answer": "t"},
    {"action": "program", "steps": [{"id": "s", "op": "select", "from": "data", "where": ["sales > 5"]},
                                    {"id": "t", "op": "aggregate", "from": "s", "fn": "count"}], "answer": "t"},
    {"action": "program", "steps": [{"id": "t", "op": "aggregate", "from": "data", "fn": "mean", "column": "sales",
                                     "by": ["region"]}], "answer": "t"},
    {"action": "program", "steps": [{"id": "s", "op": "select", "from": "data", "entities": {"region": [{"v": 1}]}},
                                    {"id": "t", "op": "aggregate", "from": "s", "fn": "sum", "column": "sales"}],
     "answer": "t"},
]


@pytest.mark.parametrize("prog", BAD)
def test_shape_problems_are_reported(prog):
    assert nl_planner.shape_problems(prog["steps"])


@pytest.mark.parametrize("prog", BAD)
def test_malformed_plan_is_rejected_not_raised(df, monkeypatch, prog):
    monkeypatch.setattr(nl_planner, "model_chat", lambda *a, **k: json.dumps(prog))
    out = nl_planner.plan("total sales for North", df, {})
    assert out["status"] in ("failed", "clarify")
    assert out["trace"] and out["trace"][0]["accepted"] is False


def test_malformed_then_repaired_plan_is_accepted(df, monkeypatch):
    good = {"action": "program", "steps": [
        {"id": "s", "op": "select", "from": "data", "entities": {"region": ["North"]}},
        {"id": "t", "op": "aggregate", "from": "s", "fn": "sum", "column": "sales"}], "answer": "t"}
    replies = iter([json.dumps(BAD[0]), json.dumps(good)])
    monkeypatch.setattr(nl_planner, "model_chat", lambda *a, **k: next(replies))
    out = nl_planner.plan("total sales for North", df, {})
    assert out["status"] == "ok"
    assert "entities" in out["trace"][0]["problems"][0]


def test_well_formed_plans_have_no_shape_problems():
    good = [{"id": "s", "op": "select", "from": "data", "entities": {"region": ["North", "South"]},
             "where": [{"column": "sales", "op": ">", "value": 5}], "period": {"start": "$month_start", "end": "$today"}},
            {"id": "t", "op": "aggregate", "from": "s", "fn": "sum", "column": "sales", "by": "region"},
            {"id": "r", "op": "rank", "from": "t", "n": 3, "order": "desc"},
            {"id": "c", "op": "compare", "a": "t", "target": 500, "days": "$days_left_month"}]
    assert nl_planner.shape_problems(good) == []


def test_unexpected_check_error_is_a_rejection(df, monkeypatch):
    prog = {"action": "program", "steps": [{"id": "t", "op": "aggregate", "from": "data", "fn": "sum",
                                            "column": "sales"}], "answer": "t"}
    monkeypatch.setattr(nl_planner, "model_chat", lambda *a, **k: json.dumps(prog))

    def boom(*a, **k):
        raise KeyError("x")
    monkeypatch.setattr(nl_planner, "_normalise", boom)
    out = nl_planner.plan("total sales", df, {})
    assert out["status"] == "failed" and "could not be read" in out["reason"]
