"""What-if Phase 3: the Silt Copilot and the What-If page share one calculation service.

The Copilot answers what-if and goal sentences about its dataset through the same service, so for the same numbers it must give
the same answer as the page. It must also step aside, leaving every earlier answer unchanged, whenever it is not sure.
"""
from __future__ import annotations

import pathlib
import re

import pandas as pd
import pytest
from platform_helpers import workspace

from app import scenario_service as svc
from app import scenarios
from app.analyst import orchestrator, scenario_bridge as bridge

SALES = pd.DataFrame({"region": ["East", "West"] * 4, "units": [10, 20, 30, 40, 50, 60, 70, 80],
                      "revenue": [200, 400, 600, 800, 1000, 1200, 1400, 1600], "cost": [120, 240, 360, 480, 600, 720, 840, 960]})
# totals: 360 units, 7,200 revenue, 4,320 cost -> price 20, unit cost 12, volume 360
BASE = {"price": 20.0, "volume": 360.0, "unit_cost": 12.0, "fixed_cost": 500.0}
FUNNEL = pd.DataFrame({"campaign": ["a", "b", "c", "d"], "leads": [1000, 2000, 3000, 4000], "conversions": [20, 50, 60, 70],
                       "revenue": [2000.0, 5500.0, 6000.0, 7000.0], "spend": [400.0, 700.0, 900.0, 1000.0]})


def ask(message, df=SALES, memory=None):
    memory = {} if memory is None else memory
    return orchestrator.ask(message, df, memory, use_model=False), memory


# ============================================================ one service, two doors ==

def test_the_copilot_gives_the_numbers_the_what_if_service_gives():
    mem = {"scenario_inputs": {"fixed_cost": 500.0}}
    out, _ = ask("What happens if I increase the price by 10%?", memory=mem)
    direct = svc.evaluate("unit_economics", BASE, {"your scenario": {"price": {"type": "pct", "value": 10}}})
    p = direct["scenarios"]["your scenario"]["outputs"]
    assert out["plan_source"] == "tool" and out["plan"]["type"] == "scenario" and out["plan"]["model"] == "unit_economics"
    # baseline profit: 7200 - 4320 - 500 = 2380; price +10% adds 0.1 x 7200 = 720
    assert "profit rises from 2,380 to 3,100" in out["answer"] and p["profit"] == pytest.approx(3100.0)
    assert f"Revenue: {svc_n(direct['baseline']['outputs']['revenue'])} → {svc_n(p['revenue'])}" in out["answer"]
    assert direct["explanation"]["summary"].split(".")[0] in out["answer"], "the same explanation text"


def svc_n(x):
    from app import scenario_explain as ex
    return ex.n(x)


def test_a_goal_sentence_is_solved_by_the_same_engine():
    mem = {"scenario_inputs": {"fixed_cost": 500.0}}
    out, _ = ask("What price would give me a 40% profit margin?", memory=mem)
    direct = svc.goal_seek("unit_economics", BASE, "margin_pct", 40, "price")
    assert f"{direct['solutions'][0]['value']:.4f}".rstrip("0") in out["answer"].replace(",", "") or "22.31" in out["answer"]
    # margin = ((p - 12) x 360 - 500) / (360 p) = 0.4  ->  0.6 p = 12 + 500 / 360  ->  p = 22.3148
    assert direct["solutions"][0]["value"] == pytest.approx((12 + 500 / 360) / 0.6, rel=1e-9) and direct["verification"]["status"] == "verified"
    assert out["calls"][0]["tool"] == "scenario_service" and out["calls"][0]["result"]["kind"] == "goal_seek"


def test_the_notes_say_where_the_numbers_came_from_and_that_the_arithmetic_was_checked():
    out, _ = ask("increase the price by 10%", memory={"scenario_inputs": {"fixed_cost": 500.0}})
    notes = " ".join(out["notes"])
    assert "same what-if service as the What-If page" in notes and "no number came from a model" in notes
    assert "columns used: units, revenue, cost" in notes and "Independent check of the arithmetic: verified" in notes
    assert "Fixed cost = 500, as you told me" in notes and "Interpreted as: Price per unit: up 10%" in notes
    assert 'sum("revenue") / sum("units")' in notes


def test_the_router_has_no_calculation_of_its_own():
    """A guard against the two doors drifting apart: neither the API router nor the Copilot bridge may call an engine directly."""
    root = pathlib.Path(scenarios.__file__).resolve().parent
    for rel in ("routers/scenarios.py", "analyst/scenario_bridge.py"):
        text = (root / rel).read_text(encoding="utf-8")
        for engine in ("scenarios.evaluate(", "scenario_sim.simulate(", "scenario_solve.goal_seek(", "scenario_sensitivity.two_way("):
            assert engine not in text, f"{rel} calls {engine} directly; it must go through scenario_service"
    assert "scenario_service" in (root / "routers/scenarios.py").read_text(encoding="utf-8")


# ========================================================== missing inputs are asked ==

def test_a_fixed_cost_the_data_does_not_hold_is_asked_for_and_then_used():
    out, mem = ask("What happens if I increase the price by 10%?")
    assert out["plan"]["type"] == "clarify" and "fixed cost" in out["answer"] and "I will not assume one" in out["answer"] and out["calls"] == []
    assert mem["pending_scenario"]["message"] == "What happens if I increase the price by 10%?"
    out2, _ = ask("fixed cost is 500", memory=mem)
    assert out2["plan"]["type"] == "scenario" and "profit rises from 2,380 to 3,100" in out2["answer"] and "pending_scenario" not in mem
    out3, _ = ask("What if I reduce unit cost by 5%?", memory=mem)
    assert "from 2,380 to 2,596" in out3["answer"], "the fixed cost you gave is remembered for the conversation"


def test_a_fixed_cost_stated_in_the_question_is_used_directly():
    out, mem = ask("with fixed costs of 500, what if I increase the price by 10%?")
    assert out["plan"]["type"] == "scenario" and "from 2,380 to 3,100" in out["answer"] and mem["scenario_inputs"]["fixed_cost"] == 500.0


def test_zero_is_a_value_you_may_give():
    out, mem = ask("What happens if I increase the price by 10%?")
    out2, _ = ask("fixed cost is 0", memory=mem)
    assert "from 2,880 to 3,600" in out2["answer"]


# ================================================================ conversation ==

def test_follow_ups_build_on_the_scenario_and_are_explained_from_a_fresh_run():
    mem = {"scenario_inputs": {"fixed_cost": 500.0}}
    a, _ = ask("increase the price by 10%", memory=mem)
    b, _ = ask("and also reduce unit cost by 5%", memory=mem)
    assert "In 'scenario 2', profit rises from 2,380 to 3,316" in b["answer"] and "Price per unit up 10%, Variable cost per unit down 5%" in b["answer"]
    why, _ = ask("Why did profit increase?", memory=mem)
    assert "profit rises from 2,380 to 3,316" in why["answer"] and "Price per unit, which adds 720" in why["answer"] and why["plan"]["type"] == "scenario"
    top, _ = ask("Which variable had the biggest impact?", memory=mem)
    assert top["plan_source"] == "tool" and "Price per unit, which adds 720" in top["answer"]
    fresh, _ = ask("what if unit cost were 5% lower?", memory=mem)
    assert "Price per unit" not in fresh["answer"].split("What was changed")[1].split(".")[0], "no carry-over without 'also'"


def test_a_why_question_with_no_scenario_is_left_to_the_copilot_as_before():
    out, _ = ask("Why did profit increase?")
    assert out["plan"].get("type") != "scenario" and out["plan_source"] != "tool" or "scenario_service" not in str(out["calls"])


# ======================================================== it steps aside otherwise ==

@pytest.mark.parametrize("message", ["What is the average revenue?", "Which region has the highest revenue?", "increase revenue by 10%", "cut costs by 5%",
                                     "Predict revenue", "Explain this dataset in simple terms.", "Find unusual records.", "hello"])
def test_ordinary_questions_are_answered_as_before_and_never_by_the_bridge(message):
    out, _ = ask(message, memory={"scenario_inputs": {"fixed_cost": 500.0}})
    assert out["plan"].get("type") != "scenario" and not any(c["tool"] == "scenario_service" for c in out["calls"])


def test_a_dataset_that_fits_no_model_is_never_handled_by_the_bridge():
    plain = pd.DataFrame({"city": ["a", "b", "c"], "temperature": [10.0, 20.0, 30.0], "rainfall": [5.0, 6.0, 7.0]})
    assert bridge.infer_model(plain) is None
    out, _ = ask("What happens if I increase the price by 10%?", df=plain)
    assert out["plan"].get("type") != "scenario" and not any(c["tool"] == "scenario_service" for c in out["calls"])
    assert bridge.try_answer("increase the price by 10%", plain, {}) is None


def test_the_models_a_dataset_can_fit_are_found_from_its_column_names_only():
    ue = bridge.infer_model(SALES)
    assert ue == ("unit_economics", {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"}, ["units", "revenue", "cost"])
    assert bridge.infer_model(SALES.rename(columns={"cost": "unit_cost"}))[1]["unit_cost_column"] == "unit_cost"
    fn = bridge.infer_model(FUNNEL)
    assert fn[0] == "funnel" and fn[1] == {"visitors_column": "leads", "orders_column": "conversions", "revenue_column": "revenue", "spend_column": "spend"}
    assert bridge.infer_model(SALES.assign(units_sold=1)) is None, "two columns could be the quantity, so it does not guess"
    assert bridge.infer_model(SALES.assign(sales=1.0)) is None, "two columns could be the revenue, so it does not guess"


def test_a_funnel_dataset_answers_a_what_if_through_the_funnel_model():
    out, _ = ask("What if conversion improves by 2 percentage points?", df=FUNNEL)
    base = scenarios.baseline_from_dataset(FUNNEL, bridge._profile(FUNNEL), "funnel", bridge.infer_model(FUNNEL)[1])["inputs"]  # noqa: SLF001
    direct = svc.evaluate("funnel", base, {"s": {"conversion_rate": {"type": "abs", "value": 0.02}}})
    assert out["plan"]["model"] == "funnel" and f"{direct['baseline']['outputs']['profit']:,.0f}" in out["answer"]
    assert f"{direct['scenarios']['s']['outputs']['orders']:,.0f}" in out["answer"] and "Interpreted as: Conversion rate: up 2 percentage points" in " ".join(out["notes"])


def test_a_scope_the_copilot_applied_is_disclosed():
    out, _ = ask("increase the price by 10% for West", memory={"scenario_inputs": {"fixed_cost": 500.0}})
    if out["plan"].get("type") == "scenario":
        assert any("in scope" in n for n in out["notes"])


# ===================================================================== the API ==

def test_the_copilot_endpoint_uses_the_bridge_and_keeps_a_conversation(client):
    ws = workspace(client, csv=SALES.to_csv(index=False).encode(), filename="sales.csv")
    body = {"dataset_id": ws["ds"], "message": "What happens if I increase the price by 10%?", "conversation_id": "wf", "use_model": False}
    first = client.post("/api/analyst/ask", json=body, headers=ws["h"]).json()
    assert "fixed cost" in first["answer"] and first["plan"]["type"] == "clarify"
    second = client.post("/api/analyst/ask", json={**body, "message": "fixed cost is 500"}, headers=ws["h"]).json()
    assert "profit rises from 2,380 to 3,100" in second["answer"] and second["plan_source"] == "tool"
    other = client.post("/api/analyst/ask", json={**body, "message": "increase the price by 10%", "conversation_id": "another"}, headers=ws["h"]).json()
    assert "fixed cost" in other["answer"], "a different conversation does not inherit the fixed cost"
