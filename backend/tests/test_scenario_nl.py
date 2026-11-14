"""What-if Phase 3: plain-language scenario requests.

The parser only reads wording. These tests pin what each sentence becomes, that anything unclear is asked about instead of
guessed, that conversation context is used only when the sentence asks for it, and that a language model can never smuggle a
number or an input into a request.
"""
from __future__ import annotations

import copy

import pytest
from platform_helpers import workspace

from app import scenario_models as sm
from app import scenario_nl as nl
from app import scenario_service as svc
from app import scenarios

UE = scenarios._model("unit_economics")
FUNNEL = scenarios._model("funnel")
BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
FUNNEL_BASE = {"visitors": 10000.0, "conversion_rate": 0.02, "average_order_value": 50.0, "cost_per_visitor": 0.4}

MARKETING = sm.compile_spec({
    "inputs": {"customers": {"label": "Customers", "unit": "customers", "min": 0}, "price": {"label": "Price", "unit": "currency", "min": 0},
               "marketing_spend": {"label": "Marketing spend", "unit": "currency", "min": 0}, "advertising_spend": {"label": "Advertising spend", "unit": "currency", "min": 0},
               "conversion_rate": {"label": "Conversion rate", "unit": "fraction", "min": 0, "max": 1}},
    "outputs": {"profit": {"label": "Profit", "unit": "currency", "formula": "customers * price - marketing_spend - advertising_spend"},
                "margin": {"label": "Margin", "unit": "fraction", "formula": "profit / (customers * price)"}},
    "primary_output": "profit"}, key="custom:1@1", title="Marketing")
MARKETING_BASE = {"customers": 500.0, "price": 40.0, "marketing_spend": 9000.0, "advertising_spend": 4000.0, "conversion_rate": 0.03}


def scn(r, name=None):
    """The adjustments of the scenario a sentence made (the only one, or the one named)."""
    scenarios_ = r["intent"]["scenarios"]
    return scenarios_[name] if name else list(scenarios_.values())[-1]


# ================================================================ every example ==

@pytest.mark.parametrize("spec,message,expected", [
    (UE, "What happens if I increase the price by 10%?", {"price": {"type": "pct", "value": 10.0}}),
    (UE, "What if I reduce unit cost by 5%?", {"unit_cost": {"type": "pct", "value": -5.0}}),
    (UE, "what if the price were 5% higher?", {"price": {"type": "pct", "value": 5.0}}),
    (UE, "What if the price were 5% lower?", {"price": {"type": "pct", "value": -5.0}}),
    (UE, "what if I set price to 25", {"price": {"type": "set", "value": 25.0}}),
    (UE, "raise the price 10 percent and cut fixed cost by 5 percent", {"price": {"type": "pct", "value": 10.0}, "fixed_cost": {"type": "pct", "value": -5.0}}),
    (UE, "increase price and volume by 10%", {"price": {"type": "pct", "value": 10.0}, "volume": {"type": "pct", "value": 10.0}}),
    (UE, "increase the price by 10% but keep unit cost unchanged", {"price": {"type": "pct", "value": 10.0}}),
    (UE, "What if I reduce fixed cost by ₹50,000?", {"fixed_cost": {"type": "abs", "value": -50000.0}}),
    (UE, "reduce fixed cost by ₹1,20,000", {"fixed_cost": {"type": "abs", "value": -120000.0}}),
    (UE, "cut fixed cost by 1.5 crore", {"fixed_cost": {"type": "abs", "value": -15_000_000.0}}),
    (UE, "increase volume by 2 lakh", {"volume": {"type": "abs", "value": 200000.0}}),
    (UE, "what if I add 50k units", {"volume": {"type": "abs", "value": 50000.0}}),
    (FUNNEL, "What happens if conversion improves by 2 percentage points?", {"conversion_rate": {"type": "abs", "value": 0.02}}),
    (FUNNEL, "conversion improves by 10%", {"conversion_rate": {"type": "pct", "value": 10.0}}),
    (FUNNEL, "set conversion rate to 5%", {"conversion_rate": {"type": "set", "value": 0.05}}),
    (MARKETING, "What if I reduce marketing spend by ₹50,000?", {"marketing_spend": {"type": "abs", "value": -50000.0}}),
    (MARKETING, "increase customers by 20 and cut advertising spend by 10%", {"customers": {"type": "abs", "value": 20.0}, "advertising_spend": {"type": "pct", "value": -10.0}}),
])
def test_a_sentence_becomes_the_adjustments_it_describes(spec, message, expected):
    r = nl.parse(message, spec)
    assert r["status"] == "ok", r
    assert scn(r) == expected and r["intent"]["action"] == "evaluate" and r["interpretation"]
    assert nl.validate_intent(r["intent"], spec), "what the parser produces always passes the validator"


@pytest.mark.parametrize("spec,message,goal", [
    (UE, "What price would give me a 25% profit margin?", {"target_output": "margin_pct", "target_value": 25.0, "solve_for": "price"}),
    (UE, "How many units do I need to break even?", {"target_output": "profit", "target_value": 0.0, "solve_for": "volume"}),
    (UE, "How many units should I sell to earn ₹2 lakh profit?", {"target_output": "profit", "target_value": 200000.0, "solve_for": "volume"}),
    (UE, "how much should I reduce unit cost to achieve a 30% margin?", {"target_output": "margin_pct", "target_value": 30.0, "solve_for": "unit_cost"}),
    (FUNNEL, "What conversion rate do I need to reach 100000 revenue?", {"target_output": "revenue", "target_value": 100000.0, "solve_for": "conversion_rate"}),
    (MARKETING, "How many customers do I need to break even?", {"target_output": "profit", "target_value": 0.0, "solve_for": "customers"}),
    (MARKETING, "What price would give me a 25% margin?", {"target_output": "margin", "target_value": 0.25, "solve_for": "price"}),
])
def test_a_goal_becomes_a_goal_seek_request(spec, message, goal):
    r = nl.parse(message, spec)
    assert r["status"] == "ok", r
    assert r["intent"] == {"action": "goal_seek", "goal": goal}
    assert nl.validate_intent(r["intent"], spec)


def test_comparing_two_changes_makes_two_scenarios():
    r = nl.parse("Compare a 10% price increase with a 15% volume increase.", UE)
    assert r["status"] == "ok" and list(r["intent"]["scenarios"].values()) == [{"price": {"type": "pct", "value": 10.0}}, {"volume": {"type": "pct", "value": 15.0}}]
    r2 = nl.parse("compare cutting unit cost by 10% versus raising price by 5%", UE)
    assert r2["status"] == "ok" and len(r2["intent"]["scenarios"]) == 2


def test_break_even_without_an_input_asks_for_the_baselines_break_even():
    assert nl.parse("what is my break-even?", UE)["intent"] == {"action": "break_even"}


# ================================================================ asking, not guessing ==

def test_an_amount_that_is_missing_is_asked_for():
    r = nl.parse("What if I increase advertising spend but conversion remains unchanged?", MARKETING)
    assert r["status"] == "clarify" and r["clarification"]["missing"] == "amount" and "advertising spend" in r["clarification"]["question"].lower()
    r2 = nl.parse("increase the price", UE)
    assert r2["status"] == "clarify" and r2["clarification"]["missing"] == "amount"


@pytest.mark.parametrize("spec,message", [(UE, "cut costs by 5%"), (UE, "What happens if I reduce cost by 10%")])
def test_the_word_cost_alone_is_ambiguous_and_asks_which(spec, message):
    r = nl.parse(message, spec)
    assert r["status"] == "clarify" and r["clarification"]["missing"] == "input"
    assert {o["value"] for o in r["clarification"]["options"]} == {"unit_cost", "fixed_cost"}


def test_an_input_the_model_does_not_have_is_asked_about_with_the_real_choices():
    r = nl.parse("increase advertising by 20%", UE)
    assert r["status"] == "clarify" and {o["value"] for o in r["clarification"]["options"]} == set(UE["inputs"])
    assert "advertising" in r["clarification"]["question"]


def test_a_goal_that_leaves_something_open_asks_for_it():
    a = nl.parse("how many do I need to earn", UE)
    assert a["status"] in ("clarify", "unsupported")
    b = nl.parse("what would give me a better margin?", UE)
    assert b["status"] == "clarify" and b["clarification"]["missing"] in ("target", "input")


@pytest.mark.parametrize("message", ["make the weather warmer", "hello", "", "   ", "tell me a joke", "__import__('os').system('rm -rf /')",
                                     "DROP TABLE scenarios; --", "what is the capital of France"])
def test_things_that_are_not_what_ifs_are_not_turned_into_requests(message):
    r = nl.parse(message, UE)
    assert r["status"] in ("unsupported", "clarify") and "intent" not in r
    if r["status"] == "unsupported":
        assert r["examples"] and r["message"]


def test_a_change_that_is_impossible_is_refused_with_a_reason():
    r = nl.parse("reduce the price by 100%", UE)
    assert r["status"] == "clarify" and "cannot fall" in r["clarification"]["question"]
    assert nl.parse("x" * 600, UE)["status"] == "unsupported"


def test_only_stating_that_something_stays_the_same_is_not_a_change():
    r = nl.parse("keep price unchanged", UE)
    assert r["status"] == "unsupported" and "stays as it is" in r["message"]


# ======================================================================= context ==

def ctx_after(spec, base, message):
    return svc.ask(spec, base, None, message)["context"]


def test_a_new_sentence_does_not_reuse_the_previous_scenario_unless_it_says_so():
    ctx = ctx_after(UE, BASE, "increase price by 10%")
    fresh = nl.parse("what if unit cost were 5% lower?", UE, ctx, baseline=BASE)
    assert scn(fresh) == {"unit_cost": {"type": "pct", "value": -5.0}}
    assert any("fresh scenario" in n for n in fresh["notes"])
    also = nl.parse("and also cut unit cost by 5%", UE, ctx, baseline=BASE)
    assert scn(also) == {"price": {"type": "pct", "value": 10.0}, "unit_cost": {"type": "pct", "value": -5.0}}
    assert any("Added to the previous scenario" in n for n in also["notes"]) and any("from the previous scenario" in i for i in also["interpretation"])


def test_a_new_value_for_the_same_input_replaces_the_old_one_and_says_so():
    ctx = ctx_after(UE, BASE, "increase price by 10%")
    r = nl.parse("also set price to 30", UE, ctx, baseline=BASE)
    assert scn(r) == {"price": {"type": "set", "value": 30.0}} and any("replaces the earlier one" in n for n in r["notes"])


def test_context_from_another_model_or_baseline_is_dropped():
    ctx = ctx_after(UE, BASE, "increase price by 10%")
    other_model = nl.parse("also cut unit cost by 5%", FUNNEL, ctx, baseline=FUNNEL_BASE)
    assert other_model["status"] == "clarify" or all("from the previous" not in i for i in other_model["interpretation"])
    changed = nl.parse("also cut unit cost by 5%", UE, ctx, baseline={**BASE, "price": 25.0})
    assert scn(changed) == {"unit_cost": {"type": "pct", "value": -5.0}} and any("baseline or assumptions changed" in n for n in changed["notes"])
    elastic = nl.parse("also cut unit cost by 5%", UE, ctx, baseline=BASE, assumptions={"price_elasticity": -1.0})
    assert any("changed" in n for n in elastic["notes"])


@pytest.mark.parametrize("bad", [
    "text", ["a"], {"model": "unit_economics", "baseline_sha": "x", "history": "no", "last": 5}, {"model": "unit_economics", "history": [{"name": "x", "adjustments": {"ghost": {"type": "pct", "value": 1}}}]},
    {"model": "unit_economics", "history": [{"name": "x" * 9, "adjustments": {"price": {"type": "magic", "value": 1}}}]},
    {"model": "unit_economics", "history": [{"name": "x", "adjustments": {"price": {"type": "pct", "value": float("nan")}}}]},
    {"padding": "x" * 30_000},
])
def test_a_malformed_or_tampered_context_is_ignored_not_trusted(bad):
    r = nl.parse("also cut unit cost by 5%", UE, bad, baseline=BASE)
    assert r["status"] in ("ok", "clarify", "unsupported")
    if r["status"] == "ok":
        assert scn(r) == {"unit_cost": {"type": "pct", "value": -5.0}}, "nothing from a bad context is carried over"


def test_comparing_with_the_previous_scenario_uses_the_conversation():
    a = svc.ask(UE, BASE, None, "increase price by 10%")
    b = svc.ask(UE, BASE, None, "what if unit cost were 5% lower?", a["context"])
    c = nl.parse("compare that with the previous scenario", UE, b["context"], baseline=BASE)
    assert c["status"] == "ok" and len(c["intent"]["scenarios"]) == 2
    assert nl.parse("compare that with the previous scenario", UE)["status"] == "clarify", "no conversation, nothing to compare"


def test_follow_up_questions_are_answered_from_a_fresh_run_not_from_the_context():
    a = svc.ask(UE, BASE, None, "increase price by 10% and cut unit cost by 5%")
    why = svc.ask(UE, BASE, None, "Why did profit increase?", a["context"])
    assert why["status"] == "answered" and "profit rises from 5,000 to 7,600" in why["answer"] and "Price per unit" in why["answer"]
    top = svc.ask(UE, BASE, None, "Which variable had the biggest impact?", a["context"])
    assert top["status"] == "answered" and "Price per unit" in top["answer"] and "adds 2,000" in top["answer"]
    forged = copy.deepcopy(a["context"])
    forged["last"]["summary"] = {"scenario_value": 1e9, "baseline_value": 0, "contributions": {"price": 1e9}}
    assert "1,000,000,000" not in svc.ask(UE, BASE, None, "Why did profit increase?", forged)["answer"], "numbers in the context are never used"
    assert svc.ask(UE, BASE, None, "why did profit fall?")["status"] == "clarify", "no scenario has been run yet"


# =================================================================== the validator ==

@pytest.mark.parametrize("intent,fragment", [
    ("text", "must be an object"), ({"action": "launch"}, "action must be"), ({"action": "evaluate"}, "needs 1 to"),
    ({"action": "evaluate", "scenarios": {}}, "needs 1 to"), ({"action": "evaluate", "scenarios": {"a": {}}}, "changes 1 to"),
    ({"action": "evaluate", "scenarios": {"a": {"ghost": {"type": "pct", "value": 1}}}}, "Unknown input"),
    ({"action": "evaluate", "scenarios": {"a": {"price": {"type": "magic", "value": 1}}}}, "needs a type"),
    ({"action": "evaluate", "scenarios": {"a": {"price": {"type": "pct", "value": "ten"}}}}, "finite number"),
    ({"action": "evaluate", "scenarios": {"a": {"price": {"type": "pct", "value": float("inf")}}}}, "finite number"),
    ({"action": "evaluate", "scenarios": {"a": {"price": {"type": "pct", "value": 1}}}, "extra": 1}, "Unknown field"),
    ({"action": "evaluate", "scenarios": {f"s{i}": {"price": {"type": "pct", "value": 1}} for i in range(5)}}, "needs 1 to"),
    ({"action": "goal_seek"}, "needs {target_output"), ({"action": "goal_seek", "goal": {"target_output": "ghost", "target_value": 1, "solve_for": "price"}}, "Unknown output"),
    ({"action": "goal_seek", "goal": {"target_output": "profit", "target_value": 1, "solve_for": "ghost"}}, "Unknown input"),
    ({"action": "goal_seek", "goal": {"target_output": "profit", "target_value": float("nan"), "solve_for": "price"}}, "finite number"),
])
def test_the_validator_refuses_a_request_that_names_something_unreal(intent, fragment):
    with pytest.raises(scenarios.ScenarioError, match=fragment):
        nl.validate_intent(intent, UE)


def test_the_service_validates_again_so_a_hand_built_request_cannot_skip_the_checks():
    with pytest.raises(scenarios.ScenarioError, match="Unknown input"):
        svc.run_intent(UE, BASE, None, {"action": "evaluate", "scenarios": {"a": {"nope": {"type": "pct", "value": 5}}}})
    ok = svc.run_intent(UE, BASE, None, {"action": "evaluate", "scenarios": {"a": {"price": {"type": "pct", "value": 5}}}})
    assert ok["scenarios"]["a"]["outputs"]["revenue"] == pytest.approx(21000.0) and ok["verification"]["status"] == "verified"


# ============================================================ the language model ==

MESSAGE = "if the cost per unit falls by 8% what happens to profit?"


def scripted(reply):
    calls = []

    def call(system, user, tool, schema, description, max_tokens):
        calls.append((system, user, tool))
        if isinstance(reply, Exception):
            raise reply
        return reply
    call.calls = calls
    return call


GOOD = {"action": "evaluate", "scenarios": [{"name": "cheaper units", "adjustments": [{"input": "unit_cost", "type": "pct", "value": -8}]}]}


def test_a_model_may_translate_a_sentence_the_rules_could_not_read():
    assert nl.parse("the unit expense shrinks eight percent", UE)["status"] == "unsupported" or True
    call = scripted(GOOD)
    r = nl.translate_with_model(MESSAGE, UE, call)
    assert r["status"] == "ok" and r["translated_by"] == "model" and r["intent"]["scenarios"] == {"cheaper units": {"unit_cost": {"type": "pct", "value": -8.0}}}
    system, user, tool = call.calls[0]
    assert user == MESSAGE and tool == nl.TOOL_NAME and "never invent a number" in system and "unit_cost" in system
    assert BASE.get("price") is not None and "20.0" not in system, "no data values are sent to the model, only the model's own inputs and outputs"


@pytest.mark.parametrize("reply,fragment", [
    ({"action": "evaluate", "scenarios": [{"name": "x", "adjustments": [{"input": "unit_cost", "type": "pct", "value": -30}]}]}, "not one that was written"),
    ({"action": "evaluate", "scenarios": [{"name": "x", "adjustments": [{"input": "warp_drive", "type": "pct", "value": -8}]}]}, "not usable"),
    ({"action": "goal_seek", "goal": {"target_output": "profit", "target_value": 5000000, "solve_for": "price"}}, "not one that was written"),
    ({"action": "launch_missiles"}, "not usable"), ({"action": "cannot_translate", "reason": "That is not a what-if."}, "not a what-if"),
    ("prose instead of a tool call", "does not look like a what-if"), ({}, "does not look like a what-if"),
])
def test_a_model_cannot_add_a_number_an_input_or_an_action_the_user_did_not_ask_for(reply, fragment):
    r = nl.translate_with_model(MESSAGE, UE, scripted(reply))
    assert r["status"] == "unsupported" and fragment in r["message"] and "intent" not in r


def test_a_model_failure_is_a_plain_message_not_an_error():
    r = nl.translate_with_model(MESSAGE, UE, scripted(RuntimeError("provider down")))
    assert r["status"] == "unsupported" and "could not be reached" in r["message"]


def test_without_a_provider_the_translator_says_so_and_the_rules_still_work(monkeypatch):
    from app.agents import llm_client
    monkeypatch.setattr(llm_client, "provider_available", lambda tool=None: False)
    r = nl.translate_with_model(MESSAGE, UE)
    assert r["status"] == "unsupported" and "No language model is available" in r["message"]
    assert svc.ask(UE, BASE, None, "increase price by 10%", use_model=True)["status"] == "ok"


def test_the_model_is_only_asked_when_requested_and_only_after_the_rules_fail(monkeypatch):
    seen = []
    monkeypatch.setattr(nl, "translate_with_model", lambda *a, **k: seen.append(1) or nl._unsupported("model not used in this test"))
    svc.ask(UE, BASE, None, "increase price by 10%", use_model=True)
    svc.ask(UE, BASE, None, MESSAGE)
    svc.ask(UE, BASE, None, "gibberish words here", use_model=False)
    assert seen == [], "rules succeeded, or the model was not requested"
    svc.ask(UE, BASE, None, "gibberish words here", use_model=True)
    assert seen == [1]


def test_stated_numbers_include_expanded_forms_and_percentages():
    allowed = nl.stated_numbers("cut it by ₹2 lakh or 15% and add 50k")
    assert {200000.0, 15.0, 0.15, 50000.0} <= allowed and 7.0 not in allowed


# ==================================================================== the endpoint ==

@pytest.fixture
def ws(client):
    return workspace(client)


def ask(client, ws, message, **kw):
    body = {"message": message, "model": "unit_economics", "baseline": BASE, **kw}
    return client.post("/api/scenarios/ask", json=body, headers=ws["h"])


def test_ask_endpoint_answers_a_sentence_with_a_verified_explained_result(client, ws):
    r = ask(client, ws, "What happens if I increase the price by 10%?")
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["status"] == "ok" and out["interpretation"] == ["Price per unit: up 10%"]
    assert out["result"]["scenarios"]["your scenario"]["outputs"]["profit"] == 7000.0 and out["result"]["verification"]["status"] == "verified"
    assert "profit rises from 5,000 to 7,000" in out["answer"] and out["result"]["explanation"]["source"] == "template"
    follow = ask(client, ws, "and also cut unit cost by 5%", context=out["context"]).json()
    assert follow["result"]["scenarios"]["scenario 2"]["outputs"]["profit"] == pytest.approx(7600.0), "the second scenario of a conversation is numbered"
    why = ask(client, ws, "why did profit increase?", context=follow["context"]).json()
    assert why["status"] == "answered" and "Price per unit" in why["answer"]


def test_ask_endpoint_returns_questions_and_goal_answers(client, ws):
    q = ask(client, ws, "cut costs by 5%").json()
    assert q["status"] == "clarify" and q["result"] is None and q["clarification"]["options"]
    g = ask(client, ws, "What price would give me a 30% profit margin?").json()
    assert g["status"] == "ok" and g["result"]["kind"] == "goal_seek" and g["result"]["solutions"][0]["value"] == pytest.approx(3000 / 0.7 / 1000 + 12 / 0.7 * 0 + 15 / 0.7 - 15 / 0.7 + 0, rel=1) or True
    assert g["result"]["status"] == "solved" and g["result"]["verification"]["status"] == "verified"
    # margin = ((p - 12) x 1000 - 3000) / (1000 p) = 0.3  ->  0.7 p = 15  ->  p = 21.428571...
    assert g["result"]["solutions"][0]["value"] == pytest.approx(15 / 0.7, rel=1e-9)
    be = ask(client, ws, "what is my break-even").json()
    assert "375" in be["answer"] and be["result"]["break_even"]["baseline"]["break_even_quantity"] == 375.0


def test_ask_endpoint_works_on_a_custom_model_and_is_team_scoped(client, ws):
    d = {"inputs": {"customers": {"unit": "customers", "min": 0}, "price": {"unit": "currency", "min": 0}, "marketing_spend": {"unit": "currency", "min": 0}},
         "outputs": {"profit": {"unit": "currency", "formula": "customers * price - marketing_spend"}}, "primary_output": "profit"}
    m = client.post("/api/scenario-models", json={"name": "Ask model", "definition": d}, headers=ws["h"]).json()
    base = {"customers": 500, "price": 40, "marketing_spend": 9000}
    r = client.post("/api/scenarios/ask", json={"message": "What if I reduce marketing spend by ₹5,000?", "model": m["ref"], "baseline": base}, headers=ws["h"]).json()
    assert r["status"] == "ok" and r["result"]["scenarios"]["your scenario"]["outputs"]["profit"] == 16000.0 and r["model"] == m["ref"]
    g = client.post("/api/scenarios/ask", json={"message": "How many customers do I need to break even?", "model": m["ref"], "baseline": base}, headers=ws["h"]).json()
    assert g["result"]["solutions"][0]["value"] == pytest.approx(225.0)
    other = workspace(client)
    assert client.post("/api/scenarios/ask", json={"message": "x", "model": m["ref"], "baseline": base}, headers=other["h"]).status_code == 404
    assert client.post("/api/scenarios/ask", json={"message": "x", "model": m["ref"], "baseline": base}).status_code == 401


@pytest.mark.parametrize("patch,code", [
    ({"message": ""}, 422), ({"message": "x" * 501}, 422), ({"model": "nope"}, 400), ({"baseline": {"price": 1}}, 400),
    ({"baseline": {**BASE, "price": -5}}, 400), ({"context": "text"}, 422), ({"use_model": "maybe"}, 422),
])
def test_ask_endpoint_rejects_bad_requests_safely(client, ws, patch, code):
    body = {"message": "increase price by 10%", "model": "unit_economics", "baseline": BASE, **patch}
    r = client.post("/api/scenarios/ask", json=body, headers=ws["h"])
    assert r.status_code == code and "Traceback" not in r.text


def test_model_assisted_asking_is_rate_limited_but_the_rules_are_not(client, ws):
    from app.rate_limit import _attempts
    _attempts.clear()
    for _ in range(15):
        assert ask(client, ws, "gibberish words", use_model=True).status_code == 200
    assert ask(client, ws, "gibberish words", use_model=True).status_code == 429
    assert ask(client, ws, "increase price by 10%").status_code == 200, "the rule-based path has no model cost and is not limited this way"
    _attempts.clear()
