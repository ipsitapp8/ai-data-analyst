"""What-if Phase 3: plain-English explanations.

The invariants that matter, each tested across every kind of result: every number in the text is in the result, nothing
causal is claimed, signs are right, and the same result always reads the same.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from platform_helpers import workspace

from app import scenario_elasticity as el
from app import scenario_explain as ex
from app import scenario_models as sm
from app import scenario_service as svc
from app import scenarios, verification
from app.analyst import explain as grounding

BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
FUNNEL = {"visitors": 10000.0, "conversion_rate": 0.02, "average_order_value": 50.0, "cost_per_visitor": 0.4}


def all_text(e: dict) -> str:
    return " ".join([e["summary"]] + [x for v in e["sections"].values() for x in v])


def results():
    """One result of every kind the explainer handles, from the shared service."""
    yield "scenario", svc.evaluate("unit_economics", BASE, {"up": {"price": {"type": "pct", "value": 10}, "unit_cost": {"type": "pct", "value": 15}}}, {"price_elasticity": -1.0}, spread_pct=10)
    yield "scenario_bad", svc.evaluate("unit_economics", BASE, {"bad": {"price": {"type": "pct", "value": -40}}}, {"price_elasticity": -1.5})
    yield "scenario_none", svc.evaluate("unit_economics", BASE, {})
    yield "scenario_funnel", svc.evaluate("funnel", FUNNEL, {"better": {"conversion_rate": {"type": "abs", "value": 0.01}, "visitors": {"type": "pct", "value": -10}}})
    yield "simulation", svc.simulate("unit_economics", BASE, {"price_elasticity": -1.2}, adjustments={"price": {"type": "pct", "value": 5}},
                                     uncertainty={"price": {"type": "normal", "sd_pct": 15}, "volume": {"type": "uniform", "low_pct": -20, "high_pct": 10}},
                                     iterations=4000, seed=3, target_value=4000)
    yield "goal_solved", svc.goal_seek("unit_economics", BASE, "margin_pct", 30, "price")
    yield "goal_multiple", svc.goal_seek("unit_economics", BASE, "profit", 1000, "price", assumptions={"price_elasticity": -2.0})
    yield "goal_constrained", svc.goal_seek("unit_economics", BASE, "profit", 1000, "price", assumptions={"price_elasticity": -2.0},
                                            constraints=[{"metric": "volume", "op": ">=", "value": 900}])
    yield "goal_impossible", svc.goal_seek("unit_economics", BASE, "margin_pct", 150, "price")
    yield "goal_no_dependence", svc.goal_seek("unit_economics", BASE, "revenue", 5000, "unit_cost")
    yield "grid", svc.grid("unit_economics", BASE, "price", "volume", resolution=11, x_range={"mode": "pct", "low": -50, "high": 50}, y_range={"mode": "pct", "low": -50, "high": 50})
    yield "grid_no_crossing", svc.grid("unit_economics", BASE, "price", "volume", resolution=7)


@pytest.fixture(scope="module")
def made():
    return dict(results())


# ======================================================= invariants, every kind ==

@pytest.mark.parametrize("name", ["scenario", "scenario_bad", "scenario_none", "scenario_funnel", "simulation", "goal_solved", "goal_multiple",
                                  "goal_constrained", "goal_impossible", "goal_no_dependence", "grid", "grid_no_crossing"])
def test_every_number_in_an_explanation_is_in_the_result_it_explains(made, name):
    e = made[name]["explanation"]
    assert e["source"] == "template" and e["summary"] and e["sections"]["cannot_conclude"]
    bad = grounding.ungrounded(all_text(e), made[name])
    assert bad == [], f"{name}: numbers not in the result: {bad}\n{all_text(e)}"


@pytest.mark.parametrize("name", ["scenario", "scenario_bad", "simulation", "goal_solved", "goal_multiple", "goal_impossible", "grid"])
def test_no_explanation_claims_a_cause(made, name):
    text = all_text(made[name]["explanation"])
    assert not ex.CAUSAL.search(text) and not verification._CAUSAL.search(text), text  # noqa: SLF001
    assert any("not a forecast" in x or "does not say" in x or "not necessarily" in x for x in made[name]["explanation"]["sections"]["cannot_conclude"])


def test_signs_are_kept_for_levels_and_directions_are_in_words(made):
    bad = all_text(made["scenario_bad"]["explanation"])
    assert "falls from 5,000 to -3,000" in bad and "turns profit negative (-3,000)" in bad and "Price elasticity is -1.5" in bad
    assert "160.0% less" in bad and "down 40%" in bad
    good = all_text(made["scenario"]["explanation"])
    assert "-" not in good.replace("break-even", "").replace("Break-even", "").replace("5th-", "") or "Price elasticity is -1" in good
    assert "takes away" in good and "adds" in good and "work against each other" in good


def test_a_scenario_explanation_names_what_changed_the_drivers_the_risks_and_the_assumptions(made):
    s = made["scenario"]["explanation"]["sections"]
    assert any("In 'up', profit falls from 5,000 to 4,380" in x for x in s["what_changed"])
    assert any("Price per unit up 10%" in x and "Variable cost per unit up 15%" in x for x in s["what_changed"])
    assert any("between 1,020 and 9,620" in x for x in s["what_changed"]), "the optimistic and pessimistic range is stated"
    assert any("largest share" in x and "Price per unit" in x and "adds 1,900" in x for x in s["why"])
    assert any("most sensitive to Price per unit" in x for x in s["risks"]) and any("Break-even is at 375" in x for x in s["risks"])
    assert any("Price elasticity is -1 (your value)" in x for x in s["assumptions"])


def test_the_drivers_in_the_text_are_the_engines_own_attribution(made):
    res = made["scenario"]
    contrib = res["scenarios"]["up"]["contributions"]["by_input"]
    top = max(contrib, key=lambda k: abs(contrib[k]))
    assert res["input_labels"][top] in made["scenario"]["explanation"]["sections"]["why"][0]
    assert sum(contrib.values()) == pytest.approx(res["scenarios"]["up"]["delta_vs_baseline"]["profit"]["abs"])


def test_no_adjustment_yet_says_so_and_a_simulation_states_its_assumptions(made):
    assert "No adjustment has been made yet" in made["scenario_none"]["explanation"]["summary"]
    sim = made["simulation"]["explanation"]
    assert "4,000 simulated runs" in sim["summary"] and any("assumptions you supplied" in x for x in sim["sections"]["assumptions"])
    assert any("target of 4,000" in x for x in sim["sections"]["what_changed"]) and any("probabilities are only as good as those ranges" in x for x in sim["sections"]["cannot_conclude"])


def test_goal_seek_explanations_cover_every_status(made):
    assert "reaches 30 when price per unit is" in made["goal_solved"]["explanation"]["summary"]
    assert "Several solutions" not in made["goal_multiple"]["explanation"]["summary"]
    assert any("More than one value reaches the target" in x for x in made["goal_multiple"]["explanation"]["sections"]["risks"])
    assert any("Constraint units sold >= 900" in x and "met" in x for x in made["goal_constrained"]["explanation"]["sections"]["risks"])
    assert "No value of price per unit" in made["goal_impossible"]["explanation"]["summary"] and any("only reaches" in x for x in made["goal_impossible"]["explanation"]["sections"]["why"])
    assert "does not change with" in made["goal_no_dependence"]["explanation"]["summary"]


def test_grid_explanations_say_where_it_crosses_or_that_it_never_does(made):
    crossing, never = made["grid"]["explanation"], made["grid_no_crossing"]["explanation"]
    assert any("crosses 0" in x for x in crossing["sections"]["risks"]) and any("never crosses 0" in x for x in never["sections"]["risks"])
    assert "runs from" in crossing["summary"] and any("not necessarily the best possible" in x for x in crossing["sections"]["cannot_conclude"])


def test_the_same_result_always_reads_the_same(made):
    assert ex.explain(made["scenario"]) == ex.explain(made["scenario"]) == made["scenario"]["explanation"]


def test_formatting_keeps_sign_for_levels_and_drops_it_for_sizes_of_change():
    assert ex.n(-3000) == "-3,000" and ex.n(5000) == "5,000" and ex.n(0.0526) == "0.05" and ex.n(-0.001) == "0" and ex.n(None) == "n/a"
    assert ex.a(-3000) == "3,000" and ex.a(None) == "n/a" and ex.pc(-12.44) == "12.4%" and ex.pc(None) == "n/a"


# ===================================================================== other kinds ==

def monthly(n=36, e=-1.5, seed=0):
    rng = np.random.default_rng(seed)
    price = 20 * np.exp(rng.normal(0, 0.12, n))
    return pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n, freq="MS"), "price": price, "units": 1000 * (price / 20) ** e * np.exp(rng.normal(0, 0.05, n))})


def test_elasticity_explanations_are_grounded_and_honest_when_nothing_was_estimated():
    good = svc.explained(el.estimate(monthly(), "price", "units", date_column="date"))
    e = good["explanation"]
    assert "price elasticity of -1." in e["summary"] and "36 observations" in e["summary"] and "interval is -1." in e["summary"]
    assert grounding.ungrounded(all_text(e), good) == [] and not ex.CAUSAL.search(all_text(e))
    assert any("observational" in x.lower() for x in e["sections"]["cannot_conclude"])
    none = svc.explained(el.estimate(monthly(5), "price", "units", date_column="date"))
    assert none["explanation"]["summary"].startswith("No estimate was made") and "Do not use" in all_text(none["explanation"])


def csv_of(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def test_period_and_segment_explanations_come_with_their_endpoints(client):
    ws = workspace(client)
    mapping = {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"}
    cp = client.post("/api/scenarios/compare-periods", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": mapping, "overrides": {"fixed_cost": 500},
                                                             "date_column": "order_date", "period_a": {"type": "month", "year": 2025, "month": 1},
                                                             "period_b": {"type": "month", "year": 2025, "month": 3}}, headers=ws["h"]).json()
    e = cp["explanation"]
    assert " in January 2025 to " in e["summary"] and " in March 2025" in e["summary"] and "describes what happened" in all_text(e)
    assert grounding.ungrounded(all_text(e), cp) == [] and not ex.CAUSAL.search(all_text(e))
    sg = client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": mapping, "segment_column": "region",
                                                      "constant_inputs": {"fixed_cost": 0}, "segment_adjustments": {"West": {"price": {"type": "pct", "value": 10}}}}, headers=ws["h"]).json()
    es = sg["explanation"]
    assert "Total profit rises" in es["summary"] and "West accounts for 100.0% of the change" in all_text(es)
    assert grounding.ungrounded(all_text(es), sg) == [] and not ex.CAUSAL.search(all_text(es))


def test_every_endpoint_returns_an_explanation_and_a_failure_to_explain_never_hides_a_result(client, monkeypatch):
    ws = workspace(client)
    h = ws["h"]
    body = {"model": "unit_economics", "baseline": BASE}
    ev = client.post("/api/scenarios/evaluate", json={**body, "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}}, headers=h).json()
    sim = client.post("/api/scenarios/simulate", json={**body, "iterations": 2000, "seed": 1, "uncertainty": {"price": {"type": "normal", "sd_pct": 5}}}, headers=h).json()
    gs = client.post("/api/scenarios/goal-seek", json={**body, "target_output": "profit", "target_value": 9000, "solve_for": "volume"}, headers=h).json()
    gr = client.post("/api/scenarios/sensitivity-grid", json={**body, "x_input": "price", "y_input": "volume", "resolution": 5}, headers=h).json()
    for r in (ev, sim, gs, gr):
        assert r["explanation"]["source"] == "template" and r["explanation"]["summary"], r["kind"]
    assert "profit rises from 5,000 to 7,000" in ev["explanation"]["summary"]

    def boom(_):
        raise RuntimeError("template bug")
    monkeypatch.setattr(ex, "explain", boom)
    again = client.post("/api/scenarios/evaluate", json={**body, "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}}, headers=h)
    assert again.status_code == 200 and again.json()["explanation"]["source"] == "unavailable"
    assert again.json()["scenarios"]["up"]["outputs"]["profit"] == 7000.0 and again.json()["verification"]["status"] == "verified"


def test_a_custom_model_is_explained_with_its_own_labels():
    spec = sm.compile_spec({"inputs": {"customers": {"label": "Active customers", "unit": "customers", "min": 0}, "price": {"label": "Monthly price", "unit": "currency", "min": 0}},
                            "outputs": {"mrr": {"label": "Monthly revenue", "unit": "currency", "formula": "customers * price"}}, "primary_output": "mrr"}, key="custom:1@1", title="MRR")
    r = svc.evaluate(spec, {"customers": 100.0, "price": 50.0}, {"grow": {"customers": {"type": "pct", "value": 20}}})
    t = all_text(r["explanation"])
    assert "monthly revenue rises from 5,000 to 6,000" in t and "Active customers up 20%" in t and "custom model" in t
    assert grounding.ungrounded(t, r) == []


# ============================================================ the optional rewrite ==

def test_a_model_rewrite_is_kept_only_if_it_adds_no_number_and_no_cause(made):
    res, e = made["scenario"], made["scenario"]["explanation"]
    ok = ex.polish(e, res, lambda text: "Profit goes from 5,000 down to 4,380 in the 'up' scenario, mostly because of costs.".replace("because of", "alongside"))
    assert ok["polish"]["used"] is True and ok["source"] == "model_rewrite" and "4,380" in ok["summary"]
    invented = ex.polish(e, res, lambda text: "Profit falls to 4,380 and would reach 99,999 next year.")
    assert invented["polish"]["used"] is False and "99,999" in invented["polish"]["reason"] and invented["summary"] == e["summary"] and invented["source"] == "template"
    causal = ex.polish(e, res, lambda text: "Profit falls to 4,380 because costs rose.")
    assert causal["polish"]["used"] is False and "causal" in causal["polish"]["reason"] and causal["summary"] == e["summary"]
    empty = ex.polish(e, res, lambda text: "   ")
    assert empty["polish"]["used"] is False and "nothing" in empty["polish"]["reason"]


def test_without_a_model_or_when_it_fails_the_template_text_stands(made):
    res, e = made["scenario"], made["scenario"]["explanation"]
    none = ex.polish(e, res, None)
    assert none["polish"]["used"] is False and "no language model" in none["polish"]["reason"] and none["summary"] == e["summary"]

    def down(_):
        raise ConnectionError("provider down")
    failed = ex.polish(e, res, down)
    assert failed["polish"]["used"] is False and "could not be reached" in failed["polish"]["reason"] and failed["summary"] == e["summary"]
    assert ex.explain({"kind": "something_new"})["sections"] == {}, "an unknown result kind is not an error"
