"""Stateful analytical copilot (controlled operations, grounding, clarification,
stale data, access control) and the what-if simulator (explicit models,
validation, sensitivity, contributions, persistence)."""
from __future__ import annotations

import math

import pytest
from platform_helpers import DRIFTED_CSV, add_member, workspace

from app import copilot, runtime, scenarios
from app.database import SessionLocal
from app.evals.scripted import ScriptedProvider
from app.models import CopilotTurn

TOTAL_REVENUE = 277110.0
WEST_REVENUE = 67590.0


@pytest.fixture
def ws(client):
    return workspace(client)


@pytest.fixture
def session(client, ws):
    r = client.post("/api/copilot/sessions", json={"dataset_id": ws["ds"]}, headers=ws["h"])
    assert r.status_code == 201, r.text
    return r.json()["id"]


def say(client, ws, sid, message, code=200):
    r = client.post(f"/api/copilot/sessions/{sid}/messages", json={"message": message}, headers=ws["h"])
    assert r.status_code == code, r.text
    return r.json()


# ============================================================== copilot ====

def test_multi_turn_analysis_keeps_and_shows_its_state(client, ws, session):
    t1 = say(client, ws, session, "total revenue")
    assert t1["result"]["type"] == "scalar" and t1["result"]["value"] == TOTAL_REVENUE
    assert t1["action"]["status"] == "applied" and t1["action"]["interpreted_by"] == "rules"

    t2 = say(client, ws, session, "by region")
    assert t2["result"]["type"] == "table" and t2["result"]["group_by"] == ["region"]
    assert {r["region"]: r["value"] for r in t2["result"]["rows_table"]}["West"] == WEST_REVENUE
    assert t2["session"]["state"]["formula"] == 'sum("revenue")', "the metric carried over from the previous turn"
    assert t2["result"]["chart"]["data"][0]["type"] == "bar"

    t3 = say(client, ws, session, "only West")
    assert t3["session"]["state"]["filters"] == [{"column": "region", "op": "==", "value": "West"}]
    assert [r["region"] for r in t3["result"]["rows_table"]] == ["West"]

    t4 = say(client, ws, session, "drill into product")
    assert t4["result"]["group_by"] == ["region", "product"] and len(t4["result"]["rows_table"]) == 3
    assert math.isclose(sum(r["value"] for r in t4["result"]["rows_table"]), WEST_REVENUE)

    t5 = say(client, ws, session, "since 2025-03-01")
    assert any(f.get("time_range") for f in t5["session"]["state"]["filters"])
    assert sum(r["value"] for r in t5["result"]["rows_table"]) < WEST_REVENUE

    t6 = say(client, ws, session, "remove the filter on region")
    assert all(f["column"] != "region" for f in t6["session"]["state"]["filters"])
    t7 = say(client, ws, session, "overall")
    assert t7["result"]["group_by"] == ["region"]
    t8 = say(client, ws, session, "reset")
    assert t8["session"]["state"]["formula"] is None and t8["result"] is None
    assert "No metric is selected" in t8["content"]

    full = client.get(f"/api/copilot/sessions/{session}", headers=ws["h"]).json()
    assert len(full["turns"]) == 16 and [t["role"] for t in full["turns"][:2]] == ["user", "assistant"]
    assert full["turns"][5]["action"]["decisions"], "analytical decisions are part of the history"


def test_every_answer_is_recomputed_and_carries_evidence(client, ws, session):
    t = say(client, ws, session, "average unit price per product")
    ev = t["evidence"]
    assert ev["formula"] == 'avg("unit_price")' and ev["group_by"] == ["product"] and ev["rows_used"] == 720
    assert ev["dataset_version"] == 1 and len(ev["dataset_fingerprint"]) == 64
    assert ev["reference_check"] == "agrees" and "not taken from the conversation" in ev["note"]
    assert ev["computed_by"] == "deterministic metric engine"


def test_it_does_not_invent_state_or_answers(client, ws, session):
    t = say(client, ws, session, "by region")
    assert t["result"] is None and "No metric is selected yet" in t["content"]
    t = say(client, ws, session, "what about the thing we discussed earlier?")
    assert t["action"]["status"] == "not_understood" and t["result"] is None and "not changed anything" in t["content"]
    assert t["session"]["state"]["formula"] is None
    t = say(client, ws, session, "why did revenue fall?")
    assert t["action"]["status"] == "not_understood" and t["action"]["suggested_question"] == "why did revenue fall?"
    assert "full analysis" in t["content"]
    t = say(client, ws, session, "total happiness")
    assert t["result"] is None, "a column that does not exist is never answered"


def test_ambiguous_references_ask_a_clarifying_question(client, ws):
    csv = b"buyer_region,seller_region,amount,qty\nWest,East,10,1\nEast,West,20,2\nWest,West,30,3\n"
    amb = workspace(client, csv, "trade.csv")
    sid = client.post("/api/copilot/sessions", json={"dataset_id": amb["ds"]}, headers=amb["h"]).json()["id"]
    say(client, amb, sid, "total amount")
    t = say(client, amb, sid, "only West")
    assert t["action"]["status"] == "clarification" and t["result"] is None
    assert {o["label"] for o in t["action"]["options"]} == {"buyer_region = West", "seller_region = West"}
    assert t["session"]["state"]["filters"] == [], "nothing changes until the user chooses"
    chosen = say(client, amb, sid, t["action"]["options"][0]["message"])
    assert chosen["result"]["value"] == 40.0


def test_approved_metric_ambiguity_is_surfaced_in_the_copilot(client, ws, session):
    for name, formula in (("Gross margin", "sum(revenue) - sum(cost)"), ("Unit margin", "(sum(revenue) - sum(cost)) / sum(units)")):
        m = client.post("/api/semantic/metrics", json={"name": name, "formula": formula, "aliases": ["margin"]}, headers=ws["h"]).json()
        client.post(f"/api/semantic/metrics/{m['id']}/versions/1/approve", headers=ws["h"])
    t = say(client, ws, session, "margin by region")
    assert t["action"]["status"] == "clarification" and len(t["action"]["options"]) == 2
    t = say(client, ws, session, "gross margin by region")
    assert t["session"]["state"]["formula"] == "sum(revenue) - sum(cost)" and t["result"]["type"] == "table"


def test_stale_dataset_is_flagged_never_silently_switched(client, ws, session):
    say(client, ws, session, "total revenue by channel")
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    t = say(client, ws, session, "only Online")
    assert t["session"]["stale"] is True and t["session"]["dataset_version"] == 1 and t["session"]["latest_version"] == 2
    assert "newer version" in t["content"] and t["evidence"]["dataset_version"] == 1, "still computed on the pinned version"
    t = say(client, ws, session, "use latest data")
    assert t["session"]["stale"] is False and t["session"]["dataset_version"] == 2
    joined = " ".join(t["action"]["decisions"])
    assert "Moved to version 2" in joined and "no longer has" in joined, "it says what it had to drop (channel is gone)"
    assert t["session"]["state"]["group_by"] == [] and t["session"]["state"]["filters"] == []
    assert t["result"]["type"] == "scalar" and t["evidence"]["dataset_version"] == 2
    assert "already on the latest" in say(client, ws, session, "use latest data")["content"]


def test_unavailable_dataset_is_reported_not_guessed(client, ws, session):
    say(client, ws, session, "total revenue")
    db = SessionLocal()
    try:
        from app.models import CopilotSession

        s = db.get(CopilotSession, session)
        s.dataset_version_id = None
        db.commit()
    finally:
        db.close()
    t = say(client, ws, session, "by region")
    assert t["action"]["status"] == "dataset_unavailable" and t["result"] is None and "no longer available" in t["content"]


def test_model_fallback_output_is_validated_like_any_input(client, ws, session):
    say(client, ws, session, "total revenue")
    valid = ScriptedProvider({"submit_operations": [{"cannot_map": False, "operations": [{"op": "drill_down", "column": "product"}]}]})
    with runtime.provider_override(valid):
        t = say(client, ws, session, "slice it along the catalogue dimension please")
    assert t["action"]["interpreted_by"] == "model" and t["result"]["group_by"] == ["product"]

    for hostile in ([{"op": "drill_down", "column": "password_hash"}],
                    [{"op": "set_metric", "formula": "__import__('os').system('id')"}],
                    [{"op": "add_filter", "column": "region", "operator": "; DROP TABLE users", "value": "x"}],
                    [{"op": "delete_everything"}]):
        with runtime.provider_override(ScriptedProvider({"submit_operations": [{"cannot_map": False, "operations": hostile}]})):
            t = say(client, ws, session, "do something unusual with the numbers now")
        assert t["action"]["status"] in ("invalid_operation", "not_understood"), hostile
        assert t["result"] is None and "not change" in t["content"].replace("did not change", "not change").replace("have not changed", "not change")
    state = client.get(f"/api/copilot/sessions/{session}", headers=ws["h"]).json()["state"]
    assert state["formula"] == 'sum("revenue")' and state["group_by"] == ["product"], "state survives every rejected attempt"

    with runtime.provider_override(ScriptedProvider({"submit_operations": [{"cannot_map": True, "operations": []}]})):
        t = say(client, ws, session, "tell me a story about dragons")
    assert t["action"]["status"] == "not_understood"


def test_sessions_are_private_to_their_owner_and_team(client, ws, session):
    say(client, ws, session, "total revenue")
    teammate = add_member(client, ws, "admin")
    other = workspace(client)
    for headers in (teammate, other["h"]):
        assert client.get(f"/api/copilot/sessions/{session}", headers=headers).status_code == 404
        assert client.post(f"/api/copilot/sessions/{session}/messages", json={"message": "by region"}, headers=headers).status_code == 404
        assert client.delete(f"/api/copilot/sessions/{session}", headers=headers).status_code == 404
        assert all(s["id"] != session for s in client.get("/api/copilot/sessions", headers=headers).json())
    assert client.post("/api/copilot/sessions", json={"dataset_id": ws["ds"]}, headers=other["h"]).status_code == 404
    assert client.get(f"/api/copilot/sessions/{session}").status_code == 401
    db = SessionLocal()
    try:
        assert db.query(CopilotTurn).filter_by(session_id=session).count() == 2, "nobody else's message was recorded"
    finally:
        db.close()
    assert client.delete(f"/api/copilot/sessions/{session}", headers=ws["h"]).status_code == 204
    assert client.get(f"/api/copilot/sessions/{session}", headers=ws["h"]).status_code == 404


def test_copilot_input_limits(client, ws, session):
    assert client.post(f"/api/copilot/sessions/{session}/messages", json={"message": ""}, headers=ws["h"]).status_code == 422
    assert client.post(f"/api/copilot/sessions/{session}/messages", json={"message": "x" * 501}, headers=ws["h"]).status_code == 422
    assert say(client, ws, session, "units > 15 total revenue")["session"]["state"]["filters"] == [{"column": "units", "op": ">", "value": "15"}]
    with pytest.raises(copilot.CopilotError):
        copilot.apply_operations(copilot.empty_state(), [{"op": "set_time_range", "column": "region", "start": "2025-01-01"}],
                                 {"columns": [{"name": "region", "kind": "categorical"}]})
    with pytest.raises(copilot.CopilotError):
        copilot.apply_operations(copilot.empty_state(), [{"op": "set_time_range", "column": "d", "start": "not-a-date"}],
                                 {"columns": [{"name": "d", "kind": "datetime"}]})


# ============================================================ scenarios ====

BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}


def test_unit_economics_arithmetic_and_labelling():
    out = scenarios.evaluate("unit_economics", BASE, {"price_up": {"price": {"type": "pct", "value": 10}},
                                                      "cheaper": {"unit_cost": {"type": "abs", "value": -2}},
                                                      "set": {"volume": {"type": "set", "value": 1500}}})
    assert out["kind"] == "scenario" and "not a validated forecast" in out["disclaimer"]
    assert out["baseline"]["outputs"] == {"revenue": 20000.0, "variable_cost": 12000.0, "gross_profit": 8000.0,
                                          "profit": 5000.0, "margin_pct": 25.0}
    assert out["scenarios"]["price_up"]["outputs"]["profit"] == 7000.0
    assert out["scenarios"]["cheaper"]["outputs"]["profit"] == 7000.0
    assert out["scenarios"]["set"]["outputs"]["profit"] == 9000.0
    d = out["scenarios"]["price_up"]["delta_vs_baseline"]["profit"]
    assert d["abs"] == 2000.0 and d["pct"] == 40.0
    assert out["formulas"] and any("elasticity is 0" in a for a in out["unsupported_assumptions"])


def test_contributions_sum_exactly_to_the_change_even_with_interactions():
    out = scenarios.evaluate("unit_economics", BASE, {"both": {"price": {"type": "pct", "value": 10},
                                                              "volume": {"type": "pct", "value": 20},
                                                              "unit_cost": {"type": "pct", "value": 5}}})
    s = out["scenarios"]["both"]
    contrib = s["contributions"]["by_input"]
    assert set(contrib) == {"price", "volume", "unit_cost"}
    assert math.isclose(sum(contrib.values()), s["delta_vs_baseline"]["profit"]["abs"], rel_tol=1e-12)
    assert contrib["price"] > 0 and contrib["volume"] > 0 and contrib["unit_cost"] < 0
    assert scenarios.shapley_contributions(lambda i: i["a"] * i["b"], {"a": 2.0, "b": 3.0}, {"a": 4.0, "b": 5.0}) == {"a": 8.0, "b": 6.0}


def test_elasticity_is_an_explicit_assumption_that_links_volume_to_price():
    flat = scenarios.evaluate("unit_economics", BASE, {"up": {"price": {"type": "pct", "value": 10}}})
    elastic = scenarios.evaluate("unit_economics", BASE, {"up": {"price": {"type": "pct", "value": 10}}}, {"price_elasticity": -1.5})
    assert flat["scenarios"]["up"]["inputs"]["volume"] == 1000.0
    assert math.isclose(elastic["scenarios"]["up"]["inputs"]["volume"], 850.0)
    assert elastic["scenarios"]["up"]["outputs"]["profit"] < flat["scenarios"]["up"]["outputs"]["profit"]
    assert any("stated assumption" in a and "not estimated" in a for a in elastic["unsupported_assumptions"])


def test_sensitivity_ranks_inputs_and_standard_scenarios_bracket_the_base():
    out = scenarios.evaluate("unit_economics", BASE, scenarios.standard_scenarios("unit_economics", 10), sensitivity_pct=10)
    rows = out["sensitivity"]["rows"]
    assert [r["input"] for r in rows][0] == "price" and rows == sorted(rows, key=lambda r: -r["range"])
    price = rows[0]
    assert (price["low"], price["high"]) == (3000.0, 7000.0)
    profit = lambda name: out["scenarios"][name]["outputs"]["profit"]  # noqa: E731
    assert profit("pessimistic") < out["baseline"]["outputs"]["profit"] < profit("optimistic")


@pytest.mark.parametrize("baseline,scens,assumptions", [
    ({**BASE, "price": -1}, {}, {}), ({**BASE, "volume": float("inf")}, {}, {}), ({"price": 1}, {}, {}),
    ({**BASE, "extra": 1}, {}, {}), (BASE, {"s": {"price": {"type": "pct", "value": -150}}}, {}),
    (BASE, {"s": {"price": {"type": "abs", "value": -500}}}, {}), (BASE, {"s": {"nope": {"type": "pct", "value": 1}}}, {}),
    (BASE, {"s": {"price": {"type": "magic", "value": 1}}}, {}), (BASE, {"s": {"price": {"type": "pct", "value": "ten"}}}, {}),
    (BASE, {}, {"price_elasticity": 99}), (BASE, {}, {"made_up": 1}), (BASE, {f"s{i}": {} for i in range(13)}, {}),
])
def test_impossible_inputs_are_rejected_with_a_reason(baseline, scens, assumptions):
    with pytest.raises(scenarios.ScenarioError) as e:
        scenarios.evaluate("unit_economics", baseline, scens, assumptions)
    assert str(e.value)


def test_funnel_model_and_rate_bounds():
    base = {"visitors": 10000.0, "conversion_rate": 0.02, "average_order_value": 50.0, "cost_per_visitor": 0.4}
    out = scenarios.evaluate("funnel", base, {"better": {"conversion_rate": {"type": "pct", "value": 50}}})
    assert out["baseline"]["outputs"]["orders"] == 200.0 and out["baseline"]["outputs"]["profit"] == 6000.0
    assert out["scenarios"]["better"]["outputs"]["orders"] == 300.0
    with pytest.raises(scenarios.ScenarioError, match="cannot be above 1"):
        scenarios.evaluate("funnel", base, {"silly": {"conversion_rate": {"type": "set", "value": 1.4}}})
    with pytest.raises(scenarios.ScenarioError, match="Unknown model"):
        scenarios.evaluate("crystal_ball", base)


def test_baseline_is_derived_from_the_dataset_with_formulas(client, ws):
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": {
        "quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"}}, headers=ws["h"])
    assert r.status_code == 200, r.text
    b = r.json()
    import csv
    import io

    from platform_helpers import SALES_CSV

    units = sum(int(row["units"]) for row in csv.DictReader(io.StringIO(SALES_CSV.decode())))  # independent reference
    assert b["inputs"]["volume"] == units and math.isclose(b["inputs"]["price"], TOTAL_REVENUE / units)
    assert b["sources"]["price"] == {"source": "dataset", "formula": 'sum("revenue") / sum("units")'}
    assert b["missing"] == ["fixed_cost"] and b["sources"]["fixed_cost"]["source"] == "assumed"
    assert b["rows_used"] == 720 and b["dataset_version"] == 1
    west = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics",
                                                        "mapping": {"quantity_column": "units"},
                                                        "filters": [{"column": "region", "op": "==", "value": "West"}]}, headers=ws["h"]).json()
    assert west["rows_used"] == 180 and west["inputs"]["price"] is None
    for bad in ({"mapping": {"quantity_column": "region"}}, {"mapping": {}}, {"mapping": {"quantity_column": "ghost"}},
                {"model": "nope", "mapping": {"quantity_column": "units"}}):
        body = {"dataset_id": ws["ds"], "model": "unit_economics", **bad}
        assert client.post("/api/scenarios/baseline", json=body, headers=ws["h"]).status_code == 400, bad


def test_scenarios_are_saved_listed_compared_and_team_scoped(client, ws):
    body = {"model": "unit_economics", "baseline": BASE, "spread_pct": 10, "scenarios": {"promo": {"price": {"type": "pct", "value": -5}}}}
    ev = client.post("/api/scenarios/evaluate", json=body, headers=ws["h"]).json()
    assert set(ev["scenarios"]) == {"optimistic", "pessimistic", "promo"} and ev["kind"] == "scenario"
    a = client.post("/api/scenarios", json={**body, "name": "Plan A", "dataset_id": ws["ds"]}, headers=ws["h"])
    assert a.status_code == 201 and a.json()["dataset_version_id"]
    b = client.post("/api/scenarios", json={**body, "name": "Plan B", "baseline": {**BASE, "volume": 2000.0}}, headers=ws["h"]).json()
    listing = client.get("/api/scenarios", headers=ws["h"]).json()
    assert listing["total"] == 2 and listing["scenarios"][0]["name"] == "Plan B" and listing["scenarios"][0]["baseline_primary"] == 13000.0
    cmp = client.get("/api/scenarios/compare", params={"ids": f"{a.json()['id']},{b['id']}"}, headers=ws["h"]).json()
    assert cmp["kind"] == "scenario" and [i["name"] for i in cmp["items"]] == ["Plan A", "Plan B"]
    assert cmp["items"][0]["scenarios"]["promo"]["profit"] == 4000.0
    saved = client.get(f"/api/scenarios/{b['id']}", headers=ws["h"]).json()
    assert saved["results"]["disclaimer"] and saved["request"]["spread_pct"] == 10

    funnel = client.post("/api/scenarios", json={"model": "funnel", "name": "F", "baseline": {
        "visitors": 1.0, "conversion_rate": 0.5, "average_order_value": 1.0, "cost_per_visitor": 0.0}}, headers=ws["h"]).json()
    assert client.get("/api/scenarios/compare", params={"ids": f"{b['id']},{funnel['id']}"}, headers=ws["h"]).status_code == 400
    assert client.get("/api/scenarios/compare", params={"ids": str(b["id"])}, headers=ws["h"]).status_code == 400
    assert client.get("/api/scenarios/compare", params={"ids": "a,b"}, headers=ws["h"]).status_code == 400
    assert client.post("/api/scenarios", json={**body, "name": "Bad", "baseline": {**BASE, "price": -5}}, headers=ws["h"]).status_code == 400

    other = workspace(client)
    assert client.get(f"/api/scenarios/{b['id']}", headers=other["h"]).status_code == 404
    assert client.delete(f"/api/scenarios/{b['id']}", headers=other["h"]).status_code == 404
    assert client.get("/api/scenarios", headers=other["h"]).json()["total"] == 0
    assert client.get("/api/scenarios/compare", params={"ids": f"{a.json()['id']},{b['id']}"}, headers=other["h"]).status_code == 404
    assert client.post("/api/scenarios", json={**body, "name": "X", "dataset_id": ws["ds"]}, headers=other["h"]).status_code == 404
    assert client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics",
                                                        "mapping": {"quantity_column": "units"}}, headers=other["h"]).status_code == 404
    assert client.get("/api/scenarios/models").status_code == 401
    assert client.delete(f"/api/scenarios/{b['id']}", headers=ws["h"]).status_code == 204
    assert {m["model"] for m in client.get("/api/scenarios/models", headers=ws["h"]).json()["models"]} == {"unit_economics", "funnel"}
