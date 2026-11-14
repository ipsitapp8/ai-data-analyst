"""What-if Phase 3: calculation traces ("How was this calculated?") and sealed evidence records.

Traces are checked against hand-worked numbers. The seal is recomputed here with plain `json` and `hashlib`, not with the
code under test. Reproduction is shown to catch a changed engine and an altered row.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from platform_helpers import workspace

from app import scenario_evidence as ev
from app import scenario_models as sm
from app import scenarios
from app.database import SessionLocal
from app.models import ScenarioEvidence

BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
FUNNEL = {"visitors": 10000.0, "conversion_rate": 0.02, "average_order_value": 50.0, "cost_per_visitor": 0.4}
SAAS = {
    "inputs": {"customers": {"label": "Active customers", "unit": "customers", "min": 0}, "price": {"label": "Monthly price", "unit": "currency/customers", "min": 0},
               "unit_cost": {"label": "Cost per customer", "unit": "currency/customers", "min": 0}},
    "assumptions": {"churn_rate": {"label": "Monthly churn", "unit": "fraction", "default": 0.05, "min": 0, "max": 1}},
    "outputs": {"mrr": {"label": "MRR", "unit": "currency", "formula": "customers * price"},
                "churned": {"label": "Churned revenue", "unit": "currency", "formula": "customers * churn_rate * price"},
                "net": {"label": "Net revenue", "unit": "currency", "formula": "mrr - churned"},
                "costs": {"label": "Costs", "unit": "currency", "formula": "customers * unit_cost"},
                "profit": {"label": "Profit", "unit": "currency", "formula": "net - costs"}},
    "primary_output": "profit", "constraints": [{"expr": "price >= unit_cost", "message": "Price must cover cost"}],
}
SAAS_BASE = {"customers": 1000.0, "price": 50.0, "unit_cost": 20.0}


def tr(model, base, inputs, assumptions, output, **kw):
    return ev.trace(model, base, inputs, assumptions, output, **kw)


# ============================================================================ traces ==

def test_a_profit_trace_shows_the_formula_the_steps_and_the_values_substituted():
    t = tr("unit_economics", BASE, BASE, {"price_elasticity": 0.0}, "profit")
    assert t["formula"] == "gross_profit - fixed_cost" and t["value"] == 5000.0 and t["matches_engine"] is True
    assert [(s["name"], s["value"]) for s in t["steps"]] == [("revenue", 20000.0), ("variable_cost", 12000.0), ("gross_profit", 8000.0), ("profit", 5000.0)]
    assert t["steps"][-1]["substituted"] == "(8,000) - (3,000)" and t["steps"][0]["substituted"] == "(20) * (1,000)"
    assert t["steps"][-1]["uses"] == ["fixed_cost", "gross_profit"]
    assert [r["name"] for r in t["inputs"]] == ["price", "volume", "unit_cost", "fixed_cost"] and t["inputs"][0]["label"] == "Price per unit" and t["inputs"][0]["unit"] == "currency/unit"
    assert t["model"] == "unit_economics" and t["engine_version"] == scenarios.ENGINE_VERSION and t["scenario"] == "baseline"


def test_a_trace_includes_only_what_the_output_depends_on():
    t = tr("unit_economics", BASE, BASE, {"price_elasticity": 0.0}, "revenue")
    assert [s["name"] for s in t["steps"]] == ["revenue"] and [r["name"] for r in t["inputs"]] == ["price", "volume"] and t["value"] == 20000.0
    m = tr("unit_economics", BASE, BASE, {"price_elasticity": 0.0}, "margin_pct")
    assert [s["name"] for s in m["steps"]] == ["revenue", "variable_cost", "gross_profit", "profit", "margin_pct"] and m["value"] == pytest.approx(25.0)


def test_a_scenario_trace_shows_each_adjustment_and_the_elasticity_link_that_moved_volume():
    out = scenarios.evaluate("unit_economics", BASE, {"up": {"price": {"type": "pct", "value": 10}}}, {"price_elasticity": -1.5})
    s = out["scenarios"]["up"]
    t = tr("unit_economics", BASE, s["inputs"], out["assumptions"], "profit", scenario="up", adjustments=s["adjustments"])
    rows = {r["name"]: r for r in t["inputs"]}
    assert rows["price"]["baseline"] == 20.0 and rows["price"]["value"] == pytest.approx(22.0) and rows["price"]["adjustment"] == {"type": "pct", "value": 10}
    assert rows["volume"]["value"] == pytest.approx(850.0) and "price elasticity -1.5" in rows["volume"]["derived_from"] and rows["volume"]["adjustment"] is None
    assert t["value"] == pytest.approx(s["outputs"]["profit"]) and t["matches_engine"] and t["scenario"] == "up"
    assert t["assumptions"][0]["name"] == "price_elasticity" and t["assumptions"][0]["applied_by_engine"] is True and t["assumptions"][0]["value"] == -1.5
    flat = scenarios.evaluate("unit_economics", BASE, {"up": {"price": {"type": "pct", "value": 10}}})
    t2 = tr("unit_economics", BASE, flat["scenarios"]["up"]["inputs"], flat["assumptions"], "profit", adjustments=flat["scenarios"]["up"]["adjustments"])
    assert t2["assumptions"] == [] and not any("derived_from" in r for r in t2["inputs"]), "with no elasticity nothing links volume to price"


@pytest.mark.parametrize("model,base,assumptions", [("unit_economics", BASE, {"price_elasticity": -0.8}), ("funnel", FUNNEL, {})])
def test_every_output_of_a_built_in_model_traces_to_the_engines_own_number(model, base, assumptions):
    spec = scenarios._model(model)
    out = scenarios.evaluate(model, base, scenarios.standard_scenarios(model, 12), assumptions)
    for name, s in [("baseline", {"inputs": base, "adjustments": {}})] + list(out["scenarios"].items()):
        for output in spec["outputs"]:
            t = tr(model, base, s["inputs"], out["assumptions"], output, scenario=name, adjustments=s["adjustments"])
            engine = (out["baseline"]["outputs"] if name == "baseline" else out["scenarios"][name]["outputs"])[output]
            assert t["matches_engine"] is True, (name, output)
            assert (t["value"] is None) == (engine is None) and (engine is None or math.isclose(t["value"], engine, rel_tol=1e-12))


def test_a_custom_models_trace_uses_its_own_formulas_units_and_constraints():
    spec = sm.compile_spec(SAAS, key="custom:4@2", title="SaaS")
    t = tr(spec, SAAS_BASE, SAAS_BASE, {"churn_rate": 0.05}, "profit")
    # 1000 x 50 = 50000; churned 1000 x 0.05 x 50 = 2500; net 47500; costs 20000; profit 27500
    assert [(s["name"], s["value"]) for s in t["steps"]] == [("mrr", 50000.0), ("churned", 2500.0), ("net", 47500.0), ("costs", 20000.0), ("profit", 27500.0)]
    assert t["steps"][1]["substituted"] == "(1,000) * (0.05) * (50)" and t["steps"][1]["unit"] == "currency" and t["matches_engine"]
    assert [a["name"] for a in t["assumptions"]] == ["churn_rate"] and t["assumptions"][0]["label"] == "Monthly churn" and t["assumptions"][0]["used_in_formulas"] is True
    assert t["constraints"] == [{"rule": "price >= unit_cost", "message": "Price must cover cost", "severity": "error", "satisfied": True}]
    assert t["model"] == "custom:4@2" and t["model_sha256"] == sm.definition_sha256(SAAS) and [r["label"] for r in t["inputs"]][0] == "Active customers"
    cheap = tr(spec, SAAS_BASE, {**SAAS_BASE, "price": 10.0}, {"churn_rate": 0.05}, "profit")
    assert cheap["constraints"][0]["satisfied"] is False, "a rule the figures break is shown as broken"


def test_a_trace_names_where_an_input_came_from_and_undefined_values_are_shown_as_undefined():
    sources = {"price": {"source": "dataset", "formula": 'sum("revenue") / sum("units")'}}
    t = tr("unit_economics", BASE, BASE, {"price_elasticity": 0.0}, "revenue", input_sources=sources)
    assert t["inputs"][0]["source"] == sources["price"]
    zero = {**BASE, "volume": 0.0}
    m = tr("unit_economics", zero, zero, {"price_elasticity": 0.0}, "margin_pct")
    assert m["value"] is None and m["steps"][-1]["value"] is None and m["matches_engine"] is True and "undefined" in m["steps"][-1]["substituted"] or m["steps"][-1]["value"] is None


def test_an_unknown_output_is_refused():
    with pytest.raises(scenarios.ScenarioError, match="Unknown output"):
        tr("unit_economics", BASE, BASE, {"price_elasticity": 0.0}, "magic")


# ============================================================================ sealing ==

def test_clean_json_makes_results_storable_and_hashable():
    messy = {"a": np.float64(1.5), "b": [np.int64(3), float("nan"), float("inf")], "c": {"d": float("-inf"), "e": (1, 2)}}
    assert ev.clean_json(messy) == {"a": 1.5, "b": [3, None, None], "c": {"d": None, "e": [1, 2]}}
    json.dumps(ev.clean_json(messy), allow_nan=False)


def test_the_seal_depends_on_the_request_and_the_result_and_not_on_key_order():
    a = ev.seal({"x": 1, "y": 2}, {"p": 1.0})
    assert a == ev.seal({"y": 2, "x": 1}, {"p": 1.0}) and a != ev.seal({"x": 1, "y": 3}, {"p": 1.0}) and a != ev.seal({"x": 1, "y": 2}, {"p": 1.0000001})


# ==================================================================== the records ==

@pytest.fixture
def ws(client):
    return workspace(client)


def evaluate_request(**kw):
    return {"baseline": BASE, "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}, "assumptions": {"price_elasticity": -1.0}, **kw}


def seal(client, ws, kind="scenario", model="unit_economics", request=None, **kw):
    r = client.post("/api/scenario-evidence", json={"kind": kind, "model": model, "request": request or evaluate_request(), **kw}, headers=ws["h"])
    assert r.status_code == 201, r.text
    return r.json()


def test_a_sealed_record_has_everything_needed_to_audit_the_calculation(client, ws):
    rec = seal(client, ws, note="Q4 pricing review")
    assert rec["kind"] == "scenario" and rec["model"] == "unit_economics" and rec["verification_status"] == "verified" and rec["note"] == "Q4 pricing review"
    assert rec["engine_versions"]["scenarios"] == scenarios.ENGINE_VERSION and rec["engine_versions"]["service"] and len(rec["content_sha256"]) == 64
    full = client.get(f"/api/scenario-evidence/{rec['id']}", headers=ws["h"]).json()
    assert full["request"]["baseline"] == BASE and full["request"]["model"] == "unit_economics"
    # volume x 0.9 under elasticity -1: price 22 -> volume 900; profit = (22 - 12) x 900 - 3000 = 6000
    assert full["result"]["scenarios"]["up"]["outputs"]["profit"] == pytest.approx(6000.0) and full["verification"]["status"] == "verified"
    assert "verification" not in full["result"] and "explanation" not in full["result"], "derived parts are not sealed into the result"


def test_the_seal_can_be_recomputed_independently_from_the_stored_record(client, ws):
    rec = seal(client, ws)
    full = client.get(f"/api/scenario-evidence/{rec['id']}", headers=ws["h"]).json()
    body = json.dumps({"request": full["request"], "result": full["result"]}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(body.encode()).hexdigest() == full["content_sha256"]


def test_the_server_always_recomputes_a_result_and_ignores_one_a_client_supplies(client, ws):
    forged = evaluate_request(result={"scenarios": {"up": {"outputs": {"profit": 1e9}}}}, verification={"status": "verified"})
    rec = seal(client, ws, request=forged)
    full = client.get(f"/api/scenario-evidence/{rec['id']}", headers=ws["h"]).json()
    assert full["result"]["scenarios"]["up"]["outputs"]["profit"] == pytest.approx(6000.0) and full["verification"]["status"] == "verified"


def test_every_kind_can_be_sealed_and_reproduced(client, ws):
    requests = {
        "scenario": evaluate_request(),
        "simulation": {"baseline": BASE, "iterations": 3000, "seed": 5, "uncertainty": {"price": {"type": "normal", "sd_pct": 8}}, "target_value": 4000},
        "goal_seek": {"baseline": BASE, "target_output": "profit", "target_value": 9000, "solve_for": "volume"},
        "sensitivity_grid": {"baseline": BASE, "x_input": "price", "y_input": "volume", "resolution": 7},
    }
    for kind, request in requests.items():
        rec = seal(client, ws, kind=kind, request=request)
        assert rec["verification_status"] == "verified", kind
        r = client.post(f"/api/scenario-evidence/{rec['id']}/reproduce", headers=ws["h"]).json()
        assert r["reproduced"] is True and r["record_intact"] is True and r["differences"] == [] and "same result today" in r["explanation"], (kind, r)


def test_a_simulation_record_keeps_the_seed_the_server_used_even_when_none_was_given(client, ws):
    rec = seal(client, ws, kind="simulation", request={"baseline": BASE, "iterations": 2000, "uncertainty": {"price": {"type": "normal", "sd_pct": 5}}})
    assert isinstance(rec["seed"], int)
    full = client.get(f"/api/scenario-evidence/{rec['id']}", headers=ws["h"]).json()
    assert full["request"]["seed"] == rec["seed"] == full["result"]["config"]["seed"]
    assert client.post(f"/api/scenario-evidence/{rec['id']}/reproduce", headers=ws["h"]).json()["reproduced"] is True


def test_reproduction_notices_a_changed_engine(client, ws, monkeypatch):
    rec = seal(client, ws)
    monkeypatch.setattr(scenarios, "ENGINE_VERSION", "scenarios.99")
    r = client.post(f"/api/scenario-evidence/{rec['id']}/reproduce", headers=ws["h"]).json()
    assert r["reproduced"] is False and r["record_intact"] is True and "engine_version" in r["differences"]
    assert "engine or the model changed" in r["explanation"]


def test_reproduction_notices_an_altered_row(client, ws):
    rec = seal(client, ws)
    with SessionLocal() as db:
        row = db.get(ScenarioEvidence, rec["id"])
        row.result_json = {**row.result_json, "primary_output": "tampered"}
        db.commit()
    r = client.post(f"/api/scenario-evidence/{rec['id']}/reproduce", headers=ws["h"]).json()
    assert r["reproduced"] is False and r["record_intact"] is False and "altered" in r["explanation"] and "primary_output" in r["differences"]


def test_a_custom_model_record_is_pinned_to_its_version_and_survives_edits(client, ws):
    m = client.post("/api/scenario-models", json={"name": "Pinned", "definition": SAAS}, headers=ws["h"]).json()
    rec = seal(client, ws, model=m["ref"], request={"baseline": SAAS_BASE, "assumptions": {"churn_rate": 0.05}, "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}})
    assert rec["model"] == m["ref"] and rec["model_sha256"] == sm.definition_sha256(SAAS)
    edited = json.loads(json.dumps(SAAS))
    edited["outputs"]["costs"]["formula"] = "customers * unit_cost + 1000"
    assert client.post(f"/api/scenario-models/{m['id']}/versions", json={"definition": edited}, headers=ws["h"]).status_code == 201
    r = client.post(f"/api/scenario-evidence/{rec['id']}/reproduce", headers=ws["h"]).json()
    assert r["reproduced"] is True, "the record points at version 1, which an edit never changes"


def test_records_are_append_only(client, ws):
    rec = seal(client, ws)
    for method in (client.put, client.patch, client.delete):
        assert method(f"/api/scenario-evidence/{rec['id']}", headers=ws["h"]).status_code == 405
    assert client.get(f"/api/scenario-evidence/{rec['id']}", headers=ws["h"]).status_code == 200


@pytest.mark.parametrize("body,code", [
    ({"kind": "teleport", "model": "unit_economics", "request": evaluate_request()}, 400),
    ({"kind": "scenario", "model": "unit_economics", "request": {"baseline": {**BASE, "price": -1}}}, 400),
    ({"kind": "scenario", "model": "nope", "request": evaluate_request()}, 400),
    ({"kind": "scenario", "model": "custom:99999", "request": evaluate_request()}, 404),
    ({"kind": "scenario", "model": "unit_economics", "request": {"baseline": BASE, "padding": "x" * 120_000}}, 413),
    ({"kind": "scenario", "model": "unit_economics", "request": "text"}, 422), ({"kind": "x" * 40, "model": "unit_economics", "request": {}}, 422),
    ({"kind": "scenario", "model": "unit_economics", "request": {"baseline": BASE}, "dataset_id": 98765}, 404),
])
def test_bad_evidence_requests_are_refused_safely_and_leave_no_record(client, ws, body, code):
    r = client.post("/api/scenario-evidence", json=body, headers=ws["h"])
    assert r.status_code == code and "Traceback" not in r.text
    assert client.get("/api/scenario-evidence", headers=ws["h"]).json()["total"] == 0


def test_records_can_be_listed_filtered_and_paged(client, ws):
    for _ in range(3):
        seal(client, ws)
    seal(client, ws, kind="goal_seek", request={"baseline": BASE, "target_output": "profit", "target_value": 9000, "solve_for": "volume"})
    all_ = client.get("/api/scenario-evidence", headers=ws["h"]).json()
    assert all_["total"] == 4 and all_["records"][0]["id"] > all_["records"][-1]["id"] and "request" not in all_["records"][0], "lists are summaries"
    assert client.get("/api/scenario-evidence", params={"kind": "goal_seek"}, headers=ws["h"]).json()["total"] == 1
    page = client.get("/api/scenario-evidence", params={"limit": 2, "offset": 1}, headers=ws["h"]).json()
    assert len(page["records"]) == 2 and page["total"] == 4
    assert client.get("/api/scenario-evidence", params={"limit": 500}, headers=ws["h"]).status_code == 422


# ===================================================== saved scenarios and datasets ==

def test_saving_a_scenario_seals_its_calculation_and_its_simulation_with_the_data_it_used(client, ws):
    body = {"model": "unit_economics", "baseline": BASE, "name": "Plan A", "dataset_id": ws["ds"], "assumptions": {"price_elasticity": -1.0},
            "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}, "baseline_source": {"dataset_version": 1, "date_column": None},
            "simulation": {"uncertainty": {"price": {"type": "normal", "sd_pct": 6}}, "iterations": 2500, "seed": 77, "adjustments": {"price": {"type": "pct", "value": 10}}}}
    saved = client.post("/api/scenarios", json=body, headers=ws["h"])
    assert saved.status_code == 201, saved.text
    sid = saved.json()["id"]
    records = client.get("/api/scenario-evidence", params={"scenario_id": sid}, headers=ws["h"]).json()["records"]
    assert sorted(r["kind"] for r in records) == ["scenario", "simulation"] and all(r["scenario_id"] == sid and r["verification_status"] == "verified" for r in records)
    prov = saved.json()["request"]["baseline_source"]["provenance"]
    assert all(r["dataset"]["content_sha256"] == prov["content_sha256"] and r["dataset"]["dataset_version_id"] == prov["dataset_version_id"] for r in records)
    sim = next(r for r in records if r["kind"] == "simulation")
    assert sim["seed"] == 77
    for r in records:
        assert client.post(f"/api/scenario-evidence/{r['id']}/reproduce", headers=ws["h"]).json()["reproduced"] is True
    plain = client.post("/api/scenarios", json={"model": "unit_economics", "baseline": BASE, "name": "Plan B"}, headers=ws["h"]).json()
    assert [r["kind"] for r in client.get("/api/scenario-evidence", params={"scenario_id": plain["id"]}, headers=ws["h"]).json()["records"]] == ["scenario"]


def test_a_record_can_name_the_dataset_version_it_was_built_from(client, ws):
    rec = seal(client, ws, dataset_id=ws["ds"], dataset_version=1)
    assert rec["dataset"]["dataset_id"] == ws["ds"] and len(rec["dataset"]["content_sha256"]) == 64 and rec["dataset"]["dataset_version_id"]
    assert client.post("/api/scenario-evidence", json={"kind": "scenario", "model": "unit_economics", "request": evaluate_request(), "dataset_id": ws["ds"], "dataset_version": 5}, headers=ws["h"]).status_code == 404


# ==================================================================== the inspector ==

def inspect(client, ws, **kw):
    body = {"model": "unit_economics", "baseline": BASE, "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}, "assumptions": {"price_elasticity": -1.0}, **kw}
    return client.post("/api/scenarios/inspect", json=body, headers=ws["h"])


def test_inspect_explains_one_number_of_the_baseline_or_a_scenario(client, ws):
    base = inspect(client, ws, output="profit").json()
    assert base["trace"]["value"] == 5000.0 and base["trace"]["scenario"] == "baseline" and base["verification"]["status"] == "verified" and base["explanation"]["summary"]
    up = inspect(client, ws, scenario="up", output="revenue").json()
    assert up["trace"]["value"] == pytest.approx(22 * 900) and {r["name"] for r in up["trace"]["inputs"]} == {"price", "volume"}
    assert next(r for r in up["trace"]["inputs"] if r["name"] == "volume")["derived_from"]
    assert inspect(client, ws).json()["trace"]["output"] == "profit", "the headline output by default"


@pytest.mark.parametrize("patch,code", [({"scenario": "ghost"}, 400), ({"output": "ghost"}, 400), ({"model": "nope"}, 400), ({"baseline": {"price": 1}}, 400), ({"output": "x" * 100}, 422)])
def test_inspect_refuses_bad_requests(client, ws, patch, code):
    assert inspect(client, ws, **patch).status_code == code


def test_inspect_can_seal_what_it_shows(client, ws):
    r = inspect(client, ws, scenario="up", persist=True, dataset_id=ws["ds"], note="checked with finance").json()
    assert r["evidence"]["kind"] == "scenario" and r["evidence"]["note"] == "checked with finance" and r["evidence"]["dataset"]["dataset_id"] == ws["ds"]
    assert client.get("/api/scenario-evidence", headers=ws["h"]).json()["total"] == 1
    assert "evidence" not in inspect(client, ws).json()


# ===================================================================== isolation ==

def test_evidence_is_team_scoped_and_needs_a_signed_in_member(client, ws):
    rec = seal(client, ws)
    other = workspace(client)
    h = other["h"]
    for call in (client.get(f"/api/scenario-evidence/{rec['id']}", headers=h), client.post(f"/api/scenario-evidence/{rec['id']}/reproduce", headers=h),
                 client.get("/api/scenario-evidence", params={"scenario_id": 1}, headers=h)):
        assert call.status_code == 404
    assert client.get("/api/scenario-evidence", headers=h).json()["total"] == 0
    assert client.get("/api/scenario-evidence/987654", headers=h).json() == client.get(f"/api/scenario-evidence/{rec['id']}", headers=h).json()
    assert client.get("/api/scenario-evidence").status_code == 401 and client.post("/api/scenario-evidence", json={}).status_code == 401
    assert client.post("/api/scenarios/inspect", json={"model": "unit_economics", "baseline": BASE}).status_code == 401
    m = client.post("/api/scenario-models", json={"name": "Private", "definition": SAAS}, headers=ws["h"]).json()
    assert client.post("/api/scenario-evidence", json={"kind": "scenario", "model": m["ref"], "request": {"baseline": SAAS_BASE}}, headers=h).status_code == 404
    seal(client, other)
    assert client.get("/api/scenario-evidence", headers=ws["h"]).json()["total"] == 1, "each team sees only its own"


# ================================================================== the migration ==

def test_the_evidence_migration_upgrades_downgrades_and_matches_the_orm(tmp_path):
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect as sa_inspect

    from app import models

    path = Path(models.__file__).resolve().parents[1] / "alembic" / "versions" / "d5f7b9c1e3a2_scenario_evidence.py"
    spec = importlib.util.spec_from_file_location("rev_d5f7b9c1e3a2", path)
    rev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rev)
    assert rev.down_revision == "c4e6a8b0d2f1"
    engine = create_engine(f"sqlite:///{(tmp_path / 'scratch.db').as_posix()}")
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            rev.upgrade()
        insp = sa_inspect(conn)
        table = models.ScenarioEvidence.__table__
        cols = {c["name"]: c for c in insp.get_columns("scenario_evidence")}
        assert set(cols) == {c.name for c in table.columns}
        for c in table.columns:
            assert cols[c.name]["nullable"] == c.nullable, c.name
        assert {i["column_names"][0] for i in insp.get_indexes("scenario_evidence")} >= {"team_id", "scenario_id"}
        assert str(cols["seed"]["type"]).upper().startswith(("BIGINT", "INTEGER")), "a 32-bit seed must fit"
        with Operations.context(MigrationContext.configure(conn)):
            rev.downgrade()
        assert "scenario_evidence" not in sa_inspect(conn).get_table_names()
