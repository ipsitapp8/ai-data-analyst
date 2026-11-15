"""What-if Phase 4: saved scenarios you come back to - edit, rename, duplicate, stale-data warnings, comparison and export.

Numbers are worked by hand. Isolation, immutability of sealed evidence and the "never silently re-base" rule are tested directly.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json

import pytest
from platform_helpers import workspace

from app import scenario_library as lib
from app.database import SessionLocal
from app.models import ScenarioEvidence

BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
# baseline: revenue 20,000; variable cost 12,000; fixed 3,000 -> profit 5,000. Price +10% -> revenue 22,000 -> profit 7,000.


def body(name="Price test", pct=10.0, **extra):
    return {"name": name, "model": "unit_economics", "baseline": BASE, "scenarios": {"your scenario": {"price": {"type": "pct", "value": pct}}},
            "assumptions": {"price_elasticity": 0.0}, **extra}


def save(client, ws, **kw):
    r = client.post("/api/scenarios", json=body(**kw), headers=ws["h"])
    assert r.status_code == 201, r.text
    return r.json()


def sales_csv(price_shift=0.0):
    rows = ["date,units,revenue,cost"]
    for i in range(1, 7):
        rows.append(f"2024-0{i}-01,{10 * i},{(20 + price_shift) * 10 * i},{12 * 10 * i}")
    return ("\n".join(rows) + "\n").encode()


# ====================================================================== rename ==

def test_renaming_changes_only_the_name(client):
    ws = workspace(client)
    s = save(client, ws)
    before = client.get(f"/api/scenarios/{s['id']}", headers=ws["h"]).json()
    r = client.patch(f"/api/scenarios/{s['id']}", json={"name": "  Renamed  "}, headers=ws["h"])
    assert r.status_code == 200 and r.json()["name"] == "Renamed"
    after = client.get(f"/api/scenarios/{s['id']}", headers=ws["h"]).json()
    assert after["results"] == before["results"] and after["request"] == before["request"] and after["revision"] == before["revision"] == 1


@pytest.mark.parametrize("name", ["", "   ", "x" * 121])
def test_a_scenario_cannot_be_renamed_to_nothing_or_to_something_huge(client, name):
    ws = workspace(client)
    s = save(client, ws)
    assert client.patch(f"/api/scenarios/{s['id']}", json={"name": name}, headers=ws["h"]).status_code == 422
    assert client.get(f"/api/scenarios/{s['id']}", headers=ws["h"]).json()["name"] == "Price test"


# ======================================================================== edit ==

def test_editing_recalculates_on_the_server_and_raises_the_revision(client):
    ws = workspace(client)
    s = save(client, ws)
    assert s["results"]["scenarios"]["your scenario"]["outputs"]["profit"] == pytest.approx(7000.0)
    r = client.put(f"/api/scenarios/{s['id']}", json=body(pct=20.0), headers=ws["h"])
    assert r.status_code == 200
    e = r.json()
    # price +20% adds 0.2 x 20,000 = 4,000 -> 9,000
    assert e["id"] == s["id"] and e["revision"] == 2 and e["results"]["scenarios"]["your scenario"]["outputs"]["profit"] == pytest.approx(9000.0)
    assert e["created_at"] == s["created_at"]


def test_editing_keeps_the_evidence_of_earlier_revisions_unchanged(client):
    ws = workspace(client)
    s = save(client, ws)
    first = client.get("/api/scenario-evidence", params={"scenario_id": s["id"]}, headers=ws["h"]).json()["records"]
    assert len(first) == 1
    client.put(f"/api/scenarios/{s['id']}", json=body(pct=20.0), headers=ws["h"])
    records = client.get("/api/scenario-evidence", params={"scenario_id": s["id"]}, headers=ws["h"]).json()["records"]
    assert len(records) == 2 and {r["id"] for r in first} <= {r["id"] for r in records}
    old = next(r for r in records if r["id"] == first[0]["id"])
    assert old["content_sha256"] == first[0]["content_sha256"], "the sealed record of revision 1 is untouched"
    assert client.post(f"/api/scenario-evidence/{old['id']}/reproduce", headers=ws["h"]).json()["reproduced"] is True


def test_an_edit_with_an_invalid_input_changes_nothing(client):
    ws = workspace(client)
    s = save(client, ws)
    bad = body(pct=20.0)
    bad["baseline"] = {**BASE, "price": -5.0}
    assert client.put(f"/api/scenarios/{s['id']}", json=bad, headers=ws["h"]).status_code == 400
    after = client.get(f"/api/scenarios/{s['id']}", headers=ws["h"]).json()
    assert after["revision"] == 1 and after["results"]["scenarios"]["your scenario"]["outputs"]["profit"] == pytest.approx(7000.0)


# ==================================================================== duplicate ==

def test_duplicate_makes_an_independent_copy_with_its_own_evidence(client):
    ws = workspace(client)
    s = save(client, ws)
    d = client.post(f"/api/scenarios/{s['id']}/duplicate", headers=ws["h"])
    assert d.status_code == 201
    d = d.json()
    assert d["id"] != s["id"] and d["name"] == "Copy of Price test" and d["revision"] == 1
    assert d["results"]["scenarios"] == s["results"]["scenarios"] and d["baseline"] == s["baseline"]
    assert len(client.get("/api/scenario-evidence", params={"scenario_id": d["id"]}, headers=ws["h"]).json()["records"]) == 1
    client.patch(f"/api/scenarios/{d['id']}", json={"name": "Other"}, headers=ws["h"])
    assert client.get(f"/api/scenarios/{s['id']}", headers=ws["h"]).json()["name"] == "Price test", "editing the copy leaves the original"


def test_duplicate_names_do_not_collide_and_a_name_can_be_given(client):
    ws = workspace(client)
    s = save(client, ws)
    names = [client.post(f"/api/scenarios/{s['id']}/duplicate", headers=ws["h"]).json()["name"] for _ in range(3)]
    assert names == ["Copy of Price test", "Copy of Price test (2)", "Copy of Price test (3)"]
    assert client.post(f"/api/scenarios/{s['id']}/duplicate", json={"name": "Mine"}, headers=ws["h"]).json()["name"] == "Mine"


def test_duplicate_keeps_a_simulation_and_its_seed(client):
    ws = workspace(client)
    sim = {"uncertainty": {"price": {"type": "normal", "sd_pct": 5}}, "iterations": 1000, "seed": 7}
    s = save(client, ws, simulation=sim)
    d = client.post(f"/api/scenarios/{s['id']}/duplicate", headers=ws["h"]).json()
    assert d["request"]["simulation"]["seed"] == 7
    assert d["results"]["simulation"]["outputs"]["profit"]["mean"] == s["results"]["simulation"]["outputs"]["profit"]["mean"]


def test_duplicate_stays_on_the_data_version_the_original_used(client):
    ws = workspace(client, csv=sales_csv(), filename="sales.csv")
    src = {"dataset_version": 1}
    s = save(client, ws, dataset_id=ws["ds"], baseline_source=src)
    up = client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", sales_csv(3.0), "text/csv")}, headers=ws["h"])
    assert up.status_code == 200, up.text
    d = client.post(f"/api/scenarios/{s['id']}/duplicate", headers=ws["h"]).json()
    assert d["dataset_version_id"] == s["dataset_version_id"], "the copy is not silently moved to the newer version"
    assert d["request"]["baseline_source"]["dataset_version"] == 1


# ================================================================== freshness ==

def test_a_scenario_with_typed_inputs_has_no_data_to_go_stale(client):
    ws = workspace(client)
    assert save(client, ws)["freshness"] == {"status": "typed_inputs", "reasons": [], "pinned_version": None, "latest_version": None}


def test_a_scenario_is_current_until_newer_data_arrives_then_stale_without_changing(client):
    ws = workspace(client, csv=sales_csv(), filename="sales.csv")
    s = save(client, ws, dataset_id=ws["ds"], baseline_source={"dataset_version": 1})
    assert s["freshness"]["status"] == "current" and s["freshness"]["reasons"] == []
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", sales_csv(3.0), "text/csv")}, headers=ws["h"])
    g = client.get(f"/api/scenarios/{s['id']}", headers=ws["h"]).json()
    assert g["freshness"]["status"] == "stale" and g["freshness"]["pinned_version"] == 1 and g["freshness"]["latest_version"] == 2
    assert "version 1" in g["freshness"]["reasons"][0] and "version 2" in g["freshness"]["reasons"][0] and "was not changed" in g["freshness"]["reasons"][0]
    assert g["baseline"] == s["baseline"] and g["results"] == s["results"], "stale data never rewrites the saved numbers"
    listed = client.get("/api/scenarios", headers=ws["h"]).json()["scenarios"][0]
    assert listed["freshness"]["status"] == "stale"


def test_a_changed_fingerprint_is_reported_as_changed(client):
    ws = workspace(client, csv=sales_csv(), filename="sales.csv")
    s = save(client, ws, dataset_id=ws["ds"], baseline_source={"dataset_version": 1})
    from app.models import DatasetVersion
    db = SessionLocal()
    try:
        v = db.get(DatasetVersion, s["dataset_version_id"])
        v.content_sha256 = "0" * 64
        db.commit()
    finally:
        db.close()
    f = client.get(f"/api/scenarios/{s['id']}", headers=ws["h"]).json()["freshness"]
    assert f["status"] == "changed" and "no longer matches" in f["reasons"][0]


def test_a_revised_custom_model_is_flagged_but_the_saved_numbers_stay(client):
    ws = workspace(client)
    d = {"name": "Margin model", "definition": {"inputs": {"price": {"label": "Price", "unit": "currency", "min": 0}, "cost": {"label": "Cost", "unit": "currency", "min": 0}},
                                                "outputs": {"profit": {"label": "Profit", "unit": "currency", "formula": "price - cost"}}, "primary_output": "profit"}}
    m = client.post("/api/scenario-models", json=d, headers=ws["h"])
    assert m.status_code == 201, m.text
    m = m.json()
    saved = client.post("/api/scenarios", json={"name": "c", "model": m["ref"], "baseline": {"price": 10.0, "cost": 4.0}, "scenarios": {}}, headers=ws["h"])
    assert saved.status_code == 201, saved.text
    assert saved.json()["freshness"]["reasons"] == []
    v2 = {"note": "v2", "definition": {**d["definition"], "outputs": {"profit": {"label": "Profit", "unit": "currency", "formula": "price - cost - 1"}}}}
    assert client.post(f"/api/scenario-models/{m['id']}/versions", json=v2, headers=ws["h"]).status_code in (200, 201)
    f = client.get(f"/api/scenarios/{saved.json()['id']}", headers=ws["h"]).json()
    assert "version 2 exists now" in " ".join(f["freshness"]["reasons"]) and f["results"]["baseline"]["outputs"]["profit"] == 6.0


# ================================================================== comparison ==

def save_pair(client, ws, base_b=None, pct_b=20.0, **kw):
    a = save(client, ws, name="A", pct=10.0)
    b_body = body(name="B", pct=pct_b)
    if base_b:
        b_body["baseline"] = base_b
    b = client.post("/api/scenarios", json=b_body, headers=ws["h"]).json()
    return a, b


def test_comparison_gives_each_scenarios_change_against_its_own_baseline(client):
    ws = workspace(client)
    a, b = save_pair(client, ws)
    r = client.get("/api/scenarios/compare", params={"ids": f"{a['id']},{b['id']}"}, headers=ws["h"]).json()
    rows = {(x["name"], x["scenario"]): x for x in r["analysis"]["rows"]}
    assert rows[("A", "baseline")]["value"] == 5000.0 and rows[("A", "your scenario")]["value"] == pytest.approx(7000.0)
    assert rows[("A", "your scenario")]["change"] == pytest.approx(2000.0) and rows[("A", "your scenario")]["change_pct"] == pytest.approx(40.0)
    assert rows[("B", "your scenario")]["change"] == pytest.approx(4000.0) and rows[("B", "your scenario")]["change_pct"] == pytest.approx(80.0)
    assert r["analysis"]["highest"]["name"] == "B" and r["analysis"]["lowest"]["name"] == "A" and r["analysis"]["warnings"] == []
    assert "describe the number only" in r["analysis"]["note"] and r["items"][0]["baseline_outputs"]["profit"] == 5000.0, "the earlier keys remain"


def test_comparison_warns_when_the_starting_point_differs(client):
    ws = workspace(client)
    a, b = save_pair(client, ws, base_b={**BASE, "price": 25.0})
    r = client.get("/api/scenarios/compare", params={"ids": f"{a['id']},{b['id']}"}, headers=ws["h"]).json()
    assert any("different baseline inputs" in w for w in r["analysis"]["warnings"])


def test_comparison_warns_about_stale_data_and_different_versions(client):
    ws = workspace(client, csv=sales_csv(), filename="sales.csv")
    a = save(client, ws, name="A", dataset_id=ws["ds"], baseline_source={"dataset_version": 1})
    client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", sales_csv(3.0), "text/csv")}, headers=ws["h"])
    b = save(client, ws, name="B", dataset_id=ws["ds"], baseline_source={"dataset_version": 2})
    r = client.get("/api/scenarios/compare", params={"ids": f"{a['id']},{b['id']}"}, headers=ws["h"]).json()
    w = " ".join(r["analysis"]["warnings"])
    assert "different versions of the data" in w and "'A': This scenario uses version 1" in w


def test_comparison_csv_is_a_file_and_defuses_formulas_in_names(client):
    ws = workspace(client)
    a = save(client, ws, name="=HYPERLINK(\"http://x\")", pct=10.0)
    b = save(client, ws, name="B", pct=20.0)
    r = client.get("/api/scenarios/compare", params={"ids": f"{a['id']},{b['id']}", "format": "csv"}, headers=ws["h"])
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv") and "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0][:3] == ["saved_as", "scenario", "primary_output"]
    assert all(not row[0].startswith("=") for row in rows if row), "no cell starts a formula"
    assert any(row[0] == "'=HYPERLINK(\"http://x\")" for row in rows)
    assert any(row and row[0].startswith("# ") and "not a validated forecast" in row[0] for row in rows)


# ======================================================================= export ==

def test_the_export_is_what_was_saved_with_its_disclaimer_evidence_and_a_checkable_fingerprint(client):
    ws = workspace(client)
    s = save(client, ws)
    r = client.get(f"/api/scenarios/{s['id']}/export", headers=ws["h"])
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    doc = r.json()
    assert doc["data_kind"] == "scenario" and "not a validated forecast" in doc["disclaimer"] and doc["baseline_inputs"] == BASE
    assert doc["scenario_outputs"]["your scenario"]["profit"] == pytest.approx(7000.0) and doc["baseline_outputs"]["profit"] == 5000.0
    assert doc["verification"]["status"] == "verified" and len(doc["evidence"]) == 1 and doc["changes"] == {"your scenario": {"price": {"type": "pct", "value": 10.0}}}
    claimed = doc.pop("content_sha256")
    again = hashlib.sha256(json.dumps(doc, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
    assert claimed == again == r.headers["x-content-sha256"], "recomputed here with plain json and hashlib"


def test_the_export_does_not_recalculate_it_reports_what_is_stored(client):
    ws = workspace(client)
    s = save(client, ws)
    from app.models import Scenario
    db = SessionLocal()
    try:
        row = db.get(Scenario, s["id"])
        res = json.loads(json.dumps(row.results_json))
        res["scenarios"]["your scenario"]["outputs"]["profit"] = 1.0
        row.results_json = res
        db.commit()
    finally:
        db.close()
    assert client.get(f"/api/scenarios/{s['id']}/export", headers=ws["h"]).json()["scenario_outputs"]["your scenario"]["profit"] == 1.0


def test_csv_and_markdown_exports_carry_the_numbers_the_disclaimer_and_the_fingerprint(client):
    ws = workspace(client)
    s = save(client, ws, name="Q3 | pricing\nplan")
    c = client.get(f"/api/scenarios/{s['id']}/export", params={"format": "csv"}, headers=ws["h"])
    rows = list(csv.reader(io.StringIO(c.text)))
    profit = next(r for r in rows if r[1] == "profit")
    assert float(profit[3]) == 5000.0 and float(profit[4]) == pytest.approx(7000.0) and float(profit[5]) == pytest.approx(2000.0)
    assert any(r and r[0] == "# content_sha256" and r[1] == c.headers["x-content-sha256"] for r in rows)
    m = client.get(f"/api/scenarios/{s['id']}/export", params={"format": "md"}, headers=ws["h"])
    assert m.headers["content-type"].startswith("text/markdown") and "# Q3 \\| pricing plan" in m.text
    assert "| your scenario | Profit | 5,000.00 | 7,000.00 | 2,000.00 |" in m.text and f"`{m.headers['x-content-sha256']}`" in m.text
    assert "not a validated forecast" in m.text and "Independent check:** verified" in m.text


def test_an_unknown_export_format_is_refused(client):
    ws = workspace(client)
    s = save(client, ws)
    assert client.get(f"/api/scenarios/{s['id']}/export", params={"format": "pdf"}, headers=ws["h"]).status_code == 422


@pytest.mark.parametrize("cell,expected", [("=1+1", "'=1+1"), ("+x", "'+x"), ("-x", "'-x"), ("@x", "'@x"), ("\tx", "'\tx"), ("plain", "plain"), (5, 5)])
def test_spreadsheet_formula_defusing(cell, expected):
    assert lib._safe(cell) == expected  # noqa: SLF001


# =================================================================== isolation ==

def test_another_team_cannot_see_edit_copy_export_or_compare_a_scenario(client):
    mine, other = workspace(client), workspace(client)
    s = save(client, mine)
    mine_two = save(client, mine, name="two", pct=5.0)
    h = other["h"]
    assert client.get(f"/api/scenarios/{s['id']}", headers=h).status_code == 404
    assert client.patch(f"/api/scenarios/{s['id']}", json={"name": "x"}, headers=h).status_code == 404
    assert client.put(f"/api/scenarios/{s['id']}", json=body(), headers=h).status_code == 404
    assert client.post(f"/api/scenarios/{s['id']}/duplicate", headers=h).status_code == 404
    assert client.get(f"/api/scenarios/{s['id']}/export", headers=h).status_code == 404
    assert client.get("/api/scenarios/compare", params={"ids": f"{s['id']},{mine_two['id']}"}, headers=h).status_code == 404
    assert client.get("/api/scenarios", headers=h).json()["total"] == 0
    assert client.get(f"/api/scenarios/{s['id']}", headers=mine["h"]).json()["name"] == "Price test"


def test_the_new_endpoints_need_a_login(client):
    assert client.patch("/api/scenarios/1", json={"name": "x"}).status_code in (401, 403)
    assert client.put("/api/scenarios/1", json=body()).status_code in (401, 403)
    assert client.post("/api/scenarios/1/duplicate").status_code in (401, 403)
    assert client.get("/api/scenarios/1/export").status_code in (401, 403)


def test_evidence_of_a_copy_is_its_own_and_scoped_to_the_team(client):
    ws = workspace(client)
    s = save(client, ws)
    d = client.post(f"/api/scenarios/{s['id']}/duplicate", headers=ws["h"]).json()
    db = SessionLocal()
    try:
        rows = db.query(ScenarioEvidence).filter(ScenarioEvidence.scenario_id.in_([s["id"], d["id"]])).all()
        assert {r.scenario_id for r in rows} == {s["id"], d["id"]} and len({r.team_id for r in rows}) == 1
        assert next(r for r in rows if r.scenario_id == d["id"]).note == f"copied from scenario {s['id']}"
    finally:
        db.close()
