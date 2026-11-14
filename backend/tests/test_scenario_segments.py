"""What-if Phase 2, feature 4: segment-level scenarios.

Every expected figure is a plain-Python sum over the rows written to the file. The properties that matter most are
the ones that prevent double counting: segments partition the rows, an input is changed once, a shared cost is split
once, and the segments add up to the total.
"""
from __future__ import annotations

import copy
import csv
import io
import random

import pytest
from platform_helpers import workspace

from app import scenario_segments as sg
from app import scenario_verify, scenarios

MAP = {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"}
NO_FIXED = {"fixed_cost": 0.0}


def make_rows():
    rng = random.Random(7)
    rows = []
    for region, base in (("East", 20.0), ("West", 30.0), ("North", 25.0)):
        for m in (1, 2, 3):
            for k in range(4):
                units = rng.randint(5, 40)
                price = round(base + rng.random() * 3, 2)
                rows.append({"order_date": f"2025-{m:02d}-{k * 7 + 1:02d}", "region": region, "product": ("A", "B")[k % 2], "units": units,
                             "unit_price": price, "revenue": round(units * price, 2), "cost": round(units * 12.0, 2)})
    rows += [{"order_date": "2025-01-05", "region": "", "product": "A", "units": 10, "unit_price": 22.0, "revenue": 220.0, "cost": 120.0},
             {"order_date": "2025-02-05", "region": "", "product": "B", "units": 4, "unit_price": 22.0, "revenue": 88.0, "cost": 48.0}]
    rows += [{"order_date": "2025-03-05", "region": "South", "product": "A", "units": 3, "unit_price": 50.0, "revenue": 150.0, "cost": 36.0}] * 3
    return rows


def csv_of(rows) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


ROWS = make_rows()


def seg_totals(rows, column="region"):
    out = {}
    for r in rows:
        key = r[column] or "(missing)"
        t = out.setdefault(key, {"units": 0.0, "revenue": 0.0, "cost": 0.0, "rows": 0})
        t["units"] += r["units"]
        t["revenue"] += r["revenue"]
        t["cost"] += r["cost"]
        t["rows"] += 1
    return out


@pytest.fixture
def ws(client):
    return workspace(client, csv=csv_of(ROWS), filename="regions.csv")


def run(client, ws, **kw):
    body = {"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "segment_column": "region", "constant_inputs": NO_FIXED, **kw}
    return client.post("/api/scenarios/segments", json=body, headers=ws["h"])


def by_value(out):
    return {s["value"]: s for s in out["segments"]}


# ==================================================== the partition and baselines ==

def test_segments_partition_the_rows_and_each_baseline_is_the_sums_over_its_rows(client, ws):
    r = run(client, ws)
    assert r.status_code == 200, r.text
    out = r.json()
    want = seg_totals(ROWS)
    assert set(by_value(out)) == {"East", "West", "North", "South", "(missing)"} and out["partition"]["segments"] == 5
    for name, t in want.items():
        s = by_value(out)[name]
        assert s["rows"] == t["rows"] and s["baseline"]["inputs"]["volume"] == t["units"]
        assert s["baseline"]["inputs"]["price"] == pytest.approx(t["revenue"] / t["units"], rel=1e-12)
        assert s["baseline"]["inputs"]["unit_cost"] == pytest.approx(t["cost"] / t["units"], rel=1e-12)
        assert s["baseline"]["outputs"]["profit"] == pytest.approx(t["revenue"] - t["cost"], rel=1e-9)
    assert out["partition"] == {"rows_in_scope": len(ROWS), "rows_in_segments": len(ROWS), "segments": 5, "complete": True, "missing_segment": True}
    assert sum(s["rows"] for s in out["segments"]) == len(ROWS), "a blank value is its own segment, not dropped"
    assert out["verification"]["status"] == "verified", [c for c in out["verification"]["checks"] if c["status"] != "pass"]
    assert out["data_kind"] == "scenario" and out["baseline_data_kind"] == "historical_actual"


def test_the_total_is_the_sum_of_the_segments_and_the_ratio_is_recomputed_from_them(client, ws):
    out = run(client, ws).json()
    revenue = sum(r["revenue"] for r in ROWS)
    cost = sum(r["cost"] for r in ROWS)
    o = out["overall"]["baseline"]["outputs"]
    assert o["revenue"] == pytest.approx(revenue, rel=1e-12) and o["profit"] == pytest.approx(revenue - cost, rel=1e-9)
    assert o["margin_pct"] == pytest.approx((revenue - cost) / revenue * 100, rel=1e-9), "margin is profit over revenue of the totals, not an average of margins"
    assert out["overall"]["additive_outputs"] == ["revenue", "variable_cost", "gross_profit", "profit"]
    assert all(c["agrees"] for c in out["consistency"].values()), "the sum over segments equals the all-rows figure"


def test_the_segments_of_a_column_can_be_listed_with_their_sizes(client, ws):
    r = client.get("/api/scenarios/segment-values", params={"dataset_id": ws["ds"], "segment_column": "region"}, headers=ws["h"]).json()
    want = seg_totals(ROWS)
    assert {s["value"]: s["rows"] for s in r["segments"]} == {k: v["rows"] for k, v in want.items()} and r["truncated"] is False
    assert r["segments"][0]["rows"] >= r["segments"][-1]["rows"] and r["provenance"]["dataset_version"] == 1
    assert client.get("/api/scenarios/segment-values", params={"dataset_id": ws["ds"], "segment_column": "units"}, headers=ws["h"]).status_code == 400


# ======================================================= adjusting some segments ==

def test_changing_one_segments_price_moves_only_that_segment_and_the_total_by_the_same_amount(client, ws):
    r = run(client, ws, segment_adjustments={"West": {"price": {"type": "pct", "value": 10}}})
    assert r.status_code == 200, r.text
    out = r.json()
    west = seg_totals(ROWS)["West"]
    s = by_value(out)
    assert s["West"]["scenario"]["outputs"]["revenue"] == pytest.approx(west["revenue"] * 1.1, rel=1e-9)
    assert s["West"]["delta"]["profit"]["abs"] == pytest.approx(0.1 * west["revenue"], rel=1e-9)
    for other in ("East", "North", "South", "(missing)"):
        assert s[other]["scenario"]["inputs"] == s[other]["baseline"]["inputs"] and s[other]["delta"]["profit"]["abs"] == 0
    total_change = out["overall"]["delta"]["profit"]["abs"]
    assert total_change == pytest.approx(0.1 * west["revenue"], rel=1e-9)
    c = out["contributions"]["by_segment"]
    assert c["West"]["share_pct"] == pytest.approx(100.0) and c["East"]["share_pct"] == 0 and sum(v["abs"] for v in c.values()) == pytest.approx(total_change, rel=1e-9)
    assert out["verification"]["status"] == "verified"
    assert any("Segments with no adjustment keep their baseline" in d for d in out["disclosures"])


def test_price_elasticity_applies_inside_the_adjusted_segment_only(client, ws):
    out = run(client, ws, assumptions={"price_elasticity": -1.5}, segment_adjustments={"East": {"price": {"type": "pct", "value": 10}}}).json()
    east = seg_totals(ROWS)["East"]
    s = by_value(out)["East"]
    assert s["scenario"]["inputs"]["volume"] == pytest.approx(east["units"] * (1 - 1.5 * 0.1), rel=1e-12)        # 10% dearer, 15% fewer sold... x0.85
    want_rev = (east["revenue"] / east["units"]) * 1.1 * east["units"] * 0.85
    assert s["scenario"]["outputs"]["revenue"] == pytest.approx(want_rev, rel=1e-9)
    assert by_value(out)["West"]["scenario"]["inputs"]["volume"] == seg_totals(ROWS)["West"]["units"]
    assert out["verification"]["status"] == "verified"


def test_a_shared_adjustment_applies_everywhere_and_a_segments_own_replaces_it_rather_than_stacking(client, ws):
    out = run(client, ws, shared_adjustments={"volume": {"type": "pct", "value": 5}},
              segment_adjustments={"West": {"volume": {"type": "pct", "value": 20}}}).json()
    t = seg_totals(ROWS)
    s = by_value(out)
    assert s["East"]["scenario"]["inputs"]["volume"] == pytest.approx(t["East"]["units"] * 1.05)
    assert s["West"]["scenario"]["inputs"]["volume"] == pytest.approx(t["West"]["units"] * 1.20), "1.2x, not 1.05 x 1.2"
    assert s["West"]["adjustments"] == {"volume": {"type": "pct", "value": 20}} and s["East"]["adjustments"] == {"volume": {"type": "pct", "value": 5}}
    assert out["verification"]["status"] == "verified"
    assert any("replaces the shared one" in d for d in out["disclosures"])


def test_an_adjustment_can_target_the_missing_value_segment(client, ws):
    out = run(client, ws, segment_adjustments={"(missing)": {"unit_cost": {"type": "pct", "value": -10}}}).json()
    t = seg_totals(ROWS)["(missing)"]
    assert by_value(out)["(missing)"]["delta"]["profit"]["abs"] == pytest.approx(0.1 * t["cost"], rel=1e-9)


# ===================================================== costs the data does not split ==

def test_a_fixed_cost_total_is_shared_out_once_in_proportion_to_volume(client, ws):
    out = run(client, ws, constant_inputs={}, shared_inputs={"fixed_cost": {"total": 3000, "allocate_by": "volume"}}).json()
    t = seg_totals(ROWS)
    units = sum(v["units"] for v in t.values())
    s = by_value(out)
    for name, v in t.items():
        assert s[name]["baseline"]["inputs"]["fixed_cost"] == pytest.approx(3000 * v["units"] / units, rel=1e-12)
    a = out["allocations"]["fixed_cost"]
    assert a["shares_sum"] == pytest.approx(1.0, abs=1e-12) and sum(s_["baseline"]["inputs"]["fixed_cost"] for s_ in out["segments"]) == pytest.approx(3000)
    revenue, cost = sum(r["revenue"] for r in ROWS), sum(r["cost"] for r in ROWS)
    assert out["overall"]["baseline"]["outputs"]["profit"] == pytest.approx(revenue - cost - 3000, rel=1e-9)
    assert out["overall"]["baseline"]["inputs"]["fixed_cost"] == 3000
    assert out["verification"]["status"] == "verified" and any("allocated across segments" in d for d in out["disclosures"])
    equal = run(client, ws, constant_inputs={}, shared_inputs={"fixed_cost": {"total": 3000, "allocate_by": "equal"}}).json()
    assert all(s_["baseline"]["inputs"]["fixed_cost"] == pytest.approx(600) for s_ in equal["segments"])


@pytest.mark.parametrize("kw,fragment", [
    ({"constant_inputs": {}}, "has no value in the data"),
    ({"constant_inputs": {"fixed_cost": 5}, "shared_inputs": {"fixed_cost": {"total": 1, "allocate_by": "equal"}}}, "choose one"),
    ({"constant_inputs": {}, "shared_inputs": {"fixed_cost": {"total": 1, "allocate_by": "colour"}}}, "must be 'equal' or another input"),
    ({"constant_inputs": {}, "shared_inputs": {"fixed_cost": {"total": 1, "allocate_by": "fixed_cost"}}}, "must be 'equal' or another input"),
    ({"constant_inputs": {}, "shared_inputs": {"fixed_cost": {"allocate_by": "equal"}}}, "needs {total, allocate_by}"),
    ({"constant_inputs": {}, "shared_inputs": {"fixed_cost": {"total": "lots"}}}, "finite number"),
    ({"constant_inputs": {"ghost": 1}}, "Unknown input"),
    ({"segment_adjustments": {"Atlantis": {"price": {"type": "pct", "value": 5}}}}, "There is no segment 'Atlantis'"),
    ({"segment_adjustments": {"West": {"price": {"type": "pct", "value": -150}}}}, "more than 100%"),
    ({"segment_adjustments": {"West": {"nope": {"type": "pct", "value": 5}}}}, "unknown input"),
    ({"shared_adjustments": {"price": {"type": "magic", "value": 5}}}, "type pct, abs or set"),
    ({"segment_column": "units"}, "is numeric"), ({"segment_column": "order_date"}, "does not exist|holds dates"),
    ({"segment_column": "ghost"}, "does not exist"), ({"model": "crystal_ball"}, "Unknown model"),
])
def test_invalid_segment_requests_are_refused_with_a_reason(client, ws, kw, fragment):
    r = run(client, ws, **kw)
    assert r.status_code in (400, 404) and __import__("re").search(fragment, r.json()["detail"]), r.text
    assert "Traceback" not in r.text


def test_a_shared_input_that_cannot_be_split_is_refused():
    frame = __import__("pandas").DataFrame({"g": ["a", "b"], "v": [0.0, 0.0]})
    # the driver (volume) sums to zero across the segments, so a proportional split has no meaning
    with pytest.raises(sg.SegmentError, match="add up to more than zero"):
        sg.run("unit_economics", frame, lambda d: {"inputs": {"price": 1.0, "volume": float(d["v"].sum()), "unit_cost": 1.0, "fixed_cost": None},
                                                 "sources": {k: {"source": "dataset", "formula": "x"} for k in ("price", "volume", "unit_cost")}},
               segment_column="g", shared_inputs={"fixed_cost": {"total": 10, "allocate_by": "volume"}}, verify=False)


# ===================================================== scope, thin and too many ==

def test_a_period_and_filters_define_the_rows_that_are_split(client, ws):
    r = run(client, ws, date_column="order_date", period={"type": "month", "year": 2025, "month": 2}, filters=[{"column": "product", "op": "==", "value": "A"}])
    assert r.status_code == 200, r.text
    out = r.json()
    scope = [x for x in ROWS if x["order_date"].startswith("2025-02") and x["product"] == "A"]
    assert out["partition"]["rows_in_scope"] == len(scope) and out["period"]["label"] == "February 2025"
    assert {s["value"]: s["rows"] for s in out["segments"]} == {k: v["rows"] for k, v in seg_totals(scope).items()}
    assert out["verification"]["status"] == "verified"
    assert run(client, ws, date_column="order_date", period={"type": "month", "year": 2030, "month": 1}).status_code == 400
    assert run(client, ws, filters=[{"column": "region", "op": "==", "value": "Nowhere"}]).status_code == 400


def test_a_segment_with_very_few_rows_is_marked_thin(client, ws):
    out = run(client, ws).json()
    assert by_value(out)["South"]["thin"] is True and by_value(out)["East"]["thin"] is False
    assert any("marked thin" in d for d in out["disclosures"])


def test_too_many_segments_is_refused_with_a_way_out(client):
    rows = [{"order_date": "2025-01-01", "code": f"c{i:03d}", "units": 5, "revenue": 50.0, "cost": 20.0} for i in range(60)]
    ws = workspace(client, csv=csv_of(rows), filename="many.csv")
    r = client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "segment_column": "code",
                                                     "constant_inputs": NO_FIXED}, headers=ws["h"])
    assert r.status_code == 400 and "60 distinct values; at most 50" in r.json()["detail"] and "Filter the data" in r.json()["detail"]
    listed = client.get("/api/scenarios/segment-values", params={"dataset_id": ws["ds"], "segment_column": "code"}, headers=ws["h"]).json()
    assert listed["truncated"] is True and listed["total_segments"] == 60 and len(listed["segments"]) == 50
    ok = client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "segment_column": "code",
                                                      "constant_inputs": NO_FIXED, "filters": [{"column": "code", "op": "in", "value": [f"c{i:03d}" for i in range(40)]}]}, headers=ws["h"])
    assert ok.status_code == 200 and ok.json()["partition"]["segments"] == 40 and ok.json()["verification"]["status"] == "verified"


# ============================================================ double counting ==

def test_an_input_that_is_not_additive_across_segments_is_caught_not_summed_silently(client, ws):
    """Price taken as a simple average of a price column is not additive: summing segment revenues disagrees with the
    revenue computed from all rows, and the result says it cannot be verified."""
    mapping = {"quantity_column": "units", "price_column": "unit_price", "total_cost_column": "cost"}
    out = run(client, ws, mapping=mapping).json()
    assert out["consistency"]["revenue"]["agrees"] is False
    v = out["verification"]
    assert v["status"] == "unverified"
    check = {c["check"]: c for c in v["checks"]}["segments_add_up_to_all_rows"]
    assert check["status"] == "fail" and "simple average" in check["detail"]


def test_tampering_with_any_segment_result_is_caught(client, ws):
    out = run(client, ws, constant_inputs={}, shared_inputs={"fixed_cost": {"total": 3000, "allocate_by": "volume"}},
              segment_adjustments={"West": {"price": {"type": "pct", "value": 8}}}).json()
    assert scenario_verify.verify_segments(copy.deepcopy(out))["status"] == "verified"
    tampers = [
        lambda o: o["segments"][0]["scenario"]["outputs"].__setitem__("profit", 1.0),
        lambda o: o["overall"]["scenario"]["outputs"].__setitem__("revenue", 1.0),
        lambda o: o["overall"]["scenario"]["outputs"].__setitem__("margin_pct", 1.0),
        lambda o: o["contributions"]["by_segment"]["West"].__setitem__("abs", 1.0),
        lambda o: o["allocations"]["fixed_cost"]["shares"].__setitem__("West", 0.9),
        lambda o: o["partition"].__setitem__("rows_in_segments", 1),
        lambda o: o["segments"][1]["adjustments"].__setitem__("volume", {"type": "pct", "value": 9}),
        lambda o: o["consistency"]["profit"].__setitem__("agrees", False),
    ]
    for tamper in tampers:
        bad = copy.deepcopy(out)
        tamper(bad)
        assert scenario_verify.verify_segments(bad)["status"] == "unverified"


def test_segment_sizes_and_inputs_are_recounted_from_the_file(client, ws):
    out = run(client, ws).json()
    checks = {c["check"]: c["status"] for c in out["verification"]["checks"]}
    assert checks["segment_sizes"] == "pass" and checks["segment_inputs"] == "pass" and checks["partition"] == "pass"
    wrong = copy.deepcopy(out)
    wrong["segments"][0]["rows"] += 1
    from app.database import SessionLocal
    from app.models import DatasetVersion
    with SessionLocal() as db:
        path = db.query(DatasetVersion).filter_by(dataset_id=ws["ds"]).first().filepath
    v = scenario_verify.verify_segments(wrong, None, {"csv_path": path, "filters": []})
    assert {c["check"]: c["status"] for c in v["checks"]}["segment_sizes"] == "fail"
    wrong2 = copy.deepcopy(out)
    wrong2["segments"][1]["baseline"]["inputs"]["volume"] += 3
    assert {c["check"]: c["status"] for c in scenario_verify.verify_segments(wrong2, None, {"csv_path": path, "filters": []})["checks"]}["segment_inputs"] == "fail"


# ===================================================================== custom ==

def test_a_custom_model_splits_by_segment_using_its_additive_and_ratio_outputs(client, ws):
    d = {"inputs": {"units_sold": {"unit": "units", "min": 0, "source": {"type": "formula", "formula": "sum(units)"}},
                    "revenue_total": {"unit": "currency", "min": 0, "source": {"type": "formula", "formula": "sum(revenue)"}},
                    "cost_total": {"unit": "currency", "min": 0, "source": {"type": "formula", "formula": "sum(cost)"}}},
         "outputs": {"revenue": {"unit": "currency", "formula": "revenue_total", "additive": True},
                     "cost": {"unit": "currency", "formula": "cost_total", "additive": True},
                     "profit": {"unit": "currency", "formula": "revenue_total - cost_total", "additive": True},
                     "margin": {"unit": "fraction", "formula": "profit / revenue", "ratio_of": {"numerator": "profit", "denominator": "revenue"}},
                     "yield": {"unit": "currency/units", "formula": "revenue_total / units_sold"}},
         "primary_output": "profit"}
    m = client.post("/api/scenario-models", json={"name": "Split model", "definition": d}, headers=ws["h"]).json()
    r = client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": m["ref"], "segment_column": "product",
                                                     "segment_adjustments": {"B": {"revenue_total": {"type": "pct", "value": 10}}}}, headers=ws["h"])
    assert r.status_code == 200, r.text
    out = r.json()
    t = seg_totals(ROWS, "product")
    assert {s["value"] for s in out["segments"]} == {"A", "B"}
    b = by_value(out)["B"]
    assert b["delta"]["profit"]["abs"] == pytest.approx(0.1 * t["B"]["revenue"], rel=1e-9)
    assert out["overall"]["baseline"]["outputs"]["margin"] == pytest.approx(sum(r_["revenue"] - r_["cost"] for r_ in ROWS) / sum(r_["revenue"] for r_ in ROWS), rel=1e-9)
    assert out["overall"]["baseline"]["outputs"]["yield"] is None and out["overall"]["not_additive"] == ["yield"], "a non-additive output is not summed"
    assert out["overall"]["ratio_outputs"]["margin"]["numerator"] == "profit"
    assert out["verification"]["status"] == "verified", [c for c in out["verification"]["checks"] if c["status"] != "pass"]


def test_a_custom_model_with_a_constraint_rejects_a_segment_scenario_that_breaks_it(client, ws):
    d = {"inputs": {"price": {"unit": "currency", "min": 0, "source": {"type": "formula", "formula": "sum(revenue) / sum(units)"}},
                    "cost": {"unit": "currency", "min": 0, "source": {"type": "formula", "formula": "sum(cost) / sum(units)"}}},
         "outputs": {"margin": {"unit": "currency", "formula": "price - cost", "additive": False}}, "primary_output": "margin",
         "constraints": [{"expr": "price >= cost", "message": "Price must cover cost"}]}
    m = client.post("/api/scenario-models", json={"name": "Guarded", "definition": d}, headers=ws["h"]).json()
    ok = client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": m["ref"], "segment_column": "region"}, headers=ws["h"])
    assert ok.status_code == 200 and ok.json()["contributions"] is None
    assert any(c["status"] == "not_applicable" for c in ok.json()["verification"]["checks"] if c["check"] == "contributions")
    bad = client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": m["ref"], "segment_column": "region",
                                                       "segment_adjustments": {"West": {"price": {"type": "pct", "value": -90}}}}, headers=ws["h"])
    assert bad.status_code == 400 and "Price must cover cost" in bad.json()["detail"] and "Segment 'West'" in bad.json()["detail"]


# ====================================================== versions, teams, security ==

def test_a_chosen_dataset_version_is_the_one_split(client, ws):
    changed = [dict(r, revenue=round(r["revenue"] * 2, 2)) for r in ROWS]
    up = client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", csv_of(changed), "text/csv")}, headers=ws["h"])
    assert up.status_code in (200, 201), up.text
    v1 = run(client, ws, dataset_version=1).json()
    v2 = run(client, ws).json()
    assert v1["provenance"]["dataset_version"] == 1 and v2["provenance"]["dataset_version"] == 2
    assert v2["overall"]["baseline"]["outputs"]["revenue"] == pytest.approx(2 * v1["overall"]["baseline"]["outputs"]["revenue"], rel=1e-9)
    assert run(client, ws, dataset_version=7).status_code == 404


def test_segment_endpoints_are_team_scoped_and_need_a_signed_in_member(client, ws):
    other = workspace(client)
    assert run(client, {**ws, "h": other["h"]}).status_code == 404
    assert client.get("/api/scenarios/segment-values", params={"dataset_id": ws["ds"], "segment_column": "region"}, headers=other["h"]).status_code == 404
    assert client.get("/api/scenarios/segment-values", params={"dataset_id": ws["ds"], "segment_column": "region"}).status_code == 401
    assert client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": "unit_economics", "segment_column": "region"}).status_code == 401
    custom = client.post("/api/scenario-models", json={"name": "Mine", "definition": {"inputs": {"a": {}}, "outputs": {"o": {"formula": "a"}}, "primary_output": "o"}}, headers=ws["h"]).json()
    assert client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": custom["ref"], "segment_column": "region"}, headers=other["h"]).status_code == 404


def test_a_dataset_with_many_rows_is_split_quickly(client):
    import time
    rng = random.Random(3)
    rows = [{"order_date": "2025-01-01", "region": f"r{rng.randint(0, 29)}", "units": rng.randint(1, 20), "revenue": 0.0, "cost": 0.0} for _ in range(40_000)]
    for r in rows:
        r["revenue"], r["cost"] = r["units"] * 20.0, r["units"] * 12.0
    ws = workspace(client, csv=csv_of(rows), filename="big.csv")
    started = time.perf_counter()
    r = client.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "segment_column": "region",
                                                     "constant_inputs": NO_FIXED, "shared_adjustments": {"price": {"type": "pct", "value": 3}}}, headers=ws["h"])
    assert r.status_code == 200 and r.json()["partition"]["segments"] == 30 and r.json()["verification"]["status"] == "verified"
    assert time.perf_counter() - started < 30
