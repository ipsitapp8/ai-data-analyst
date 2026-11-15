"""What-if Phase 2, feature 3: time-aware baselines and period comparison.

The dataset is generated here, so every expected number is a plain-Python sum over the rows written to the file.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import io

import pandas as pd
import pytest
from platform_helpers import workspace

from app import scenario_time as tm
from app import scenario_verify, scenarios

MONTHS = [(2025, m) for m in range(1, 13)] + [(2026, m) for m in range(1, 4)]
FIXED = 500.0


def make_rows(shift: float = 0.0):
    rows = []
    for idx, (y, m) in enumerate(MONTHS):
        price = 20.0 + idx + shift
        for day, region in ((1, "East"), (8, "West"), (15, "East"), (28, "West")):
            units = 10 * (idx + 1) + day
            rows.append({"order_date": f"{y}-{m:02d}-{day:02d}", "region": region, "units": units,
                         "revenue": round(units * price, 2), "cost": round(units * 12.0, 2)})
    return rows


def csv_of(rows) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode()


def inside(rows, start, end, region=None):
    return [r for r in rows if start <= r["order_date"] <= end and (region is None or r["region"] == region)]


def totals(rows):
    units = sum(r["units"] for r in rows)
    revenue = sum(r["revenue"] for r in rows)
    cost = sum(r["cost"] for r in rows)
    return {"volume": float(units), "price": revenue / units, "unit_cost": cost / units}


MAP = {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"}


@pytest.fixture
def rows():
    return make_rows()


@pytest.fixture
def ws(client, rows):
    return workspace(client, csv=csv_of(rows), filename="orders.csv")


def baseline(client, ws, period=None, **extra):
    body = {"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "overrides": {"fixed_cost": FIXED},
            "date_column": "order_date", **extra}
    if period is not None:
        body["period"] = period
    return client.post("/api/scenarios/baseline", json=body, headers=ws["h"])


# ============================================================== period parsing ==

@pytest.mark.parametrize("spec,label,start,end", [
    ({"type": "year", "year": 2025}, "2025", "2025-01-01", "2025-12-31"),
    ({"type": "quarter", "year": 2025, "quarter": 2}, "2025 Q2", "2025-04-01", "2025-06-30"),
    ({"type": "quarter", "year": 2025, "quarter": 4}, "2025 Q4", "2025-10-01", "2025-12-31"),
    ({"type": "month", "year": 2025, "month": 3}, "March 2025", "2025-03-01", "2025-03-31"),
    ({"type": "month", "year": 2024, "month": 2}, "February 2024", "2024-02-01", "2024-02-29"),
    ({"type": "month", "year": 2025, "month": 12}, "December 2025", "2025-12-01", "2025-12-31"),
    ({"type": "custom", "start": "2025-02-10", "end": "2025-03-05"}, "2025-02-10 to 2025-03-05", "2025-02-10", "2025-03-05"),
])
def test_periods_cover_exactly_their_calendar_span(spec, label, start, end):
    p = tm.parse_period(spec)
    assert p["label"] == label and str(p["start"].date()) == start and str(p["end"].date()) == end
    assert p["start"].time() == dt.time(0, 0) and p["end"].time() == dt.time(23, 59, 59, 999999)
    assert p["spec"]["type"] == spec["type"]


def test_all_data_has_no_bounds_and_no_filter():
    p = tm.parse_period(None)
    assert p["type"] == "all" and p["start"] is None and tm.period_filters("order_date", p) == []


@pytest.mark.parametrize("spec", [
    {"type": "decade"}, {"type": "year", "year": 1800}, {"type": "year", "year": "2025"}, {"type": "year", "year": True},
    {"type": "quarter", "year": 2025, "quarter": 5}, {"type": "quarter", "year": 2025}, {"type": "month", "year": 2025, "month": 13},
    {"type": "month", "year": 2025, "month": 0}, {"type": "custom", "start": "2025-05-01", "end": "2025-04-01"},
    {"type": "custom", "start": "not a date", "end": "2025-04-01"}, {"type": "custom"}, {"type": "month", "month": 3},
])
def test_invalid_periods_are_refused_with_a_reason(spec):
    with pytest.raises(tm.TimeError):
        tm.parse_period(spec)


def test_the_last_moment_of_a_period_is_inside_it_and_the_next_moment_is_not():
    df = pd.DataFrame({"d": pd.to_datetime(["2025-03-31 23:30:00", "2025-04-01 00:00:00", "2025-03-01 00:00:00", "2025-02-28 23:59:59"]), "v": [1, 2, 4, 8]})
    from app import metric_engine as me
    p = tm.parse_period({"type": "month", "year": 2025, "month": 3})
    got = me.apply_filters(df, tm.period_filters("d", p))
    assert sorted(got["v"]) == [1, 4]


def test_the_periods_on_offer_come_from_the_datas_own_calendar(rows):
    df = pd.DataFrame(rows).assign(order_date=lambda d: pd.to_datetime(d["order_date"]))
    a = tm.available_periods(df, "order_date")
    assert a["min"] == "2025-01-01" and a["max"] == "2026-03-28" and a["rows_with_date"] == 60 and a["rows_without_date"] == 0
    assert [y["year"] for y in a["years"]] == [2025, 2026] and [y["rows"] for y in a["years"]] == [48, 12]
    assert len(a["months"]) == 15 and all(m["rows"] == 4 for m in a["months"]) and len(a["quarters"]) == 5
    assert a["quarters"][-1] == {"year": 2026, "quarter": 1, "rows": 12}
    gaps = df[df["order_date"].dt.month != 6]
    assert not any(m["month"] == 6 and m["year"] == 2025 for m in tm.available_periods(gaps, "order_date")["months"]), "an empty month is not offered"
    with pytest.raises(tm.TimeError, match="not made of dates"):
        tm.available_periods(df, "region")
    with pytest.raises(tm.TimeError, match="does not exist"):
        tm.available_periods(df, "ghost")


# =========================================================== baseline by period ==

@pytest.mark.parametrize("period,start,end", [
    ({"type": "month", "year": 2025, "month": 3}, "2025-03-01", "2025-03-31"),
    ({"type": "quarter", "year": 2025, "quarter": 3}, "2025-07-01", "2025-09-30"),
    ({"type": "year", "year": 2025}, "2025-01-01", "2025-12-31"),
    ({"type": "custom", "start": "2025-02-10", "end": "2025-04-20"}, "2025-02-10", "2025-04-20"),
    ({"type": "all"}, "2000-01-01", "2100-01-01"),
])
def test_a_period_baseline_equals_the_sums_over_that_periods_rows(client, ws, rows, period, start, end):
    r = baseline(client, ws, period)
    assert r.status_code == 200, r.text
    b = r.json()
    want = totals(inside(rows, start, end))
    for k, v in want.items():
        assert b["inputs"][k] == pytest.approx(v, rel=1e-12), k
    assert b["inputs"]["fixed_cost"] == FIXED and b["sources"]["fixed_cost"]["source"] == "override"
    assert b["rows_used"] == len(inside(rows, start, end)) and b["data_kind"] == "historical_actual"
    assert b["period"]["type"] == period["type"] and b["period"]["rows"] == b["rows_used"] and b["period"]["date_column"] == "order_date"
    assert b["verification"]["status"] == "verified", [c for c in b["verification"]["checks"] if c["status"] != "pass"]
    assert set(b["data_kinds"]) == {"historical_actual", "scenario", "statistical_estimate", "forecast"}
    assert "Not produced" in b["data_kinds"]["forecast"]


def test_a_period_can_be_combined_with_other_filters(client, ws, rows):
    r = baseline(client, ws, {"type": "quarter", "year": 2025, "quarter": 1}, filters=[{"column": "region", "op": "==", "value": "West"}])
    assert r.status_code == 200
    want = totals(inside(rows, "2025-01-01", "2025-03-31", "West"))
    assert r.json()["inputs"]["volume"] == want["volume"] and r.json()["rows_used"] == 6
    assert r.json()["verification"]["status"] == "verified"


def test_the_period_and_its_provenance_are_returned_with_the_dataset_version(client, ws):
    b = baseline(client, ws, {"type": "month", "year": 2025, "month": 5}).json()
    p = b["provenance"]
    assert p["dataset_id"] == ws["ds"] and p["dataset_version"] == 1 and b["dataset_version"] == 1 and len(p["content_sha256"]) == 64
    assert b["period"]["data_covers"] == ["2025-01-01", "2026-03-28"] and b["period"]["start"] == "2025-05-01"


@pytest.mark.parametrize("period,date_column,code,fragment", [
    ({"type": "month", "year": 2024, "month": 5}, "order_date", 400, "No rows fall in May 2024. The data covers 2025-01-01 to 2026-03-28"),
    ({"type": "month", "year": 2025, "month": 5}, None, 400, "Choose the date column"),
    ({"type": "month", "year": 2025, "month": 5}, "ghost", 400, "does not exist"),
    ({"type": "month", "year": 2025, "month": 5}, "region", 400, "not made of dates"),
    ({"type": "month", "year": 2025, "month": 13}, "order_date", 400, "month must be"),
    ({"type": "forever"}, "order_date", 400, "period type"),
])
def test_unusable_periods_are_refused_not_turned_into_zeros(client, ws, period, date_column, code, fragment):
    body = {"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "period": period, "date_column": date_column}
    r = client.post("/api/scenarios/baseline", json=body, headers=ws["h"])
    assert r.status_code == code and fragment in r.json()["detail"], r.text


def test_all_data_needs_no_date_column_and_keeps_the_old_behaviour(client, ws, rows):
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP}, headers=ws["h"])
    assert r.status_code == 200 and r.json()["inputs"]["volume"] == totals(rows)["volume"] and r.json()["period"]["label"] == "All data"
    assert r.json()["missing"] == ["fixed_cost"]


def test_a_chosen_dataset_version_is_the_one_used(client, ws):
    v2 = make_rows(shift=5.0)
    up = client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", csv_of(v2), "text/csv")}, headers=ws["h"])
    assert up.status_code in (200, 201), up.text
    period = {"type": "month", "year": 2025, "month": 4}
    old = baseline(client, ws, period, dataset_version=1).json()
    new = baseline(client, ws, period).json()
    assert old["provenance"]["dataset_version"] == 1 and new["provenance"]["dataset_version"] == 2 and old["provenance"]["latest_version"] == 2
    assert new["inputs"]["price"] == pytest.approx(old["inputs"]["price"] + 5.0, rel=1e-9)
    assert old["provenance"]["content_sha256"] != new["provenance"]["content_sha256"]
    assert baseline(client, ws, period, dataset_version=9).status_code == 404


def test_a_baseline_that_does_not_match_the_file_is_not_verified(client, ws, rows):
    from app.database import SessionLocal
    from app.models import DatasetVersion
    b = baseline(client, ws, {"type": "month", "year": 2025, "month": 2}).json()
    with SessionLocal() as db:
        path = db.query(DatasetVersion).filter_by(dataset_id=ws["ds"]).first().filepath
    assert scenario_verify.verify_baseline(path, copy.deepcopy(b), [*tm.period_filters("order_date", tm.parse_period({"type": "month", "year": 2025, "month": 2}))])["status"] == "verified"
    wrong = copy.deepcopy(b)
    wrong["inputs"]["volume"] += 1
    v = scenario_verify.verify_baseline(path, wrong, tm.period_filters("order_date", tm.parse_period({"type": "month", "year": 2025, "month": 2})))
    assert v["status"] == "unverified" and {c["check"]: c["status"] for c in v["checks"]}["input_volume"] == "fail"


# ============================================================ period comparison ==

def compare(client, ws, a, b, **extra):
    body = {"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "overrides": {"fixed_cost": FIXED}, "date_column": "order_date",
            "period_a": a, "period_b": b, **extra}
    return client.post("/api/scenarios/compare-periods", json=body, headers=ws["h"])


def test_two_periods_are_compared_with_the_change_and_what_drove_it(client, ws, rows):
    a, b = {"type": "month", "year": 2025, "month": 1}, {"type": "month", "year": 2025, "month": 6}
    r = compare(client, ws, a, b)
    assert r.status_code == 200, r.text
    out = r.json()
    ta, tb = totals(inside(rows, "2025-01-01", "2025-01-31")), totals(inside(rows, "2025-06-01", "2025-06-30"))
    ref = lambda t: (t["price"] - t["unit_cost"]) * t["volume"] - FIXED  # noqa: E731
    assert out["a"]["outputs"]["profit"] == pytest.approx(ref(ta), rel=1e-9) and out["b"]["outputs"]["profit"] == pytest.approx(ref(tb), rel=1e-9)
    change = ref(tb) - ref(ta)
    assert out["output_changes"]["profit"]["abs"] == pytest.approx(change, rel=1e-9)
    assert out["input_changes"]["volume"]["abs"] == pytest.approx(tb["volume"] - ta["volume"]) and out["input_changes"]["fixed_cost"]["abs"] == 0
    contrib = out["contributions"]["by_input"]
    assert sum(contrib.values()) == pytest.approx(change, rel=1e-9) and "fixed_cost" not in contrib, "an input that did not move has no share"
    assert contrib["volume"] > 0 and contrib["price"] > 0, "volume and price both rose between January and June"
    assert out["data_kind"] == "historical_actual" and out["verification"]["status"] == "verified"
    assert out["periods"]["a"]["period"]["label"] == "January 2025" and out["periods"]["b"]["rows_used"] == 4
    assert out["provenance"]["dataset_version"] == 1 and any("not a forecast" in x for x in out["limits"])


def test_a_period_compared_with_itself_has_no_change(client, ws):
    p = {"type": "quarter", "year": 2025, "quarter": 2}
    out = compare(client, ws, p, p).json()
    assert out["contributions"]["by_input"] == {} and all(c["abs"] == 0 for c in out["input_changes"].values())
    assert out["output_changes"]["profit"]["abs"] == 0 and out["verification"]["status"] == "verified"


def test_comparison_refuses_incomplete_periods_and_empty_ones(client, ws):
    ok = {"type": "month", "year": 2025, "month": 2}
    r = client.post("/api/scenarios/compare-periods", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP,
                                                            "date_column": "order_date", "period_a": ok, "period_b": ok}, headers=ws["h"])
    assert r.status_code == 400 and "fixed_cost" in r.json()["detail"], "an input the data lacks is asked for, not assumed"
    empty = compare(client, ws, ok, {"type": "month", "year": 2019, "month": 1})
    assert empty.status_code == 400 and "No rows fall in January 2019" in empty.json()["detail"]
    assert compare(client, ws, ok, {"type": "weekly"}).status_code == 400
    assert compare(client, ws, ok, ok, date_column="ghost").status_code == 400


def test_a_tampered_comparison_is_not_verified(client, ws):
    out = compare(client, ws, {"type": "month", "year": 2025, "month": 1}, {"type": "month", "year": 2025, "month": 9}).json()
    for tamper in (lambda o: o["b"]["outputs"].__setitem__("profit", 1.0),
                   lambda o: o["output_changes"]["profit"].__setitem__("abs", 1.0),
                   lambda o: o["contributions"]["by_input"].__setitem__("volume", 1.0),
                   lambda o: o["input_changes"]["price"].__setitem__("pct", 99.0)):
        bad = copy.deepcopy(out)
        tamper(bad)
        assert scenario_verify.verify_period_comparison(bad)["status"] == "unverified"


def test_a_custom_model_compares_periods_too(client, ws, rows):
    d = {"inputs": {"units_sold": {"unit": "units", "min": 0, "source": {"type": "formula", "formula": "sum(units)"}},
                    "revenue_total": {"unit": "currency", "min": 0, "source": {"type": "formula", "formula": "sum(revenue)"}}},
         "outputs": {"revenue_per_unit": {"unit": "currency/units", "formula": "revenue_total / units_sold"}}, "primary_output": "revenue_per_unit"}
    m = client.post("/api/scenario-models", json={"name": "Yield", "definition": d}, headers=ws["h"]).json()
    r = client.post("/api/scenarios/compare-periods", json={"dataset_id": ws["ds"], "model": m["ref"], "date_column": "order_date",
                                                            "period_a": {"type": "quarter", "year": 2025, "quarter": 1},
                                                            "period_b": {"type": "quarter", "year": 2025, "quarter": 4}}, headers=ws["h"])
    assert r.status_code == 200, r.text
    out = r.json()
    ya = totals(inside(rows, "2025-01-01", "2025-03-31"))["price"]
    yb = totals(inside(rows, "2025-10-01", "2025-12-31"))["price"]
    assert out["a"]["outputs"]["revenue_per_unit"] == pytest.approx(ya) and out["b"]["outputs"]["revenue_per_unit"] == pytest.approx(yb)
    assert out["verification"]["status"] == "verified" and out["model"] == m["ref"]


def test_the_periods_endpoint_lists_what_the_data_covers(client, ws):
    r = client.get("/api/scenarios/periods", params={"dataset_id": ws["ds"], "date_column": "order_date"}, headers=ws["h"])
    assert r.status_code == 200 and len(r.json()["months"]) == 15 and r.json()["provenance"]["dataset_version"] == 1
    assert client.get("/api/scenarios/periods", params={"dataset_id": ws["ds"], "date_column": "region"}, headers=ws["h"]).status_code == 400
    assert client.get("/api/scenarios/periods", params={"dataset_id": ws["ds"], "date_column": "x", "dataset_version": 4}, headers=ws["h"]).status_code == 404


def test_time_endpoints_are_team_scoped_and_need_a_signed_in_member(client, ws):
    other = workspace(client)
    month = {"type": "month", "year": 2025, "month": 3}
    assert baseline(client, ws, month).status_code == 200
    assert client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics", "mapping": MAP, "period": month,
                                                        "date_column": "order_date"}, headers=other["h"]).status_code == 404
    assert compare(client, {**ws, "h": other["h"]}, month, month).status_code == 404
    assert client.get("/api/scenarios/periods", params={"dataset_id": ws["ds"], "date_column": "order_date"}, headers=other["h"]).status_code == 404
    assert client.get("/api/scenarios/periods", params={"dataset_id": ws["ds"], "date_column": "order_date"}).status_code == 401
    assert client.post("/api/scenarios/compare-periods", json={}).status_code == 401


# ================================================ provenance kept with a scenario ==

def test_a_saved_scenario_keeps_the_period_and_the_dataset_version_it_came_from(client, ws):
    period = {"type": "quarter", "year": 2025, "quarter": 2}
    b = baseline(client, ws, period).json()
    up = client.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", csv_of(make_rows(shift=3.0)), "text/csv")}, headers=ws["h"])
    assert up.status_code in (200, 201)
    inputs = {k: v for k, v in b["inputs"].items()}
    src = {"dataset_version": 1, "date_column": "order_date", "period": b["period"]["spec"], "mapping": MAP}
    saved = client.post("/api/scenarios", json={"model": "unit_economics", "baseline": inputs, "name": "Q2 plan", "dataset_id": ws["ds"],
                                                "scenarios": {"up": {"price": {"type": "pct", "value": 5}}}, "baseline_source": src}, headers=ws["h"])
    assert saved.status_code == 201, saved.text
    got = client.get(f"/api/scenarios/{saved.json()['id']}", headers=ws["h"]).json()
    kept = got["request"]["baseline_source"]
    assert kept["period"] == {"type": "quarter", "year": 2025, "quarter": 2} and kept["date_column"] == "order_date"
    assert kept["provenance"]["dataset_version"] == 1 and kept["provenance"]["latest_version"] == 2
    assert kept["provenance"]["content_sha256"] == b["provenance"]["content_sha256"], "the saved scenario points at the data it used"
    from app.database import SessionLocal
    from app.models import DatasetVersion
    with SessionLocal() as db:
        v1 = db.query(DatasetVersion).filter_by(dataset_id=ws["ds"], version_number=1).first()
    assert got["dataset_version_id"] == v1.id
    assert client.post("/api/scenarios", json={"model": "unit_economics", "baseline": inputs, "name": "Bad", "dataset_id": ws["ds"],
                                               "baseline_source": {"dataset_version": 9}}, headers=ws["h"]).status_code == 404
    huge = {"dataset_version": 1, "note": "x" * 30_000}
    assert client.post("/api/scenarios", json={"model": "unit_economics", "baseline": inputs, "name": "Huge", "dataset_id": ws["ds"],
                                               "baseline_source": huge}, headers=ws["h"]).status_code == 413


def test_every_result_says_what_kind_of_figure_it_is(client, ws):
    ev = client.post("/api/scenarios/evaluate", json={"model": "unit_economics", "baseline": {"price": 20, "volume": 100, "unit_cost": 12, "fixed_cost": 0}},
                     headers=ws["h"]).json()
    assert ev["data_kind"] == "scenario" and ev["kind"] == "scenario"
    b = baseline(client, ws, {"type": "month", "year": 2025, "month": 1}).json()
    assert b["data_kind"] == "historical_actual"
    assert scenarios.evaluate("unit_economics", {"price": 1, "volume": 1, "unit_cost": 1, "fixed_cost": 0})["data_kind"] == "scenario"
