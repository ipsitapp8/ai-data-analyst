"""What-if Phase 2, feature 5: two-way sensitivity grids (heatmaps) and scalable Shapley attribution.

Unit economics gives closed forms to check against: profit(p, v) = (p - 12) x v - 3000, so the break-even price at volume
v is exactly 12 + 3000 / v, and the break-even volume at price p is 3000 / (p - 12).
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pytest
from platform_helpers import workspace

from app import scenario_models as sm
from app import scenario_sensitivity as ss
from app import scenario_verify, scenarios

BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
FUNNEL = {"visitors": 10000.0, "conversion_rate": 0.02, "average_order_value": 50.0, "cost_per_visitor": 0.4}
WIDE = {"mode": "pct", "low": -50, "high": 50}


def grid(**kw):
    args = dict(model="unit_economics", baseline=BASE, x_input="price", y_input="volume", resolution=11)
    args.update(kw)
    return ss.two_way(**args)


# ================================================================= the values ==

def test_every_cell_is_the_models_output_for_that_pair_of_values():
    g = grid()
    assert g["x"]["values"] == pytest.approx(np.linspace(16, 24, 11)) and g["y"]["values"] == pytest.approx(np.linspace(800, 1200, 11))
    z = g["z"]
    assert len(z) == 11 and all(len(row) == 11 for row in z), "rows follow the y input, columns the x input"
    for i, j in ((0, 0), (0, 10), (10, 0), (5, 5), (10, 10), (3, 7)):
        p, v = g["x"]["values"][j], g["y"]["values"][i]
        assert z[i][j] == pytest.approx((p - 12) * v - 3000, rel=1e-12), (i, j)
    assert g["verification"]["status"] == "verified", [c for c in g["verification"]["checks"] if c["status"] != "pass"]
    assert g["data_kind"] == "scenario" and g["cells"] == 121 and g["empty_cells"] == 0


def test_the_best_and_worst_cells_are_the_extremes_and_are_said_to_be_grid_extremes_only():
    g = grid()
    assert g["best"]["value"] == pytest.approx((24 - 12) * 1200 - 3000) == pytest.approx(11400.0) and (g["best"]["x"], g["best"]["y"]) == (24.0, 1200.0)
    assert g["worst"]["value"] == pytest.approx((16 - 12) * 800 - 3000) == pytest.approx(200.0) and (g["worst"]["x"], g["worst"]["y"]) == (16.0, 800.0)
    assert any("not claimed to be a global optimum" in d for d in g["disclosures"])


def test_the_current_position_is_computed_exactly_not_read_off_the_nearest_cell():
    g = grid(adjustments={"unit_cost": {"type": "pct", "value": 25}}, resolution=10)       # 10 points: the centre is between cells
    assert g["current"] == {"x": 20.0, "y": 1000.0, "value": pytest.approx((20 - 15) * 1000 - 3000)}
    assert g["baseline_value"] == pytest.approx(5000.0)
    assert all(abs(x - 20.0) > 1e-9 for x in g["x"]["values"]), "the centre is not on the grid, yet its value is still exact"


def test_other_adjustments_stay_and_the_axes_are_centred_on_the_scenario_value():
    g = grid(adjustments={"price": {"type": "pct", "value": 10}, "unit_cost": {"type": "set", "value": 14.0}})
    assert g["x"]["center"] == pytest.approx(22.0) and g["x"]["values"][0] == pytest.approx(17.6) and g["x"]["values"][-1] == pytest.approx(26.4)
    p, v = g["x"]["values"][4], g["y"]["values"][7]
    assert g["z"][7][4] == pytest.approx((p - 14.0) * v - 3000), "the cost adjustment is kept; the price adjustment is replaced by the axis"
    assert any("replace any adjustment" in d for d in g["disclosures"])


def test_absolute_limits_are_accepted():
    g = grid(x_range={"mode": "absolute", "low": 10, "high": 30}, y_range={"mode": "absolute", "low": 500, "high": 1500})
    assert g["x"]["values"][0] == 10.0 and g["x"]["values"][-1] == 30.0 and g["y"]["values"][-1] == 1500.0
    assert g["z"][0][0] == pytest.approx((10 - 12) * 500 - 3000)


def test_price_elasticity_moves_volume_along_a_price_axis():
    g = grid(x_input="price", y_input="unit_cost", assumptions={"price_elasticity": -1.5}, y_range={"mode": "pct", "low": -25, "high": 25})
    for i, j in ((0, 0), (4, 6), (10, 10)):
        p, c = g["x"]["values"][j], g["y"]["values"][i]
        volume = max(0.0, 1000 * (1 + -1.5 * (p / 20 - 1)))
        assert g["z"][i][j] == pytest.approx((p - c) * volume - 3000, rel=1e-12)
    assert any("price elasticity of -1.5" in d for d in g["disclosures"]) and g["verification"]["status"] == "verified"
    flat = grid(x_input="price", y_input="volume", assumptions={"price_elasticity": -1.5})
    assert not any("price elasticity" in d for d in flat["disclosures"]), "volume is an axis here, so nothing links it to price"


def test_a_funnel_grid_and_other_outputs():
    g = ss.two_way("funnel", FUNNEL, "conversion_rate", "average_order_value", "revenue", resolution=7)
    i, j = 2, 5
    assert g["z"][i][j] == pytest.approx(10000 * g["x"]["values"][j] * g["y"]["values"][i], rel=1e-12) and g["output"]["label"] == "Revenue"
    margin = grid(output="margin_pct", resolution=5)
    p, v = margin["x"]["values"][2], margin["y"]["values"][3]
    assert margin["z"][3][2] == pytest.approx(((p - 12) * v - 3000) / (p * v) * 100, rel=1e-12) and margin["verification"]["status"] == "verified"


# ===================================================================== boundary ==

def test_the_break_even_boundary_lies_exactly_where_profit_is_zero():
    g = grid(x_range=WIDE, y_range=WIDE, resolution=21)
    b = g["boundary"]
    assert b["threshold"] == 0.0 and len(b["points"]) > 10 and b["cells_above"] > 0 and b["cells_below"] > 0
    assert b["cells_above"] + b["cells_below"] + b["cells_on"] == 441
    for pt in b["points"]:
        if pt["edge"] == "row":                    # profit is linear in price along a row, so interpolation is exact
            assert pt["x"] == pytest.approx(12 + 3000 / pt["y"], rel=1e-9)
        else:                                      # and linear in volume down a column
            assert pt["y"] == pytest.approx(3000 / (pt["x"] - 12), rel=1e-9)
    assert g["verification"]["status"] == "verified"


def test_another_threshold_moves_the_boundary():
    g = grid(x_range=WIDE, y_range=WIDE, resolution=21, threshold=5000.0)
    assert g["boundary"]["points"] and all(pt["x"] == pytest.approx(12 + 8000 / pt["y"], rel=1e-9) for pt in g["boundary"]["points"] if pt["edge"] == "row")
    none = grid(threshold=None)
    assert none["boundary"] is None and none["verification"]["status"] == "verified"


def test_no_crossing_is_said_plainly():
    g = grid()                                      # +-20%: profit stays positive everywhere
    assert g["boundary"]["points"] == [] and any("never crosses 0" in d for d in g["disclosures"])
    assert g["boundary"]["cells_below"] == 0


# ================================================================ invalid requests ==

@pytest.mark.parametrize("kw,fragment", [
    ({"x_input": "ghost"}, "Unknown x_input"), ({"y_input": "ghost"}, "Unknown y_input"), ({"y_input": "price"}, "two different inputs"),
    ({"output": "magic"}, "Unknown output"), ({"resolution": 2}, "resolution"), ({"resolution": 62}, "resolution"), ({"resolution": True}, "resolution"),
    ({"resolution": 5.5}, "resolution"),
    ({"x_range": {"mode": "pct", "low": 10, "high": -10}}, "must be below"), ({"x_range": {"mode": "pct", "low": -100, "high": 10}}, "-100%"),
    ({"x_range": {"mode": "pct", "low": -10}}, "needs"), ({"x_range": {"mode": "pct", "low": -10, "high": 10, "x": 1}}, "needs"),
    ({"x_range": {"mode": "weekly", "low": -10, "high": 10}}, "mode must be"), ({"x_range": "wide"}, "needs"),
    ({"x_range": {"mode": "pct", "low": float("nan"), "high": 10}}, "finite number"),
    ({"x_range": {"mode": "absolute", "low": -5, "high": 10}}, "goes outside its allowed range"),
    ({"threshold": float("nan")}, "finite number"), ({"model": "crystal_ball"}, "Unknown model"),
    ({"baseline": {**BASE, "price": -1.0}}, "cannot be below"),
    ({"adjustments": {"price": {"type": "pct", "value": -150}}}, "more than 100%"),
])
def test_invalid_grid_requests_are_refused_with_a_reason(kw, fragment):
    with pytest.raises(scenarios.ScenarioError, match=fragment):
        grid(**kw)


def test_a_percentage_range_around_zero_and_beyond_a_physical_limit_are_refused():
    with pytest.raises(ss.SensitivityError, match="use absolute limits"):
        grid(baseline={**BASE, "fixed_cost": 0.0}, x_input="fixed_cost", y_input="volume")
    with pytest.raises(ss.SensitivityError, match="goes outside its allowed range"):
        ss.two_way("funnel", FUNNEL, "conversion_rate", "average_order_value", x_range={"mode": "pct", "low": -10, "high": 5000})


# ====================================================================== custom ==

SAAS = {"inputs": {"customers": {"unit": "customers", "min": 0}, "price": {"unit": "currency/customers", "min": 0},
                   "unit_cost": {"unit": "currency/customers", "min": 0}},
        "outputs": {"profit": {"unit": "currency", "formula": "customers * (price - unit_cost)"}}, "primary_output": "profit",
        "constraints": [{"expr": "price >= unit_cost", "message": "Price must cover cost"}]}


def test_a_custom_models_grid_leaves_cells_that_break_a_constraint_empty():
    spec = sm.compile_spec(SAAS, key="custom:5@1", title="SaaS")
    g = ss.two_way(spec, {"customers": 100.0, "price": 30.0, "unit_cost": 20.0}, "price", "unit_cost", x_range={"mode": "pct", "low": -40, "high": 20},
                   y_range={"mode": "pct", "low": -20, "high": 40}, resolution=15)
    assert g["constraint_rejected_cells"] > 0 and g["empty_cells"] == g["constraint_rejected_cells"]
    for i, y in enumerate(g["y"]["values"]):
        for j, x in enumerate(g["x"]["values"]):
            assert (g["z"][i][j] is None) == (x < y - 1e-12), (x, y)
            if g["z"][i][j] is not None:
                assert g["z"][i][j] == pytest.approx(100 * (x - y))
    assert any("break one of the model's constraints" in d for d in g["disclosures"])
    assert g["verification"]["status"] == "verified", [c for c in g["verification"]["checks"] if c["status"] != "pass"]
    assert g["worst"]["value"] >= 0 and g["model"] == "custom:5@1"


# ================================================================ verification ==

def test_a_tampered_grid_is_not_verified():
    g = grid(x_range=WIDE, y_range=WIDE, resolution=15)
    assert scenario_verify.verify_two_way(copy.deepcopy(g))["status"] == "verified"
    tampers = [
        lambda o: o["z"][3].__setitem__(4, o["z"][3][4] + 1.0),
        lambda o: o["best"].__setitem__("value", o["best"]["value"] - 1.0),
        lambda o: o["worst"].__setitem__("value", o["worst"]["value"] + 1.0),
        lambda o: o["boundary"]["points"].pop(),
        lambda o: o["boundary"].__setitem__("cells_above", o["boundary"]["cells_above"] + 1),
        lambda o: o["boundary"]["points"][0].__setitem__("x", o["boundary"]["points"][0]["x"] + 0.5),
        lambda o: o["current"].__setitem__("value", 0.0),
        lambda o: o["z"][0].__setitem__(0, None),
    ]
    for tamper in tampers:
        bad = copy.deepcopy(g)
        tamper(bad)
        assert scenario_verify.verify_two_way(bad)["status"] == "unverified"


def test_the_largest_grid_is_fast_and_verified():
    started = time.perf_counter()
    g = grid(resolution=ss.MAX_RESOLUTION, x_range=WIDE, y_range=WIDE)
    assert g["cells"] == 3721 and g["verification"]["status"] == "verified" and time.perf_counter() - started < 20
    spec = sm.compile_spec(SAAS, key="custom:5@1", title="SaaS")
    started = time.perf_counter()
    c = ss.two_way(spec, {"customers": 100.0, "price": 30.0, "unit_cost": 20.0}, "price", "customers", resolution=ss.MAX_RESOLUTION)
    assert c["verification"]["status"] == "verified" and time.perf_counter() - started < 40


# ================================================== Shapley at scale, in practice ==

def test_attribution_stays_exact_to_ten_changed_inputs_and_is_sampled_beyond():
    names = [f"x{i}" for i in range(12)]
    d = {"inputs": {n: {"min": 0} for n in names}, "outputs": {"total": {"formula": " + ".join(f"{a} * {b}" for a, b in zip(names, names[1:] + names[:1]))}},
         "primary_output": "total"}
    spec = sm.compile_spec(d, key="custom:6@1", title="Ring12")
    base = {n: 1.0 for n in names}
    ten = scenarios.evaluate(spec, base, {"s": {n: {"type": "pct", "value": 20} for n in names[:10]}}, {})["scenarios"]["s"]["contributions"]
    twelve = scenarios.evaluate(spec, base, {"s": {n: {"type": "pct", "value": 20} for n in names}}, {})["scenarios"]["s"]["contributions"]
    assert ten["computation"]["method"] == "exact" and "approximate" not in ten["method"]
    assert twelve["computation"]["method"] == "sampled" and twelve["computation"]["samples"] == 2000 and "approximate" in twelve["method"]
    assert set(twelve["computation"]["standard_errors"]) == set(names)
    started = time.perf_counter()
    scenarios.evaluate(spec, base, {"s": {n: {"type": "pct", "value": 20} for n in names}}, {})
    assert time.perf_counter() - started < 10


# ===================================================================== the API ==

@pytest.fixture
def ws(client):
    return workspace(client)


def post(client, ws, **kw):
    body = {"model": "unit_economics", "baseline": BASE, "x_input": "price", "y_input": "volume", "resolution": 11, **kw}
    return client.post("/api/scenarios/sensitivity-grid", json=body, headers=ws["h"])


def test_grid_endpoint_returns_the_grid_with_its_check(client, ws):
    r = post(client, ws, x_range=WIDE, y_range=WIDE)
    assert r.status_code == 200, r.text
    g = r.json()
    assert g["verification"]["status"] == "verified" and g["best"]["value"] > g["worst"]["value"] and len(g["z"]) == 11
    assert g["boundary"]["points"] and g["current"]["value"] == pytest.approx(5000.0)


def test_grid_endpoint_works_for_a_custom_model_and_refuses_bad_requests(client, ws):
    m = client.post("/api/scenario-models", json={"name": "Grid model", "definition": SAAS}, headers=ws["h"]).json()
    ok = client.post("/api/scenarios/sensitivity-grid", json={"model": m["ref"], "baseline": {"customers": 100, "price": 30, "unit_cost": 20},
                                                              "x_input": "price", "y_input": "customers", "resolution": 9}, headers=ws["h"])
    assert ok.status_code == 200 and ok.json()["model"] == m["ref"] and ok.json()["verification"]["status"] == "verified"
    for patch, code in (({"x_input": "ghost"}, 400), ({"y_input": "price"}, 400), ({"resolution": 100}, 422), ({"resolution": 1}, 422),
                        ({"x_range": {"mode": "pct", "low": 5, "high": 1}}, 400), ({"model": "custom:999999"}, 404), ({"model": "custom:x"}, 400),
                        ({"baseline": {"price": 1}}, 400), ({"threshold": "high"}, 422)):
        r = post(client, ws, **patch)
        assert r.status_code == code, (patch, r.text)
        assert "Traceback" not in r.text


def test_grid_endpoint_is_team_scoped_and_needs_a_signed_in_member(client, ws):
    m = client.post("/api/scenario-models", json={"name": "Mine", "definition": SAAS}, headers=ws["h"]).json()
    other = workspace(client)
    body = {"model": m["ref"], "baseline": {"customers": 100, "price": 30, "unit_cost": 20}, "x_input": "price", "y_input": "customers"}
    assert client.post("/api/scenarios/sensitivity-grid", json=body, headers=other["h"]).status_code == 404
    assert client.post("/api/scenarios/sensitivity-grid", json=body).status_code == 401
    assert post(client, other).status_code == 200, "the built-in models are available to every team"
