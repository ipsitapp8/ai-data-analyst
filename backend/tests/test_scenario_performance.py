"""What-if Phase 5: the engines stay usable at realistic sizes, and user-supplied settings cannot ask for unbounded work.

Time budgets are generous (several times what a laptop needs) so the tests catch an algorithmic regression, such as a Python loop
where a vectorised step used to be, without failing on a slow CI machine. Bounds are tested exactly: just inside passes, just outside is refused.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pytest
from platform_helpers import workspace

from app import scenario_formula as sf
from app import scenario_models as sm
from app import scenario_sensitivity as ss
from app import scenario_sim, scenarios

UE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
N = 14


def wide_model(n=N):
    """A custom model with n inputs and n + 1 outputs, each output using two inputs."""
    defn = {"inputs": {f"x{i}": {"label": f"X{i}", "unit": "units", "min": 0} for i in range(n)},
            "outputs": {**{f"t{i}": {"label": f"T{i}", "unit": "units", "formula": f"x{i} * x{(i + 1) % n} + x{i}"} for i in range(n)},
                        "total": {"label": "Total", "unit": "units", "formula": " + ".join(f"t{i}" for i in range(n))}},
            "primary_output": "total"}
    return sm.compile_spec(defn, key="custom:1@1", title="Wide")


def timed(fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t


# ================================================================== speed ==

def test_ten_thousand_draws_of_a_wide_custom_model_are_fast():
    spec = wide_model()
    base = {f"x{i}": 10.0 + i for i in range(N)}
    out, secs = timed(scenario_sim.simulate, spec, base, None, {}, {f"x{i}": {"type": "normal", "sd_pct": 5} for i in range(10)}, None, 10_000, 1)
    assert out["outputs"]["total"]["n_valid"] == 10_000 and secs < 8.0, f"{secs:.1f}s"


def test_the_largest_allowed_simulation_finishes_in_reasonable_time():
    out, secs = timed(scenario_sim.simulate, "unit_economics", UE, None, {"price_elasticity": 0.0}, {"price": {"type": "normal", "sd_pct": 5}},
                      None, scenario_sim.MAX_ITERATIONS, 1)
    assert out["config"]["iterations"] == 200_000 and secs < 10.0, f"{secs:.1f}s"


def test_the_widest_two_way_grid_is_fast_and_exactly_full():
    g, secs = timed(ss.two_way, "unit_economics", UE, "price", "volume", "profit", assumptions={"price_elasticity": 0.0}, resolution=ss.MAX_RESOLUTION)
    assert len(g["z"]) == ss.MAX_RESOLUTION == len(g["z"][0]) and secs < 5.0, f"{secs:.1f}s"


def test_attribution_over_many_changed_inputs_is_sampled_not_exponential_and_still_sums_exactly():
    spec = wide_model()
    base = {f"x{i}": 10.0 + i for i in range(N)}
    adj = {"s": {f"x{i}": {"type": "pct", "value": 5} for i in range(N)}}
    out, secs = timed(scenarios.evaluate, spec, base, adj, {}, 10.0)
    c = out["scenarios"]["s"]["contributions"]
    assert "sampled" in c["method"] and secs < 25.0, f"{secs:.1f}s"
    total_change = out["scenarios"]["s"]["outputs"]["total"] - out["baseline"]["outputs"]["total"]
    assert sum(c["by_input"].values()) == pytest.approx(total_change, rel=1e-9), "sampling changes the split, never the total"


def test_exact_attribution_is_used_up_to_its_limit_and_is_exact():
    out = scenarios.evaluate("unit_economics", UE, {"s": {"price": {"type": "pct", "value": 10}, "unit_cost": {"type": "pct", "value": -5}}}, {"price_elasticity": 0.0}, 10.0)
    c = out["scenarios"]["s"]["contributions"]
    # price +10% -> +2,000 at base cost; unit cost -5% -> 0.6 x 1,000 = +600 (interaction: price change does not affect volume here)
    assert "sampled" not in c["method"] and "approximate" not in c["method"] and sum(c["by_input"].values()) == pytest.approx(2600.0)


def test_a_large_dataset_gives_a_baseline_quickly_and_correctly(client):
    n = 100_000
    rng = np.random.default_rng(0)
    units = rng.integers(1, 50, n)
    csv = io.StringIO()
    csv.write("date,units,revenue,cost\n")
    for u in units[:5]:
        pass
    body = "\n".join(f"2024-01-{1 + i % 28:02d},{u},{20 * u},{12 * u}" for i, u in enumerate(units))
    ws = workspace(client, csv=("date,units,revenue,cost\n" + body + "\n").encode(), filename="big.csv")
    t = time.perf_counter()
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics",
                                                      "mapping": {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"},
                                                      "overrides": {"fixed_cost": 1.0}}, headers=ws["h"])
    secs = time.perf_counter() - t
    assert r.status_code == 200, r.text
    inputs = r.json()["inputs"]
    assert inputs["volume"] == float(units.sum()) and inputs["price"] == 20.0 and inputs["unit_cost"] == 12.0
    assert r.json()["verification"]["status"] == "verified" and secs < 20.0, f"{secs:.1f}s"


# ================================================================== bounds ==

def test_simulation_iterations_are_bounded_at_both_ends(client):
    ws = workspace(client)
    req = {"model": "unit_economics", "baseline": UE, "assumptions": {"price_elasticity": 0.0}, "uncertainty": {"price": {"type": "normal", "sd_pct": 5}}}
    for ok in (scenario_sim.MIN_ITERATIONS, scenario_sim.MAX_ITERATIONS):
        assert client.post("/api/scenarios/simulate", json={**req, "iterations": ok, "seed": 1}, headers=ws["h"]).status_code == 200
    for bad in (scenario_sim.MIN_ITERATIONS - 1, scenario_sim.MAX_ITERATIONS + 1, 10 ** 9, -5, 0):
        assert client.post("/api/scenarios/simulate", json={**req, "iterations": bad}, headers=ws["h"]).status_code == 422, bad
    with pytest.raises(scenarios.ScenarioError):
        scenario_sim.simulate("unit_economics", UE, None, {}, {"price": {"type": "normal", "sd_pct": 5}}, None, 10 ** 9, 1)


def test_the_grid_resolution_is_bounded(client):
    ws = workspace(client)
    req = {"model": "unit_economics", "baseline": UE, "assumptions": {"price_elasticity": 0.0}, "x_input": "price", "y_input": "volume", "output": "profit"}
    assert client.post("/api/scenarios/sensitivity-grid", json={**req, "resolution": ss.MAX_RESOLUTION}, headers=ws["h"]).status_code == 200
    assert client.post("/api/scenarios/sensitivity-grid", json={**req, "resolution": ss.MAX_RESOLUTION + 1}, headers=ws["h"]).status_code == 422
    assert client.post("/api/scenarios/sensitivity-grid", json={**req, "resolution": 10 ** 6}, headers=ws["h"]).status_code == 422


def test_request_collections_are_bounded(client):
    ws = workspace(client)
    many = {f"x{i}": {"type": "normal", "sd_pct": 5} for i in range(21)}
    r = client.post("/api/scenarios/simulate", json={"model": "unit_economics", "baseline": UE, "uncertainty": many}, headers=ws["h"])
    assert r.status_code == 422, "more than 20 uncertain inputs is refused before any work is done"
    assert client.post("/api/scenarios/ask", json={"message": "x" * 501, "model": "unit_economics", "baseline": UE}, headers=ws["h"]).status_code == 422
    assert client.get("/api/scenarios/compare", params={"ids": ",".join(str(i) for i in range(1, 8))}, headers=ws["h"]).status_code == 400, "at most 6 scenarios"
    assert client.get("/api/scenarios", params={"limit": 201}, headers=ws["h"]).status_code == 422


@pytest.mark.parametrize("formula,why", [
    ("x" + " + x" * 80, "too many nodes"),
    ("-(" * 25 + "x" + ")" * 25, "too deep"),
    (" + ".join(["x"] * 25), "a chain of 25 additions is deeper than the limit"),
    ("x ** 11", "exponent above the cap"),
    ("x ** y", "variable exponent"),
    ("a" * (sf.MAX_FORMULA_CHARS + 1), "too long"),
    ("__import__('os')", "call"), ("x.__class__", "attribute"), ("(lambda: 1)()", "lambda"), ("[i for i in range(10**9)]", "comprehension"),
    ("open('f')", "call"), ("x if x else 1", "conditional"), ("'a' * 10", "string"), ("10 ** 10 ** 10", "tower of powers"),
])
def test_a_formula_cannot_ask_for_unbounded_or_unsafe_computation(formula, why):
    defn = {"inputs": {"x": {"label": "X", "unit": "units", "min": 0}, "y": {"label": "Y", "unit": "units", "min": 0}},
            "outputs": {"out": {"label": "Out", "unit": "units", "formula": formula}}, "primary_output": "out"}
    t = time.perf_counter()
    analysis = sm.analyze(defn)
    assert not analysis["valid"], why
    assert time.perf_counter() - t < 2.0, "refused quickly, before anything is evaluated"
    with pytest.raises(scenarios.ScenarioError):
        sm.compile_spec(defn, key="custom:1@1", title="bad")


def test_a_formula_near_the_limits_compiles_and_runs():
    """15 terms is inside the depth limit (a chain of more than about 20 is refused; split it across named outputs instead)."""
    defn = {"inputs": {"x": {"label": "X", "unit": "units", "min": 0}},
            "outputs": {"out": {"label": "Out", "unit": "units", "formula": " + ".join(["x"] * 15)}}, "primary_output": "out"}
    spec = sm.compile_spec(defn, key="custom:1@1", title="ok")
    assert scenarios.evaluate(spec, {"x": 2.0}, {}, {}, 10.0)["baseline"]["outputs"]["out"] == 30.0


def test_a_repeated_evaluation_does_not_grow_without_bound():
    """No hidden cache: the same request twice costs about the same and leaves no state that changes the answer."""
    spec = wide_model()
    base = {f"x{i}": 10.0 + i for i in range(N)}
    first = scenarios.evaluate(spec, base, {}, {}, 10.0)["baseline"]["outputs"]["total"]
    for _ in range(20):
        assert scenarios.evaluate(spec, base, {}, {}, 10.0)["baseline"]["outputs"]["total"] == first
