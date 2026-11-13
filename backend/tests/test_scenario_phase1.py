"""What-if Phase 1: break-even, Monte Carlo simulation, goal seek and their independent verification.

Every expected number below is derived on paper or with plain Python from the model's formulas, never read back from
the code under test: the closed forms are written out in the comments.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import pytest
from platform_helpers import workspace

from app import scenario_sim, scenario_solve, scenario_verify, scenarios

BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
FUNNEL = {"visitors": 10000.0, "conversion_rate": 0.02, "average_order_value": 50.0, "cost_per_visitor": 0.4}


@pytest.fixture
def ws(client):
    return workspace(client)


def sim(**kw):
    args = dict(model="unit_economics", baseline=BASE, uncertainty={"price": {"type": "normal", "sd_pct": 5}},
                iterations=5000, seed=1)
    args.update(kw)
    return scenario_sim.simulate(**args)


# ============================================================ vector formulas ==

@pytest.mark.parametrize("model", ["unit_economics", "funnel"])
def test_array_formulas_equal_the_scalar_formulas(model):
    rng = np.random.default_rng(3)
    inputs = {k: rng.uniform(0.01, 1.0, 300) if "max" in r else rng.uniform(0.1, 500.0, 300)
              for k, r in scenarios.MODELS[model]["inputs"].items()}
    vec = scenarios.outputs_vec(model, inputs, {})
    for j in range(300):
        scalar = scenarios._outputs(model, {k: float(v[j]) for k, v in inputs.items()}, {})
        for k, want in scalar.items():
            assert math.isclose(float(vec[k][j]), want, rel_tol=1e-12, abs_tol=1e-12), (k, j)


def test_a_margin_on_zero_revenue_is_not_a_number_never_zero():
    vec = scenarios.outputs_vec("unit_economics", {**{k: np.array([v]) for k, v in BASE.items()}, "price": np.array([0.0])}, {})
    assert np.isnan(vec["margin_pct"][0])
    assert scenarios._outputs("unit_economics", {**BASE, "price": 0.0}, {})["margin_pct"] is None


# ================================================================= break-even ==

def test_unit_economics_break_even_follows_fixed_cost_over_contribution():
    be = scenarios.break_even("unit_economics", BASE)
    # contribution 20 - 12 = 8; 3000 / 8 = 375 units; 375 x 20 = 7500 revenue; 1000 - 375 = 625 above; 625/1000 = 62.5%
    assert be["status"] == "reachable" and be["break_even_quantity"] == 375.0 and be["break_even_revenue"] == 7500.0
    assert be["distance_units"] == 625.0 and be["additional_units_needed"] == 0.0 and be["safety_margin_pct"] == 62.5
    assert be["above_break_even"] is True
    # profit really is zero at that quantity, by the scenario engine itself
    assert math.isclose(scenarios._outputs("unit_economics", {**BASE, "volume": 375.0}, {})["profit"], 0.0, abs_tol=1e-9)


def test_below_break_even_reports_the_units_still_needed():
    be = scenarios.break_even("unit_economics", {**BASE, "volume": 200.0})
    assert be["above_break_even"] is False and be["additional_units_needed"] == 175.0
    assert be["distance_units"] == -175.0 and be["safety_margin_pct"] == -87.5


@pytest.mark.parametrize("override,status", [
    ({"price": 12.0}, "never"),                     # a unit adds nothing towards fixed cost
    ({"price": 10.0}, "never"),                     # a unit loses money
    ({"price": 12.0, "fixed_cost": 0.0}, "flat"),   # zero everywhere
])
def test_break_even_that_does_not_exist_says_so_and_returns_no_number(override, status):
    be = scenarios.break_even("unit_economics", {**BASE, **override})
    assert be["status"] == status and "break_even_quantity" not in be and be["explanation"]


def test_no_fixed_cost_breaks_even_at_zero_units():
    be = scenarios.break_even("unit_economics", {**BASE, "fixed_cost": 0.0})
    assert be["break_even_quantity"] == 0.0 and be["safety_margin_pct"] == 100.0


def test_funnel_break_even_is_a_threshold_because_the_model_has_no_fixed_cost():
    be = scenarios.break_even("funnel", FUNNEL)
    # 0.4 / 50 = 0.008 conversion; 0.4 / 0.02 = 20 order value; 0.02 x 50 = 1.0 cost per visitor
    assert be["status"] == "profitable_at_any_scale" and be["break_even_visitors"] == 0.0
    assert math.isclose(be["break_even_conversion_rate"], 0.008) and math.isclose(be["break_even_average_order_value"], 20.0)
    assert math.isclose(be["break_even_cost_per_visitor"], 1.0) and "no fixed cost" in be["explanation"]
    assert math.isclose(be["safety_margin_pct"]["conversion_rate"], 60.0)
    losing = scenarios.break_even("funnel", {**FUNNEL, "conversion_rate": 0.005})
    assert losing["status"] == "never" and losing["break_even_visitors"] is None


def test_break_even_curve_crosses_zero_where_the_break_even_is():
    be = scenarios.break_even("unit_economics", BASE)
    x, profit = np.array(be["curve"]["x"]), np.array(be["curve"]["profit"])
    assert profit[0] == -3000.0 and profit[-1] > 0 and be["curve"]["break_even_x"] == 375.0
    crossing = x[np.argmax(profit >= 0)]
    assert crossing >= 375.0 and x[np.argmax(profit >= 0) - 1] < 375.0


def test_evaluate_stays_compatible_and_adds_break_even():
    out = scenarios.evaluate("unit_economics", BASE, {"up": {"price": {"type": "pct", "value": 10}}})
    for key in ("kind", "disclaimer", "baseline", "scenarios", "sensitivity", "formulas", "unsupported_assumptions"):
        assert key in out
    assert out["engine_version"] and out["break_even"]["baseline"]["break_even_quantity"] == 375.0
    assert out["break_even"]["scenarios"]["up"]["break_even_quantity"] == 3000 / 10.0   # price 22: contribution 10


# ======================================================== evaluation verification ==

def test_a_normal_evaluation_verifies_every_check():
    out = scenarios.evaluate("unit_economics", BASE, scenarios.standard_scenarios("unit_economics", 10),
                             {"price_elasticity": -1.2})
    v = scenario_verify.verify_evaluation(out)
    assert v["status"] == "verified", [c for c in v["checks"] if c["status"] != "pass"]
    assert {c["check"] for c in v["checks"]} >= {"baseline_outputs", "scenario_outputs", "shapley_contributions",
                                                 "sensitivity_table", "break_even"}
    f = scenarios.evaluate("funnel", FUNNEL, {"better": {"conversion_rate": {"type": "pct", "value": 50}}})
    assert scenario_verify.verify_evaluation(f)["status"] == "verified"


@pytest.mark.parametrize("tamper", [
    lambda o: o["baseline"]["outputs"].__setitem__("profit", 1.0),
    lambda o: o["scenarios"]["up"]["outputs"].__setitem__("revenue", 1.0),
    lambda o: o["scenarios"]["up"]["contributions"]["by_input"].__setitem__("price", 1.0),
    lambda o: o["sensitivity"]["rows"][0].__setitem__("low", 1.0),
    lambda o: o["break_even"]["baseline"].__setitem__("break_even_quantity", 1.0),
])
def test_a_tampered_evaluation_is_not_reported_as_verified(tamper):
    out = scenarios.evaluate("unit_economics", BASE, {"up": {"price": {"type": "pct", "value": 10}}})
    tampered = copy.deepcopy(out)
    tamper(tampered)
    assert scenario_verify.verify_evaluation(tampered)["status"] == "unverified"


def test_a_verifier_that_cannot_run_marks_the_result_unverified():
    def broken(_):
        raise RuntimeError("boom")
    v = scenario_verify.safe(broken, {})
    assert v["status"] == "unverified" and "could not run" in v["error"]


# =============================================================== Monte Carlo ==

def test_the_same_seed_reproduces_the_run_exactly_and_another_seed_does_not():
    a, b, c = sim(seed=11), sim(seed=11), sim(seed=12)
    assert a["target"]["histogram"] == b["target"]["histogram"] and a["outputs"] == b["outputs"]
    assert a["outputs"]["profit"]["mean"] != c["outputs"]["profit"]["mean"]
    assert a["config"]["seed"] == 11 and a["config"]["iterations"] == 5000


def test_a_missing_seed_is_chosen_returned_and_replayable():
    a = sim(seed=None)
    assert isinstance(a["config"]["seed"], int)
    assert sim(seed=a["config"]["seed"])["outputs"]["profit"] == a["outputs"]["profit"]


def test_normal_price_uncertainty_gives_the_analytic_mean_spread_and_probability():
    # profit = 1000 x (price - 12) - 3000 with price ~ Normal(20, 5): mean 5000, sd 5000, P(profit > 0) = P(price > 15)
    out = sim(uncertainty={"price": {"type": "normal", "sd_pct": 25}}, iterations=100_000)
    p = out["outputs"]["profit"]
    assert abs(p["mean"] - 5000) < 4 * 5000 / math.sqrt(100_000) * 1.2     # truncation at price 0 moves it a hair
    assert abs(p["std"] - 5000) / 5000 < 0.03
    pos = out["target"]["probabilities"]["positive"]
    assert abs(pos["probability"] - 0.8413) < 4 * pos["standard_error"] + 5e-4
    assert out["target"]["low"] < out["target"]["median"] < out["target"]["high"]
    assert p["min"] <= p["p5"] <= p["p25"] <= p["median"] <= p["p75"] <= p["p95"] <= p["max"]


def test_uniform_and_triangular_stay_inside_their_ranges_with_the_right_mean():
    u = sim(uncertainty={"price": {"type": "uniform", "low_pct": -10, "high_pct": 10}}, iterations=50_000)
    # price in [18, 22] -> profit = 1000 x (price - 12) - 3000 in [3000, 7000]; mean 5000
    assert 3000 - 1e-6 <= u["outputs"]["profit"]["min"] and u["outputs"]["profit"]["max"] <= 7000 + 1e-6
    assert abs(u["outputs"]["profit"]["mean"] - 5000) < 30
    t = sim(uncertainty={"price": {"type": "triangular", "low_pct": -10, "mode_pct": 0, "high_pct": 30}}, iterations=100_000)
    # price triangular(18, 20, 26): mean (18+20+26)/3; profit mean = 1000 x (mean - 12) - 3000
    assert abs(t["outputs"]["profit"]["mean"] - (1000 * ((18 + 20 + 26) / 3 - 12) - 3000)) < 20


def test_absolute_parameters_work_and_cannot_be_mixed_with_percentages():
    out = sim(uncertainty={"price": {"type": "uniform", "low": 19.0, "high": 21.0}}, iterations=20_000)
    # price in [19, 21] -> profit in [4000, 6000]
    assert 4000 - 1e-6 <= out["outputs"]["profit"]["min"] and out["outputs"]["profit"]["max"] <= 6000 + 1e-6
    with pytest.raises(scenario_sim.SimulationError, match="not both"):
        sim(uncertainty={"price": {"type": "uniform", "low_pct": -5, "high": 21.0}})


def test_a_distribution_is_truncated_to_the_allowed_range_and_says_so():
    out = sim(model="funnel", baseline=FUNNEL, uncertainty={"conversion_rate": {"type": "normal", "sd_pct": 100}},
              iterations=20_000)
    # Normal(0.02, 0.02): P(< 0) = Phi(-1) = 0.1587 is cut off
    assert abs(out["truncated_probability"]["conversion_rate"] - 0.1587) < 1e-3
    assert any("truncated" in d for d in out["disclosures"])
    assert out["outputs"]["orders"]["min"] >= 0 and out["verification"]["status"] == "verified"
    raw = scenario_sim._raw(scenario_sim._prepare("funnel", FUNNEL, {}, {}, {"conversion_rate": {"type": "normal", "sd_pct": 100}},
                                                   None, 5000, 1, None, None, "at_least"))
    cr = raw["samples"]["conversion_rate"]
    assert cr.min() >= 0 and cr.max() <= 1 and (cr == 0).mean() < 0.001      # truncated, not piled up at the limit


def test_target_probabilities_and_the_exceeds_baseline_event():
    out = sim(uncertainty={"price": {"type": "normal", "sd_pct": 25}}, iterations=50_000, target_value=5000)
    # profit >= 5000 <=> price >= 20, the mean: about one half; exceeding the baseline profit of 5000 is the strict version
    reach = out["target"]["probabilities"]["reaches_target"]
    assert abs(reach["probability"] - 0.5) < 4 * reach["standard_error"] + 1e-3 and reach["target"] == 5000.0
    assert out["target"]["probabilities"]["exceeds_baseline"]["probability"] <= reach["probability"]
    at_most = sim(uncertainty={"price": {"type": "normal", "sd_pct": 25}}, iterations=20_000, target_value=0, target_direction="at_most")
    assert abs(at_most["target"]["probabilities"]["reaches_target"]["probability"]
               - at_most["target"]["probabilities"]["negative"]["probability"]) < 0.01
    assert "reaches_target" in out["target"]["probabilities"] and sim()["target"]["probabilities"]["reaches_target"] is None


def test_correlation_changes_the_spread_and_is_checked_against_the_request():
    unc = {"price": {"type": "normal", "sd_pct": 10}, "volume": {"type": "normal", "sd_pct": 10}}
    independent = sim(uncertainty=unc, iterations=40_000)
    negative = sim(uncertainty=unc, iterations=40_000,
                   correlation={"inputs": ["price", "volume"], "matrix": [[1, -0.8], [-0.8, 1]]})
    # revenue = price x volume: opposite moves cancel, so the spread of revenue falls
    assert negative["outputs"]["revenue"]["std"] < 0.8 * independent["outputs"]["revenue"]["std"]
    chk = {c["check"]: c for c in negative["verification"]["checks"]}
    assert chk["correlation"]["status"] == "pass" and negative["verification"]["status"] == "verified"
    assert any("Gaussian copula" in d for d in negative["disclosures"])
    assert any("independently" in d for d in independent["disclosures"]) is True


@pytest.mark.parametrize("corr,message", [
    ({"inputs": ["price", "volume"], "matrix": [[1, 0.5], [0.4, 1]]}, "symmetric"),
    ({"inputs": ["price", "volume"], "matrix": [[1, 0.5], [0.5, 0.9]]}, "diagonal"),
    ({"inputs": ["price", "volume"], "matrix": [[1, 1.5], [1.5, 1]]}, "between -1 and 1"),
    ({"inputs": ["price", "volume"], "matrix": [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1]]}, "2 × 2"),
    ({"inputs": ["price", "unit_cost"], "matrix": [[1, 0.5], [0.5, 1]]}, "needs an uncertainty distribution"),
    ({"inputs": ["price", "price"], "matrix": [[1, 0.5], [0.5, 1]]}, "repeat"),
    ({"inputs": ["price"], "matrix": [[1]]}, "at least two"),
    ({"inputs": ["price", "volume"]}, "2 × 2"),
])
def test_invalid_correlation_matrices_are_rejected_with_the_reason(corr, message):
    unc = {"price": {"type": "normal", "sd_pct": 5}, "volume": {"type": "normal", "sd_pct": 5}}
    with pytest.raises(scenario_sim.SimulationError, match=message):
        sim(uncertainty=unc, correlation=corr)


def test_a_matrix_that_is_not_positive_semidefinite_is_rejected():
    unc = {k: {"type": "normal", "sd_pct": 5} for k in ("price", "volume", "unit_cost")}
    impossible = [[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]]       # A~B, A~C strongly but B~C strongly opposite
    with pytest.raises(scenario_sim.SimulationError, match="positive semidefinite"):
        sim(uncertainty=unc, correlation={"inputs": ["price", "volume", "unit_cost"], "matrix": impossible})


@pytest.mark.parametrize("kwargs,message", [
    ({"uncertainty": {}}, "No ranges are assumed"),
    ({"uncertainty": {"price": {"type": "fixed"}}}, "Every input was marked fixed"),
    ({"uncertainty": {"ghost": {"type": "normal", "sd_pct": 5}}}, "Unknown input"),
    ({"uncertainty": {"price": {"type": "lognormal", "sd_pct": 5}}}, "Unknown distribution"),
    ({"uncertainty": {"price": {"type": "normal"}}}, "needs 'sd"),
    ({"uncertainty": {"price": {"type": "normal", "sd_pct": 0}}}, "above 0"),
    ({"uncertainty": {"price": {"type": "normal", "sd_pct": 5, "bogus": 1}}}, "Unknown parameter"),
    ({"uncertainty": {"price": {"type": "uniform", "low_pct": 5, "high_pct": -5}}}, "low must be below high"),
    ({"uncertainty": {"price": {"type": "triangular", "low_pct": -5, "mode_pct": 9, "high_pct": 5}}}, "most likely"),
    ({"uncertainty": {"price": {"type": "normal", "sd_pct": "five"}}}, "finite number"),
    ({"uncertainty": {"price": {"type": "normal", "sd_pct": float("nan")}}}, "finite number"),
    ({"uncertainty": {"price": "wide"}}, "must be an object"),
    ({"iterations": 50}, "iterations"), ({"iterations": 10 ** 7}, "iterations"), ({"iterations": True}, "iterations"),
    ({"seed": -1}, "seed"), ({"seed": 2 ** 32}, "seed"), ({"seed": True}, "seed"),
    ({"target_output": "magic"}, "Unknown output"), ({"target_direction": "around"}, "target_direction"),
    ({"target_value": float("inf")}, "finite number"),
    ({"adjustments": {"price": {"type": "pct", "value": -150}}}, "more than 100%"),
])
def test_invalid_simulation_requests_are_refused_with_a_reason(kwargs, message):
    with pytest.raises(scenarios.ScenarioError, match=message):
        sim(**kwargs)


def test_a_percentage_spread_of_a_zero_input_is_refused_not_silently_zero():
    with pytest.raises(scenario_sim.SimulationError, match="absolute units"):
        sim(baseline={**BASE, "fixed_cost": 0.0}, uncertainty={"fixed_cost": {"type": "normal", "sd_pct": 10}})
    out = sim(baseline={**BASE, "fixed_cost": 0.0}, uncertainty={"fixed_cost": {"type": "normal", "sd": 500}})
    # a cost cannot be negative, so Normal(0, 500) is cut at 0: a half-normal, whose sd is 500 x sqrt(1 - 2/pi) = 301.4
    assert abs(out["outputs"]["profit"]["std"] - 500 * math.sqrt(1 - 2 / math.pi)) < 6
    assert any("50.00% of the stated distribution lay outside" in d for d in out["disclosures"])


def test_elasticity_moves_volume_with_each_sampled_price_unless_volume_is_uncertain():
    unc = {"price": {"type": "normal", "sd_pct": 5}}
    ctx = scenario_sim._prepare("unit_economics", BASE, {}, {"price_elasticity": -1.5}, unc, None, 2000, 5, None, None, "at_least")
    s, _ = scenario_sim._draw(ctx)
    want = np.maximum(0.0, 1000.0 * (1 + -1.5 * (s["price"] / 20.0 - 1)))
    assert np.allclose(s["volume"], want) and s["volume"].std() > 0
    out = scenario_sim.simulate("unit_economics", BASE, {}, {"price_elasticity": -1.5}, unc, iterations=2000, seed=5)
    assert any("price elasticity of -1.5" in d for d in out["disclosures"]) and out["verification"]["status"] == "verified"
    both = {**unc, "volume": {"type": "normal", "sd_pct": 5}}
    ctx2 = scenario_sim._prepare("unit_economics", BASE, {}, {"price_elasticity": -1.5}, both, None, 2000, 5, None, None, "at_least")
    s2, _ = scenario_sim._draw(ctx2)
    assert abs(np.corrcoef(s2["price"], s2["volume"])[0, 1]) < 0.1
    assert any("not applied" in d for d in scenario_sim.simulate(
        "unit_economics", BASE, {}, {"price_elasticity": -1.5}, both, iterations=1000, seed=5)["disclosures"])


def test_the_scenario_adjustments_set_the_centre_of_the_simulation():
    out = sim(adjustments={"price": {"type": "pct", "value": 10}}, uncertainty={"unit_cost": {"type": "normal", "sd_pct": 5}},
              iterations=40_000)
    # price 22 fixed, unit cost ~ N(12, 0.6): profit mean = (22-12) x 1000 - 3000 = 7000
    assert out["scenario_inputs"]["price"] == 22.0 and abs(out["outputs"]["profit"]["mean"] - 7000) < 10
    assert out["fixed_inputs"] == ["price", "volume", "fixed_cost"]
    assert out["target"]["deterministic_scenario_value"] == 7000.0 and out["target"]["baseline_value"] == 5000.0


def test_the_result_states_what_is_assumed_and_what_was_held_fixed():
    out = sim()
    text = " ".join(out["disclosures"])
    assert "assumptions you supplied" in text and "Held fixed" in text and out["kind"] == "simulation" and out["disclaimer"]
    assert set(out["uncertain_inputs"]) == {"price"} and out["config"]["correlation"] is None


# ----------------------------------------------------- simulation verification --

def test_a_simulation_verifies_its_arithmetic_statistics_probabilities_and_reproducibility():
    out = sim(uncertainty={"price": {"type": "normal", "sd_pct": 8}, "volume": {"type": "uniform", "low_pct": -10, "high_pct": 10}},
              target_value=4000, iterations=8000)
    v = out["verification"]
    assert v["status"] == "verified", [c for c in v["checks"] if c["status"] not in ("pass", "not_applicable")]
    names = {c["check"]: c["status"] for c in v["checks"]}
    for key in ("draw_arithmetic", "summary_statistics", "probabilities", "histogram", "input_distributions",
                "reproducible_with_seed", "fixed_inputs_constant"):
        assert names[key] == "pass"
    assert names["correlation"] == "not_applicable"
    assert names["input_ranges_supported_by_data"] == "not_run", "the ranges are assumptions and are never reported as checked"
    assert v["scope"] and any("assumptions" in x for x in v["limitations"])


def test_tampered_simulation_results_fail_verification(monkeypatch):
    captured = {}
    original = scenario_verify.verify_simulation

    def spy(result, raw, rerun, spec=None):
        captured.update(result=result, raw=raw, rerun=rerun)
        return original(result, raw, rerun, spec)

    monkeypatch.setattr(scenario_verify, "verify_simulation", spy)
    sim(target_value=4000)
    result, raw, rerun = captured["result"], captured["raw"], captured["rerun"]
    assert original(copy.deepcopy(result), raw, rerun)["status"] == "verified"
    for tamper in (
        lambda r: r["outputs"]["profit"].__setitem__("mean", r["outputs"]["profit"]["mean"] + 50),
        lambda r: r["outputs"]["profit"].__setitem__("p95", 0.0),
        lambda r: r["target"]["probabilities"]["positive"].__setitem__("count", 1),
        lambda r: r["target"]["histogram"]["counts"].__setitem__(0, r["target"]["histogram"]["counts"][0] + 5),
        lambda r: r["config"]["distributions"]["price"]["params"].__setitem__("sd", 0.2),
    ):
        bad = copy.deepcopy(result)
        tamper(bad)
        assert original(bad, raw, rerun)["status"] == "unverified"


def test_a_sampler_that_ignores_the_requested_correlation_fails_verification(monkeypatch):
    unc = {"price": {"type": "normal", "sd_pct": 10}, "volume": {"type": "normal", "sd_pct": 10}}
    corr = {"inputs": ["price", "volume"], "matrix": [[1, 0.9], [0.9, 1]]}
    original = scenario_sim.validate_correlation
    monkeypatch.setattr(scenario_sim, "validate_correlation", lambda spec, unc_names: None)    # the sampler drops it...
    result = scenario_sim.simulate("unit_economics", BASE, {}, {}, unc, None, 20_000, 3)
    result["config"]["correlation"] = {"inputs": corr["inputs"], "matrix": corr["matrix"]}      # ...but the result claims it
    raw = scenario_sim._raw(scenario_sim._prepare("unit_economics", BASE, {}, {}, unc, None, 20_000, 3, None, None, "at_least"))
    verdict = scenario_verify.verify_simulation(result, raw, lambda: raw)
    assert {c["check"]: c["status"] for c in verdict["checks"]}["correlation"] == "fail" and verdict["status"] == "unverified"
    monkeypatch.setattr(scenario_sim, "validate_correlation", original)


def test_the_largest_allowed_simulation_stays_fast_enough():
    started = time.perf_counter()
    out = sim(uncertainty={k: {"type": "normal", "sd_pct": 5} for k in ("price", "volume", "unit_cost", "fixed_cost")},
              correlation={"inputs": ["price", "volume"], "matrix": [[1, -0.4], [-0.4, 1]]},
              iterations=scenario_sim.MAX_ITERATIONS)
    elapsed = time.perf_counter() - started
    assert out["verification"]["status"] == "verified" and out["config"]["iterations"] == 200_000
    assert elapsed < 20, f"200,000 draws took {elapsed:.1f}s"


# ================================================================== goal seek ==

def seek(**kw):
    args = dict(model="unit_economics", baseline=BASE, target_output="margin_pct", target_value=20.0, solve_for="price")
    args.update(kw)
    return scenario_solve.goal_seek(**args)


def reach(model, baseline, output, **set_inputs):
    """The output the scenario engine gives when the named inputs are set: an independent recheck of a solution."""
    out = scenarios.evaluate(model, baseline, {"x": {k: {"type": "set", "value": v} for k, v in set_inputs.items()}})
    return out["scenarios"]["x"]["outputs"][output]


def test_price_for_a_twenty_percent_margin_matches_the_closed_form():
    # margin = ((p - 12) x 1000 - 3000) / (p x 1000) = 0.2  ->  0.8 p = 15  ->  p = 18.75
    r = seek()
    s = r["solutions"][0]
    assert r["status"] == "solved" and math.isclose(s["value"], 18.75, rel_tol=1e-9)
    assert math.isclose(reach("unit_economics", BASE, "margin_pct", price=s["value"]), 20.0, rel_tol=1e-9)
    assert r["recommended_index"] == 0 and "18.75" in r["message"] and r["verification"]["status"] == "verified"
    assert r["search_range"]["upper_assumed"] is True and any("assumed ceiling" in d for d in r["disclosures"])


@pytest.mark.parametrize("kwargs,expected", [
    # profit = 8 v - 3000 = 200000 -> v = 25375
    ({"target_output": "profit", "target_value": 200000.0, "solve_for": "volume"}, 25375.0),
    # margin 30%: ((20 - c) x 1000 - 3000) / 20000 = 0.3 -> c = 11
    ({"target_output": "margin_pct", "target_value": 30.0, "solve_for": "unit_cost"}, 11.0),
    # profit = 8000 - F = 1000 -> F = 7000
    ({"target_output": "profit", "target_value": 1000.0, "solve_for": "fixed_cost"}, 7000.0),
    # revenue = price x volume = 40000 -> price = 40
    ({"target_output": "revenue", "target_value": 40000.0, "solve_for": "price"}, 40.0),
])
def test_unit_economics_goals_match_their_closed_forms(kwargs, expected):
    r = seek(**kwargs)
    assert r["status"] == "solved" and math.isclose(r["solutions"][0]["value"], expected, rel_tol=1e-9)
    assert r["verification"]["status"] == "verified"


def test_funnel_goals_match_their_closed_forms():
    # revenue = visitors x cr x aov = 10000 x cr x 50 = 100000 -> cr = 0.2
    r = seek(model="funnel", baseline=FUNNEL, target_output="revenue", target_value=100000.0, solve_for="conversion_rate")
    assert r["status"] == "solved" and math.isclose(r["solutions"][0]["value"], 0.2, rel_tol=1e-9)
    # profit = visitors x (1.0 - 0.4) = 12000 -> visitors = 20000
    v = seek(model="funnel", baseline=FUNNEL, target_output="profit", target_value=12000.0, solve_for="visitors")
    assert math.isclose(v["solutions"][0]["value"], 20000.0, rel_tol=1e-9) and v["verification"]["status"] == "verified"


def test_a_goal_beyond_a_physical_limit_has_no_solution_and_reports_the_reachable_range():
    # conversion cannot exceed 1: the most revenue is 10000 x 1 x 50 = 500000
    r = seek(model="funnel", baseline=FUNNEL, target_output="revenue", target_value=1_000_000.0, solve_for="conversion_rate")
    assert r["status"] == "no_solution" and r["solutions"] == [] and r["recommended_index"] is None
    assert math.isclose(r["achievable_range"]["max"], 500000.0) and "above" in r["message"]
    assert r["may_exist_beyond_search_range"] is False and r["verification"]["status"] == "verified"


def test_a_margin_above_100_percent_is_impossible_and_is_not_blamed_on_the_search_ceiling():
    r = seek(target_value=150.0)
    assert r["status"] == "no_solution" and r["achievable_range"]["max"] < 100
    assert r["may_exist_beyond_search_range"] is False, "the margin flattens towards 100%, it is not heading for 150%"


def test_a_goal_that_a_wider_search_would_reach_says_to_widen_it():
    r = seek(target_output="profit", target_value=5_000_000_000.0, solve_for="volume", upper=None)
    assert r["status"] == "no_solution" and r["may_exist_beyond_search_range"] is True and "widen" in r["message"]
    bounded = seek(target_output="profit", target_value=200000.0, solve_for="volume", upper=500.0)
    assert bounded["status"] == "no_solution" and bounded["may_exist_beyond_search_range"] is False
    assert bounded["search_range"]["upper_assumed"] is False


def test_an_input_that_does_not_affect_the_target_is_reported_not_solved():
    r = seek(target_output="revenue", target_value=5000.0, solve_for="unit_cost")
    assert r["status"] == "no_dependence" and r["solutions"] == [] and "does not change" in r["message"]
    assert r["verification"]["status"] == "verified"


def test_an_output_that_divides_by_zero_everywhere_is_undefined_not_solved():
    r = seek(baseline={**BASE, "volume": 0.0})            # no units: revenue 0 at every price, so margin is undefined
    assert r["status"] == "undefined" and r["solutions"] == [] and "divides by zero" in r["message"]
    assert r["verification"]["status"] == "verified"


def test_with_elasticity_there_can_be_two_prices_and_both_are_returned():
    # volume = 1000 x (1 - 2 (p/20 - 1)); profit = (p - 12) x volume - 3000 = 1000 has two roots
    r = seek(target_output="profit", target_value=1000.0, assumptions={"price_elasticity": -2.0})
    assert r["status"] == "multiple_solutions" and len(r["solutions"]) == 2
    low, high = sorted(s["value"] for s in r["solutions"])
    for p in (low, high):
        volume = 1000.0 * (1 + -2.0 * (p / 20.0 - 1))
        assert math.isclose((p - 12) * volume - 3000, 1000.0, rel_tol=1e-9)
    assert low < 20 < high and r["solutions"][r["recommended_index"]]["value"] == low     # closest to the current price
    assert any("elasticity" in d for d in r["disclosures"]) and r["verification"]["status"] == "verified"


def test_constraints_choose_between_solutions_and_can_rule_them_all_out():
    kw = dict(target_output="profit", target_value=1000.0, assumptions={"price_elasticity": -2.0})
    keep_volume = seek(**kw, constraints=[{"metric": "volume", "op": ">=", "value": 900}])
    assert keep_volume["status"] == "solved"
    assert [s["feasible"] for s in keep_volume["solutions"]] == [True, False]
    above = seek(**kw, constraints=[{"metric": "price", "op": ">=", "value": 20}])
    assert above["status"] == "solved" and above["solutions"][above["recommended_index"]]["value"] > 20
    none = seek(**kw, constraints=[{"metric": "price", "op": ">=", "value": 100}])
    assert none["status"] == "no_feasible_solution" and none["recommended_index"] is None and len(none["solutions"]) == 2
    assert none["verification"]["status"] == "verified"


def test_a_solver_failure_is_reported_never_turned_into_an_answer(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("did not converge")
    monkeypatch.setattr(scenario_solve.optimize, "brentq", boom)
    r = seek()
    assert r["status"] == "convergence_failure" and r["solutions"] == [] and r["failures"]
    assert "No solution is reported" in r["message"]


def test_a_pole_is_not_mistaken_for_a_solution():
    # cost_per_order = acquisition / orders blows up as conversion -> 0: a sign change with no root
    r = seek(model="funnel", baseline=FUNNEL, target_output="cost_per_order", target_value=-5.0, solve_for="conversion_rate")
    assert r["status"] in ("no_solution", "no_dependence") and r["solutions"] == []


@pytest.mark.parametrize("kwargs,message", [
    ({"target_output": "magic"}, "Unknown target metric"),
    ({"target_value": float("nan")}, "finite number"), ({"target_value": float("inf")}, "finite number"),
    ({"solve_for": "ghost"}, "Unknown input"),
    ({"adjustments": {"price": {"type": "pct", "value": 5}}}, "variable being solved"),
    ({"adjustments": {"volume": {"type": "pct", "value": -300}}}, "more than 100%"),
    ({"lower": 10.0, "upper": 5.0}, "lower must be below upper"),
    ({"solve_for": "volume", "lower": -5.0}, "allowed range"),
    ({"model": "funnel", "baseline": FUNNEL, "target_output": "profit", "solve_for": "conversion_rate", "upper": 3.0}, "allowed range"),
    ({"constraints": [{"metric": "ghost", "op": ">=", "value": 1}]}, "Unknown constraint metric"),
    ({"constraints": [{"metric": "price", "op": "~", "value": 1}]}, "Unknown constraint operator"),
    ({"constraints": [{"metric": "price", "op": ">="}]}, "exactly"),
    ({"constraints": [{"metric": "price", "op": ">=", "value": 1}] * 6}, "at most"),
    ({"model": "crystal_ball"}, "Unknown model"),
    ({"baseline": {**BASE, "price": -1.0}}, "cannot be below"),
])
def test_invalid_goal_seek_requests_are_refused_with_a_reason(kwargs, message):
    with pytest.raises(scenarios.ScenarioError, match=message):
        seek(**kwargs)


def test_other_adjustments_set_the_ground_the_goal_is_solved_on():
    # volume cut by 50% -> 500 units: margin 20% needs ((p - 12) x 500 - 3000) / (500 p) = 0.2 -> 0.8 p = 18 -> p = 22.5
    r = seek(adjustments={"volume": {"type": "pct", "value": -50}})
    assert math.isclose(r["solutions"][0]["value"], 22.5, rel_tol=1e-9) and r["adjustments"]["volume"]["value"] == -50


def test_goal_seek_verification_catches_a_wrong_or_missing_solution():
    r = seek(target_output="profit", target_value=1000.0, assumptions={"price_elasticity": -2.0})
    assert scenario_verify.verify_goal_seek(r)["status"] == "verified"
    wrong = copy.deepcopy(r)
    wrong["solutions"][0]["value"] += 0.5
    assert {c["check"]: c["status"] for c in scenario_verify.verify_goal_seek(wrong)["checks"]}["target_reached"] == "fail"
    missing = copy.deepcopy(r)
    missing["solutions"] = missing["solutions"][:1]
    v = scenario_verify.verify_goal_seek(missing)
    assert {c["check"]: c["status"] for c in v["checks"]}["no_solution_missed"] == "fail" and v["status"] == "unverified"
    claimed = seek(target_output="revenue", target_value=5000.0, solve_for="unit_cost")
    claimed["status"] = "undefined"
    assert scenario_verify.verify_goal_seek(claimed)["status"] == "unverified"


# ===================================================================== the API ==

def test_evaluate_returns_break_even_and_a_verification_block(client, ws):
    r = client.post("/api/scenarios/evaluate", json={"model": "unit_economics", "baseline": BASE, "spread_pct": 10}, headers=ws["h"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verification"]["status"] == "verified" and body["break_even"]["baseline"]["break_even_quantity"] == 375.0
    assert set(body["break_even"]["scenarios"]) == {"optimistic", "pessimistic"}


def test_simulate_endpoint_runs_and_repeats_with_the_same_seed(client, ws):
    body = {"model": "unit_economics", "baseline": BASE, "iterations": 4000, "seed": 99, "target_value": 4000,
            "adjustments": {"price": {"type": "pct", "value": 5}},
            "uncertainty": {"price": {"type": "normal", "sd_pct": 6}, "volume": {"type": "triangular", "low_pct": -15, "high_pct": 10}},
            "correlation": {"inputs": ["price", "volume"], "matrix": [[1, -0.3], [-0.3, 1]]}}
    a = client.post("/api/scenarios/simulate", json=body, headers=ws["h"])
    assert a.status_code == 200, a.text
    out = a.json()
    assert out["kind"] == "simulation" and out["verification"]["status"] == "verified" and out["config"]["seed"] == 99
    assert out["target"]["probabilities"]["reaches_target"]["target"] == 4000.0
    assert client.post("/api/scenarios/simulate", json=body, headers=ws["h"]).json()["outputs"] == out["outputs"]
    assert "samples" not in out and len(str(out)) < 60_000, "raw draws are never returned"


@pytest.mark.parametrize("patch,code", [
    ({"uncertainty": {}}, 400), ({"iterations": 10 ** 7}, 422), ({"iterations": 5}, 422), ({"seed": -3}, 422),
    ({"uncertainty": {"price": {"type": "normal", "sd_pct": -1}}}, 400), ({"model": "nope"}, 400),
    ({"baseline": {**BASE, "price": -9.0}}, 400), ({"target_direction": "sideways"}, 400),
    ({"correlation": {"inputs": ["price", "volume"], "matrix": [[1, 2], [2, 1]]}}, 400),
])
def test_simulate_endpoint_rejects_bad_requests_safely(client, ws, patch, code):
    body = {"model": "unit_economics", "baseline": BASE, "iterations": 1000, "uncertainty": {"price": {"type": "normal", "sd_pct": 5}},
            "correlation": None, **patch}
    r = client.post("/api/scenarios/simulate", json=body, headers=ws["h"])
    assert r.status_code == code, r.text
    assert "Traceback" not in r.text and "File " not in r.text


def test_goal_seek_endpoint_solves_and_rejects(client, ws):
    ok = client.post("/api/scenarios/goal-seek", json={"model": "unit_economics", "baseline": BASE, "target_output": "margin_pct",
                                                       "target_value": 20, "solve_for": "price"}, headers=ws["h"])
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["status"] == "solved" and math.isclose(body["solutions"][0]["value"], 18.75) and body["verification"]["status"] == "verified"
    with_constraint = client.post("/api/scenarios/goal-seek", json={
        "model": "unit_economics", "baseline": BASE, "assumptions": {"price_elasticity": -2}, "target_output": "profit",
        "target_value": 1000, "solve_for": "price", "constraints": [{"metric": "volume", "op": ">=", "value": 900}]}, headers=ws["h"])
    assert with_constraint.json()["status"] == "solved"
    for bad, code in (({"solve_for": "ghost"}, 400), ({"target_output": "x"}, 400), ({"lower": 9, "upper": 1}, 400),
                      ({"constraints": [{"metric": "price", "op": "!!", "value": 1}]}, 400),
                      ({"constraints": [{"metric": "price", "op": ">=", "value": 1}] * 9}, 422), ({"target_value": "high"}, 422)):
        body = {"model": "unit_economics", "baseline": BASE, "target_output": "profit", "target_value": 1000, "solve_for": "price", **bad}
        assert client.post("/api/scenarios/goal-seek", json=body, headers=ws["h"]).status_code == code, bad


def test_the_new_endpoints_need_a_signed_in_member(client, ws):
    sim_body = {"model": "unit_economics", "baseline": BASE, "uncertainty": {"price": {"type": "normal", "sd_pct": 5}}}
    seek_body = {"model": "unit_economics", "baseline": BASE, "target_output": "profit", "target_value": 1, "solve_for": "price"}
    assert client.post("/api/scenarios/simulate", json=sim_body).status_code == 401
    assert client.post("/api/scenarios/goal-seek", json=seek_body).status_code == 401
    no_team = {k: v for k, v in ws["h"].items() if k.lower() != "x-team-id"}
    assert client.post("/api/scenarios/simulate", json=sim_body, headers=no_team).status_code in (400, 403, 422)
    other = workspace(client)
    assert client.post("/api/scenarios/simulate", json=sim_body, headers=other["h"]).status_code == 200


def test_a_saved_simulation_keeps_its_seed_and_can_be_reproduced(client, ws):
    unc = {"price": {"type": "normal", "sd_pct": 7}}
    body = {"model": "unit_economics", "baseline": BASE, "name": "Risky promo", "dataset_id": ws["ds"],
            "scenarios": {"promo": {"price": {"type": "pct", "value": -5}}},
            "simulation": {"uncertainty": unc, "iterations": 3000, "target_value": 3000,
                           "adjustments": {"price": {"type": "pct", "value": -5}}}}
    saved = client.post("/api/scenarios", json=body, headers=ws["h"])
    assert saved.status_code == 201, saved.text
    got = client.get(f"/api/scenarios/{saved.json()['id']}", headers=ws["h"]).json()
    config = got["request"]["simulation"]
    assert isinstance(config["seed"], int), "the server's seed is stored so the run can be repeated"
    stored = got["results"]["simulation"]
    assert stored["config"]["seed"] == config["seed"] and stored["verification"]["status"] == "verified"
    again = client.post("/api/scenarios/simulate", json={"model": "unit_economics", "baseline": BASE, **config}, headers=ws["h"]).json()
    assert again["outputs"] == stored["outputs"] and again["target"]["histogram"] == stored["target"]["histogram"]
    assert got["results"]["verification"]["status"] == "verified" and got["results"]["break_even"]
    other = workspace(client)
    assert client.get(f"/api/scenarios/{saved.json()['id']}", headers=other["h"]).status_code == 404


def test_saving_without_a_simulation_is_unchanged_and_a_bad_simulation_blocks_the_save(client, ws):
    plain = client.post("/api/scenarios", json={"model": "unit_economics", "baseline": BASE, "name": "Plain"}, headers=ws["h"])
    assert plain.status_code == 201 and "simulation" not in plain.json()["results"]
    bad = client.post("/api/scenarios", json={"model": "unit_economics", "baseline": BASE, "name": "Bad",
                                              "simulation": {"uncertainty": {}}}, headers=ws["h"])
    assert bad.status_code == 400
    assert client.get("/api/scenarios", headers=ws["h"]).json()["total"] == 1
