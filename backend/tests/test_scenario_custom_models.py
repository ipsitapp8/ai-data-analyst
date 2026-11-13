"""What-if Phase 2, feature 2: custom business models.

Covers the formula language (safety, evaluation, dependency order, cycles, units), definition validation, running a
custom model on every scenario engine, equivalence with the built-in model, verification, versioning, team isolation
and binding to the semantic layer. Expected values are worked out by hand in the comments.
"""
from __future__ import annotations

import builtins
import copy
import csv
import io
import math

import numpy as np
import pytest
from platform_helpers import SALES_CSV, workspace

from app import scenario_formula as sf
from app import scenario_models as sm
from app import scenario_sim, scenario_solve, scenario_verify, scenarios

SAAS = {
    "description": "Monthly recurring revenue with churn",
    "inputs": {
        "customers": {"label": "Active customers", "unit": "customers", "min": 0},
        "price": {"label": "Monthly price", "unit": "currency/customers", "min": 0, "good": "up"},
        "unit_cost": {"label": "Cost per customer", "unit": "currency/customers", "min": 0, "good": "down"},
    },
    "assumptions": {"churn_rate": {"label": "Monthly churn", "unit": "fraction", "default": 0.05, "min": 0, "max": 1}},
    "outputs": {
        "mrr": {"label": "MRR", "unit": "currency", "formula": "customers * price", "additive": True},
        "churned": {"label": "Churned revenue", "unit": "currency", "formula": "customers * churn_rate * price", "additive": True},
        "net_revenue": {"label": "Net revenue", "unit": "currency", "formula": "mrr - churned", "additive": True},
        "costs": {"label": "Costs", "unit": "currency", "formula": "customers * unit_cost", "additive": True},
        "profit": {"label": "Profit", "unit": "currency", "formula": "net_revenue - costs", "additive": True},
        "margin": {"label": "Margin", "unit": "fraction", "formula": "profit / net_revenue"},
    },
    "primary_output": "profit",
    "constraints": [{"expr": "price >= unit_cost", "message": "Price must cover the cost per customer", "severity": "error"}],
}
SAAS_BASE = {"customers": 1000.0, "price": 50.0, "unit_cost": 20.0}

UNIT_ECONOMICS = {
    "strict_units": True,
    "inputs": {"price": {"unit": "currency/unit", "min": 0, "good": "up"}, "volume": {"unit": "unit", "min": 0, "good": "up"},
               "unit_cost": {"unit": "currency/unit", "min": 0, "good": "down"}, "fixed_cost": {"unit": "currency", "min": 0, "good": "down"}},
    "outputs": {"revenue": {"unit": "currency", "formula": "price * volume"},
                "variable_cost": {"unit": "currency", "formula": "unit_cost * volume"},
                "gross_profit": {"unit": "currency", "formula": "revenue - variable_cost"},
                "profit": {"unit": "currency", "formula": "gross_profit - fixed_cost"},
                "margin_pct": {"unit": "percent", "formula": "profit / revenue * 100"}},
    "primary_output": "profit",
}
UE_BASE = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}


def saas_spec(**patch):
    d = copy.deepcopy(SAAS)
    d.update(patch)
    return sm.compile_spec(d, key="custom:1@1", title="SaaS")


# ============================================================== formula safety ==

@pytest.mark.parametrize("text", [
    "customers * price", "(a + b) * c / d", "-a + +b", "sqrt(a) + log(b) - exp(c)", "min(a, b, c) + max(a, b)",
    "abs(a - b) * floor(c) / ceil(d)", "a ** 2 + b ** -1 + c ** 0.5", "1.5e3 * a",
])
def test_valid_formulas_parse(text):
    assert sf.parse(text).tree


@pytest.mark.parametrize("text", [
    "__import__('os').system('echo hi')", "a.real", "a[0]", "(lambda: 1)()", "[x for x in a]", "'text'", "True + a", "a if b else c",
    "eval('1')", "open('x')", "a ** b", "a ** 99", "a ** (1 + 1)", "min(a)", "max()", "abs(a, b)", "sqrt(a=1)", "a and b",
    "a < b", "{1: 2}", "a; b", "", "   ", "a +", "x" * 400, "1e999", "not a", "a @ b", "a // b", "a % b", "-" * 30 + "a",
    "min", "f(x)", "sum(a)",
])
def test_unsafe_or_invalid_formulas_are_rejected_when_parsed(text):
    with pytest.raises(sf.FormulaError):
        sf.parse(text)


def test_a_formula_can_never_reach_python_eval_or_exec(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("eval/exec must never run")
    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    spec = saas_spec()
    assert spec["compute"](SAAS_BASE, {"churn_rate": 0.05})["profit"] == 27500.0
    assert spec["compute_vec"]({k: np.array([v, v * 2]) for k, v in SAAS_BASE.items()}, {"churn_rate": 0.05})["profit"].shape == (2,)


def test_a_formula_with_too_many_parts_or_too_deep_is_refused():
    def tree(depth):
        return "a" if depth == 0 else f"({tree(depth - 1)}+{tree(depth - 1)})"
    wide = tree(6)                                             # 64 names, 63 additions, nested only 6 deep, under 300 characters
    assert len(wide) < sf.MAX_FORMULA_CHARS
    with pytest.raises(sf.FormulaError, match="too long"):
        sf.parse(wide)
    with pytest.raises(sf.FormulaError, match="nested too deeply"):
        sf.parse(" + ".join(["a"] * 70))
    with pytest.raises(sf.FormulaError, match="nested too deeply"):
        sf.parse("-" * 25 + "a")           # brackets add no nesting to the syntax tree; a chain of signs does


def test_evaluation_matches_python_arithmetic_and_marks_undefined_as_none():
    f = sf.parse("(a + b) * c / d - e ** 2 + sqrt(a) + abs(-b)")
    env = {"a": 9.0, "b": 3.0, "c": 4.0, "d": 8.0, "e": 2.0}
    assert sf.evaluate_scalar(f, env) == pytest.approx((9 + 3) * 4 / 8 - 4 + 3 + 3)
    assert sf.evaluate_scalar(sf.parse("a / b"), {"a": 1.0, "b": 0.0}) is None
    assert sf.evaluate_scalar(sf.parse("sqrt(a)"), {"a": -4.0}) is None
    assert sf.evaluate_scalar(sf.parse("log(a)"), {"a": 0.0}) is None
    assert sf.evaluate_scalar(sf.parse("exp(a)"), {"a": 1e6}) is None
    assert sf.evaluate_scalar(sf.parse("a ** -1"), {"a": 0.0}) is None
    vec = sf.evaluate(sf.parse("a / b").tree, {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([2.0, 0.0, 4.0])})
    assert vec[0] == 0.5 and np.isnan(vec[1]) and vec[2] == 0.75


# =========================================================== dependency order ==

def test_outputs_are_ordered_by_what_they_use_whatever_order_they_are_written_in():
    deps = {"profit": {"revenue", "costs"}, "revenue": {"price", "qty"}, "costs": {"qty", "unit"}}
    order = sf.order_outputs(deps, {"price", "qty", "unit"})
    assert order.index("revenue") < order.index("profit") and order.index("costs") < order.index("profit")


@pytest.mark.parametrize("deps,loop", [
    ({"a": {"b"}, "b": {"a"}}, "a → b → a"), ({"a": {"a"}}, "a → a"),
    ({"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"x"}}, "a → b → c → a"),
])
def test_circular_definitions_are_refused_and_the_loop_is_named(deps, loop):
    with pytest.raises(sf.FormulaError, match="circle") as e:
        sf.order_outputs(deps, {"x"})
    assert loop in str(e.value)


def test_an_unknown_name_is_reported_with_a_suggestion():
    with pytest.raises(sf.FormulaError, match=r"uses 'custmers'.*did you mean 'customers'"):
        sf.order_outputs({"mrr": {"custmers", "price"}}, {"customers", "price"})


# ==================================================================== units ==

@pytest.mark.parametrize("text,dim", [
    ("currency/unit", {"currency": 1, "unit": -1}), ("units*units", {"units": 2}), ("fraction", {}), ("percent", {}), ("", {}),
    ("currency/unit/month", {"currency": 1, "unit": -1, "month": -1}), ("km^2", {"km": 2}),
])
def test_unit_text_is_read_as_exponents_of_base_units(text, dim):
    assert {k: int(v) for k, v in sf.parse_unit(text).items()} == dim


@pytest.mark.parametrize("bad", ["currency / ", "£", "a b", "x" * 50, "currency//unit"])
def test_malformed_units_are_refused(bad):
    with pytest.raises(sf.FormulaError):
        sf.parse_unit(bad)


def test_unit_checking_catches_adding_unlike_things_and_wrong_declared_units():
    units = {"price": sf.parse_unit("currency/unit"), "volume": sf.parse_unit("unit"), "fixed": sf.parse_unit("currency")}
    ok = sf.unit_problems(sf.parse("price * volume - fixed"), units, sf.parse_unit("currency"))
    assert ok == []
    mixed = sf.unit_problems(sf.parse("price + volume"), units, sf.parse_unit("currency"))
    assert any("adds currency/unit and unit" in p for p in mixed)
    wrong = sf.unit_problems(sf.parse("price * volume"), units, sf.parse_unit("unit"))
    assert any("works out to currency but is declared as unit" in p for p in wrong)
    assert sf.unit_problems(sf.parse("sqrt(price * price)"), units, sf.parse_unit("currency/unit")) == []
    assert any("log()" in p for p in sf.unit_problems(sf.parse("log(volume)"), units, {}))
    assert sf.unit_problems(sf.parse("fixed + 0"), units, sf.parse_unit("currency")) == []


# ======================================================== definition analysis ==

def test_a_valid_definition_is_analysed_ordered_and_unit_checked():
    a = sm.analyze(SAAS)
    assert a["valid"] and a["errors"] == []
    assert a["order"].index("mrr") < a["order"].index("net_revenue") < a["order"].index("profit") < a["order"].index("margin")
    assert a["units"]["mrr"] == "currency" and a["units"]["price"] == "currency/customers" and a["units"]["margin"] == "no unit"
    assert a["dependencies"]["net_revenue"] == ["churned", "mrr"]


def test_every_problem_is_reported_at_once():
    bad = {"inputs": {"Price": {"min": 5, "max": 1}, "min": {}, "x": {"good": "sideways"}},
           "assumptions": {"rate": {"default": 5, "min": 0, "max": 1}},
           "outputs": {"profit": {"formula": "price * qty"}, "x": {"formula": "1/0 +"}}, "primary_output": "nope"}
    a = sm.analyze(bad)
    text = " | ".join(a["errors"])
    assert not a["valid"] and len(a["errors"]) >= 6
    for fragment in ("'Price' is not a valid name", "'min' is reserved", "minimum is above maximum", "'good' must be",
                     "the default 5 is outside", "primary_output must name"):
        assert fragment in text, fragment
    assert "The formula for 'x'" in text


@pytest.mark.parametrize("patch,fragment", [
    ({"unknown_field": 1}, "Unknown field"), ({"inputs": []}, "must be an object"), ({"inputs": {}}, "1 to 20 inputs"),
    ({"outputs": {}}, "1 to 30 outputs"),
    ({"outputs": {"a": {"formula": "b"}, "b": {"formula": "a"}}, "primary_output": "a"}, "depend on each other in a circle"),
    ({"outputs": {"profit": {"formula": "customers * nothing"}}}, "'nothing'"),
    ({"outputs": {"profit": {"formula": "customers", "additive": True, "ratio_of": {"numerator": "profit", "denominator": "profit"}}}},
     "two different outputs"),
    ({"outputs": {"a": {"formula": "customers"}, "profit": {"formula": "a", "additive": True,
                                                         "ratio_of": {"numerator": "a", "denominator": "profit"}}}},
     "both additive and a ratio"),
    ({"constraints": [{"expr": "price >> 2"}]}, "one comparison"), ({"constraints": [{"expr": "price >= ghost"}]}, "unknown name"),
    ({"constraints": [{"expr": "price >= unit_cost", "severity": "fatal"}]}, "severity"),
    ({"inputs": {"customers": {"source": {"type": "metric"}}}}, "metric_id"),
    ({"inputs": {"customers": {"source": {"type": "formula", "formula": "sum(a) +"}}}}, "source formula is not valid"),
    ({"inputs": {"customers": {"source": {"type": "teleport"}}}}, "a source is"),
    ({"inputs": {"customers": {"min": 5, "max": 1}}}, "minimum is above maximum"),
    ({"outputs": {"profit": {"formula": "customers", "reconcile": {"metric_id": "x"}}}}, "reconcile"),
])
def test_invalid_definitions_are_refused_with_the_reason(patch, fragment):
    base = {"inputs": {"customers": {"unit": "customers"}, "price": {}, "unit_cost": {}},
            "outputs": {"profit": {"formula": "customers * price"}}, "primary_output": "profit"}
    a = sm.analyze({**base, **patch})
    assert not a["valid"] and any(fragment in e for e in a["errors"]), a["errors"]


def test_duplicate_and_over_limit_names_are_refused():
    d = {"inputs": {"a": {}}, "assumptions": {"a": {"default": 0, "min": 0, "max": 1}}, "outputs": {"a": {"formula": "1"}}, "primary_output": "a"}
    assert any("used more than once" in e for e in sm.analyze(d)["errors"])
    many = {"inputs": {f"i{k}": {} for k in range(21)}, "outputs": {"o": {"formula": "i0"}}, "primary_output": "o"}
    assert any("1 to 20 inputs" in e for e in sm.analyze(many)["errors"])


def test_unit_problems_are_warnings_unless_the_model_is_strict():
    d = copy.deepcopy(SAAS)
    d["outputs"]["profit"]["unit"] = "customers"
    lenient = sm.analyze(d)
    assert lenient["valid"] and any("Output 'profit' works out to currency" in w for w in lenient["warnings"])
    strict = sm.analyze({**d, "strict_units": True})
    assert not strict["valid"] and any("Output 'profit' works out to currency" in e for e in strict["errors"])
    d2 = copy.deepcopy(SAAS)
    d2["constraints"] = [{"expr": "price >= customers", "message": "x"}]
    assert any("Constraint 1 compares currency/customers with customers" in w for w in sm.analyze(d2)["warnings"])


def test_an_output_that_is_undefined_even_at_a_sample_point_is_flagged():
    d = {"inputs": {"x": {}}, "outputs": {"y": {"formula": "x / (x - x)"}}, "primary_output": "y"}
    a = sm.analyze(d)
    assert a["valid"] and any("undefined at a sample point" in w for w in a["warnings"])


def test_the_definition_hash_ignores_key_order_and_changes_with_content():
    shuffled = {k: SAAS[k] for k in reversed(list(SAAS))}
    assert sm.definition_sha256(SAAS) == sm.definition_sha256(shuffled)
    changed = copy.deepcopy(SAAS)
    changed["outputs"]["profit"]["formula"] = "net_revenue - costs - 1"
    assert sm.definition_sha256(changed) != sm.definition_sha256(SAAS)


# ============================================ a custom model on the engines ==

def test_a_custom_model_evaluates_to_the_hand_worked_numbers():
    spec = saas_spec()
    out = scenarios.evaluate(spec, SAAS_BASE, {"up": {"price": {"type": "pct", "value": 10}}}, {"churn_rate": 0.05})
    b = out["baseline"]["outputs"]
    # 1000 x 50 = 50000; churned 1000 x 0.05 x 50 = 2500; net 47500; costs 20000; profit 27500; margin 27500/47500
    assert b["mrr"] == 50000.0 and b["churned"] == 2500.0 and b["net_revenue"] == 47500.0 and b["costs"] == 20000.0
    assert b["profit"] == 27500.0 and math.isclose(b["margin"], 27500 / 47500)
    up = out["scenarios"]["up"]["outputs"]          # price 55: mrr 55000, churned 2750, net 52250, profit 32250
    assert up["profit"] == pytest.approx(32250.0, rel=1e-12) and out["model"] == "custom:1@1" and out["break_even"] is None
    assert out["scenarios"]["up"]["contributions"]["by_input"]["price"] == pytest.approx(4750.0, rel=1e-12)
    assert any("custom model" in n for n in out["unsupported_assumptions"])


def test_a_custom_model_is_independently_verified_with_a_separate_evaluator():
    spec = saas_spec()
    out = scenarios.evaluate(spec, SAAS_BASE, scenarios.standard_scenarios(spec, 10), {"churn_rate": 0.07})
    v = scenario_verify.verify_evaluation(out, spec)
    assert v["status"] == "verified", [c for c in v["checks"] if c["status"] != "pass"]
    names = {c["check"] for c in v["checks"]}
    assert {"baseline_outputs", "scenario_outputs", "shapley_contributions", "model_constraints"} <= names and "break_even" not in names
    tampered = copy.deepcopy(out)
    tampered["scenarios"]["optimistic"]["outputs"]["net_revenue"] += 1
    assert scenario_verify.verify_evaluation(tampered, spec)["status"] == "unverified"


def test_constraints_reject_an_impossible_scenario_and_report_warnings_beside_the_result():
    spec = saas_spec()
    with pytest.raises(scenarios.ScenarioError, match="Price must cover the cost per customer"):
        scenarios.evaluate(spec, SAAS_BASE, {"cheap": {"price": {"type": "set", "value": 10}}}, {"churn_rate": 0.05})
    with pytest.raises(scenarios.ScenarioError, match="The baseline breaks"):
        scenarios.evaluate(spec, {**SAAS_BASE, "price": 5.0}, {}, {"churn_rate": 0.05})
    d = copy.deepcopy(SAAS)
    d["constraints"] = [{"expr": "margin >= 0.5", "message": "Margin below 50%", "severity": "warning"}]
    soft = sm.compile_spec(d, key="custom:2@1", title="Soft")
    out = scenarios.evaluate(soft, SAAS_BASE, {"thin": {"price": {"type": "set", "value": 25}}}, {"churn_rate": 0.05})
    assert "thin" in out["constraint_notes"] and out["constraint_notes"]["thin"][0]["message"] == "Margin below 50%"
    assert "baseline" not in out["constraint_notes"]


def test_the_same_model_written_as_a_custom_definition_matches_the_built_in_exactly():
    built_in = scenarios.evaluate("unit_economics", UE_BASE, scenarios.standard_scenarios("unit_economics", 12), sensitivity_pct=15)
    spec = sm.compile_spec(UNIT_ECONOMICS, key="custom:9@1", title="UE")
    custom = scenarios.evaluate(spec, UE_BASE, scenarios.standard_scenarios(spec, 12), {}, sensitivity_pct=15)
    assert custom["baseline"]["outputs"] == built_in["baseline"]["outputs"]
    for name in built_in["scenarios"]:
        for k, v in built_in["scenarios"][name]["outputs"].items():
            assert custom["scenarios"][name]["outputs"][k] == pytest.approx(v, rel=1e-12), (name, k)
        assert custom["scenarios"][name]["contributions"]["by_input"] == pytest.approx(built_in["scenarios"][name]["contributions"]["by_input"], rel=1e-9)
    assert [r["input"] for r in custom["sensitivity"]["rows"]] == [r["input"] for r in built_in["sensitivity"]["rows"]]
    a = scenario_sim.simulate("unit_economics", UE_BASE, {}, {}, {"price": {"type": "normal", "sd_pct": 6}, "volume": {"type": "uniform", "low_pct": -8, "high_pct": 8}}, None, 4000, 21)
    b = scenario_sim.simulate(spec, UE_BASE, {}, {}, {"price": {"type": "normal", "sd_pct": 6}, "volume": {"type": "uniform", "low_pct": -8, "high_pct": 8}}, None, 4000, 21)
    for k in a["outputs"]:
        assert a["outputs"][k]["mean"] == pytest.approx(b["outputs"][k]["mean"], rel=1e-12) and a["outputs"][k]["p95"] == pytest.approx(b["outputs"][k]["p95"], rel=1e-12)
    assert b["verification"]["status"] == "verified"
    ga = scenario_solve.goal_seek("unit_economics", UE_BASE, "margin_pct", 20, "price")
    gb = scenario_solve.goal_seek(spec, UE_BASE, "margin_pct", 20, "price")
    assert gb["solutions"][0]["value"] == pytest.approx(ga["solutions"][0]["value"], rel=1e-9) and gb["verification"]["status"] == "verified"


def test_monte_carlo_on_a_custom_model_drops_draws_that_break_a_constraint():
    spec = saas_spec()
    # price ~ Normal(22, 4) against a cost of 20: P(price < 20) = Phi(-0.5) = 0.3085 of the draws break the constraint
    out = scenario_sim.simulate(spec, {**SAAS_BASE, "price": 22.0}, {}, {"churn_rate": 0.05},
                                {"price": {"type": "normal", "sd": 4.0}}, None, 60_000, 5)
    assert abs(out["constraint_rejected_fraction"] - 0.3085) < 0.01
    assert any("broke one of the model's constraints" in d for d in out["disclosures"])
    assert out["outputs"]["profit"]["n_valid"] == pytest.approx(60_000 * (1 - out["constraint_rejected_fraction"]), abs=1)
    assert out["verification"]["status"] == "verified", [c for c in out["verification"]["checks"] if c["status"] not in ("pass", "not_applicable", "not_run")]


def test_goal_seek_on_a_custom_model_respects_its_constraints():
    spec = saas_spec()
    r = scenario_solve.goal_seek(spec, SAAS_BASE, "profit", 40000.0, "price", assumptions={"churn_rate": 0.05})
    # profit = 1000 x 0.95 x p - 20000 = 40000  ->  p = 60000 / 950
    assert r["status"] == "solved" and r["solutions"][0]["value"] == pytest.approx(60000 / 950, rel=1e-9)
    assert r["verification"]["status"] == "verified"
    below_cost = scenario_solve.goal_seek(spec, SAAS_BASE, "profit", -5000.0, "price", assumptions={"churn_rate": 0.05})
    # profit -5000 would need p = 15000/950 = 15.8, below the cost of 20: the model forbids it, so there is no solution
    assert below_cost["status"] in ("no_solution", "undefined") and below_cost["solutions"] == []


def test_shapley_is_exact_for_small_models_and_sampled_for_large_ones_and_always_sums_to_the_change():
    f = lambda i: i["a"] * i["b"] * i["c"] + i["d"] * i["e"] + i["f"] ** 2 + i["g"] * i["h"]  # noqa: E731
    base = {k: 1.0 for k in "abcdefgh"}
    scen = {k: 1.0 + 0.3 * (n + 1) for n, k in enumerate("abcdefgh")}
    exact = scenarios.shapley(f, base, scen, max_exact=8)
    assert exact["method"] == "exact" and exact["samples"] is None
    sampled = scenarios.shapley(f, base, scen, max_exact=4, samples=8000, seed=3)
    assert sampled["method"] == "sampled" and sampled["samples"] == 8000 and sampled["changed_inputs"] == 8
    change = f(scen) - f(base)
    assert sum(exact["values"].values()) == pytest.approx(change, rel=1e-9)
    assert sum(sampled["values"].values()) == pytest.approx(change, rel=1e-9), "every sampled order sums to the total, so the estimate does too"
    for k, v in exact["values"].items():
        assert abs(sampled["values"][k] - v) < 4 * sampled["standard_errors"][k] + 1e-9, k
    assert scenarios.shapley(f, base, scen, max_exact=4, seed=3) == scenarios.shapley(f, base, scen, max_exact=4, seed=3)
    assert scenarios.shapley(f, base, base) == {"values": {}, "method": "exact", "changed_inputs": 0, "samples": None, "standard_errors": None}
    assert scenarios.shapley_contributions(lambda i: i["a"] * i["b"], {"a": 2.0, "b": 3.0}, {"a": 4.0, "b": 5.0}) == {"a": 8.0, "b": 6.0}


def test_a_model_with_many_inputs_evaluates_quickly_and_says_its_shapley_is_approximate():
    inputs = {f"x{i}": {"min": 0} for i in range(16)}
    d = {"inputs": inputs, "outputs": {"total": {"formula": " + ".join(f"x{i} * x{(i + 1) % 16}" for i in range(16))}}, "primary_output": "total"}
    spec = sm.compile_spec(d, key="custom:3@1", title="Ring")
    base = {f"x{i}": 1.0 for i in range(16)}
    out = scenarios.evaluate(spec, base, {"all_up": {k: {"type": "pct", "value": 20} for k in base}}, {})
    c = out["scenarios"]["all_up"]["contributions"]
    assert c["computation"]["method"] == "sampled" and "approximate" in c["method"]
    assert sum(c["by_input"].values()) == pytest.approx(out["scenarios"]["all_up"]["delta_vs_baseline"]["total"]["abs"], rel=1e-9)
    assert scenario_verify.verify_evaluation(out, spec)["status"] == "verified"


# ================================================================== the API ==

def metric(client, ws, name, formula, aliases=(), approve=True):
    m = client.post("/api/semantic/metrics", json={"name": name, "formula": formula, "aliases": list(aliases)}, headers=ws["h"]).json()
    if approve:
        assert client.post(f"/api/semantic/metrics/{m['id']}/versions/1/approve", headers=ws["h"]).status_code == 200
    return m


@pytest.fixture
def ws(client):
    return workspace(client)


def create(client, ws, definition=None, name="SaaS economics"):
    r = client.post("/api/scenario-models", json={"name": name, "description": "d", "definition": definition or SAAS}, headers=ws["h"])
    assert r.status_code == 201, r.text
    return r.json()


def test_create_list_get_and_the_pinned_reference(client, ws):
    m = create(client, ws)
    assert m["ref"] == f"custom:{m['id']}@1" and m["current_version"] == 1 and m["status"] == "active"
    assert m["describe"]["outputs"]["profit"] == "Profit" and m["describe"]["custom"]["units"]["mrr"] == "currency"
    assert m["definition"]["outputs"]["mrr"]["formula"] == "customers * price"
    listing = client.get("/api/scenario-models", headers=ws["h"]).json()["models"]
    assert [x["id"] for x in listing] == [m["id"]] and "definition" not in listing[0]
    assert client.get(f"/api/scenario-models/{m['id']}", headers=ws["h"]).json()["versions"][0]["definition_sha256"] == sm.definition_sha256(SAAS)
    models = client.get("/api/scenarios/models", headers=ws["h"]).json()["models"]
    assert {x["model"] for x in models} == {"unit_economics", "funnel"}, "the built-in list is unchanged"


def test_a_custom_model_runs_through_every_scenario_endpoint(client, ws):
    ref = create(client, ws)["ref"]
    body = {"model": ref, "baseline": SAAS_BASE, "assumptions": {"churn_rate": 0.05}}
    ev = client.post("/api/scenarios/evaluate", json={**body, "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}}, headers=ws["h"])
    assert ev.status_code == 200, ev.text
    out = ev.json()
    assert out["scenarios"]["up"]["outputs"]["profit"] == pytest.approx(32250.0) and out["verification"]["status"] == "verified" and out["model"] == ref
    sim = client.post("/api/scenarios/simulate", json={**body, "iterations": 3000, "seed": 4, "uncertainty": {"customers": {"type": "normal", "sd_pct": 5}}}, headers=ws["h"])
    assert sim.status_code == 200 and sim.json()["verification"]["status"] == "verified" and sim.json()["model"] == ref
    gs = client.post("/api/scenarios/goal-seek", json={**body, "target_output": "profit", "target_value": 40000, "solve_for": "price"}, headers=ws["h"])
    assert gs.status_code == 200 and gs.json()["status"] == "solved" and abs(gs.json()["solutions"][0]["value"] - 60000 / 950) < 1e-6
    bad = client.post("/api/scenarios/evaluate", json={**body, "scenarios": {"cheap": {"price": {"type": "set", "value": 5}}}}, headers=ws["h"])
    assert bad.status_code == 400 and "Price must cover" in bad.json()["detail"]
    saved = client.post("/api/scenarios", json={**body, "name": "SaaS +10%", "scenarios": {"up": {"price": {"type": "pct", "value": 10}}}}, headers=ws["h"])
    assert saved.status_code == 201 and saved.json()["model"] == ref


def test_editing_a_model_adds_a_version_and_never_changes_an_earlier_result(client, ws):
    m = create(client, ws)
    v1_ref, id_ = m["ref"], m["id"]
    body = {"baseline": SAAS_BASE, "assumptions": {"churn_rate": 0.05}}
    v1 = client.post("/api/scenarios/evaluate", json={"model": v1_ref, **body}, headers=ws["h"]).json()["baseline"]["outputs"]["profit"]
    edited = copy.deepcopy(SAAS)
    edited["outputs"]["costs"]["formula"] = "customers * unit_cost + 1000"
    r = client.post(f"/api/scenario-models/{id_}/versions", json={"definition": edited, "note": "add a fixed cost"}, headers=ws["h"])
    assert r.status_code == 201, r.text
    assert r.json()["current_version"] == 2 and r.json()["ref"] == f"custom:{id_}@2" and len(r.json()["versions"]) == 2
    assert client.post("/api/scenarios/evaluate", json={"model": v1_ref, **body}, headers=ws["h"]).json()["baseline"]["outputs"]["profit"] == v1
    latest = client.post("/api/scenarios/evaluate", json={"model": f"custom:{id_}", **body}, headers=ws["h"]).json()
    assert latest["model"] == f"custom:{id_}@2" and latest["baseline"]["outputs"]["profit"] == v1 - 1000
    old = client.get(f"/api/scenario-models/{id_}/versions/1", headers=ws["h"]).json()
    assert old["definition"]["outputs"]["costs"]["formula"] == "customers * unit_cost" and old["note"] == "First version"
    assert client.get(f"/api/scenario-models/{id_}/versions/9", headers=ws["h"]).status_code == 404
    assert client.post("/api/scenarios/evaluate", json={"model": f"custom:{id_}@7", **body}, headers=ws["h"]).status_code == 404


def test_validate_endpoint_reports_every_problem_and_saves_nothing(client, ws):
    ok = client.post("/api/scenario-models/validate", json={"definition": SAAS}, headers=ws["h"]).json()
    assert ok["valid"] and ok["order"][0] in ("mrr", "churned", "costs") and ok["units"]["profit"] == "currency"
    bad = client.post("/api/scenario-models/validate", json={"definition": {"inputs": {"X": {}}, "outputs": {"a": {"formula": "__import__('os')"}}, "primary_output": "z"}}, headers=ws["h"]).json()
    assert not bad["valid"] and len(bad["errors"]) >= 3
    assert client.get("/api/scenario-models", headers=ws["h"]).json()["models"] == []


@pytest.mark.parametrize("body,code,fragment", [
    ({"name": "Bad", "definition": {"inputs": {"a": {}}, "outputs": {"o": {"formula": "a.b"}}, "primary_output": "o"}}, 400, "formula"),
    ({"name": "", "definition": SAAS}, 422, ""), ({"name": "x", "definition": "text"}, 422, ""),
    ({"name": "Cycle", "definition": {"inputs": {"a": {}}, "outputs": {"p": {"formula": "q"}, "q": {"formula": "p"}}, "primary_output": "p"}}, 400, "circle"),
])
def test_invalid_models_are_refused_and_not_saved(client, ws, body, code, fragment):
    r = client.post("/api/scenario-models", json=body, headers=ws["h"])
    assert r.status_code == code and fragment in r.text
    assert client.get("/api/scenario-models", headers=ws["h"]).json()["models"] == []


def test_names_are_unique_per_team_and_oversized_definitions_are_refused(client, ws):
    create(client, ws, name="Plan A")
    assert client.post("/api/scenario-models", json={"name": "plan a", "definition": SAAS}, headers=ws["h"]).status_code == 400
    other = workspace(client)
    assert create(client, other, name="Plan A")["name"] == "Plan A", "another team may use the same name"
    huge = copy.deepcopy(SAAS)
    huge["description"] = "x" * 70_000
    assert client.post("/api/scenario-models", json={"name": "Huge", "definition": huge}, headers=ws["h"]).status_code == 413


def test_archiving_hides_a_model_but_keeps_its_saved_results_reproducible(client, ws):
    m = create(client, ws)
    body = {"model": m["ref"], "baseline": SAAS_BASE, "assumptions": {"churn_rate": 0.05}}
    assert client.post(f"/api/scenario-models/{m['id']}/archive", headers=ws["h"]).json()["status"] == "archived"
    assert client.get("/api/scenario-models", headers=ws["h"]).json()["models"] == []
    assert len(client.get("/api/scenario-models", params={"include_archived": True}, headers=ws["h"]).json()["models"]) == 1
    assert client.post("/api/scenarios/evaluate", json=body, headers=ws["h"]).status_code == 200
    assert client.post(f"/api/scenario-models/{m['id']}/versions", json={"definition": SAAS}, headers=ws["h"]).status_code == 400
    assert client.post(f"/api/scenario-models/{m['id']}/restore", headers=ws["h"]).json()["status"] == "active"


def test_another_team_cannot_see_use_change_or_even_detect_a_model(client, ws):
    m = create(client, ws)
    other = workspace(client)
    h = other["h"]
    for call in (client.get(f"/api/scenario-models/{m['id']}", headers=h), client.get(f"/api/scenario-models/{m['id']}/versions/1", headers=h),
                 client.post(f"/api/scenario-models/{m['id']}/versions", json={"definition": SAAS}, headers=h),
                 client.post(f"/api/scenario-models/{m['id']}/archive", headers=h), client.get(f"/api/scenario-models/{m['id']}", headers=h),
                 client.post("/api/scenarios/evaluate", json={"model": m["ref"], "baseline": SAAS_BASE, "assumptions": {"churn_rate": 0.05}}, headers=h),
                 client.post("/api/scenarios/simulate", json={"model": m["ref"], "baseline": SAAS_BASE, "uncertainty": {"price": {"type": "normal", "sd_pct": 5}}}, headers=h),
                 client.post("/api/scenarios/goal-seek", json={"model": m["ref"], "baseline": SAAS_BASE, "target_output": "profit", "target_value": 1, "solve_for": "price"}, headers=h)):
        assert call.status_code == 404, call.text
    assert client.get("/api/scenario-models", headers=h).json()["models"] == []
    assert client.get("/api/scenario-models").status_code == 401
    missing = client.get("/api/scenario-models/987654", headers=h)
    assert missing.status_code == 404 and missing.json() == client.get(f"/api/scenario-models/{m['id']}", headers=h).json()


@pytest.mark.parametrize("ref", ["custom:", "custom:abc", "custom:1@", "custom:1@x", "custom:-3"])
def test_malformed_model_references_are_refused_cleanly(client, ws, ref):
    r = client.post("/api/scenarios/evaluate", json={"model": ref, "baseline": SAAS_BASE}, headers=ws["h"])
    assert r.status_code == 400 and "Traceback" not in r.text


# ------------------------------------------------------------ semantic layer --

def bound(metric_id, **patch):
    d = copy.deepcopy(SAAS)
    d["inputs"]["customers"] = {"label": "Revenue base", "unit": "customers", "min": 0, "source": {"type": "metric", "metric_id": metric_id}}
    d.update(patch)
    return d


def csv_sum(col):
    return sum(float(r[col]) for r in csv.DictReader(io.StringIO(SALES_CSV.decode())))


def test_only_an_approved_metric_can_drive_a_model(client, ws):
    draft = metric(client, ws, "Order revenue", "sum(revenue)", approve=False)
    r = client.post("/api/scenario-models", json={"name": "Uses draft", "definition": bound(draft["id"])}, headers=ws["h"])
    assert r.status_code == 400 and "draft, not approved" in r.text
    approved = metric(client, ws, "Order revenue two", "sum(revenue)")
    m = create(client, ws, bound(approved["id"]), name="Uses approved")
    assert m["definition"]["inputs"]["customers"]["source"]["metric_id"] == approved["id"]
    ghost = client.post("/api/scenario-models", json={"name": "Ghost", "definition": bound(987654)}, headers=ws["h"])
    assert ghost.status_code == 400 and "Metric not found" in ghost.text
    other = workspace(client)
    foreign = client.post("/api/scenario-models", json={"name": "Foreign", "definition": bound(approved["id"])}, headers=other["h"])
    assert foreign.status_code == 400 and "Metric not found" in foreign.text, "another team's metric cannot be used"


def test_the_baseline_comes_from_the_approved_metric_with_its_provenance(client, ws):
    mt = metric(client, ws, "Total revenue", "sum(revenue)")
    m = create(client, ws, bound(mt["id"]))
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"], "overrides": {"price": 50, "unit_cost": 20}}, headers=ws["h"])
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["inputs"]["customers"] == pytest.approx(csv_sum("revenue")) == 277110.0
    src = b["sources"]["customers"]
    assert src["source"] == "semantic_metric" and src["metric_id"] == mt["id"] and src["formula"] == "sum(revenue)" and src["metric_version"] == 1
    assert b["sources"]["price"]["source"] == "override" and b["missing"] == [] and b["dataset_version"] == 1 and b["rows_used"] == 720
    unbound = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"]}, headers=ws["h"]).json()
    assert unbound["missing"] == ["price", "unit_cost"] and unbound["sources"]["price"]["source"] == "assumed", "nothing is filled in silently"
    west = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"], "filters": [{"column": "region", "op": "==", "value": "West"}]}, headers=ws["h"]).json()
    assert west["inputs"]["customers"] == 67590.0 and west["rows_used"] == 180


def test_a_metric_that_stops_being_approved_stops_driving_the_baseline(client, ws):
    mt = metric(client, ws, "Total revenue", "sum(revenue)")
    m = create(client, ws, bound(mt["id"]))
    assert client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"]}, headers=ws["h"]).status_code == 200
    assert client.post(f"/api/semantic/metrics/{mt['id']}/deprecate", headers=ws["h"]).status_code == 200
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"]}, headers=ws["h"])
    assert r.status_code == 400 and "deprecated, not approved" in r.json()["detail"]


def test_a_formula_binding_is_allowed_but_labelled_as_not_an_approved_metric(client, ws):
    d = copy.deepcopy(SAAS)
    d["inputs"]["price"]["source"] = {"type": "formula", "formula": "sum(revenue) / sum(units)"}
    m = create(client, ws, d, name="Formula bound")
    b = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"], "overrides": {"customers": 1000, "unit_cost": 5}}, headers=ws["h"]).json()
    assert b["inputs"]["price"] == pytest.approx(csv_sum("revenue") / csv_sum("units"))
    assert b["sources"]["price"]["source"] == "formula" and "not an approved metric" in b["sources"]["price"]["note"]
    d["inputs"]["price"]["source"] = {"type": "formula", "formula": "sum(ghost)"}
    broken = create(client, ws, d, name="Broken binding")
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": broken["ref"]}, headers=ws["h"])
    assert r.status_code == 400 and "ghost" in r.text


def test_a_term_that_matches_two_approved_metrics_asks_the_user_to_choose(client, ws):
    metric(client, ws, "Gross margin", "sum(revenue) - sum(cost)", ["margin"])
    metric(client, ws, "Unit margin", "(sum(revenue) - sum(cost)) / sum(units)", ["margin"])
    d = copy.deepcopy(SAAS)
    d["inputs"]["customers"]["source"] = {"type": "term", "term": "margin"}
    m = create(client, ws, d, name="Term bound")
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"]}, headers=ws["h"])
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "more than one approved metric" in detail["message"] and {c["name"] for c in detail["candidates"]} == {"Gross margin", "Unit margin"}
    single = metric(client, ws, "Net sales total", "sum(revenue)", ["net sales"])
    d["inputs"]["customers"]["source"] = {"type": "term", "term": "net sales"}
    ok = create(client, ws, d, name="Term unique")
    b = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": ok["ref"]}, headers=ws["h"]).json()
    assert b["sources"]["customers"]["metric_id"] == single["id"]
    d["inputs"]["customers"]["source"] = {"type": "term", "term": "nothing like this"}
    none = create(client, ws, d, name="Term none")
    assert "No approved metric matches" in client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": none["ref"]}, headers=ws["h"]).json()["detail"]


def test_a_model_output_is_reconciled_with_an_approved_metric(client, ws):
    net = metric(client, ws, "Net revenue", "sum(revenue) - sum(cost)")
    d = {"inputs": {"revenue_total": {"unit": "currency", "min": 0, "source": {"type": "formula", "formula": "sum(revenue)"}},
                    "cost_total": {"unit": "currency", "min": 0, "source": {"type": "formula", "formula": "sum(cost)"}}},
         "outputs": {"net": {"unit": "currency", "formula": "revenue_total - cost_total", "reconcile": {"metric_id": net["id"]}},
                     "wrong": {"unit": "currency", "formula": "revenue_total - cost_total - 1", "reconcile": {"metric_id": net["id"]}}},
         "primary_output": "net"}
    m = create(client, ws, d, name="Reconciled")
    b = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": m["ref"]}, headers=ws["h"]).json()
    rec = {r["output"]: r for r in b["reconciliation"]}
    expected = csv_sum("revenue") - csv_sum("cost")
    assert rec["net"]["agrees"] is True and rec["net"]["observed"] == pytest.approx(expected) and rec["net"]["difference"] == pytest.approx(0, abs=1e-6)
    assert rec["wrong"]["agrees"] is False and rec["wrong"]["difference"] == pytest.approx(-1.0)


def test_built_in_baselines_still_work_alongside_custom_models(client, ws):
    r = client.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics",
                                                     "mapping": {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"}}, headers=ws["h"])
    assert r.status_code == 200 and r.json()["inputs"]["volume"] == csv_sum("units")


# ============================================================== the migration ==

def test_the_scenario_models_migration_upgrades_downgrades_and_matches_the_orm(tmp_path):
    """Run the revision's own upgrade() and downgrade() on a scratch database, then compare it with the ORM tables."""
    import importlib.util
    from pathlib import Path

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import create_engine, inspect

    from app import models

    path = Path(models.__file__).resolve().parents[1] / "alembic" / "versions" / "c4e6a8b0d2f1_scenario_models.py"
    spec = importlib.util.spec_from_file_location("rev_c4e6a8b0d2f1", path)
    rev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rev)
    assert rev.down_revision == "b3d5f7a9c1e2"
    engine = create_engine(f"sqlite:///{(tmp_path / 'scratch.db').as_posix()}")
    with engine.begin() as conn:
        with Operations.context(MigrationContext.configure(conn)):
            rev.upgrade()
        insp = inspect(conn)
        assert {"scenario_models", "scenario_model_versions"} <= set(insp.get_table_names())
        for table in (models.ScenarioModel.__table__, models.ScenarioModelVersion.__table__):
            cols = {c["name"]: c for c in insp.get_columns(table.name)}
            assert set(cols) == {c.name for c in table.columns}, table.name
            for c in table.columns:
                assert cols[c.name]["nullable"] == c.nullable, (table.name, c.name)
        uniques = {tuple(u["column_names"]) for t in ("scenario_models", "scenario_model_versions") for u in insp.get_unique_constraints(t)}
        assert ("team_id", "slug") in uniques and ("model_id", "version") in uniques
        assert any(i["column_names"] == ["team_id"] for i in insp.get_indexes("scenario_models"))
        with Operations.context(MigrationContext.configure(conn)):
            rev.downgrade()
        assert not {"scenario_models", "scenario_model_versions"} & set(inspect(conn).get_table_names())
