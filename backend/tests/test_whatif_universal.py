"""Universal what-if: profiling, domain packs, discovery, the shared engine and the API.

Expected values are worked out by hand in the comments."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.whatif import discovery, engine, operations, service
from app.whatif import scenario as sc
from app.whatif.domains import get_domain, list_domains
from app.whatif.profile import build_profile


def students(n: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = rng.integers(1, 8, n)
    return pd.DataFrame({"Student ID": range(1001, 1001 + n), "Attendance": rng.integers(50, 100, n),
                         "Maths Marks": (hours * 6 + rng.normal(30, 8, n)).clip(0, 100).round(),
                         "Science Marks": rng.integers(20, 95, n), "Study Hours": hours})


SMALL = pd.DataFrame({"Student ID": range(1, 11), "Attendance": [60, 70, 75, 80, 90, 95, 75, 85, 70, 80], "Marks": [30, 40, 45, 50, 55, 60, 40, 50, 45, 60]})


def run_q(text: str, df: pd.DataFrame, **kw):
    q = service.from_question(text, df, filename="x.csv", **kw)
    assert q["status"] == "ready", q
    return service.run(q["scenario"], df)


# ------------------------------------------------------------------ profiling --

def test_profile_reads_types_missing_duplicates_and_time():
    df = pd.DataFrame({"id": range(1, 9), "city": list("aabbccdd"), "score": [1.0, 2, 3, None, 5, 6, 7, 100],
                       "when": pd.date_range("2024-01-01", periods=8), "ok": [True, False] * 4})
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    p = build_profile(df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["score"]["missing"] == 1 and "score" in p["numeric"]
    assert cols["when"]["role"] == "datetime" and p["time_ranges"][0]["span_days"] == 7
    assert p["duplicates"] == 1 and p["rows"] == 9
    assert cols["score"]["outliers"] >= 1                       # 100 is far outside the other values
    assert "city" in p["categorical"] and "ok" in p["boolean"]


def test_education_is_detected_with_an_explanation_and_entities():
    p = build_profile(students())
    assert p["domain"]["key"] == "education" and p["domain"]["source"] == "detected"
    assert "Attendance" in p["domain"]["explanation"] and "Student" in p["domain"]["explanation"]
    assert p["entities"][0]["noun"] == "student"
    assert {"Maths Marks", "Science Marks"} == set(p["concepts"]["marks"]["columns"])
    assert any({r["a"], r["b"]} == {"Maths Marks", "Study Hours"} for r in p["relationships"])


def test_one_matching_column_is_not_enough_and_unfamiliar_data_is_custom():
    p = build_profile(pd.DataFrame({"revenue": [1, 2, 3, 4], "gizmo": [5, 6, 7, 8], "wibble": [9, 8, 7, 6]}))
    assert p["domain"]["key"] == "custom" and p["domain"]["source"] == "fallback"
    assert "No domain has enough evidence" in p["domain"]["explanation"]


def test_manual_domain_override_is_recorded_and_unknown_domains_refused():
    p = build_profile(students(), domain="business")
    assert p["domain"]["key"] == "business" and p["domain"]["source"] == "override" and p["domain"]["detected"] == "education"
    with pytest.raises(KeyError):
        build_profile(students(), domain="astrology")


def test_ambiguous_column_is_returned_as_candidates_never_chosen():
    df = students().rename(columns={"Study Hours": "Lecture Attendance"})
    m = build_profile(df)["concepts"]["attendance"]
    assert m["ambiguous"] and m["columns"] == [] and set(m["candidates"]) == {"Attendance", "Lecture Attendance"}


def test_personal_columns_are_marked_sensitive_and_never_listed():
    df = pd.DataFrame({"Patient Name": ["Asha Rao", "Ravi Kumar", "Meena Iyer"] * 5, "Phone": ["1", "2", "3"] * 5, "Ward": ["A", "B", "C"] * 5, "Wait Minutes": range(15)})
    p = build_profile(df)
    cols = {c["name"]: c for c in p["columns"]}
    assert cols["Patient Name"]["sensitive"] and "values" not in cols["Patient Name"] and "values" not in cols["Phone"]
    assert "Patient Name" in p["sensitive_columns"] and "Ward" in p["dimensions"]


# ---------------------------------------------------------------- education --

def test_pass_marks_counts_are_exact():
    # marks 30,40,45,50,55,60,40,50,45,60: >=40 -> 9 pass, >=50 -> 5 pass (50,55,60,50,60)
    r = run_q("What if passing marks increase from 40 to 50?", SMALL)
    m = {x["key"]: x for x in r["results"]["metrics"]}
    assert (m["Marks__count"]["baseline"], m["Marks__count"]["scenario"]) == (9, 5)
    assert r["results"]["evidence_kind"] == "exact_calculation" and r["results"]["details"]["columns"]["Marks"]["lost"] == 4


def test_attendance_requirement_uses_the_records():
    # attendance 60,70,75,80,90,95,75,85,70,80: >=75 -> 7 (75,80,90,95,75,85,80), >=85 -> 3 (90,95,85)
    r = run_q("What if attendance requirements increase from 75% to 85%?", SMALL)
    m = {x["key"]: x for x in r["results"]["metrics"]}
    assert (m["Attendance__count"]["baseline"], m["Attendance__count"]["scenario"]) == (7, 3)


def test_fraction_attendance_is_scaled_and_disclosed():
    df = SMALL.assign(Attendance=SMALL["Attendance"] / 100)
    r = run_q("What if attendance requirements increase from 75% to 85%?", df)
    assert r["results"]["metrics"][0]["scenario"] == 3
    assert any("0-1 fraction" in a for a in r["results"]["assumptions"])


def test_missing_current_threshold_is_asked_for_not_invented():
    q = service.from_question("What if passing marks increase to 50?", SMALL)
    out = service.run(q["scenario"], SMALL)
    assert out["validation"]["status"] == "needs_input" and out["results"] is None
    assert [m["name"] for m in out["validation"]["missing_inputs"]] == ["current_threshold"]


def test_capacity_needs_faculty_and_seats_then_is_exact():
    df = students(120)
    q = service.from_question("What if 100 additional students enroll?", df)
    out = service.run(q["scenario"], df)
    assert out["validation"]["status"] == "needs_input"
    assert {m["name"] for m in out["validation"]["missing_inputs"]} == {"faculty", "seats"}
    s = q["scenario"]
    s["inputs"].update({"faculty": 10, "seats": 200})
    out = service.run(s, df)
    m = {x["key"]: x for x in out["results"]["metrics"]}
    assert out["baseline_sources"]["students"].startswith("read from your data")
    assert m["students_per_faculty"]["baseline"] == 12 and m["students_per_faculty"]["scenario"] == 22   # 120/10 -> 220/10
    assert m["seats_short"]["scenario"] == 20 and m["seat_utilisation_pct"]["scenario"] == pytest.approx(110)


def test_study_hours_effect_is_an_estimate_with_an_interval_and_needs_a_column_choice():
    df = students(200)
    q = service.from_question("What if students study two additional hours every day?", df)
    first = service.run(q["scenario"], df)
    assert first["validation"]["status"] == "needs_input" and first["validation"]["missing_inputs"][0]["kind"] == "column"
    s = q["scenario"]
    s["columns"] = {"marks": ["Maths Marks"]}
    out = service.run(s, df)
    r = out["results"]
    assert r["evidence_kind"] == "statistical_estimate" and r["uncertainty"]["kind"] == "confidence_interval"
    assert r["uncertainty"]["change_low"] < r["metrics"][1]["delta"] < r["uncertainty"]["change_high"]
    assert 8 < r["metrics"][1]["delta"] < 16                    # about 6 marks per hour x 2 hours
    assert "cause" in " ".join(out["explanation"]["sections"]["cannot_conclude"])


def test_no_relationship_means_no_estimate():
    df = students(200)
    df["Science Marks"] = np.random.default_rng(9).integers(20, 95, len(df))
    q = service.from_question("What if students study two additional hours every day?", df)
    q["scenario"]["columns"] = {"marks": ["Science Marks"]}
    out = service.run(q["scenario"], df)
    assert out["results"]["status"] == "insufficient_evidence" and out["results"]["metrics"] == []
    assert out["results"]["evidence_kind"] == "unsupported_conclusion" and "cannot be estimated" in out["results"]["reason"]


def test_forecast_needs_enough_history():
    few = pd.DataFrame({"Student ID": range(1, 9), "Enrolled Date": pd.date_range("2024-01-01", periods=8, freq="W")})
    s = sc.new_scenario("education", "edu_enrolment_forecast", inputs={"periods_ahead": 1})
    out = service.run(s, few)
    assert out["results"]["status"] == "insufficient_evidence"
    many = pd.DataFrame({"Student ID": range(1, 241), "Enrolled Date": pd.date_range("2023-01-01", periods=240, freq="3D")})
    out = service.run(sc.new_scenario("education", "edu_enrolment_forecast", inputs={"periods_ahead": 2}), many)
    assert out["results"]["evidence_kind"] == "forecast" and out["results"]["uncertainty"]["kind"] == "prediction_interval"
    u = out["results"]["uncertainty"]
    assert u["low"] < out["results"]["metrics"][0]["scenario"] < u["high"]


def test_invalid_inputs_are_rejected():
    s = sc.new_scenario("education", "edu_attendance_requirement", inputs={"current_threshold": 75, "new_threshold": 140})
    assert service.run(s, SMALL)["validation"]["status"] == "invalid"
    s = sc.new_scenario("education", "edu_attendance_requirement", inputs={"current_threshold": "x", "new_threshold": 80})
    assert service.run(s, SMALL)["validation"]["status"] == "invalid"
    s = sc.new_scenario("education", "edu_pass_marks", inputs={"current_threshold": 40, "new_threshold": 50, "bogus": 1})
    assert "Unknown input" in service.run(s, SMALL)["validation"]["errors"][0]


# ----------------------------------------------------------------- business --

def test_business_template_matches_the_existing_engine_exactly():
    from app import scenario_service
    base = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
    direct = scenario_service.evaluate("unit_economics", base, {"s": {"price": {"type": "pct", "value": 10}}}, None)
    s = sc.new_scenario("business", "biz_price_change", inputs={**base, "price_change": 10})
    out = service.run(s)
    prof = {m["key"]: m for m in out["results"]["metrics"]}["profit"]
    assert prof["baseline"] == direct["baseline"]["outputs"]["profit"] == 5000          # 20*1000 - 12*1000 - 3000
    assert prof["scenario"] == direct["scenarios"]["s"]["outputs"]["profit"] == 7000    # price 22: 22000 - 12000 - 3000
    assert out["results"]["engine"] == "scenario_service.evaluate" and out["results"]["verification"]


def test_break_even_and_sensitivity_methods_for_the_price_model():
    base = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
    be = service.run(sc.new_scenario("business", "biz_break_even", inputs=base))
    assert be["results"]["details"]["break_even"]["baseline"]["break_even_quantity"] == pytest.approx(375)        # 3000 / (20 - 12)
    sen = service.run(sc.new_scenario("business", "biz_sensitivity", inputs=base))
    assert sen["results"]["details"]["sensitivity"]["rows"]


def test_marketing_template_estimates_from_the_data_when_it_can():
    rng = np.random.default_rng(3)
    spend = rng.uniform(100, 1000, 80)
    df = pd.DataFrame({"Marketing Spend": spend, "Revenue": spend * 4 + rng.normal(0, 200, 80), "Units Sold": rng.integers(1, 50, 80)})
    assert build_profile(df)["domain"]["key"] == "business"
    out = run_q("What if we increase marketing spending by 20%?", df)
    r = out["results"]
    assert r["evidence_kind"] == "statistical_estimate" and r["metrics"][1]["delta"] > 0


# ---------------------------------------------------------- other domains --

HC = {"admissions_per_day": 40, "avg_length_of_stay": 5, "beds": 250}


def test_healthcare_capacity_is_exact_and_monte_carlo_is_reproducible():
    s = sc.new_scenario("healthcare", "hc_add_beds", inputs={**HC, "extra_beds": 10})
    out = service.run(s)
    m = {x["key"]: x for x in out["results"]["metrics"]}
    assert m["occupancy_pct"]["baseline"] == pytest.approx(80) and m["occupancy_pct"]["scenario"] == pytest.approx(200 / 260 * 100)
    mc = lambda seed: service.run(sc.new_scenario("healthcare", "hc_admissions_surge", inputs={**HC, "admissions_change": 25}, seed=seed,
                                                  uncertainty={"admissions_per_day": {"type": "normal", "sd_pct": 15}}))
    a, b, c = mc(7), mc(7), mc(8)
    assert a["results"]["details"]["simulation"] == b["results"]["details"]["simulation"] and a["results"]["seed"] == 7
    assert a["results"]["details"]["simulation"]["p5"] != c["results"]["details"]["simulation"]["p5"]
    assert a["results"]["evidence_kind"] == "hypothetical_assumption" and any("assumptions, not measurements" in x for x in a["results"]["assumptions"])


def test_monte_carlo_without_a_distribution_asks_for_one():
    out = service.run(sc.new_scenario("healthcare", "hc_admissions_surge", inputs={**HC, "admissions_change": 25}))
    assert out["validation"]["status"] == "needs_input" and out["validation"]["missing_inputs"][0]["name"] == "uncertainty"


def test_clinical_questions_are_unsupported():
    out = service.run(sc.new_scenario("healthcare", "hc_treatment"))
    assert out["validation"]["status"] == "unsupported" and "clinical" in out["validation"]["errors"][0]
    q = service.from_question("What if we change the treatment protocol?", None)
    assert q["status"] == "ready" and service.run(q["scenario"])["validation"]["status"] == "unsupported"


def test_travel_budget_from_typed_values_only():
    inputs = {"days": 5, "hotel_per_night": 100, "food_per_day": 40, "local_transport_per_day": 10, "fixed_costs": 500}
    base = service.run(sc.new_scenario("travel", "tr_extend", inputs={**inputs, "extra_days": 0}))
    ext = service.run(sc.new_scenario("travel", "tr_extend", inputs={**inputs, "extra_days": 3}))
    t = lambda o: {x["key"]: x for x in o["results"]["metrics"]}["total_cost"]
    assert t(base)["baseline"] == 5 * 150 + 500 and t(ext)["scenario"] == 8 * 150 + 500
    assert ext["results"]["evidence_kind"] == "hypothetical_assumption"
    live = service.run(sc.new_scenario("travel", "tr_live_prices"))
    assert live["validation"]["status"] == "unsupported" and "live" in live["validation"]["errors"][0]


def test_government_population_and_budget():
    out = service.run(sc.new_scenario("government", "gov_population", inputs={"population": 100000, "usage_rate": 0.2, "capacity": 25000, "population_change": 30}))
    m = {x["key"]: x for x in out["results"]["metrics"]}
    assert m["utilisation_pct"]["baseline"] == pytest.approx(80) and m["utilisation_pct"]["scenario"] == pytest.approx(104)
    assert m["shortfall"]["scenario"] == pytest.approx(1000)
    bud = service.run(sc.new_scenario("government", "gov_budget", inputs={"budget": 1_000_000, "cost_per_unit": 50_000, "budget_change": -10}))
    assert bud["results"]["metrics"][0]["scenario"] == pytest.approx(18)


CUSTOM = {"inputs": {"visitors": {"label": "Visitors", "unit": "visitors", "min": 0}, "ticket": {"label": "Ticket", "unit": "currency/visitors", "min": 0}},
          "outputs": {"takings": {"label": "Takings", "unit": "currency", "formula": "visitors * ticket"}}, "primary_output": "takings",
          "constraints": [{"expr": "ticket >= 1", "message": "Tickets must cost at least 1", "severity": "error"}]}


def test_custom_domain_runs_a_user_defined_model_and_validates_it():
    s = sc.new_scenario("custom", "custom_model", custom_model=CUSTOM, inputs={})
    s["baseline"] = {"visitors": 500, "ticket": 8}
    s["changes"] = {"ticket": {"type": "pct", "value": 25}}
    out = service.run(s)
    m = out["results"]["metrics"][0]
    assert (m["baseline"], m["scenario"]) == (4000, 5000)
    missing = sc.new_scenario("custom", "custom_model", custom_model=CUSTOM)
    assert service.run(missing)["validation"]["status"] == "needs_input"
    bad = sc.new_scenario("custom", "custom_model", custom_model={**CUSTOM, "outputs": {"takings": {"label": "T", "unit": "x", "formula": "visitors * __import__('os')"}}})
    bad["baseline"] = {"visitors": 1, "ticket": 1}
    assert service.run(bad)["validation"]["status"] == "invalid"


def test_goal_seek_success_and_failure_states():
    base = {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0}
    ok = sc.new_scenario("business", "biz_price_change", method="goal_seek", inputs=base,
                         target={"output": "profit", "value": 8000, "solve_for": "price"})
    out = service.run(ok)
    assert out["results"]["status"] == "ok" and out["results"]["details"]["goal_seek"]["solution"] == pytest.approx(23)    # (p - 12) * 1000 - 3000 = 8000
    impossible = sc.new_scenario("business", "biz_price_change", method="goal_seek", inputs=base,
                                 target={"output": "profit", "value": 8000, "solve_for": "price", "lower": 0, "upper": 15})
    assert service.run(impossible)["results"]["status"] != "ok"
    assert service.run(sc.new_scenario("business", "biz_price_change", method="goal_seek", inputs=base))["validation"]["status"] == "needs_input"


def test_wrong_method_for_the_question_is_refused():
    s = sc.new_scenario("education", "edu_pass_marks", method="monte_carlo", inputs={"current_threshold": 40, "new_threshold": 50})
    assert "cannot be used" in service.run(s, SMALL)["validation"]["errors"][0]


def test_compare_lines_up_metrics_and_flags_mismatches():
    a = run_q("What if passing marks increase from 40 to 50?", SMALL)
    b = run_q("What if passing marks increase from 40 to 55?", SMALL)
    cmp = service.compare([a, b])
    row = next(r for r in cmp["metrics"] if r["label"].endswith("meeting the pass mark") and "share" not in r["label"])
    assert row["scenario"][a["id"]] == 5 and row["scenario"][b["id"]] == 3        # >=50: 5 pass, >=55: 3 pass
    with pytest.raises(ValueError):
        service.compare([a])


# ---------------------------------------------------------------- discovery --

def test_dataset_aware_suggestions_are_classified_and_never_invent_columns():
    df = students()
    out = service.suggest(df, None, filename="students.csv")
    cols = set(df.columns)
    by = {s["template_id"]: s for s in out["suggestions"]}
    assert by["edu_pass_marks"]["feasibility"] == "ready"
    assert by["edu_enrolment_capacity"]["feasibility"] == "needs_assumptions" and by["edu_enrolment_capacity"]["missing_inputs"]
    assert by["edu_enrolment_forecast"]["feasibility"] == "needs_external_data"      # no date column in this table
    assert by["edu_extra_exam"]["feasibility"] == "needs_external_data"
    for s in out["suggestions"]:
        for cs in list(s["available_variables"].values()) + list(s["columns"].values()):
            assert set(cs) <= cols, s
        assert s["feasibility"] in ("ready", "needs_assumptions", "needs_external_data", "unsupported")
        assert s["reason"] and s["method"] and "limitations" in s


def test_relationship_with_no_support_is_unsupported_not_ready():
    df = students(200)
    df["Study Hours"] = np.random.default_rng(5).integers(1, 8, len(df))             # break the relationship with marks
    df["Maths Marks"] = np.random.default_rng(6).integers(20, 95, len(df))
    df = df.drop(columns=["Science Marks"])
    by = {s["template_id"]: s for s in service.suggest(df, None)["suggestions"]}
    assert by["edu_study_hours"]["feasibility"] == "unsupported" and "relationship" in by["edu_study_hours"]["limitations"][0]


def test_domain_templates_without_a_dataset_and_without_any_model():
    out = service.suggest(None, "education")
    assert out["meta"]["llm"] is None and {s["feasibility"] for s in out["suggestions"]} <= {"needs_external_data", "needs_assumptions"}
    assert any(s["template_id"] == "edu_enrolment_capacity" and s["feasibility"] == "needs_assumptions" for s in out["suggestions"])
    trav = service.suggest(None, "travel")["suggestions"]
    assert any(s["feasibility"] == "needs_assumptions" for s in trav) and any(s["feasibility"] == "needs_external_data" for s in trav)


def test_objective_ranks_templates_by_rules():
    out = service.suggest(students(), "education", objective="I am a college administrator planning enrollment and class capacity for the next academic year")
    assert out["suggestions"][0]["template_id"] in ("edu_enrolment_capacity", "edu_enrolment_forecast", "edu_faculty_change")
    nothing = service.suggest(students(), "education", objective="zzz qqq")
    assert nothing["meta"]["unmatched"] is True


def test_a_failing_or_inventive_model_cannot_break_or_extend_discovery():
    def broken(objective, candidates):
        raise RuntimeError("model down")

    def inventive(objective, candidates):
        return ["education:made_up_template", candidates[-1]["id"]]
    base = service.suggest(students(), "education", objective="attendance")
    down = service.suggest(students(), "education", objective="attendance", ranker=broken)
    assert down["meta"]["llm"] == "unavailable" and [s["id"] for s in down["suggestions"]] == [s["id"] for s in base["suggestions"]]
    inv = service.suggest(students(), "education", objective="attendance", ranker=inventive)
    assert {s["id"] for s in inv["suggestions"]} == {s["id"] for s in base["suggestions"]}
    assert all("made_up" not in s["id"] for s in inv["suggestions"])


def test_unmatched_questions_are_unsupported_and_dataset_questions_do_not_cross_domains_silently():
    assert service.from_question("What will the stock market do?", students())["status"] == "unsupported"
    q = service.from_question("What if hotel costs increase by 15%?", students())
    assert q["scenario"]["domain"] == "travel"            # classified into the right domain, not run on student columns


def test_unfamiliar_dataset_gets_generic_column_questions():
    df = pd.DataFrame({"widgets": np.arange(60) * 2.0, "gadgets": np.arange(60) * 3.0 + np.random.default_rng(2).normal(0, 3, 60), "colour": ["r", "g"] * 30})
    out = service.suggest(df, None)
    gen = [s for s in out["suggestions"] if s["template_id"].startswith("generic_")]
    assert gen and all(set(sum(s["columns"].values(), [])) <= set(df.columns) for s in gen)
    s = sc.new_scenario("custom", "generic_association", inputs={"change": 10}, columns={"x": ["widgets"], "y": ["gadgets"]})
    assert service.run(s, df)["results"]["status"] == "ok"


# ----------------------------------------------------------------- registry --

def test_registry_lists_the_six_domains_and_every_template_is_well_formed():
    assert [d.key for d in list_domains()] == ["business", "education", "healthcare", "travel", "government", "custom"]
    ids = set()
    for d in list_domains():
        for t in d.templates:
            assert t.id not in ids and t.method and t.reason
            ids.add(t.id)
            if t.operation not in ("none", "formula_model", "custom_model"):
                assert t.operation in operations.OPERATIONS
            if t.operation == "formula_model":
                assert t.model is not None
    with pytest.raises(KeyError):
        get_domain("nope")


# ---------------------------------------------------------------------- API --

def _csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


@pytest.fixture
def ws(client):
    from platform_helpers import workspace
    return workspace(client, _csv(students()), "students.csv")


def test_api_standalone_flow_without_any_model(client, ws):
    h, ds = ws["h"], ws["ds"]
    assert [d["key"] for d in client.get("/api/whatif/domains", headers=h).json()["domains"]][:2] == ["business", "education"]
    p = client.post("/api/whatif/profile", json={"dataset_id": ds}, headers=h).json()
    assert p["domain"]["key"] == "education"
    p2 = client.post("/api/whatif/profile", json={"dataset_id": ds, "domain": "business"}, headers=h).json()
    assert p2["domain"]["source"] == "override"
    s = client.post("/api/whatif/suggest", json={"dataset_id": ds, "objective": "attendance rules", "use_model": True}, headers=h).json()
    assert s["suggestions"] and s["meta"]["llm"] in ("unavailable", None)        # no model configured in tests: rules answered
    q = client.post("/api/whatif/from-question", json={"message": "What if passing marks increase from 40 to 50?", "dataset_id": ds}, headers=h).json()
    run = client.post("/api/whatif/run", json={"scenario": q["scenario"], "dataset_id": ds}, headers=h).json()
    assert run["validation"]["status"] == "valid" and run["results"]["status"] == "ok" and run["explanation"]["sections"]["cannot_conclude"]
    saved = client.post("/api/whatif/scenarios", json={"scenario": q["scenario"], "dataset_id": ds, "name": "Pass mark 50"}, headers=h)
    assert saved.status_code == 201, saved.text
    sid = saved.json()["id"]
    assert client.get(f"/api/whatif/scenarios/{sid}", headers=h).json()["scenario"]["results"]["status"] == "ok"
    assert [r["id"] for r in client.get("/api/whatif/scenarios", headers=h).json()["scenarios"]] == [sid]
    assert sid not in [r["id"] for r in client.get("/api/scenarios", headers=h).json()["scenarios"]]      # business library is untouched
    assert client.get(f"/api/scenarios/{sid}", headers=h).status_code == 404
    q2 = client.post("/api/whatif/from-question", json={"message": "What if passing marks increase from 40 to 60?", "dataset_id": ds}, headers=h).json()
    saved2 = client.post("/api/whatif/scenarios", json={"scenario": q2["scenario"], "dataset_id": ds}, headers=h).json()
    cmp = client.post("/api/whatif/compare", json={"ids": [sid, saved2["id"]]}, headers=h)
    assert cmp.status_code == 200 and cmp.json()["metrics"]
    upd = client.put(f"/api/whatif/scenarios/{sid}", json={"scenario": q2["scenario"], "dataset_id": ds, "name": "Renamed"}, headers=h)
    assert upd.status_code == 200 and upd.json()["name"] == "Renamed"
    assert client.post(f"/api/whatif/scenarios/{sid}/rerun", headers=h).json()["results"]["status"] == "ok"
    assert client.delete(f"/api/whatif/scenarios/{sid}", headers=h).status_code == 204
    assert client.get(f"/api/whatif/scenarios/{sid}", headers=h).status_code == 404


def test_api_rejects_invalid_and_never_trusts_client_results(client, ws):
    h, ds = ws["h"], ws["ds"]
    q = client.post("/api/whatif/from-question", json={"message": "What if passing marks increase from 40 to 50?", "dataset_id": ds}, headers=h).json()
    forged = {**q["scenario"], "results": {"status": "ok", "evidence_kind": "exact_calculation", "metrics": [{"label": "x", "baseline": 1, "scenario": 999}]}}
    run = client.post("/api/whatif/run", json={"scenario": forged, "dataset_id": ds}, headers=h).json()
    assert all(m["scenario"] != 999 for m in run["results"]["metrics"])
    missing = client.post("/api/whatif/from-question", json={"message": "What if passing marks increase to 50?", "dataset_id": ds}, headers=h).json()
    r = client.post("/api/whatif/scenarios", json={"scenario": missing["scenario"], "dataset_id": ds}, headers=h)
    assert r.status_code == 422
    assert client.post("/api/whatif/run", json={"scenario": {"domain": "nope", "template_id": "x"}}, headers=h).status_code == 422
    assert client.post("/api/whatif/profile", json={"dataset_id": ds, "domain": "nope"}, headers=h).status_code == 422


def test_api_team_isolation_and_authentication(client, ws):
    from platform_helpers import workspace
    other = workspace(client, _csv(students(60, 4)), "other.csv")
    q = client.post("/api/whatif/from-question", json={"message": "What if passing marks increase from 40 to 50?", "dataset_id": ws["ds"]}, headers=ws["h"]).json()
    mine = client.post("/api/whatif/scenarios", json={"scenario": q["scenario"], "dataset_id": ws["ds"]}, headers=ws["h"]).json()["id"]
    assert client.post("/api/whatif/profile", json={"dataset_id": ws["ds"]}, headers=other["h"]).status_code == 404
    assert client.post("/api/whatif/run", json={"scenario": q["scenario"], "dataset_id": ws["ds"]}, headers=other["h"]).status_code == 404
    assert client.post("/api/whatif/suggest", json={"dataset_id": ws["ds"]}, headers=other["h"]).status_code == 404
    for call in (lambda: client.get(f"/api/whatif/scenarios/{mine}", headers=other["h"]), lambda: client.delete(f"/api/whatif/scenarios/{mine}", headers=other["h"]),
                 lambda: client.post(f"/api/whatif/scenarios/{mine}/rerun", headers=other["h"]),
                 lambda: client.put(f"/api/whatif/scenarios/{mine}", json={"scenario": q["scenario"]}, headers=other["h"]),
                 lambda: client.post("/api/whatif/compare", json={"ids": [mine]}, headers=other["h"])):
        assert call().status_code == 404
    assert other["team"] != ws["team"] and client.get("/api/whatif/scenarios", headers=other["h"]).json()["total"] == 0
    for method, path in (("get", "/api/whatif/domains"), ("post", "/api/whatif/suggest"), ("post", "/api/whatif/run"), ("get", f"/api/whatif/scenarios/{mine}"),
                         ("post", "/api/whatif/scenarios"), ("delete", f"/api/whatif/scenarios/{mine}")):
        assert getattr(client, method)(path).status_code in (401, 403)


def test_api_manual_scenarios_need_no_dataset(client):
    from platform_helpers import signup, make_team
    token, _ = signup(client)
    h = make_team(client, token)
    s = sc.new_scenario("travel", "tr_extend", inputs={"days": 5, "hotel_per_night": 100, "food_per_day": 40, "local_transport_per_day": 10, "fixed_costs": 500, "extra_days": 3})
    out = client.post("/api/whatif/run", json={"scenario": s}, headers=h).json()
    assert {m["key"]: m for m in out["results"]["metrics"]}["total_cost"]["scenario"] == 8 * 150 + 500
    sug = client.post("/api/whatif/suggest", json={"domain": "travel"}, headers=h).json()
    assert sug["suggestions"] and sug["domain"]["key"] == "travel"
