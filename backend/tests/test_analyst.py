"""Analyst Copilot (app/analyst): tool validation, deterministic tools, Random
Forest suitability, business recommendations, grounding of written numbers,
model-plan fallback, and the API end to end. All data here is invented
(app/analyst/demo_data.py). No model is called: the model planner is either
off or replaced by a stub."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from platform_helpers import workspace

from app.analyst import business, demo_data as dd, explain, ml, orchestrator, planner, tools

E, C, K, INV, TINY = dd.employees(), dd.customers(), dd.campaigns(), dd.no_marketing(), dd.tiny()


def ask(q, df, mem=None, **kw):
    return orchestrator.ask(q, df, {} if mem is None else mem, use_model=kw.pop("use_model", False), **kw)


# ------------------------------------------------------------- validation --

@pytest.mark.parametrize("call, fragment", [
    ({"tool": "drop_table", "args": {}}, "unknown tool"),
    ({"tool": "compute_metric", "args": {"aggregation": "avg", "column": "wage"}}, "no column named 'wage'"),
    ({"tool": "compute_metric", "args": {"aggregation": "avg", "column": "department"}}, "not numeric"),
    ({"tool": "compute_metric", "args": {"aggregation": "mode", "column": "salary"}}, "is not one of"),
    ({"tool": "compute_metric", "args": {"aggregation": "sum"}}, "needs a column"),
    ({"tool": "compare_groups", "args": {"outcome": "salary", "group": "years_experience"}}, "categorical column is needed"),
    ({"tool": "random_forest", "args": {}}, "'target' is required"),
    ({"tool": "compute_metric", "args": {"aggregation": "avg", "column": "salary", "evil": 1}}, "unknown argument"),
    ({"tool": "budget_scenario", "args": {"change_pct": "a lot"}}, "a number is needed"),
])
def test_invalid_tool_calls_are_rejected_before_running(call, fragment):
    with pytest.raises(tools.InvalidCall, match=fragment):
        tools.validate_call(call, E)


# ---------------------------------------------------------------- metrics --

def test_average_salary_is_computed_from_the_data():
    r = ask("What is the average salary?", E)
    call = r["calls"][0]
    assert call["tool"] == "compute_metric" and call["result"]["value"] == pytest.approx(E["salary"].mean())
    assert f"{E['salary'].mean():,.2f}" in r["answer"]


def test_which_department_pays_the_most_groups_and_tests():
    r = ask("Which department pays the most?", E)
    groups = r["calls"][0]["result"]["groups"]
    truth = E.groupby("department")["salary"].mean().sort_values(ascending=False)
    assert [g["group"] for g in groups] == list(truth.index)
    assert groups[0]["value"] == pytest.approx(truth.iloc[0])
    assert r["calls"][1]["result"]["test"]["name"] == "Welch's one-way test"


def test_describe_column_reports_spread_and_interval():
    out = tools.run(tools.validate_call({"tool": "describe_column", "args": {"column": "salary"}}, E), E, {})
    assert out["sd"] == pytest.approx(E["salary"].std()) and out["variance"] == pytest.approx(E["salary"].var())
    assert out["mean_ci95"][0] < out["mean"] < out["mean_ci95"][1]


def test_categorical_comparison_checks_expected_counts():
    out = tools.run({"tool": "compare_groups", "args": {"outcome": "churn", "group": "contract"}}, C, {})
    assert out["test"]["name"].startswith("chi-square") and "cramers_v" in out["test"]
    small = pd.DataFrame({"g": ["a"] * 3 + ["b"] * 3, "y": ["x", "x", "z", "z", "z", "x"]})
    out = tools.run({"tool": "compare_groups", "args": {"outcome": "y", "group": "g"}}, small, {})
    assert out["test"]["name"] == "Fisher's exact test", "too few expected counts for chi-square"


def test_correlation_and_regression_state_they_are_not_causal():
    out = tools.run({"tool": "correlation", "args": {"x": "years_experience", "y": "salary"}}, E, {})
    assert out["pearson_r"] == pytest.approx(E["years_experience"].corr(E["salary"])) and "causation" in out["note"]
    out = tools.run({"tool": "linear_regression", "args": {"target": "salary", "features": ["years_experience"]}}, E, {})
    slope = np.polyfit(E["years_experience"], E["salary"], 1)[0]
    assert out["coefficients"][1]["estimate"] == pytest.approx(slope) and "not causal" in out["note"]


# ----------------------------------------------------------- random forest --

def test_churn_classification_with_metrics_importance_and_identifier_removed():
    mem = {}
    r = ask("Predict customer churn.", C, mem)
    res = r["calls"][0]["result"]
    assert res["task"] == "classification" and res["target"] == "churn"
    assert {"accuracy", "balanced_accuracy", "roc_auc", "baseline_accuracy"} <= res["metrics"].keys()
    assert any(d["column"] == "customer_id" and "identifier" in d["reason"] for d in res["dropped"])
    assert res["importance"][0]["feature"] == "contract", "the invented data makes contract the main driver"
    r = ask("Which factors influenced your prediction?", C, mem)
    assert r["calls"][0]["tool"] == "feature_importance" and "not what causes" in r["answer"]


def test_salary_regression_beats_its_baseline():
    res = ask("Predict salary.", E)["calls"][0]["result"]
    assert res["task"] == "regression" and res["metrics"]["mae"] < res["metrics"]["baseline_mae"] and res["beats_baseline"]


def test_unclear_target_asks_which_column():
    r = ask("Predict churn.", E)
    assert r["plan"]["type"] == "clarify" and "salary" in r["plan"]["options"] and not r["calls"]


@pytest.mark.parametrize("df, target, fragment", [
    (TINY, "churn", "at least 40"),
    (pd.DataFrame({"y": ["a"] * 60, "x": range(60)}), "y", "single value"),
    (pd.DataFrame({"y": ["a"] * 55 + ["b"] * 5, "x": range(60)}), "y", "only 5 rows"),
])
def test_unsuitable_data_is_refused_with_the_reason(df, target, fragment):
    with pytest.raises(ml.Unsuitable, match=fragment):
        ml.train(df, target)


def test_leaking_columns_are_left_out():
    df = C.copy()
    df["churn_flag"] = (df["churn"] == "Yes").astype(int)
    feats, dropped = ml.screen_features(df, "churn", "classification")
    assert "churn_flag" not in feats and any(d["column"] == "churn_flag" and "leakage" in d["reason"] for d in dropped)


def test_prediction_for_new_rows_lists_what_it_assumed():
    mem = {}
    ask("Predict churn", C, mem)
    msg = "Predict churn for a Monthly contract, 3 months tenure, 95 monthly charges and 4 support tickets"
    out = tools.run(tools.validate_call({"tool": "predict", "args": {"rows": [
        {"tenure_months": 3, "contract": "Monthly", "monthly_charges": 95, "support_tickets": 4}]}}, C, msg), C, mem)
    p = out["predictions"][0]
    assert p["prediction"] in ("Yes", "No") and abs(sum(p["probabilities"].values()) - 1) < 1e-9
    assert p["imputed"] == ["region"]


# ------------------------------------------------------- business analysis --

def test_recommendation_for_the_next_campaign_is_evidence_based():
    mem = {}
    r = ask("Which area should receive our next advertising campaign?", K, mem)
    res = r["calls"][0]["result"]
    a = next(g for g in res["performance"]["groups"] if g["group"] == "Area A")
    sub = K[K["area"] == "Area A"]
    assert a["roi"] == pytest.approx((sub["revenue"].sum() - sub["ad_spend"].sum()) / sub["ad_spend"].sum())
    assert a["cac"] == pytest.approx(sub["ad_spend"].sum() / sub["conversions"].sum())
    assert res["recommendation"]["focus"] == "Area A"
    for section in ("### Recommendation", "### Evidence", "### Methodology", "### Limitations", "### Next action"):
        assert section in r["answer"]
    assert "Area E" in res["performance"]["too_few_rows"], "two campaigns are too few to rank"


def test_why_explains_the_previous_recommendation_from_its_evidence():
    mem = {}
    ask("Should we increase the advertising budget in Area A?", K, mem)
    r = ask("Why do you recommend Area A?", K, mem)
    assert r["plan"]["type"] == "explain_previous"
    for part in ("business question", "data used", "analytical method", "computed results", "statistical evidence",
                 "uncertainty", "supports the recommendation", "Limitations"):
        assert part in r["answer"]
    assert ask("Why do you recommend that?", K, {})["answer"].startswith("I haven't made a recommendation")


def test_a_money_losing_area_is_not_recommended_for_more_budget():
    rec = tools.run({"tool": "recommend", "args": {"focus": "Area D", "kind": "increase"}}, K, {})["recommendation"]
    assert rec["decision"] == "do_not_expand"


def test_budget_scenario_states_assumptions_and_bounds():
    r = ask("What if we increase the budget by 20%?", K)
    s = r["calls"][0]["result"]
    assert s["new_spend"] == pytest.approx(K["ad_spend"].sum() * 1.2)
    assert s["if_returns_scale"]["roi"] == pytest.approx(s["baseline_roi"])
    assert s["if_no_response"]["profit"] == pytest.approx(K["revenue"].sum() - K["ad_spend"].sum() * 1.2)
    assert "not a prediction" in r["answer"]


def test_no_advertising_history_is_declined():
    r = ask("Which area should receive our next advertising campaign?", INV)
    assert r["plan"]["type"] == "decline" and "not sufficient" in r["answer"]
    with pytest.raises(business.Insufficient, match="spend"):
        business.performance(INV)


# ---------------------------------------------------- conversation, grounding --

@pytest.mark.parametrize("q, fragment", [("Hello.", "Silt's analyst"), ("What is Random Forest?", "decision trees")])
def test_conversational_questions_are_answered_without_tools(q, fragment):
    r = ask(q, E)
    assert r["plan"]["type"] == "chat" and fragment in r["answer"] and not r["calls"]


def test_grounding_accepts_tool_numbers_and_rejects_invented_ones():
    ev = [{"value": 98087.5, "roi": 1.093, "p_value": 0.0004}]
    assert explain.ungrounded("The average is 98,087.50 and ROI is 109.3%.", ev) == []
    assert explain.ungrounded("ROI was 1.09.", ev) == []
    assert explain.ungrounded("Expect 250,000 in new revenue.", ev) == ["250,000"]


class _Stub:
    def __init__(self, plan_text, summary="Area A returned the most."):
        self.plan_text, self.summary, self.calls = plan_text, summary, 0

    def __call__(self, messages, max_tokens=700):
        self.calls += 1
        return self.plan_text if self.calls == 1 else self.summary


def test_a_valid_model_plan_is_executed_and_its_summary_checked(monkeypatch):
    stub = _Stub('{"type":"tools","intent":"metric","calls":[{"tool":"compute_metric","args":{"aggregation":"avg","column":"salary"}}]}',
                 summary="The average salary is 12,345.")
    monkeypatch.setattr(planner, "model_chat", stub)
    r = ask("average pay?", E, use_model=True)
    assert r["plan_source"] == "model" and r["calls"][0]["result"]["value"] == pytest.approx(E["salary"].mean())
    assert r["grounding"]["ungrounded"] == ["12,345"] and "12,345" not in r["answer"], "invented number never shown"


@pytest.mark.parametrize("plan_text", ["not json at all",
                                       '{"type":"tools","calls":[{"tool":"compute_metric","args":{"aggregation":"avg","column":"pay"}}]}',
                                       '{"type":"tools","calls":[{"tool":"rm_rf","args":{}}]}'])
def test_an_invalid_model_plan_falls_back_to_the_deterministic_planner(monkeypatch, plan_text):
    monkeypatch.setattr(planner, "model_chat", _Stub(plan_text))
    r = ask("What is the average salary?", E, use_model=True)
    assert r["plan_source"] == "rules" and r["calls"][0]["status"] == "ok" and r["notes"]


def test_a_model_chat_reply_with_invented_numbers_is_replaced(monkeypatch):
    monkeypatch.setattr(planner, "model_chat", _Stub('{"type":"chat","reply":"Your revenue will grow by 37,500 next year."}'))
    r = ask("Hello.", E, use_model=True)
    assert "37,500" not in r["answer"] and r["notes"]


# ------------------------------------------------------------------- API --

def test_api_end_to_end_with_csv_and_excel(client):
    csv = K.to_csv(index=False).encode()
    ws = workspace(client, csv=csv, filename="campaigns.csv")
    r = client.post("/api/analyst/ask", json={"dataset_id": ws["ds"], "message": "Which area should receive our next "
                    "advertising campaign?", "conversation_id": "t1", "use_model": False}, headers=ws["h"])
    assert r.status_code == 200, r.text
    assert "### Recommendation" in r.json()["answer"] and r.json()["plan_source"] == "rules"
    r = client.post("/api/analyst/ask", json={"dataset_id": ws["ds"], "message": "Why do you recommend Area A?",
                    "conversation_id": "t1", "use_model": False}, headers=ws["h"])
    assert r.json()["plan"]["type"] == "explain_previous"

    buf = io.BytesIO()
    E.to_excel(buf, index=False)
    up = client.post("/api/datasets/upload", files={"file": ("staff.xlsx", buf.getvalue(), "application/octet-stream")},
                     headers=ws["h"])
    assert up.status_code == 200, up.text
    r = client.post("/api/analyst/ask", json={"dataset_id": up.json()["id"], "message": "What is the average salary?",
                    "use_model": False}, headers=ws["h"])
    assert r.json()["calls"][0]["result"]["value"] == pytest.approx(E["salary"].mean())
    assert client.get("/api/analyst/status", headers=ws["h"]).status_code == 200
    other = workspace(client)
    assert client.post("/api/analyst/ask", json={"dataset_id": ws["ds"], "message": "hi"},
                       headers=other["h"]).status_code == 404, "another team's dataset is not reachable"


# ------------------------------------------------- safety fixes (post-v1) --

@pytest.mark.parametrize("text, ok", [
    ("The average salary is 98,087.50.", True), ("about 98,088", True), ("roughly 98,000", True),
    ("The average salary is 98,087.40.", False), ("98,087.6", False), ("98,100", True), ("98,200", False),
    ("ROI 109.27%", True), ("ROI 109.3%", True), ("ROI 109%", True), ("ROI 109.2%", False), ("ROI 1.09", True),
    ("p < 0.001", True), ("at the 5% level", True), ("a 95% interval", True), ("expect 250,000 more", False),
])
def test_grounding_accepts_only_roundings_at_the_written_precision(text, ok):
    ev = [{"value": 98087.5, "roi": 1.0927}]
    assert (explain.ungrounded(text, ev) == []) is ok


class _Seq:
    """A stand-in for the model: returns the given replies in order."""
    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, messages, max_tokens=700):
        self.prompts.append(messages)
        return self.replies.pop(0) if self.replies else "ok"


def test_prediction_inputs_must_be_the_users_own():
    row = {"department": "Engineering", "level": "Senior", "years_experience": 10}
    call = {"tool": "predict", "args": {"rows": [row]}}
    with pytest.raises(tools.InvalidCall, match="not given in your message"):
        tools.validate_call(call, E, "Predict salary.")
    with pytest.raises(tools.InvalidCall, match="must be given in your message"):
        tools.validate_call(call, E)
    msg = "Predict the salary of a Senior in Engineering with 10 years experience"
    assert tools.validate_call(call, E, msg)["tool"] == "predict"
    with pytest.raises(tools.InvalidCall, match="value 12"):
        tools.validate_call({"tool": "predict", "args": {"rows": [{**row, "years_experience": 12}]}}, E, msg)


def test_a_model_plan_with_an_invented_prediction_row_is_not_executed(monkeypatch):
    mem = {}
    ask("Predict salary.", E, mem)
    invented = '{"type":"tools","calls":[{"tool":"predict","args":{"rows":[{"department":"Engineering","level":"Senior","years_experience":10,"age":45}]}}]}'
    monkeypatch.setattr(planner, "model_chat", _Seq(invented))
    r = orchestrator.ask("Predict salary.", E, mem, use_model=True)
    assert r["plan_source"] == "rules" and not any(c["tool"] == "predict" for c in r["calls"])
    assert r["first_pass"]["valid"] is False and "not given in your message" in r["first_pass"]["error"]


def test_a_decline_that_contradicts_the_schema_gets_one_recorded_reconsideration(monkeypatch):
    decline = '{"type":"decline","reason":"This dataset has no advertising history."}'
    good = '{"type":"tools","intent":"recommendation","calls":[{"tool":"recommend","args":{"kind":"where"}}]}'
    stub = _Seq(decline, good, "Area A had the highest return.")
    monkeypatch.setattr(planner, "model_chat", stub)
    r = orchestrator.ask("Which area should receive the next campaign?", K, {}, use_model=True)
    assert r["plan_source"] == "model_retry" and r["calls"][0]["tool"] == "recommend"
    fp = r["first_pass"]
    assert fp["valid"] is False and "advertising history" in fp["error"] and fp["plan"]["type"] == "decline"
    assert fp["retry"]["valid"] is True and fp["retry"]["plan"]["type"] == "tools"
    assert "rejected by validation" in stub.prompts[1][-1]["content"], "the model is told why"


def test_a_second_contradiction_falls_back_visibly_and_never_retries_twice(monkeypatch):
    decline = '{"type":"decline","reason":"no budget column"}'
    stub = _Seq(decline, decline)
    monkeypatch.setattr(planner, "model_chat", stub)
    r = orchestrator.ask("Should we increase the budget?", K, {}, use_model=True)
    assert r["plan_source"] == "rules" and r["first_pass"]["retry"]["valid"] is False
    assert any("did not resolve" in n for n in r["notes"])
    assert len([p for p in stub.prompts if p[0]["role"] == "system" and "JSON" in p[0]["content"]]) == 2


def test_a_false_no_column_clarification_is_caught(monkeypatch):
    clar = '{"type":"clarify","question":"There is no churn column. Which column should I predict?","options":["contract"]}'
    monkeypatch.setattr(planner, "model_chat", _Seq(clar, clar))
    r = orchestrator.ask("Predict customer churn.", C, {}, use_model=True)
    assert "no 'churn' column" in r["first_pass"]["error"]


def test_a_correct_decline_is_not_flagged(monkeypatch):
    from app.analyst import plan_checks
    decline = {"type": "decline", "reason": "This dataset has no advertising history."}
    assert plan_checks.contradiction(decline, INV, "Should we advertise more?") is None
    clar = {"type": "clarify", "question": "There is no churn column. Which column should I predict?"}
    assert plan_checks.contradiction(clar, E, "Predict churn") is None
    monkeypatch.setattr(planner, "model_chat", _Seq(json_decline := '{"type":"decline","reason":"This dataset has no advertising history."}'))
    r = orchestrator.ask("Should we advertise more?", INV, {}, use_model=True)
    assert r["plan_source"] == "model" and r["first_pass"]["valid"] is True and "not sufficient" in r["answer"]
