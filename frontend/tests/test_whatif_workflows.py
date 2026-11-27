"""What-if: the four starting points are four different journeys, each run through the page and the real API to a result.

Only the language model is replaced (tests never reach a paid model); everything between the click and the calculation is real."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from test_pages import _InProcessRequests, api, healthy, page, text_of  # noqa: F401 - `api` is a fixture; imported first: it sets the paths

import api_client
from app.agents import llm_client
from app.rate_limit import _attempts
from platform_helpers import workspace

EXPLORE, DATASET, MANUAL, BRAINSTORM = "Explore suggestions", "Use my dataset", "Build a scenario manually", "Brainstorm with Silt AI"
STEP2, STEP3 = "2 · Questions", "3 · Scenario"
OBJECTIVE = "I run a school and want more students to pass this year"


def _students() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 80
    hours = rng.uniform(0.5, 6, n).round(1)
    return pd.DataFrame({"student_id": [f"S{i:03d}" for i in range(n)], "attendance_percentage": rng.uniform(50, 100, n).round(1),
                         "study_hours_per_day": hours, "assignment_completion_percentage": rng.uniform(30, 100, n).round(1),
                         "current_exam_score": (hours * 8 + 35 + rng.normal(0, 5, n)).clip(0, 100).round(1)})


@pytest.fixture
def ws(api, monkeypatch):  # noqa: F811
    monkeypatch.setattr(api_client, "requests", _InProcessRequests(api))
    monkeypatch.setattr(llm_client, "provider_available", lambda *a, **k: False)
    _attempts.clear()
    buf = io.StringIO()
    _students().to_csv(buf, index=False)
    return workspace(api, buf.getvalue().encode(), "students.csv")


def _model(monkeypatch, ideas):
    asked = []

    def call_tool(system, user_content, tool_name, *a, **k):
        asked.append(user_content)
        return {"ideas": ideas}

    monkeypatch.setattr(llm_client, "provider_available", lambda *a, **k: True)
    monkeypatch.setattr(llm_client, "call_tool", call_tool)
    monkeypatch.setattr(llm_client, "LAST_PROVIDER", "stub-model")
    return asked


IDEAS = [
    {"title": "Raise study time", "question": "What if students study more each day?", "why": "Study time moves with scores.", "usefulness": 4,
     "kind": "column_relationship", "x_column": "study_hours_per_day", "y_column": "current_exam_score"},
    {"title": "Lift attendance", "question": "What if attendance rises across the school?", "why": "Attendance supports passing.", "usefulness": 5,
     "kind": "calculation", "calculation_id": "edu_attendance_increase"},
    {"title": "Tutoring budget", "question": "What if we fund extra tutoring?", "why": "Targets weak students.", "usefulness": 5, "kind": "own_model",
     "variables": [{"name": "tutors", "label": "Tutors"}, {"name": "pupils_each", "label": "Students per tutor"}],
     "outcomes": [{"name": "covered", "label": "Students covered", "formula": "tutors * pupils_each"}]},
    {"title": "Invented column", "question": "What if parent income rises?", "why": "x", "usefulness": 5, "kind": "column_change", "column": "parent_income"},
    {"title": "Bad formula", "question": "What if y?", "why": "x", "usefulness": 5, "kind": "own_model",
     "variables": [{"name": "a", "label": "A"}], "outcomes": [{"name": "b", "label": "B", "formula": "__import__('os')"}]},
    {"title": "Made-up weights", "question": "What is the expected score?", "why": "x", "usefulness": 5, "kind": "own_model",
     "variables": [{"name": "attendance", "label": "Attendance"}, {"name": "hours", "label": "Hours"}],
     "outcomes": [{"name": "score", "label": "Score", "formula": "attendance * 0.3 + (hours / 6) * 40"}]},
    {"title": "New curriculum","question": "What if the curriculum changes?", "why": "Could matter.", "usefulness": 5, "kind": "outside_evidence",
     "data_needed": ["results from schools that changed curriculum"]},
]


def _open(ws, start: str, dataset: bool = True, domain: str | None = None, objective: str | None = None):
    """Step 1 on the real page: domain, starting point, dataset, objective, then on to step 2."""
    at = healthy(page("14_What_If.py", ws))
    if domain:
        at.radio(key="wxu_pick_domain").set_value(domain).run()
    if start != EXPLORE:
        at.radio(key="wxu_pick_start").set_value(start).run()
    if dataset:
        at.selectbox(key="wxu_ds").select_index(1).run()
    if objective:
        at.text_area(key="wxu_obj").input(objective).run()
    at.button(key="wxu_next_setup").click().run()
    return healthy(at)


def _uses(at) -> list[str]:
    return [b.key.removeprefix("wxu_use_") for b in at.button if b.key and b.key.startswith("wxu_use_")]


def _choose(at, prefix: str) -> tuple:
    key = next(k for k in _uses(at) if k.startswith(prefix))
    at.button(key=f"wxu_use_{key}").click().run()
    at = healthy(at)
    return at, at.session_state["wxu_sel"]["id"]


def _go(at, step: str):
    at.session_state["wxu_goto"] = step
    return healthy(at.run())


def _result(at) -> dict:
    return at.session_state["wxu_result"]


def _has(at, key: str) -> bool:
    return key in at.session_state


def test_a_explore_discovers_by_capability_then_runs_a_choice(ws):
    at = _open(ws, EXPLORE)
    text = text_of(at)
    assert "Discover what you could explore in Education" in text and "Move a rule or cut-off" in text and "Needs outside evidence" in text
    assert "Explores" in text and "Data needed" in text and "What is in" not in text and "Ideas from Silt AI" not in text
    assert not any("generic_" in k for k in _uses(at)), "column-by-column scenarios belong to Use my dataset"
    at, sid = _choose(at, "education:edu_attendance_requirement")
    at.number_input(key=f"wxu_in_{sid}_current_threshold").set_value(75.0).run()
    at.number_input(key=f"wxu_in_{sid}_new_threshold").set_value(85.0).run()
    at.button(key=f"wxu_run_{sid}").click().run()
    res = _result(healthy(at))
    values = _students()["attendance_percentage"]
    metric = res["results"]["metrics"][0]
    assert (metric["baseline"], metric["scenario"]) == ((values >= 75).sum(), (values >= 85).sum())


def test_a_explore_suggestions_differ_between_domains(ws):
    education = _uses(_open(ws, EXPLORE, dataset=False, domain="education"))
    travel = _uses(_open(ws, EXPLORE, dataset=False, domain="travel"))
    assert education and travel and not set(education) & set(travel)
    assert all(k.startswith("travel:") for k in travel)


def test_b_dataset_inspects_real_columns_and_runs_a_grounded_scenario(ws):
    at = _open(ws, DATASET)
    text = text_of(at)
    assert "What is in students.csv" in text and "Scenarios built from these columns" in text
    assert "Exact calculations from your columns" in text and "Estimates from your data" in text and "Calculations that also need your assumptions" in text
    assert "What if current_exam_score changes by" in text and "What if … changes" not in text, "each question names its column"
    table = at.dataframe[0].value
    assert list(table["Column"]) == list(_students().columns)
    assert "introduce an additional examination" not in text, "a question no column feeds is not offered here"
    columns = set(_students().columns)
    at, sid = _choose(at, "education:generic_threshold:current_exam_score")
    assert set(at.session_state["wxu_sel"]["used_columns"]) <= columns
    at.number_input(key=f"wxu_in_{sid}_current_threshold").set_value(40.0).run()
    at.number_input(key=f"wxu_in_{sid}_new_threshold").set_value(60.0).run()
    at.button(key=f"wxu_run_{sid}").click().run()
    res = _result(healthy(at))
    scores = _students()["current_exam_score"]
    m = {x["key"]: x for x in res["results"]["metrics"]}["current_exam_score__count"]
    assert (m["baseline"], m["scenario"]) == ((scores >= 40).sum(), (scores >= 60).sum())
    assert res["results"]["evidence_kind"] == "exact_calculation" and res["columns"] == {"metric": ["current_exam_score"]}


def test_b_dataset_estimate_is_labelled_as_an_estimate(ws):
    at = _open(ws, DATASET)
    at, sid = _choose(at, "education:generic_association:")
    at.number_input(key=f"wxu_in_{sid}_change").set_value(10.0).run()
    at.button(key=f"wxu_run_{sid}").click().run()
    assert _result(healthy(at))["results"]["evidence_kind"] == "statistical_estimate"


def test_b_dataset_without_a_dataset_asks_for_one(ws):
    at = _open(ws, DATASET, dataset=False)
    assert "Choose a dataset in step 1" in text_of(at) and not _uses(at)


def test_c_manual_has_no_suggestions_and_runs_the_persons_own_numbers(ws, api):  # noqa: F811
    at = _open(ws, MANUAL, dataset=False)
    text = text_of(at)
    assert "Build your own scenario" in text and not _uses(at) and "What if" not in text
    assert at.text_input(key="wxm_v_obj").value == "" and at.number_input(key="wxm_v_var_0_base").value is None, "nothing is filled in"
    at.text_input(key="wxm_v_obj").input("Cost of running extra revision classes").run()
    at.text_input(key="wxm_v_var_0_name").input("Classes").run()
    at.number_input(key="wxm_v_var_0_base").set_value(4.0).run()
    at.number_input(key="wxm_v_var_0_new").set_value(6.0).run()
    at.text_input(key="wxm_v_var_1_name").input("Cost per class").run()
    at.number_input(key="wxm_v_var_1_base").set_value(250.0).run()
    at.text_input(key="wxm_v_out_0_name").input("Total cost").run()
    at.text_input(key="wxm_v_out_0_formula").input("classes * cost_per_class").run()
    at.text_area(key="wxm_v_assume").input("A teacher is available").run()
    at.text_area(key="wxm_v_cons").input("total_cost <= 2000 | Over the budget").run()
    at.button(key="wxm_btn_run").click().run()
    at = healthy(at)
    res = _result(at)
    m = {x["key"]: x for x in res["results"]["metrics"]}["total_cost"]
    assert (m["baseline"], m["scenario"]) == (1000, 1500)
    assert at.session_state["wxu_step"] == STEP3
    assert any("Stated by you" in a and "A teacher is available" in a for a in res["explanation"]["sections"]["assumptions"])
    assert "What this cannot tell you" in text_of(at)
    # the result survives another click (saving needs a second run of the page), and the draft survives going back to edit it
    at.text_input(key="wxu_save_name").set_value("Revision classes").run()
    at.button(key="wxu_save").click().run()
    assert [s["name"] for s in api.get("/api/whatif/scenarios", headers=ws["h"]).json()["scenarios"]] == ["Revision classes"]
    at.button(key="wxu_back").click().run()
    at = healthy(at)
    assert at.text_input(key="wxm_v_obj").value == "Cost of running extra revision classes" and at.number_input(key="wxm_v_var_0_new").value == 6.0


def test_c_manual_says_what_is_missing_instead_of_running(ws):
    at = _open(ws, MANUAL, dataset=False, domain="education")
    at.button(key="wxm_btn_run").click().run()
    assert not at.exception
    errors = " ".join(str(e.value) for e in at.error)
    assert "State the scenario objective" in errors and "Add at least one variable" in errors and not _has(at, "wxu_scn")
    at.text_input(key="wxm_v_obj").input("x").run()
    at.text_input(key="wxm_v_var_0_name").input("price").run()
    at.number_input(key="wxm_v_var_0_base").set_value(10.0).run()
    at.number_input(key="wxm_v_var_0_new").set_value(12.0).run()
    at.text_input(key="wxm_v_out_0_name").input("revenue").run()
    at.text_input(key="wxm_v_out_0_formula").input("price * unknown_thing").run()
    at.button(key="wxm_btn_run").click().run()
    assert any("unknown_thing" in str(e.value) for e in at.error) and not _has(at, "wxu_scn")


def test_c_manual_rows_can_be_added_and_removed(ws):
    at = _open(ws, MANUAL, dataset=False)
    at.button(key="wxm_btn_add_var").click().run()
    at.text_input(key="wxm_v_var_2_name").input("third").run()
    at.button(key="wxm_btn_del_var").click().run()
    at = healthy(at)
    assert "wxm_v_var_2_name" not in [t.key for t in at.text_input]
    at.button(key="wxm_btn_add_var").click().run()
    assert healthy(at).text_input(key="wxm_v_var_2_name").value == ""


def test_d_brainstorm_shows_ai_ideas_ranked_and_runs_one(ws, monkeypatch):
    asked = _model(monkeypatch, IDEAS)
    at = _open(ws, BRAINSTORM, objective=OBJECTIVE)
    assert not _uses(at) and not asked, "nothing is listed before the model is asked"
    at.button(key="wxu_go").click().run()
    at = healthy(at)
    assert OBJECTIVE in asked[0] and "study_hours_per_day" in asked[0] and "S000" not in asked[0], "columns are sent, rows are not"
    text = text_of(at)
    assert "Ideas from Silt AI" in text and "written by stub-model" in text and "3 idea(s) were left out" in text
    assert "parent_income" not in text and "Invented column" not in text and "Bad formula" not in text and "Made-up weights" not in text
    assert "results from schools that changed curriculum" in text
    ideas = at.session_state["wxu_ideas"]["out"]["suggestions"]
    assert [i["ai"]["title"] for i in ideas] == ["Lift attendance", "Raise study time", "Tutoring budget", "New curriculum"]
    assert [i["feasibility"] for i in ideas] == ["ready", "ready", "needs_assumptions", "needs_external_data"]
    assert len(_uses(at)) == 3, "an idea that needs outside evidence cannot be picked"
    at, sid = _choose(at, "ai2:")
    at.number_input(key=f"wxu_in_{sid}_change").set_value(20.0).run()
    at.button(key=f"wxu_run_{sid}").click().run()
    res = _result(healthy(at))
    assert res["results"]["status"] == "ok" and res["results"]["details"]["x"] == "study_hours_per_day"


def test_d_brainstorm_own_model_idea_takes_values_and_runs(ws, monkeypatch):
    _model(monkeypatch, IDEAS)
    at = _open(ws, BRAINSTORM, objective=OBJECTIVE)
    at.button(key="wxu_go").click().run()
    at, sid = _choose(at, "ai3:")
    assert at.number_input(key=f"wxb_{sid}_0_base").value is None
    at.number_input(key=f"wxb_{sid}_0_base").set_value(2.0).run()
    at.number_input(key=f"wxb_{sid}_0_new").set_value(5.0).run()
    at.number_input(key=f"wxb_{sid}_1_base").set_value(8.0).run()
    at.button(key=f"wxbtn_run_{sid}").click().run()
    m = _result(healthy(at))["results"]["metrics"][0]
    assert (m["baseline"], m["scenario"]) == (16, 40)


def test_d_brainstorm_unavailable_says_so_and_the_fallback_is_a_separate_choice(ws):
    at = _open(ws, BRAINSTORM, objective=OBJECTIVE)
    at.button(key="wxu_go").click().run()
    assert not at.exception and any("not set up on this server" in str(e.value) for e in at.error)
    assert not _uses(at) and "Ideas from Silt AI" not in text_of(at) and not _has(at, "wxu_ideas")
    at.button(key="wxu_fallback_btn").click().run()
    assert not at.exception
    assert "Rules-based scenarios (not written by Silt AI)" in text_of(at) and _uses(at) and not any(k.startswith("ai") for k in _uses(at))


def test_d_brainstorm_api_never_substitutes_rules(ws, api, monkeypatch):  # noqa: F811
    body = {"objective": OBJECTIVE, "dataset_id": ws["ds"]}
    assert api.post("/api/whatif/brainstorm", json=body, headers=ws["h"]).status_code == 503
    _model(monkeypatch, [IDEAS[3]])
    r = api.post("/api/whatif/brainstorm", json=body, headers=ws["h"])
    assert r.status_code == 502 and "none of its ideas matched" in r.json()["detail"]
    monkeypatch.setattr(llm_client, "call_tool", lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
    assert api.post("/api/whatif/brainstorm", json=body, headers=ws["h"]).status_code == 503


def test_switching_starting_points_leaks_nothing(ws, monkeypatch):
    _model(monkeypatch, IDEAS)
    at = _open(ws, EXPLORE)
    at, sid = _choose(at, "education:edu_attendance_requirement")
    at.number_input(key=f"wxu_in_{sid}_current_threshold").set_value(75.0).run()
    at.number_input(key=f"wxu_in_{sid}_new_threshold").set_value(85.0).run()
    at.button(key=f"wxu_run_{sid}").click().run()
    explore = set(_uses(_go(at, STEP2)))

    def switch(start: str):
        a = _go(at, "1 · Setup")
        a.radio(key="wxu_pick_start").set_value(start).run()
        a = healthy(a)
        for key in ("wxu_sel", "wxu_result", "wxu_scn", "wxu_ideas", f"wxu_in_{sid}_new_threshold"):
            assert not _has(a, key) or a.session_state[key] is None, f"{key} survived the switch to {start}"
        third = _go(a, STEP3)
        shown = text_of(third)
        assert "What if the attendance requirement changes" not in shown, f"the old question showed under {start}"
        assert not [n for n in third.number_input if str(n.key).startswith("wxu_")], f"the old inputs showed under {start}"
        return _go(third, STEP2)

    at = switch(DATASET)
    dataset = set(_uses(at))
    assert dataset and dataset != explore and "Discover what you could explore" not in text_of(at)
    at = switch(MANUAL)
    assert not _uses(at) and "Build your own scenario" in text_of(at) and "What is in" not in text_of(at)
    at.text_input(key="wxm_v_obj").input("my own draft").run()
    at = switch(BRAINSTORM)
    assert not _uses(at) and "Build your own scenario" not in text_of(at)
    at = _go(at, "1 · Setup")
    at.text_area(key="wxu_obj").input(OBJECTIVE).run()
    at = _go(at, STEP2)
    at.button(key="wxu_go").click().run()
    assert any(k.startswith("ai") for k in _uses(healthy(at)))
    at = switch(EXPLORE)
    assert set(_uses(at)) == explore and "Ideas from Silt AI" not in text_of(at)
    at = switch(MANUAL)
    assert at.text_input(key="wxm_v_obj").value == "my own draft", "each starting point keeps its own state"


def test_brainstorming_again_asks_the_model_again(ws, monkeypatch):
    asked = _model(monkeypatch, IDEAS)
    at = _open(ws, BRAINSTORM, objective=OBJECTIVE)
    at.button(key="wxu_go").click().run()
    healthy(at).button(key="wxu_go").click().run()
    assert len(asked) == 2 and _uses(healthy(at))


def test_changing_the_objective_drops_ideas_made_for_the_old_one(ws, monkeypatch):
    _model(monkeypatch, IDEAS)
    at = _open(ws, BRAINSTORM, objective=OBJECTIVE)
    at.button(key="wxu_go").click().run()
    at = _go(healthy(at), "1 · Setup")
    at.text_area(key="wxu_obj").input("I want to cut the cost of exams").run()
    at = _go(at, STEP2)
    assert not _uses(at) and not _has(at, "wxu_ideas")
