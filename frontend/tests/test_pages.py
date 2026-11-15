"""UI integration: every page touched by the platform work is run headlessly
(Streamlit AppTest) against the real FastAPI app.

The page code is the production code and so is the API: only the transport is
swapped -- `api_client`'s HTTP calls are routed into the app in-process instead
of over a socket. A page passes when it renders without an exception or an
error box and shows what the backend actually holds.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

FRONTEND_DIR = Path(__file__).resolve().parents[1]
BACKEND_TESTS = FRONTEND_DIR.parent / "backend" / "tests"

if "app.main" not in sys.modules:
    # Running the frontend tests on their own: load the backend's test setup
    # (scratch database, env) exactly as its own suite does.
    spec = importlib.util.spec_from_file_location("backend_test_setup", BACKEND_TESTS / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
sys.path.insert(0, str(BACKEND_TESTS))

from fastapi.testclient import TestClient  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import api_client  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402
from platform_helpers import DRIFTED_CSV, ask_and_run, seed_dashboard, workspace  # noqa: E402


class _Response:
    def __init__(self, r):
        self._r, self.status_code, self.text = r, r.status_code, r.text
        self.ok = r.status_code < 400
        self.content, self.headers = r.content, r.headers

    def json(self):
        return self._r.json()


class _InProcessRequests:
    """`requests`-shaped facade over the FastAPI TestClient."""

    def __init__(self, client: TestClient):
        self.client = client

    def _call(self, method: str, url: str, **kw):
        kw.pop("timeout", None)
        path = url.replace(api_client.BACKEND_BASE_URL, "")
        return _Response(self.client.request(method, path, **kw))

    def get(self, url, **kw):
        return self._call("GET", url, **kw)

    def post(self, url, **kw):
        return self._call("POST", url, **kw)

    def put(self, url, **kw):
        return self._call("PUT", url, **kw)

    def patch(self, url, **kw):
        return self._call("PATCH", url, **kw)

    def delete(self, url, **kw):
        return self._call("DELETE", url, **kw)


@pytest.fixture(scope="module")
def api():
    with TestClient(fastapi_app) as client:
        yield client


@pytest.fixture
def ws(api, monkeypatch):
    monkeypatch.setattr(api_client, "requests", _InProcessRequests(api))
    from app.rate_limit import _attempts

    _attempts.clear()
    return workspace(api)


def page(name: str, ws: dict, **state) -> AppTest:
    at = AppTest.from_file(str(FRONTEND_DIR / "pages" / name), default_timeout=60)
    at.session_state["auth_user"] = {"id": 0, "email": ws["email"], "display_name": "T"}
    at.session_state["auth_token"] = ws["token"]
    at.session_state["active_team_id"] = ws["team"]
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def text_of(at: AppTest) -> str:
    parts = [str(m.value) for m in at.markdown] + [str(c.value) for c in at.caption]
    parts += [str(x.value) for x in at.info] + [str(x.value) for x in at.warning] + [str(x.value) for x in at.success]
    return "\n".join(parts)


def healthy(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    return at


# ----------------------------------------------------------------- analyses --

def test_analyses_shows_job_route_and_trace_for_a_finished_run(api, ws):
    out = ask_and_run(api, ws, "Total revenue by region")
    at = healthy(page("3_Analyses.py", ws, active_question_id=out["question_id"]))
    text = text_of(at)
    assert "Fast path" in text and f"Job #{out['job_id']}" in text and "succeeded" in text
    assert "Model calls:</b> 0" in text, "the run trace is shown"
    assert not [b for b in at.button if b.key == "an_cancel"], "nothing to cancel on a finished run"


def test_analyses_offers_cancel_while_queued_and_cancels(api, ws):
    created = api.post("/api/questions", json={"dataset_id": ws["ds"], "question": "Summarise the business"}, headers=ws["h"]).json()
    at = healthy(page("3_Analyses.py", ws, active_question_id=created["question_id"]))
    assert "queued" in text_of(at)
    at.button(key="an_cancel").click().run()
    healthy(at)
    assert api.get(f"/api/jobs/{created['job_id']}", headers=ws["h"]).json()["state"] == "cancelled"
    assert "Cancelled" in text_of(at)


def test_analyses_presents_clarification_options(api, ws):
    for name, formula in (("Gross margin", "sum(revenue) - sum(cost)"), ("Unit margin", "(sum(revenue) - sum(cost)) / sum(units)")):
        m = api.post("/api/semantic/metrics", json={"name": name, "formula": formula, "aliases": ["margin"]}, headers=ws["h"]).json()
        api.post(f"/api/semantic/metrics/{m['id']}/versions/1/approve", headers=ws["h"])
    out = ask_and_run(api, ws, "What is our margin?")
    at = healthy(page("3_Analyses.py", ws, active_question_id=out["question_id"]))
    assert "more than one possible meaning" in text_of(at)
    options = [b for b in at.button if (b.key or "").startswith("an_clar_")]
    assert len(options) == 2 and any("Gross margin" in b.label for b in options)
    options[0].click().run()
    healthy(at)
    assert "active_question_id" not in at.session_state, "choosing an option goes back to the ask form, pre-filled"


# --------------------------------------------------------------- dashboards --

def test_dashboard_shows_verdict_evidence_and_run_record(api, ws):
    out = ask_and_run(api, ws, "Total revenue by region")
    at = healthy(page("4_Dashboards.py", ws, active_question_id=out["question_id"]))
    text = text_of(at)
    assert "Fully Verified" in text
    assert any("5 of 5 claims fully verified" in e.label for e in at.expander)
    assert "sha256" in text and "none (deterministic)" in text, "the run record is rendered"
    assert any("verified" in b.label for b in at.button if (b.key or "").startswith("kpibtn_")), "KPIs carry their evidence status"


# ---------------------------------------------------------------- scheduled --

def test_scheduled_alert_lifecycle_from_the_page(api, ws):
    from unittest.mock import patch

    from app import scheduler

    sa = api.post("/api/scheduled-analyses", json={"dataset_id": ws["ds"], "question": "Revenue?"}, headers=ws["h"]).json()
    with patch.object(scheduler, "send_email", return_value=True):
        for value in (100, 180):
            qid, _ = seed_dashboard(ws, [{"label": "Revenue", "value": f"${value}"}], trigger="scheduled", scheduled_analysis_id=sa["id"])
            scheduler.process_completed_run(qid)
    alert = api.get("/api/alerts", headers=ws["h"]).json()["alerts"][0]
    at = healthy(page("10_Scheduled.py", ws))
    text = text_of(at)
    assert "Revenue" in text and "Open" in text and "email sent" in text and "vs previous" in text
    at.button(key=f"al_ack_{alert['id']}").click().run()
    healthy(at)
    assert api.get(f"/api/alerts/{alert['id']}", headers=ws["h"]).json()["status"] == "acknowledged"
    at.button(key=f"al_res_{alert['id']}").click().run()
    healthy(at)
    assert api.get(f"/api/alerts/{alert['id']}", headers=ws["h"]).json()["status"] == "resolved"


# ------------------------------------------------------------------ copilot --

def test_copilot_page_renders_history_and_sends_a_message(api, ws):
    sid = api.post("/api/copilot/sessions", json={"dataset_id": ws["ds"]}, headers=ws["h"]).json()["id"]
    for message in ("total revenue by region", "only West"):
        api.post(f"/api/copilot/sessions/{sid}/messages", json={"message": message}, headers=ws["h"])
    at = healthy(page("19_Drilldown_Sessions.py", ws, copilot_session_id=sid))
    text = text_of(at)
    assert "region == West" in text and "an independent recomputation agrees" in text
    assert len(at.chat_message) == 4 and len(at.dataframe) == 2
    at.chat_input[0].set_value("drill into product").run()
    healthy(at)
    session = api.get(f"/api/copilot/sessions/{sid}", headers=ws["h"]).json()
    assert session["state"]["group_by"] == ["region", "product"] and len(session["turns"]) == 6
    assert len(at.chat_message) == 6


def test_copilot_page_empty_and_stale_states(api, ws):
    at = healthy(page("19_Drilldown_Sessions.py", ws))
    assert "No sessions yet" in text_of(at) and "Pick or start a session" in text_of(at)
    sid = api.post("/api/copilot/sessions", json={"dataset_id": ws["ds"]}, headers=ws["h"]).json()["id"]
    api.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    at = healthy(page("19_Drilldown_Sessions.py", ws, copilot_session_id=sid))
    assert "newer version of this dataset exists" in text_of(at)


def test_a_run_that_could_not_reach_a_model_shows_which_provider_failed_and_how(api, ws):
    """Was: only "No AI model could be reached", with the recorded reason filtered out of the trace table."""
    from platform_helpers import ask, run_job

    created = ask(api, ws, "Summarise what this dataset says about the business")
    assert run_job(created["job_id"]) == "failed"
    at = page("3_Analyses.py", ws, active_question_id=created["question_id"]).run()
    assert not at.exception, [e.value for e in at.exception]
    shown = " ".join(str(e.value) for e in at.error)
    assert "Gemini: no API key is configured." in shown and "Llama: no API key is configured." in shown
    rows = [row for frame in at.dataframe for row in frame.value.to_dict("records")]
    failed = [r for r in rows if r.get("Kind") == "model request failed"]
    assert [r["Step"] for r in failed] == ["gemini:not_configured", "llama:not_configured"], rows
    assert all("action=not_tried" in r["Detail"] for r in failed)


def test_settings_names_the_configured_models_next_to_the_key_status(api, ws):
    from app import config

    at = page("7_Settings.py", ws).run()       # the page's own warning about the test sandbox runner is an st.error
    assert not at.exception, [e.value for e in at.exception]
    text = text_of(at)
    assert f"Gemini · {config.GEMINI_MODEL}" in text and f"Llama · {config.LLAMA_MODEL}" in text and "No key" in text


# ----------------------------------------------------------- investigations --

def test_investigations_page_shows_the_report(api, ws):
    out = ask_and_run(api, ws, "Why did revenue change?")
    at = healthy(page("13_Investigations.py", ws))
    text = text_of(at)
    assert "West" in text and "Hypotheses tested" in text and "Not established" in text
    assert "not what caused it" in text, "the caveats are on the page"
    assert len(at.metric) >= 6 and len(at.dataframe) >= 1
    assert api.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]["question_id"] == out["question_id"]


def test_investigations_page_asks_without_a_model_and_shows_the_evidence_table(api, ws):
    from platform_helpers import run_job

    at = healthy(page("13_Investigations.py", ws))
    assert [b.key for b in at.button if b.key in ("inv_ask_go", "inv_why_go")] == ["inv_ask_go", "inv_why_go"]
    assert "uses no AI model" in text_of(at)

    created = api.post("/api/investigations/ask", headers=ws["h"],
                       json={"dataset_id": ws["ds"], "question": "Why did revenue decrease last month?"}).json()
    assert run_job(created["job_id"]) == "succeeded"
    at = healthy(page("13_Investigations.py", ws, investigation_id=created["investigation_id"]))
    text = text_of(at)
    for expected in ("Evidence for each candidate explanation", "region = West", "Strong evidence of location",
                     "Hypothesis, not a finding", "not testable with this dataset", "never evidence of what",
                     "Recommended follow-up analyses", "last month", "larger than the metric's ordinary variation"):
        assert expected in text, expected
    assert "interval" in text and "after adjustment" in text, "uncertainty is shown, adjusted for the segments tested"
    assert "31 days against 28" in text and "like for like the change is" in text and "Exact arithmetic" in text
    assert "--dash-surface" not in _styles(at), "the palette is borrowed for the table; the page keeps its own look"


def _hypothesis_script() -> dict:
    import test_hypothesis_engine as fixtures

    return fixtures.script()


def test_investigations_page_shows_a_hypothesis_report(api, ws):
    from platform_helpers import ask, run_job

    created = ask(api, ws, "Why did revenue drop by 20% in the last two months?")
    assert run_job(created["job_id"], _hypothesis_script()) == "succeeded"
    at = healthy(page("13_Investigations.py", ws))
    text = text_of(at)
    assert "Supported" in text and "Weakened" in text and "H1" in text and "West" in text
    for section in ("Observed result", "Statistical evidence", "Practical significance", "How to read these statistics",
                    "Possible explanations (not established)",
                    "Suggested follow-up analyses", "Data quality", "Limitations", "Assumptions made"):
        assert section in text, section
    assert "2 supported, 2 weakened, 1 inconclusive" in text and "inside this finding" in text
    assert "do not prove what caused it" in "\n".join(str(s.value) for s in at.success), "the determination is the first thing shown"
    assert "6 of 8 model calls" in text and "12 of 24 analyses" in text and "every number checked" in text
    assert [m.value for m in at.metric][:3] == ["145,800.00", "131,310.00", "-14,490.00"]
    tables = {tuple(d.value.columns)[:2]: d.value for d in at.dataframe}
    assert len(tables[("ID", "Analysis")]) == 12, "the executed analyses are listed"
    technical = tables[("Analysis", "For")]
    assert list(technical["For"])[0] == "headline" and "Welch's t-test" in set(technical["Method"])
    assert set(technical["Interval"]) >= {"—"} or all(" to " in v for v in technical["Interval"])
    assert "0" not in set(technical["Adjusted p"]) and "< 0.001" in set(technical["p"]) | set(technical["Adjusted p"]) | {"< 0.001"}
    assert "Why this method:" in text and "Null hypothesis:" in text and "Assumption —" in text
    assert "not evidence that there is no difference" in text or "not evidence that nothing changed" in text
    assert "does not say the difference is large, important or caused" in text
    assert "refund_status" in text, "rejected hypotheses are shown with the reason"

    analyses = healthy(page("3_Analyses.py", ws, active_question_id=created["question_id"]))
    assert "Hypothesis investigation" in text_of(analyses)


def test_investigations_page_starts_a_hypothesis_investigation(api, ws, monkeypatch):
    from app.agents import llm_client

    at = healthy(page("13_Investigations.py", ws))
    assert at.button(key="inv_why_go").disabled, "no question yet"
    at.text_area(key="inv_question").set_value("Why did revenue drop in the last two months?").run()
    at.button(key="inv_why_go").click().run()
    assert any("No AI model provider" in str(e.value) for e in at.error), "without a model the page says so"
    assert api.get("/api/investigations", headers=ws["h"]).json()["investigations"] == []

    # The page and the API run on their own threads, out of reach of a scripted provider, so the
    # provider check alone is switched on here; the endpoint's own gate is covered in the backend tests.
    monkeypatch.setattr(llm_client, "provider_available", lambda tool_name=None: True)
    at = healthy(page("13_Investigations.py", ws))
    at.text_area(key="inv_question").set_value("Why did revenue drop in the last two months?").run()
    at.button(key="inv_why_go").click().run()
    inv = api.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]
    assert inv["engine"] == "hypothesis" and inv["status"] == "queued"
    waiting = healthy(page("13_Investigations.py", ws))
    assert "still running" in text_of(waiting)


def test_investigations_page_empty_state(api, ws):
    at = healthy(page("13_Investigations.py", ws))
    assert "No investigations yet" in text_of(at)


# ------------------------------------------------------------------ what-if --

def test_what_if_page_computes_scenarios_and_rejects_nothing_silently(api, ws):
    at = healthy(page("14_What_If.py", ws))
    assert "not a validated forecast" in text_of(at)
    for name, value in (("price", 20.0), ("volume", 1000.0), ("unit_cost", 12.0), ("fixed_cost", 3000.0)):
        at.number_input(key=f"wi_in_unit_economics_{name}").set_value(value)
    at.number_input(key="wi_adj_unit_economics_price").set_value(10.0)
    at.run()
    healthy(at)
    frame = at.dataframe[0].value
    assert frame.loc["Profit", "baseline"] == 5000.0 and frame.loc["Profit", "your scenario"] == 7000.0
    assert {"optimistic", "pessimistic"} <= set(frame.columns)
    assert "contributions sum exactly" in text_of(at) and "elasticity is 0" in text_of(at)
    at.text_input(key="wi_name").set_value("Price test").run()
    at.button(key="wi_save").click().run()
    healthy(at)
    saved = api.get("/api/scenarios", headers=ws["h"]).json()
    assert saved["total"] == 1 and saved["scenarios"][0]["name"] == "Price test"


def _unit_economics(at, price=20.0, volume=1000.0, unit_cost=12.0, fixed_cost=3000.0):
    for name, value in (("price", price), ("volume", volume), ("unit_cost", unit_cost), ("fixed_cost", fixed_cost)):
        at.number_input(key=f"wi_in_unit_economics_{name}").set_value(value)
    return at.run()


def test_what_if_page_shows_break_even_and_the_independent_check(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    healthy(at)
    text = text_of(at)
    # 3000 / (20 - 12) = 375 units; 375 x 20 = 7500 revenue; 625 units above; 62.5% safety margin
    assert "Break-even units" in text and "375.00" in text and "7,500.00" in text and "625.00" in text and "62.5%" in text
    assert "Verified" in text and "Independent check of the scenario arithmetic" in text
    at.selectbox(key="wi_model").select("funnel").run()
    for name, value in (("visitors", 10000.0), ("conversion_rate", 0.02), ("average_order_value", 50.0), ("cost_per_visitor", 0.4)):
        at.number_input(key=f"wi_in_funnel_{name}").set_value(value)
    at.run()
    healthy(at)
    assert "Break-even conversion" in text_of(at) and "no fixed cost" in text_of(at)


def test_what_if_simulation_runs_repeats_and_is_saved_with_its_seed(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    assert at.button(key="wi_mc_run").proto.disabled, "nothing is uncertain yet, so there is nothing to run"
    assert "at least one input" in text_of(at)
    at.selectbox(key="wi_mc_dist_unit_economics_price").select("normal").run()
    at.number_input(key="wi_mc_sd_unit_economics_price").set_value(25.0)
    at.number_input(key="wi_mc_iter").set_value(4000)
    at.number_input(key="wi_mc_target").set_value(4000.0)
    at.checkbox(key="wi_mc_target_on").check().run()
    at.button(key="wi_mc_run").click().run()
    healthy(at)
    stored = at.session_state["wi_mc_result_unit_economics"]
    result = stored["result"]
    text = text_of(at)
    assert "seed 42" in text and "4,000 runs" in text and "Expected profit" in text and "5th percentile" in text
    assert "Chance profit is above zero" in text and "at least 4,000.00" in text and "assumptions you supplied" in text
    assert result["verification"]["status"] == "verified" and "Verified" in text
    assert any("Mean" in list(df.value.columns) for df in at.dataframe), "the per-output statistics table is shown"
    again = api.post("/api/scenarios/simulate", json=stored["body"], headers=ws["h"]).json()
    assert again["outputs"] == result["outputs"], "the same seed and settings reproduce the page's numbers"

    at.number_input(key="wi_mc_sd_unit_economics_price").set_value(10.0).run()
    assert "changed since this simulation ran" in text_of(at)

    at.number_input(key="wi_mc_sd_unit_economics_price").set_value(25.0).run()
    at.text_input(key="wi_name").set_value("Risky").run()
    at.button(key="wi_save").click().run()
    healthy(at)
    saved = api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"][0]
    full = api.get(f"/api/scenarios/{saved['id']}", headers=ws["h"]).json()
    assert full["request"]["simulation"]["seed"] == 42 and full["results"]["simulation"]["config"]["iterations"] == 4000


def test_what_if_simulation_errors_are_shown_not_raised(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    for name in ("price", "volume", "unit_cost"):
        at.selectbox(key=f"wi_mc_dist_unit_economics_{name}").select("normal")
    at.run()
    for pair, value in ((("price", "volume"), 0.9), (("price", "unit_cost"), 0.9), (("volume", "unit_cost"), -0.9)):
        at.slider(key=f"wi_mc_corr_unit_economics_{pair[0]}_{pair[1]}").set_value(value)
    at.run()
    at.button(key="wi_mc_run").click().run()
    assert not at.exception
    assert any("positive semidefinite" in str(e.value) for e in at.error), [e.value for e in at.error]
    assert "wi_mc_result_unit_economics" not in at.session_state


def test_what_if_goal_seek_solves_and_reports_an_impossible_goal(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.selectbox(key="wi_gs_out_unit_economics").select("margin_pct")
    at.number_input(key="wi_gs_value_unit_economics").set_value(20.0)
    at.selectbox(key="wi_gs_var_unit_economics").select("price").run()
    at.button(key="wi_gs_run").click().run()
    healthy(at)
    text = text_of(at)
    # margin = ((p - 12) x 1000 - 3000) / (1000 p) = 0.2  ->  p = 18.75
    assert "Solved" in text and "18.75" in text and "Independent check of the solution" in text
    assert at.session_state["wi_gs_result_unit_economics"]["result"]["verification"]["status"] == "verified"
    at.number_input(key="wi_gs_value_unit_economics").set_value(150.0).run()
    assert "changed since this was solved" in text_of(at)
    at.button(key="wi_gs_run").click().run()
    healthy(at)
    assert "No solution" in text_of(at) and "only reach" in text_of(at)


def test_what_if_goal_seek_respects_adjusted_inputs_and_constraints(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.number_input(key="wi_adj_unit_economics_volume").set_value(-50.0).run()
    assert "stay at their adjusted values" in text_of(at)
    assert "volume" not in [o for o in at.selectbox(key="wi_gs_var_unit_economics").options], "an adjusted input is not offered"
    at.selectbox(key="wi_gs_out_unit_economics").select("margin_pct")
    at.number_input(key="wi_gs_value_unit_economics").set_value(20.0)
    at.selectbox(key="wi_gs_var_unit_economics").select("price").run()
    at.button(key="wi_gs_run").click().run()
    healthy(at)
    assert "22.5" in text_of(at)          # volume 500: 0.8 p = 18  ->  p = 22.5
    at.selectbox(key="wi_gs_cm_unit_economics_1").select("price")
    at.selectbox(key="wi_gs_co_unit_economics_1").select("<=")
    at.number_input(key="wi_gs_cv_unit_economics_1").set_value(20.0).run()
    at.button(key="wi_gs_run").click().run()
    healthy(at)
    assert "No feasible solution" in text_of(at)


def _demand_csv(n=36, e=-1.5, seed=0) -> bytes:
    import io

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    price = 20 * np.exp(rng.normal(0, 0.12, n))
    frame = pd.DataFrame({"date": pd.date_range("2022-01-01", periods=n, freq="MS"), "price": price,
                          "units": 1000 * (price / 20) ** e * np.exp(rng.normal(0, 0.05, n))})
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    return buf.getvalue().encode()


def _pick(at, key: str, label: str):
    box = at.selectbox(key=key)
    return box.select_index([str(o) for o in box.options].index(label))


def test_what_if_elasticity_estimate_is_a_suggestion_the_user_must_choose_to_use(api, ws):
    up = api.post("/api/datasets/upload", files={"file": ("demand.csv", _demand_csv(), "text/csv")}, headers=ws["h"])
    assert up.status_code == 200, up.text
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.slider(key="wi_as_unit_economics_price_elasticity").set_value(-0.7).run()
    assert at.button(key="wi_el_run").proto.disabled, "no columns chosen yet"
    _pick(at, "wi_el_ds", "demand.csv").run()
    _pick(at, "wi_el_price", "price").run()
    _pick(at, "wi_el_qty", "units").run()
    _pick(at, "wi_el_date", "date")
    at.run()
    at.button(key="wi_el_run").click().run()
    healthy(at)
    text = text_of(at)
    result = at.session_state["wi_el_result_unit_economics"]
    assert result["status"] == "estimated" and abs(result["elasticity"]["estimate"] + 1.5) < 0.25
    assert "Estimated elasticity" in text and "36 observations" in text and "Statistical estimate" in text and "Usable" in text
    assert at.slider(key="wi_as_unit_economics_price_elasticity").value == -0.7, "the manual value is untouched by an estimate"
    assert "Your value in the slider: <b>-0.7</b>" in text
    saved_manual = _save(at, api, ws, "Manual elasticity")
    assert saved_manual["request"]["assumption_sources"] == {}

    at.button(key="wi_el_use").click().run()
    healthy(at)
    chosen = at.slider(key="wi_as_unit_economics_price_elasticity").value
    assert chosen == round(result["elasticity"]["estimate"], 2)
    saved_used = _save(at, api, ws, "Estimated elasticity")
    src = saved_used["request"]["assumption_sources"]["price_elasticity"]
    assert src["source"] == "estimate" and src["provenance"]["dataset_version"] == 1 and src["chosen_value"] == chosen

    at.slider(key="wi_as_unit_economics_price_elasticity").set_value(-2.0).run()
    assert _save(at, api, ws, "Back to manual")["request"]["assumption_sources"] == {}, "an edited value is no longer the estimate"


def _save(at, api, ws, name: str) -> dict:
    at.text_input(key="wi_name").set_value(name).run()
    at.button(key="wi_save").click().run()
    healthy(at)
    listing = api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"]
    return api.get(f"/api/scenarios/{next(s for s in listing if s['name'] == name)['id']}", headers=ws["h"]).json()


def test_what_if_elasticity_refuses_unsuitable_data_and_offers_no_button(api, ws):
    api.post("/api/datasets/upload", files={"file": ("tiny.csv", _demand_csv(n=5), "text/csv")}, headers=ws["h"])
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    _pick(at, "wi_el_ds", "tiny.csv").run()
    _pick(at, "wi_el_price", "price").run()
    _pick(at, "wi_el_qty", "units").run()
    _pick(at, "wi_el_date", "date")
    at.run()
    at.button(key="wi_el_run").click().run()
    healthy(at)
    text = text_of(at)
    assert "Not estimated" in text and "only 5 usable observation" in text
    assert not [b for b in at.button if b.key == "wi_el_use"], "an unusable estimate cannot be applied"


def _fill_model(at, name="Revenue plan", formula="customers * price", profit_formula="revenue - 1000"):
    at.text_input(key="cm_name").set_value(name)
    for i, (n, unit) in enumerate((("customers", "customers"), ("price", "currency/customers"))):
        at.text_input(key=f"cm_in_{i}_name").set_value(n)
        at.text_input(key=f"cm_in_{i}_label").set_value(n.title())
        at.text_input(key=f"cm_in_{i}_unit").set_value(unit)
    for i, (n, unit, f) in enumerate((("revenue", "currency", formula), ("profit", "currency", profit_formula))):
        at.text_input(key=f"cm_out_{i}_name").set_value(n)
        at.text_input(key=f"cm_out_{i}_unit").set_value(unit)
        at.text_input(key=f"cm_out_{i}_formula").set_value(f)
    at.run()
    at.selectbox(key="cm_primary").select("profit")
    return at.run()


def test_what_if_custom_model_is_built_checked_saved_and_run_from_the_page(api, ws):
    at = healthy(page("14_What_If.py", ws))
    _fill_model(at, formula="customers * price", profit_formula="revenue - nothing")
    at.button(key="cm_validate").click().run()
    healthy(at)
    text = text_of(at)
    assert "Not valid" in text and "nothing" in text, "an unknown name is reported with its reason, nothing is saved"
    assert api.get("/api/scenario-models", headers=ws["h"]).json()["models"] == []

    at.text_input(key="cm_out_1_formula").set_value("revenue - 1000").run()
    at.button(key="cm_validate").click().run()
    healthy(at)
    assert "Valid" in text_of(at) and "revenue → profit" in text_of(at)
    at.button(key="cm_save_new").click().run()
    healthy(at)
    models = api.get("/api/scenario-models", headers=ws["h"]).json()["models"]
    assert [m["name"] for m in models] == ["Revenue plan"] and models[0]["ref"].endswith("@1")
    assert "Saved 'Revenue plan' as version 1." in text_of(at)

    ref = models[0]["ref"]
    assert "Revenue plan (custom, v1)" in [str(o) for o in at.selectbox(key="wi_model").options]
    at.selectbox(key="wi_model").select(ref).run()
    at.number_input(key=f"wi_in_{ref}_customers").set_value(200.0)
    at.number_input(key=f"wi_in_{ref}_price").set_value(15.0)
    at.number_input(key=f"wi_adj_{ref}_price").set_value(10.0).run()
    healthy(at)
    frame = at.dataframe[0].value
    # 200 x 15 = 3000 revenue, profit 2000; price +10% = 16.5 -> revenue 3300, profit 2300
    assert frame.loc["profit", "baseline"] == 2000.0 and frame.loc["profit", "your scenario"] == 2300.0
    assert "Independent check of the scenario arithmetic" in text_of(at) and "Verified" in text_of(at)
    assert "custom model" in text_of(at), "the custom-model note replaces the built-in elasticity wording"
    assert "Break-even" not in text_of(at), "a custom model has no built-in break-even"
    at.button(key="wi_mc_run")  # the simulation and goal-seek tools are present for custom models too
    at.selectbox(key=f"wi_mc_dist_{ref}_price").select("normal").run()
    at.number_input(key="wi_mc_iter").set_value(2000)
    at.button(key="wi_mc_run").click().run()
    healthy(at)
    assert at.session_state[f"wi_mc_result_{ref}"]["result"]["verification"]["status"] == "verified"
    at.number_input(key=f"wi_adj_{ref}_price").set_value(0.0).run()       # an adjusted input is not offered as the one to solve for
    at.selectbox(key=f"wi_gs_out_{ref}").select("profit")
    at.number_input(key=f"wi_gs_value_{ref}").set_value(5000.0)
    at.selectbox(key=f"wi_gs_var_{ref}").select("price").run()
    at.button(key="wi_gs_run").click().run()
    healthy(at)
    assert abs(at.session_state[f"wi_gs_result_{ref}"]["result"]["solutions"][0]["value"] - 30.0) < 1e-9   # 200 x p - 1000 = 5000


def test_what_if_custom_model_can_be_edited_into_a_new_version_without_changing_the_old_one(api, ws):
    at = healthy(page("14_What_If.py", ws))
    _fill_model(at)
    at.button(key="cm_save_new").click().run()
    healthy(at)
    model = api.get("/api/scenario-models", headers=ws["h"]).json()["models"][0]
    at.button(key=f"cm_edit_{model['id']}").click().run()
    healthy(at)
    assert at.text_input(key="cm_name").value == "Revenue plan" and at.text_input(key="cm_out_1_formula").value == "revenue - 1000"
    at.text_input(key="cm_out_1_formula").set_value("revenue - 2000").run()
    at.text_input(key="cm_note").set_value("fixed cost is 2000").run()
    at.button(key="cm_save_version").click().run()
    healthy(at)
    got = api.get(f"/api/scenario-models/{model['id']}", headers=ws["h"]).json()
    assert got["current_version"] == 2 and got["definition"]["outputs"]["profit"]["formula"] == "revenue - 2000"
    old = api.get(f"/api/scenario-models/{model['id']}/versions/1", headers=ws["h"]).json()
    assert old["definition"]["outputs"]["profit"]["formula"] == "revenue - 1000" and got["versions"][1]["note"] == "fixed cost is 2000"
    assert "Revenue plan (custom, v2)" in [str(o) for o in at.selectbox(key="wi_model").options]


def test_what_if_custom_baseline_comes_from_an_approved_metric_and_says_so(api, ws):
    m = api.post("/api/semantic/metrics", json={"name": "Total revenue", "formula": "sum(revenue)"}, headers=ws["h"]).json()
    assert api.post(f"/api/semantic/metrics/{m['id']}/versions/1/approve", headers=ws["h"]).status_code == 200
    definition = {"inputs": {"base": {"label": "Revenue base", "unit": "currency", "min": 0, "source": {"type": "metric", "metric_id": m["id"]}},
                             "uplift": {"label": "Uplift", "unit": "fraction", "min": 0}},
                  "outputs": {"projected": {"unit": "currency", "formula": "base * (1 + uplift)"}}, "primary_output": "projected"}
    made = api.post("/api/scenario-models", json={"name": "Revenue uplift", "definition": definition}, headers=ws["h"]).json()
    ref = made["ref"]
    at = healthy(page("14_What_If.py", ws))
    at.selectbox(key="wi_model").select(ref).run()
    at.button(key="wi_base_go").click().run()
    healthy(at)
    assert at.number_input(key=f"wi_in_{ref}_base").value == 277110.0
    text = text_of(at)
    assert "approved metric “Total revenue” v1" in text and "Revenue base" in text and "not in the data" in text
    api.post(f"/api/semantic/metrics/{m['id']}/deprecate", headers=ws["h"])
    at.button(key="wi_base_go").click().run()
    assert any("deprecated, not approved" in str(e.value) for e in at.error), [e.value for e in at.error]


def test_what_if_baseline_can_be_taken_from_one_period_and_the_scenario_remembers_it(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    for f, col in (("quantity_column", "units"), ("amount_column", "revenue"), ("total_cost_column", "cost")):
        _pick(at, f"wi_map_unit_economics_{f}", col)
    at.run()
    _pick(at, "wi_date", "order_date").run()
    at.selectbox(key="wi_ptype").select("month").run()
    months = api.get("/api/scenarios/periods", params={"dataset_id": ws["ds"], "date_column": "order_date"}, headers=ws["h"]).json()["months"]
    assert "(" in str(at.selectbox(key="wi_month").options[0]) and f"{months[0]['rows']:,} rows" in str(at.selectbox(key="wi_month").options[0])
    at.selectbox(key="wi_month").select_index(0).run()
    at.button(key="wi_base_go").click().run()
    healthy(at)
    text = text_of(at)
    want = api.post("/api/scenarios/baseline", json={"dataset_id": ws["ds"], "model": "unit_economics", "date_column": "order_date",
                    "mapping": {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"},
                    "period": {"type": "month", "year": months[0]["year"], "month": months[0]["month"]}}, headers=ws["h"]).json()
    assert at.number_input(key="wi_in_unit_economics_volume").value == want["inputs"]["volume"]
    assert at.number_input(key="wi_in_unit_economics_volume").value < sum(m["rows"] for m in months) * 100
    assert "Historical actuals" in text and want["period"]["label"] in text and "dataset version 1" in text and "Independent check of the baseline" in text
    at.number_input(key="wi_in_unit_economics_fixed_cost").set_value(500.0).run()
    saved = _save(at, api, ws, "March baseline")
    src = saved["request"]["baseline_source"]
    assert src["period"] == {"type": "month", "year": months[0]["year"], "month": months[0]["month"]} and src["date_column"] == "order_date"
    assert src["provenance"]["dataset_version"] == 1 and len(src["provenance"]["content_sha256"]) == 64


def test_what_if_compare_periods_shows_actuals_the_change_and_its_drivers(api, ws):
    at = healthy(page("14_What_If.py", ws))
    assert "Historical actuals" in text_of(at) and "not produced here" in text_of(at), "the kinds of figure are explained"
    for f, col in (("quantity_column", "units"), ("amount_column", "revenue"), ("total_cost_column", "cost")):
        _pick(at, f"cp_map_unit_economics_{f}", col)
    at.run()
    at.selectbox(key="cp_a_ptype").select("month")
    at.selectbox(key="cp_b_ptype").select("month").run()
    at.selectbox(key="cp_a_month").select_index(0)
    at.selectbox(key="cp_b_month").select_index(2).run()
    at.button(key="cp_run").click().run()
    assert any("fixed_cost" in str(e.value) for e in at.error), "an input the data does not supply is asked for, not assumed"
    at.number_input(key="cp_ov_unit_economics_fixed_cost").set_value(500.0).run()
    at.button(key="cp_run").click().run()
    healthy(at)
    r = at.session_state["cp_result"]
    assert r["data_kind"] == "historical_actual" and r["verification"]["status"] == "verified"
    assert r["periods"]["a"]["period"]["type"] == "month" and r["periods"]["a"]["period"]["label"] != r["periods"]["b"]["period"]["label"]
    text = text_of(at)
    assert "Historical actuals" in text and "dataset version 1" in text and "not a forecast" in text
    assert any("Period A" in list(df.value.columns) for df in at.dataframe)
    assert "Independent check of the comparison" in text and "Verified" in text


def test_what_if_segments_tab_adjusts_one_segment_and_shows_each_segment_and_the_total(api, ws):
    listing = api.get("/api/scenarios/segment-values", params={"dataset_id": ws["ds"], "segment_column": "region"}, headers=ws["h"]).json()
    target = listing["segments"][0]["value"]
    at = healthy(page("14_What_If.py", ws))
    _pick(at, "sg_column", "region").run()
    for f, col in (("quantity_column", "units"), ("amount_column", "revenue"), ("total_cost_column", "cost")):
        _pick(at, f"sg_map_unit_economics_{f}", col)
    at.run()
    at.button(key="sg_run").click().run()
    assert any("has no value in the data" in str(e.value) for e in at.error), "an input the data cannot supply is asked for, not assumed"
    at.selectbox(key="sg_mode_unit_economics_fixed_cost").select("constant").run()
    at.multiselect(key="sg_pick").select(target).run()
    at.number_input(key=f"sg_adj_unit_economics_{target}_price").set_value(10.0).run()
    at.button(key="sg_run").click().run()
    healthy(at)
    r = at.session_state["sg_result"]
    assert r["verification"]["status"] == "verified" and r["partition"]["complete"]
    seg = {s["value"]: s for s in r["segments"]}
    assert seg[target]["scenario"]["outputs"]["revenue"] == pytest.approx(seg[target]["baseline"]["outputs"]["revenue"] * 1.1, rel=1e-9)
    assert r["contributions"]["by_segment"][target]["share_pct"] == pytest.approx(100.0)
    assert all(s["delta"]["profit"]["abs"] == 0 for v, s in seg.items() if v != target)
    text = text_of(at)
    assert "Every row counted once" in text and "Independent check of the segments" in text and "Verified" in text
    assert "each row is in exactly one segment" in text.replace("every row", "each row") or "exactly one segment" in text
    assert any("Share of total change %" in list(df.value.columns) for df in at.dataframe)
    again = api.post("/api/scenarios/segments", json={"dataset_id": ws["ds"], "model": "unit_economics", "segment_column": "region", "constant_inputs": {"fixed_cost": 0.0},
                                                      "mapping": {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": "cost"},
                                                      "segment_adjustments": {target: {"price": {"type": "pct", "value": 10.0}}}}, headers=ws["h"]).json()
    assert again["overall"]["scenario"]["outputs"] == r["overall"]["scenario"]["outputs"], "the page shows exactly what the API computed"


def test_what_if_segments_tab_shares_a_total_cost_out_once(api, ws):
    at = healthy(page("14_What_If.py", ws))
    _pick(at, "sg_column", "region").run()
    for f, col in (("quantity_column", "units"), ("amount_column", "revenue"), ("total_cost_column", "cost")):
        _pick(at, f"sg_map_unit_economics_{f}", col)
    at.run()
    at.selectbox(key="sg_mode_unit_economics_fixed_cost").select("total").run()
    at.number_input(key="sg_val_unit_economics_fixed_cost").set_value(4000.0).run()
    _pick(at, "sg_by_unit_economics_fixed_cost", "In proportion to Units sold").run()
    at.button(key="sg_run").click().run()
    healthy(at)
    r = at.session_state["sg_result"]
    assert sum(s["baseline"]["inputs"]["fixed_cost"] for s in r["segments"]) == pytest.approx(4000.0)
    assert r["allocations"]["fixed_cost"]["allocate_by"] == "volume" and r["overall"]["baseline"]["inputs"]["fixed_cost"] == 4000.0
    assert "in proportion to volume" in text_of(at) and r["verification"]["status"] == "verified"


def test_what_if_heatmap_draws_the_grid_marks_break_even_and_is_checked(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.number_input(key="gr_xl_unit_economics").set_value(-50.0)
    at.number_input(key="gr_xh_unit_economics").set_value(50.0)
    at.number_input(key="gr_yl_unit_economics").set_value(-50.0)
    at.number_input(key="gr_yh_unit_economics").set_value(50.0)
    at.run()
    at.button(key="gr_run").click().run()
    healthy(at)
    g = at.session_state["gr_result_unit_economics"]
    assert g["x"]["input"] == "price" and g["y"]["input"] == "volume" and g["verification"]["status"] == "verified"
    assert g["boundary"]["points"] and all(abs(p["x"] - (12 + 3000 / p["y"])) < 1e-6 for p in g["boundary"]["points"] if p["edge"] == "row")
    text = text_of(at)
    assert "Best on this grid" in text and "crosses 0" in text and "Independent check of the grid" in text and "Verified" in text
    assert "not claimed to be a global optimum" in text
    want = api.post("/api/scenarios/sensitivity-grid", json={"model": "unit_economics", "baseline": {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0},
                                                             "x_input": "price", "y_input": "volume", "resolution": 21,
                                                             "x_range": {"mode": "pct", "low": -50.0, "high": 50.0}, "y_range": {"mode": "pct", "low": -50.0, "high": 50.0}},
                    headers=ws["h"]).json()
    assert want["z"] == g["z"], "the page draws exactly what the API computed"


def test_what_if_heatmap_reports_an_invalid_choice_instead_of_failing(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.selectbox(key="gr_y_unit_economics").select("price").run()          # the same input on both axes
    at.button(key="gr_run").click().run()
    assert not at.exception and any("two different inputs" in str(e.value) for e in at.error)
    assert "gr_result_unit_economics" not in at.session_state
    at.selectbox(key="gr_y_unit_economics").select("unit_cost").run()
    at.selectbox(key="gr_out_unit_economics").select("margin_pct").run()
    at.button(key="gr_run").click().run()
    healthy(at)
    assert at.session_state["gr_result_unit_economics"]["output"]["key"] == "margin_pct"


# ------------------------------------------------------------- data quality --

def test_data_quality_page_lists_incidents_and_resolves_one(api, ws):
    at = healthy(page("15_Data_Quality.py", ws))
    assert "No incidents" in text_of(at)
    api.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    at = healthy(page("15_Data_Quality.py", ws))
    text = text_of(at)
    assert "column removed" in text and "channel" in text and "Distribution drift" in text and "History by version" in text
    incident = api.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()["incidents"][0]
    at.button(key=f"dq_res_{incident['id']}").click().run()
    healthy(at)
    after = api.get(f"/api/datasets/{ws['ds']}/quality", headers=ws["h"]).json()
    assert next(i for i in after["incidents"] if i["id"] == incident["id"])["status"] == "resolved"
    assert after["open_incidents"] == len(after["incidents"]) - 1


# ----------------------------------------------------------- semantic layer --

def test_semantic_page_lists_metrics_and_approves_a_draft(api, ws):
    at = healthy(page("16_Semantic_Layer.py", ws))
    assert "No metrics defined yet" in text_of(at)
    m = api.post("/api/semantic/metrics", json={"name": "Net revenue", "formula": "sum(revenue) - sum(cost)", "aliases": ["net sales"]}, headers=ws["h"]).json()
    at = healthy(page("16_Semantic_Layer.py", ws))
    assert any("Net revenue" in e.label and "draft" in e.label for e in at.expander)
    at.button(key=f"sm_ap_{m['id']}").click().run()
    healthy(at)
    assert api.get(f"/api/semantic/metrics/{m['id']}", headers=ws["h"]).json()["status"] == "approved"
    at.text_input(key="sm_formula").set_value("__import__('os').system('id')").run()
    assert any("Only aggregations" in str(e.value) for e in at.error), "an invalid formula is rejected on the page"
    at.text_input(key="sm_formula").set_value("sum(revenue) / sum(units)").run()
    assert not at.error and any("Valid" in str(x.value) for x in at.success)


# --------------------------------------------------------------- evaluation --

def test_evaluation_page_shows_a_stored_run(api, ws):
    from app.database import SessionLocal
    from app.evals import runner

    report = runner.run_suite("offline", components=["routing", "verification"])
    db = SessionLocal()
    try:
        run_id = runner.store_run(db, report)
    finally:
        db.close()
    at = healthy(page("17_Evaluation.py", ws))
    assert at.metric[0].value == f"{report['summary']['passed']} / {report['summary']['cases']}"
    assert f"Suite <b>{report['suite_version']}</b>" in text_of(at)
    assert run_id > 0


# ----------------------------------------------------------------- settings --

def test_settings_page_reports_the_sandbox_honestly(api, ws):
    at = healthy_except_expected(page("7_Settings.py", ws))
    text = text_of(at) + "\n".join(str(e.value) for e in at.error)
    assert "not a security boundary" in text, "the development runner is called out on the Settings page"


def healthy_except_expected(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    return at


# ------------------------------------------------- sessions and connectivity --
# A rejected token and an unreachable server are different problems and must
# never be reported as each other.

SESSION_KEYS = ("auth_user", "auth_token", "active_team_id", "active_community_id")


def _state(at: AppTest, key: str):
    return at.session_state[key] if key in at.session_state else None


def _errors(at: AppTest) -> str:
    return "\n".join(str(e.value) for e in at.error)


@pytest.mark.parametrize("name", ["13_Investigations.py", "3_Analyses.py", "4_Dashboards.py", "1_Overview.py"])
def test_a_rejected_token_sends_the_user_to_sign_in_with_a_clear_message(api, ws, name):
    at = page(name, ws, auth_token="expired-or-signed-with-an-old-key")
    assert not at.exception, [e.value for e in at.exception]
    assert [str(w.value) for w in at.warning] == [api_client.SESSION_EXPIRED_NOTICE]
    assert "Backend unreachable" not in _errors(at) and "401" not in _errors(at), "a rejected token is not a connectivity problem"
    assert all(_state(at, k) is None for k in SESSION_KEYS), "the dead session is cleared"
    assert {t.key for t in at.text_input} >= {"login_email", "login_password"}, "the existing login gate is shown"


def test_the_session_notice_is_shown_once_and_signing_in_again_works(api, ws):
    at = page("13_Investigations.py", ws, auth_token="expired")
    assert len(at.warning) == 1
    at.run()
    assert not at.warning and {t.key for t in at.text_input} >= {"login_email"}, "no repeat of the notice, no redirect loop"

    at.text_input(key="login_email").set_value(ws["email"])
    at.text_input(key="login_password").set_value("password123")
    next(b for b in at.button if b.label == "Log in").click().run()
    assert not at.exception and _state(at, "auth_token") and _state(at, "auth_token") != "expired"
    assert not at.warning and "login_email" not in {t.key for t in at.text_input}, "signed in: the gate is gone"


def test_a_wrong_password_is_a_login_error_not_an_expired_session(api, ws):
    at = AppTest.from_file(str(FRONTEND_DIR / "pages" / "13_Investigations.py"), default_timeout=60)
    at.query_params["view"] = "login"  # the sign-in page; the bare address is the landing page
    at.run()
    at.text_input(key="login_email").set_value(ws["email"])
    at.text_input(key="login_password").set_value("definitely-wrong")
    next(b for b in at.button if b.label == "Log in").click().run()
    assert not at.exception and "Could not log in: 401: Incorrect email or password" in _errors(at)
    assert not at.warning and _state(at, "auth_notice") is None, "no token was sent, so nothing expired"
    assert {t.key for t in at.text_input} >= {"login_email"}, "still on the form; no loop"


def test_an_unreachable_backend_is_reported_as_unreachable_and_keeps_the_session(ws, monkeypatch):
    monkeypatch.setattr(api_client, "requests", api_client._Http())           # the real transport,
    monkeypatch.setattr(api_client, "BACKEND_BASE_URL", "http://127.0.0.1:9") # at an address where nothing listens
    at = page("13_Investigations.py", ws)
    assert not at.exception, "a dead server is an error message, not a traceback"
    assert "Backend unreachable: no response from http://127.0.0.1:9 (ConnectionError)" in _errors(at)
    assert not at.warning and _state(at, "auth_notice") is None, "nothing says the session expired"
    assert _state(at, "auth_token") == ws["token"] and _state(at, "active_team_id") == ws["team"], "still signed in when the server returns"
    assert "login_email" not in {t.key for t in at.text_input}


def test_the_client_tells_the_three_failures_apart(monkeypatch):
    import requests as real_requests
    import streamlit as st

    class Resp:
        def __init__(self, status, detail="x", sent_token=None):
            self.status_code, self.ok, self.text, self._detail = status, status < 400, detail, detail
            if sent_token is not None:
                self.request = type("Req", (), {"headers": {"Authorization": "Bearer t"} if sent_token else {}})()

        def json(self):
            return {"detail": self._detail}

    st.session_state.clear()
    st.session_state["auth_token"], st.session_state["auth_user"], st.session_state["active_team_id"] = "t", {"id": 1}, 7
    for status in (400, 403, 404, 409, 422, 429, 500, 503):
        with pytest.raises(api_client.ApiError) as e:
            api_client._handle(Resp(status, "nope", sent_token=True))
        assert type(e.value) is api_client.ApiError and str(e.value) == f"{status}: nope", "other errors are unchanged"
    assert st.session_state["auth_token"] == "t", "only a 401 ends the session"

    with pytest.raises(api_client.ApiError) as e:                  # 401 on a request that carried no token (the login form)
        api_client._handle(Resp(401, "Incorrect email or password", sent_token=False))
    assert type(e.value) is api_client.ApiError and st.session_state["auth_token"] == "t" and "auth_notice" not in st.session_state

    with pytest.raises(api_client.SessionExpired, match="sign in again"):
        api_client._handle(Resp(401, "Invalid or expired token", sent_token=True))
    assert not any(k in st.session_state for k in SESSION_KEYS) and st.session_state["auth_notice"] == api_client.SESSION_EXPIRED_NOTICE
    assert issubclass(api_client.SessionExpired, api_client.ApiError) and issubclass(api_client.BackendUnreachable, api_client.ApiError)

    http = api_client._Http()
    for failure in (real_requests.exceptions.ConnectionError("refused"), real_requests.exceptions.Timeout("slow")):
        def boom(*a, _f=failure, **k):
            raise _f
        monkeypatch.setattr(real_requests, "get", boom)
        with pytest.raises(api_client.BackendUnreachable, match=type(failure).__name__):
            http.get("http://x")
    monkeypatch.setattr(real_requests, "get", lambda *a, **k: (_ for _ in ()).throw(ValueError("a bug, not the network")))
    with pytest.raises(ValueError):
        http.get("http://x")
    assert http.exceptions is real_requests.exceptions
    st.session_state.clear()


# --------------------------------------------------------- dashboard design --

def _styles(at: AppTest) -> str:
    return "\n".join(str(m.value) for m in at.markdown if "<style" in str(m.value))


def _figures(at: AppTest) -> list:
    """The figures the page drew, as Plotly figures (binary-encoded arrays decoded)."""
    import plotly.io as pio

    return [pio.from_json(c.proto.spec) for c in at.get("plotly_chart")]


def _numbers(values) -> list[float]:
    """A trace's values as plain numbers, whether Plotly sent a list or a packed array."""
    import base64

    import numpy as np

    if isinstance(values, dict) and "bdata" in values:
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=values["dtype"]).tolist()
    return [float(v) for v in values]


def test_the_dashboard_look_is_applied_on_the_dashboard_page_only(api, ws):
    from style import theme

    out = ask_and_run(api, ws, "Total revenue by region")
    dash = healthy(page("4_Dashboards.py", ws, active_question_id=out["question_id"]))
    css = _styles(dash)
    assert "--dash-surface" in css and theme.DASHBOARD_TOKENS["--accent"] in css
    assert "--dash-surface" in _styles(healthy(page("4_Dashboards.py", ws))), "the list of dashboards uses it too"
    for other in ("3_Analyses.py", "1_Overview.py", "13_Investigations.py"):
        assert "--dash-surface" not in _styles(healthy(page(other, ws, active_question_id=out["question_id"]))), other

    sheet = theme.dashboard_theme_css()
    declared = sheet.split("}")[0]
    assert declared.startswith("<style>.stMain,") and ":root" not in sheet, "tokens are set on the main area, never globally"
    assert "stSidebar" not in sheet, "the sidebar keeps the base look"
    assert all(f"{name}: {value};" in declared for name, value in theme.DASHBOARD_TOKENS.items())
    assert theme.ACCENT == "#250e2c" and "--accent: #250e2c" in theme.BASE_CSS, "the base tokens are untouched"


def test_the_dashboard_shows_whatever_the_analysis_produced(api, ws):
    """Nothing on the page is fixed: a dashboard with other KPIs, more of them and no charts shows exactly those."""
    kpis = [{"label": f"Tonnes shipped via dock {i}", "value": f"{i * 1234.5:,.1f}", "element_id": f"{i:032x}"} for i in range(1, 7)]
    kpis[4].pop("element_id")
    qid, _ = seed_dashboard(ws, kpis, verdict_state="VERIFIED_WITH_CAVEATS")
    at = healthy(page("4_Dashboards.py", ws, active_question_id=qid))
    cards = [b for b in at.button if (b.key or "").startswith("kpibtn_")]
    assert len(cards) == 6, "every KPI is shown, not only the first four"
    for kpi, card in zip(kpis, cards):
        assert kpi["label"] in card.label and f"**{kpi['value']}**" in card.label
        assert card.disabled is (not kpi.get("element_id")), "a KPI without an inspectable element is not clickable"
    text = text_of(at)
    assert "Key metrics <b>6</b>" in text
    assert "Charts <b>" not in text and "Narrative Summary" not in text and "Chart Studio" not in text
    assert not _figures(at) and "Verified with caveats" in text

    empty_qid, _ = seed_dashboard(ws, [])
    bare = healthy(page("4_Dashboards.py", ws, active_question_id=empty_qid))
    assert "Key metrics" not in text_of(bare) and not [b for b in bare.button if (b.key or "").startswith("kpibtn_")]


def test_dashboard_charts_narrative_and_inspection_come_from_the_run(api, ws):
    from style import theme

    out = ask_and_run(api, ws, "Total revenue by region")
    stored = api.get(f"/api/questions/{out['question_id']}/dashboard", headers=ws["h"]).json()
    at = healthy(page("4_Dashboards.py", ws, active_question_id=out["question_id"]))
    text = text_of(at)
    assert f"Key metrics <b>{len(stored['kpis'])}</b>" in text and f"Charts <b>{len(stored['charts'])}</b>" in text
    assert stored["narrative"] and stored["narrative"][:60].replace("'", "&#x27;") in text, "the narrative is the stored one"
    for chart in stored["charts"]:
        assert chart["title"] in text

    figures = _figures(at)
    assert len(figures) == len(stored["charts"]) >= 1
    drawn, source = figures[0].data[0], theme.figure_from_json(stored["charts"][0]["plotly_json"]).data[0]
    assert list(drawn.y) == list(source.y) and list(drawn.x) == list(source.x), "the plotted values are the computed ones"
    assert drawn.marker.color == theme.DASHBOARD_COLORWAY[0], "generated charts take the dashboard's data colours"
    assert list(figures[0].layout.colorway) == theme.DASHBOARD_COLORWAY

    kpi = next(b for b in at.button if (b.key or "").startswith("kpibtn_") and not b.disabled)
    kpi.click().run()
    healthy(at)
    assert at.session_state["_inspect_open"] is True and at.session_state["_inspect_element_id"] == kpi.key[len("kpibtn_"):]
    assert "Evidence" in text_of(at), "the inspect panel still opens from a KPI, with its evidence"


def test_restyling_in_chart_studio_keeps_the_user_s_palette(api, ws):
    import chart_studio as cs
    from style import theme

    out = ask_and_run(api, ws, "Total revenue by region")
    at = healthy(page("4_Dashboards.py", ws, active_question_id=out["question_id"]))
    before = _figures(at)[0].data[0]
    pills = next(w for w in at.get("button_group") if "type_" in (w.key or ""))
    pills.set_value("hbar").run()
    healthy(at)
    after = _figures(at)[0].data[0]
    assert after.orientation == "h" and sorted(_numbers(after.x)) == sorted(_numbers(before.y)), "same numbers, drawn sideways"
    colour = after.marker.color
    first = cs.palette_colors(cs.DEFAULT_STYLE, 1)[0]                       # the studio's own palette, at its own opacity
    assert (colour if isinstance(colour, str) else colour[0]) == cs.rgba(first, cs.DEFAULT_STYLE["opacity"])
    saved = api.get(f"/api/questions/{out['question_id']}/dashboard", headers=ws["h"]).json()["view_overrides"]
    assert [v["chart_type"] for v in saved.values()] == ["hbar"], "the choice is saved for the team as before"
    assert theme.CHART_COLORWAY[0] == "#837ab6", "other pages' chart colours are unchanged"


def test_chart_helpers_keep_their_default_colours_unless_given_a_colorway():
    import plotly.graph_objects as go
    from style import theme

    def fig():
        return go.Figure([go.Bar(x=["a", "b"], y=[1, 2], marker_color="#123456"),
                          go.Scatter(x=[1, 2], y=[3, 4], mode="lines", line_color="#654321"),
                          go.Bar(x=["a", "b"], y=[1, 2], marker_color=["#111111", "#222222"])])

    default = theme.style_chart(fig())
    assert list(default.layout.colorway) == theme.CHART_COLORWAY
    assert default.data[0].marker.color == theme.CHART_COLORWAY[0] and default.data[1].line.color == theme.CHART_COLORWAY[1]
    pastel = theme.style_chart(fig(), colorway=theme.DASHBOARD_COLORWAY)
    assert list(pastel.layout.colorway) == theme.DASHBOARD_COLORWAY
    assert pastel.data[0].marker.color == theme.DASHBOARD_COLORWAY[0] and pastel.data[1].line.color == theme.DASHBOARD_COLORWAY[1]
    assert list(pastel.data[2].marker.color) == ["#111111", "#222222"], "a per-point colour encoding is left alone"
    kept = theme.style_chart(fig(), recolor=False, colorway=theme.DASHBOARD_COLORWAY)
    assert kept.data[0].marker.color == "#123456", "an already-styled figure keeps its own colours"
    assert list(pastel.data[0].y) == [1, 2] and list(pastel.data[1].y) == [3, 4], "restyling never touches the values"
    assert len(set(theme.DASHBOARD_COLORWAY)) == len(theme.DASHBOARD_COLORWAY) == 6


def test_the_dashboard_tokens_and_palette_are_one_coherent_set():
    import chart_studio as cs
    from style import theme

    t = theme.DASHBOARD_TOKENS
    assert (t["--bg-page"], t["--dash-surface"], t["--accent"], t["--text-primary"], t["--text-muted"], t["--border"]) == (
        "#B8CECA", "#FFFFFF", "#FF824D", "#202522", "#7C8581", "#DCE3E0")
    assert theme.DASHBOARD_PALETTE == {"orange": "#FF824D", "yellow": "#FFD166", "mint": "#83D8C5",
                                       "coral": "#E87983", "lavender": "#A6A2E8", "blue": "#61BDE0"}
    assert theme.DASHBOARD_COLORWAY[0] == t["--accent"], "the accent leads the data colours"
    assert [t[f"--dash-c{i}"] for i in range(1, 7)] == theme.DASHBOARD_COLORWAY, "cards and charts draw from the same six colours"
    assert cs.PALETTES["Dashboard"] == theme.DASHBOARD_COLORWAY and cs.DEFAULT_STYLE["palette"] == "Dashboard", \
        "a chart re-drawn in Chart Studio starts in the same palette"
    assert "SILT" in cs.PALETTES and cs.PALETTE_NAMES[0] == "Dashboard", "earlier palettes remain available"
    import re

    sheet = re.sub(r"/\*.*?\*/", "", theme.dashboard_theme_css())            # comments are not selectors
    for selector in sheet.replace("<style>", "").replace("</style>", "").split("}"):
        head = selector.split("{")[0].strip()
        if head and not head.startswith("@media"):
            parts, depth, start = [], 0, 0                      # split on commas outside :has(...) / :not(...)
            for i, ch in enumerate(head):
                depth += (ch == "(") - (ch == ")")
                if ch == "," and depth == 0:
                    parts.append(head[start:i])
                    start = i + 1
            for part in parts + [head[start:]]:
                assert any(scope in part for scope in (".stMain", ".st-key-dash_", ".st-key-card_studio", ".st-key-flat_pick",
                                                       ".dash-", "stDialog", "popover")), f"unscoped rule: {part.strip()}"


def test_the_dashboard_chart_finish_changes_presentation_only():
    import plotly.graph_objects as go
    from style import theme

    def fig(names=(None, None)):
        return go.Figure([go.Bar(x=["a", "b"], y=[1.5, 2.5], name=names[0], customdata=[["p"], ["q"]], hovertemplate="%{x}: %{y}"),
                          go.Scatter(x=[1, 2], y=[3, 4], mode="lines+markers", name=names[1])])

    before = fig(("Revenue", "Units"))
    after = theme.dashboard_chart(theme.style_chart(fig(("Revenue", "Units")), showlegend=True, colorway=theme.DASHBOARD_COLORWAY))
    for old, new in zip(before.data, after.data):
        assert (list(old.x), list(old.y), old.type, old.name, old.hovertemplate) == (list(new.x), list(new.y), new.type, new.name, new.hovertemplate)
    assert after.data[0].customdata is not None and after.layout.showlegend is True, "named series keep their legend"
    assert after.layout.legend.orientation == "h" and after.layout.font.color == theme.DASHBOARD_CHART["muted"]
    assert after.layout.xaxis.gridcolor == after.layout.yaxis.gridcolor == theme.DASHBOARD_CHART["grid"]
    assert after.layout.barcornerradius == 6 and after.data[1].line.width == 2.5 and after.data[1].marker.size == 7
    assert after.data[0].marker.color == "#FF824D" and after.data[1].line.color == "#FFD166"
    assert after.layout.paper_bgcolor == after.layout.plot_bgcolor == "rgba(0,0,0,0)", "the card behind the chart shows through"

    lone = theme.dashboard_chart(theme.style_chart(go.Figure([go.Bar(x=["a"], y=[1])]), showlegend=True))
    assert lone.layout.showlegend is False, "a single unnamed series has no legend to show"
    named = theme.dashboard_chart(theme.style_chart(go.Figure([go.Bar(x=["a"], y=[1], name="Revenue")]), showlegend=True))
    assert named.layout.showlegend is True

    chosen = go.Figure([go.Scatter(x=[1, 2], y=[3, 4], mode="lines", line=dict(width=6, color="#123456"))])
    kept = theme.dashboard_chart(theme.style_chart(chosen, showlegend=True, recolor=False), own_traces=True)
    assert kept.data[0].line.width == 6 and kept.data[0].line.color == "#123456" and kept.layout.barcornerradius is None
    assert kept.layout.showlegend is True, "a Chart Studio chart keeps the legend setting the user chose"
    wide = theme.style_chart(go.Figure([go.Scatter(x=[1, 2], y=[3, 4], line=dict(width=4))]))
    assert theme.dashboard_chart(wide).data[0].line.width == 4, "an explicit width is never overridden"
    plain = theme.style_chart(go.Figure([go.Bar(x=["a"], y=[1])]))
    assert plain.layout.legend.orientation is None and plain.layout.barcornerradius is None, "other pages' charts are not finished this way"


# ------------------------------------------------- organization / discussion --

def test_discussion_page_shows_a_posted_comment_and_can_post(api, ws):
    api.post("/api/discussion", json={"body": "Revenue dipped in the west"}, headers=ws["h"])
    at = healthy(page("18_Discussion.py", ws))
    assert "Revenue dipped in the west" in text_of(at)


def test_workspaces_page_renders_org_admin_controls_for_the_owner(api, ws):
    at = healthy(page("8_Workspaces.py", ws))
    text = text_of(at)
    assert "Team members" in text and "Create a team" in text


def test_a_user_with_no_team_sees_onboarding_instead_of_the_app(api, ws):
    from platform_helpers import signup

    token, email = signup(api)
    bare = {"token": token, "email": email, "team": None}
    at = AppTest.from_file(str(FRONTEND_DIR / "pages" / "2_Datasets.py"), default_timeout=60)
    at.session_state["auth_user"] = {"id": 0, "email": email, "display_name": "New Person"}
    at.session_state["auth_token"] = token
    at = at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = text_of(at)
    assert "Create a workspace" in text and "not in a team yet" not in text.lower()
    assert not [b for b in at.button if b.key == "nav_datasets"], "no app navigation before a workspace exists"


def test_session_cookie_script_sets_and_clears_the_token_safely():
    import auth

    on = auth._cookie_js('abc."</script>')
    assert "max-age=604800" in on and "SameSite=Lax" in on and "encodeURIComponent(" in on
    assert on.count("</script>") == 1, "a hostile token cannot close the script tag early"
    off = auth._cookie_js(None)
    assert "max-age=0" in off and "silt_token=" in off


def _signed_out(view: str | None) -> AppTest:
    at = AppTest.from_file(str(FRONTEND_DIR / "Home.py"), default_timeout=60)
    if view:
        at.query_params["view"] = view
    return at.run()


def test_signed_out_visitors_get_a_landing_page_and_a_separate_sign_in_page(api):
    landing = _signed_out(None)
    assert not landing.exception
    assert not landing.text_input, "the landing page has no form on it"
    page_html = " ".join(str(m.value) for m in landing.markdown)
    assert 'href="?view=signup"' in page_html and 'href="?view=login"' in page_html, "Get started and Log in are links"

    login = _signed_out("login")
    assert not login.exception and {t.key for t in login.text_input} >= {"login_email", "login_password"}
    assert [t.label for t in login.tabs][0] == "Log in"

    signup = _signed_out("signup")
    assert not signup.exception and [t.label for t in signup.tabs][0] == "Create account", "Get started opens sign-up first"


# ---------------------------------------------------- what-if: ask, explain, inspect --

def test_what_if_page_answers_a_plain_sentence_and_shows_how_it_was_read(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.text_input(key="ask_text_unit_economics").set_value("What happens if I increase the price by 10%?").run()
    at.button(key="ask_go").click().run()
    healthy(at)
    text = text_of(at)
    # profit 5,000 -> price +10% adds 0.1 x 20,000 = 2,000 -> 7,000
    assert "How I understood it" in text and "Price per unit: up 10%" in text and "profit rises from 5,000 to 7,000" in text
    at.button(key="ask_apply").click().run()
    assert at.number_input(key="wi_adj_unit_economics_price").value == 10.0, "the form now holds the same change"
    assert at.session_state["ask_ctx_unit_economics"], "the conversation context is kept by the page"


def test_what_if_page_asks_when_a_sentence_is_unclear_and_refuses_to_invent_numbers(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.text_input(key="ask_text_unit_economics").set_value("What if I change cost by 10%?").run()
    at.button(key="ask_go").click().run()
    healthy(at)
    assert at.warning, "a clarifying question is shown"
    at.text_input(key="ask_text_unit_economics").set_value("tell me a joke").run()
    at.button(key="ask_go").click().run()
    healthy(at)
    assert "what-if" in text_of(at).lower()


def test_what_if_page_explains_in_plain_english_without_claiming_a_cause(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.number_input(key="wi_adj_unit_economics_price").set_value(10.0).run()
    healthy(at)
    text = text_of(at)
    labels = [e.label for e in at.expander]
    assert "In plain English" in text and "What this cannot tell you" in labels and "What changed" in labels
    assert any("not evidence of causation" in str(x.value) for x in list(at.markdown) + list(at.caption))


def test_what_if_page_shows_the_calculation_and_seals_it_as_evidence(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.number_input(key="wi_adj_unit_economics_price").set_value(10.0).run()
    at.selectbox(key="ins_scn_unit_economics").select("your scenario")
    at.selectbox(key="ins_out_unit_economics").select("profit").run()
    at.button(key="ins_go").click().run()
    healthy(at)
    assert "Matches the engine" in text_of(at) and "With your numbers" in str(at.dataframe[-2].value.columns.tolist())
    assert api.get("/api/scenario-evidence", headers=ws["h"]).json()["records"] == [], "showing does not store anything"
    at.button(key="ins_seal").click().run()
    healthy(at)
    assert "Sealed" in text_of(at)
    recs = api.get("/api/scenario-evidence", headers=ws["h"]).json()["records"]
    assert len(recs) == 1 and recs[0]["kind"] == "scenario"


# ------------------------------------------ what-if: reopen, edit, rename, copy, compare, export --

def _save_from_page(at, name, price_pct=10.0):
    at.number_input(key="wi_adj_unit_economics_price").set_value(price_pct)
    at.run()
    at.text_input(key="wi_name").set_value(name).run()
    at.button(key="wi_save").click().run()
    return healthy(at)


def test_what_if_saved_scenario_is_reopened_edited_and_updated_in_place(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    _save_from_page(at, "Price test")
    sid = api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"][0]["id"]
    at = healthy(page("14_What_If.py", ws))
    at.button(key=f"wi_open_{sid}").click().run()
    healthy(at)
    assert at.number_input(key="wi_adj_unit_economics_price").value == 10.0 and at.number_input(key="wi_in_unit_economics_price").value == 20.0
    assert any("Editing saved scenario" in str(i.value) for i in at.info)
    at.number_input(key="wi_adj_unit_economics_price").set_value(20.0).run()
    at.button(key="wi_update").click().run()
    healthy(at)
    got = api.get(f"/api/scenarios/{sid}", headers=ws["h"]).json()
    assert got["revision"] == 2 and got["results"]["scenarios"]["your scenario"]["outputs"]["profit"] == 9000.0
    assert api.get("/api/scenarios", headers=ws["h"]).json()["total"] == 1, "updated in place, not duplicated"
    assert len(api.get("/api/scenario-evidence", params={"scenario_id": sid}, headers=ws["h"]).json()["records"]) == 2


def test_what_if_reopening_says_what_the_form_cannot_show(api, ws):
    r = api.post("/api/scenarios", headers=ws["h"], json={"name": "Set price", "model": "unit_economics", "baseline": {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0},
                                                          "scenarios": {"your scenario": {"price": {"type": "set", "value": 25.0}}}})
    assert r.status_code == 201, r.text
    at = healthy(page("14_What_If.py", ws))
    at.button(key=f"wi_open_{r.json()['id']}").click().run()
    healthy(at)
    assert any("could not load" in str(w.value) and "which the form cannot show" in str(w.value) for w in at.warning)


def test_what_if_rename_and_duplicate_from_the_saved_tab(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    _save_from_page(at, "Price test")
    sid = api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"][0]["id"]
    at = healthy(page("14_What_If.py", ws))
    at.text_input(key=f"wi_ren_{sid}").set_value("Better name").run()
    at.button(key=f"wi_rename_{sid}").click().run()
    healthy(at)
    assert api.get(f"/api/scenarios/{sid}", headers=ws["h"]).json()["name"] == "Better name"
    at.button(key=f"wi_dup_{sid}").click().run()
    healthy(at)
    names = sorted(s["name"] for s in api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"])
    assert names == ["Better name", "Copy of Better name"]


def test_what_if_saved_scenario_shows_its_data_status_and_stale_warning(api, ws):
    ds = api.get("/api/datasets", headers=ws["h"]).json()
    ds_id = (ds["datasets"] if isinstance(ds, dict) else ds)[0]["id"]
    r = api.post("/api/scenarios", headers=ws["h"], json={"name": "From data", "model": "unit_economics", "baseline": {"price": 20.0, "volume": 1000.0, "unit_cost": 12.0, "fixed_cost": 3000.0},
                                                          "scenarios": {}, "dataset_id": ds_id, "baseline_source": {"dataset_version": 1}})
    assert r.status_code == 201, r.text
    at = healthy(page("14_What_If.py", ws))
    assert "Data is current" in " ".join(str(m.value) for m in at.markdown)
    csv = b"region,units,revenue\nEast,1,2\nWest,2,4\n"
    assert api.post(f"/api/datasets/{ds_id}/versions", files={"file": ("v2.csv", csv, "text/csv")}, headers=ws["h"]).status_code == 200
    at = healthy(page("14_What_If.py", ws))
    text = " ".join(str(m.value) for m in at.markdown)
    assert "Newer data exists" in text and "version 2 exists now" in text and "was not changed" in text


def test_what_if_comparison_shows_changes_warnings_and_offers_a_file(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    _save_from_page(at, "A", 10.0)
    at.number_input(key="wi_adj_unit_economics_price").set_value(20.0).run()
    at.text_input(key="wi_name").set_value("B").run()
    at.button(key="wi_save").click().run()
    ids = [s["id"] for s in api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"]]
    at = healthy(page("14_What_If.py", ws))
    for i in ids:
        at.checkbox(key=f"wi_pick_{i}").check()
    at.run()
    healthy(at)
    frame = [d.value for d in at.dataframe if "Change %" in d.value.columns][0]
    mine = frame[(frame["Saved as"] == "B") & (frame["Scenario"] == "your scenario")].iloc[0]
    assert mine["Result"] == 9000.0 and mine["Change"] == 4000.0 and mine["Change %"] == 80.0
    assert "Highest: B" in text_of(at) and not [w for w in at.warning if "starts from" in str(w.value) or "different" in str(w.value)]
    assert [b for b in at.get("download_button")], "the comparison can be downloaded"


def test_what_if_export_is_prepared_on_request_and_is_what_was_saved(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    _save_from_page(at, "Price test")
    sid = api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"][0]["id"]
    at = healthy(page("14_What_If.py", ws))
    assert not [b for b in at.get("download_button") if "Download JSON" in str(b.proto.label)]
    at.selectbox(key=f"wi_fmt_{sid}").select("md").run()
    at.button(key=f"wi_prep_{sid}").click().run()
    healthy(at)
    fmt, (data, sha) = at.session_state[f"wi_file_{sid}"]
    assert fmt == "md" and b"not a validated forecast" in data and sha in data.decode()
    assert "Exactly what was saved" in text_of(at)


# ------------------------------------------------ what-if: new data version, error and empty states --

def test_what_if_stale_scenario_is_recalculated_on_new_data_and_the_original_is_kept(api, ws):
    ds = api.get("/api/datasets", headers=ws["h"]).json()
    ds_id = (ds["datasets"] if isinstance(ds, dict) else ds)[0]["id"]
    from api_client import ApiError  # noqa: F401  (the page module imports the same name)
    probe = api.post("/api/scenarios/baseline", headers=ws["h"], json={"dataset_id": ds_id, "model": "unit_economics",
                                                                        "mapping": {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": None},
                                                                        "overrides": {"fixed_cost": 10.0, "unit_cost": 1.0}})
    if probe.status_code != 200:
        pytest.skip("the fixture dataset does not map to unit economics")
    base = probe.json()["inputs"]
    r = api.post("/api/scenarios", headers=ws["h"], json={"name": "From data", "model": "unit_economics", "baseline": base, "scenarios": {}, "dataset_id": ds_id,
                                                          "baseline_source": {"dataset_version": 1, "mapping": {"quantity_column": "units", "amount_column": "revenue", "total_cost_column": None}}})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    at = healthy(page("14_What_If.py", ws))
    assert not [b for b in at.button if b.key == f"wi_rebase_{sid}"], "no recalculate button while the data is current"
    csv = b"region,units,revenue\nEast,1,2\nWest,2,4\n"
    assert api.post(f"/api/datasets/{ds_id}/versions", files={"file": ("v2.csv", csv, "text/csv")}, headers=ws["h"]).status_code == 200
    at = healthy(page("14_What_If.py", ws))
    at.button(key=f"wi_rebase_{sid}").click().run()
    healthy(at)
    assert "is unchanged" in text_of(at) and "data version 2" in text_of(at)
    scenarios = api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"]
    assert len(scenarios) == 2 and {s["name"] for s in scenarios} == {"From data", "From data (data v2)"}
    assert api.get(f"/api/scenarios/{sid}", headers=ws["h"]).json()["freshness"]["status"] == "stale", "the original still says it is on version 1"


def test_what_if_ask_box_shows_a_refusal_instead_of_failing(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    at.text_input(key="ask_text_unit_economics").set_value("increase price by 10000000%").run()
    at.button(key="ask_go").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("outside what the What-If form allows" in str(e.value) for e in at.error), "the reason is shown"
    assert api.get("/api/scenarios", headers=ws["h"]).json()["total"] == 0


def test_what_if_ask_button_is_off_until_there_is_a_question_and_start_over_clears_it(api, ws):
    at = _unit_economics(healthy(page("14_What_If.py", ws)))
    assert at.button(key="ask_go").disabled
    at.text_input(key="ask_text_unit_economics").set_value("increase the price by 10%").run()
    at.button(key="ask_go").click().run()
    assert at.session_state["ask_last_unit_economics"]["status"] == "ok"
    at.button(key="ask_reset").click().run()
    assert "ask_last_unit_economics" not in at.session_state and "ask_ctx_unit_economics" not in at.session_state


def test_what_if_saved_tab_empty_state_and_single_pick_hint(api, ws):
    at = healthy(page("14_What_If.py", ws))
    assert "Nothing saved yet" in " ".join(str(m.value) for m in at.markdown)
    at = _unit_economics(at)
    _save_from_page(at, "Only one")
    at = healthy(page("14_What_If.py", ws))
    sid = api.get("/api/scenarios", headers=ws["h"]).json()["scenarios"][0]["id"]
    at.checkbox(key=f"wi_pick_{sid}").check().run()
    assert "Tick two or more scenarios" in text_of(at)
