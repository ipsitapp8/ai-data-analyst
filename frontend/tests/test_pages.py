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
    at = healthy(page("12_Copilot.py", ws, copilot_session_id=sid))
    text = text_of(at)
    assert "region == West" in text and "an independent recomputation agrees" in text
    assert len(at.chat_message) == 4 and len(at.dataframe) == 2
    at.chat_input[0].set_value("drill into product").run()
    healthy(at)
    session = api.get(f"/api/copilot/sessions/{sid}", headers=ws["h"]).json()
    assert session["state"]["group_by"] == ["region", "product"] and len(session["turns"]) == 6
    assert len(at.chat_message) == 6


def test_copilot_page_empty_and_stale_states(api, ws):
    at = healthy(page("12_Copilot.py", ws))
    assert "No sessions yet" in text_of(at) and "Pick or start a session" in text_of(at)
    sid = api.post("/api/copilot/sessions", json={"dataset_id": ws["ds"]}, headers=ws["h"]).json()["id"]
    api.post(f"/api/datasets/{ws['ds']}/versions", files={"file": ("v2.csv", DRIFTED_CSV, "text/csv")}, headers=ws["h"])
    at = healthy(page("12_Copilot.py", ws, copilot_session_id=sid))
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
    at = AppTest.from_file(str(FRONTEND_DIR / "pages" / "13_Investigations.py"), default_timeout=60).run()
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
