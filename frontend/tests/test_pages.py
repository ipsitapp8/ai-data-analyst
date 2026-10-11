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


# ----------------------------------------------------------- investigations --

def test_investigations_page_shows_the_report(api, ws):
    out = ask_and_run(api, ws, "Why did revenue change?")
    at = healthy(page("13_Investigations.py", ws))
    text = text_of(at)
    assert "West" in text and "Hypotheses tested" in text and "Not established" in text
    assert "not what caused it" in text, "the caveats are on the page"
    assert len(at.metric) >= 6 and len(at.dataframe) >= 1
    assert api.get("/api/investigations", headers=ws["h"]).json()["investigations"][0]["question_id"] == out["question_id"]


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
