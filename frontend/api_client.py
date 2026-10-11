"""Thin requests-based client for the FastAPI backend.

Auth/team context is read directly from st.session_state rather than passed
into every call site -- that keeps every existing page (Datasets, Analyses,
Dashboards, Audit Trail, ...) working unchanged: they already call
list_datasets()/list_questions()/etc with no params, and now those calls are
scoped to whichever team the sidebar switcher has active.
"""
from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from config import BACKEND_BASE_URL

TIMEOUT = 30


class ApiError(Exception):
    pass


def _headers(with_team: bool = True) -> dict:
    headers = {}
    token = st.session_state.get("auth_token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if with_team:
        team_id = st.session_state.get("active_team_id")
        if team_id:
            headers["X-Team-Id"] = str(team_id)
    return headers


def _handle(resp: requests.Response) -> Any:
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise ApiError(f"{resp.status_code}: {detail}")
    return resp.json()


def health() -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/health", timeout=TIMEOUT))


# --------------------------------------------------------------------- auth --

def signup(email: str, password: str, display_name: str) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/auth/signup",
        json={"email": email, "password": password, "display_name": display_name},
        timeout=TIMEOUT,
    )
    return _handle(resp)


def login(email: str, password: str) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/auth/login",
        json={"email": email, "password": password},
        timeout=TIMEOUT,
    )
    return _handle(resp)


def me() -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/auth/me", headers=_headers(with_team=False), timeout=TIMEOUT))


# ---------------------------------------------------------- communities/teams --

def my_workspaces() -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/me/workspaces", headers=_headers(with_team=False), timeout=TIMEOUT))


def my_invites() -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/me/invites", headers=_headers(with_team=False), timeout=TIMEOUT))


def accept_invite(team_id: int) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/teams/{team_id}/accept-invite",
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


def list_communities() -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/communities", headers=_headers(with_team=False), timeout=TIMEOUT))


def create_community(name: str) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/communities", json={"name": name},
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


def list_teams(community_id: int) -> list[dict]:
    resp = requests.get(
        f"{BACKEND_BASE_URL}/api/communities/{community_id}/teams",
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


def create_team(community_id: int, name: str) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/communities/{community_id}/teams", json={"name": name},
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


def invite_member(team_id: int, email: str) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/teams/{team_id}/invite", json={"email": email},
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


def invite_members_bulk(team_id: int, emails: list[str]) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/teams/{team_id}/invite/bulk", json={"emails": emails},
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


def revoke_invite(team_id: int, member_id: int) -> dict:
    resp = requests.delete(
        f"{BACKEND_BASE_URL}/api/teams/{team_id}/members/{member_id}",
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


def list_members(team_id: int) -> list[dict]:
    resp = requests.get(
        f"{BACKEND_BASE_URL}/api/teams/{team_id}/members",
        headers=_headers(with_team=False), timeout=TIMEOUT,
    )
    return _handle(resp)


# ----------------------------------------------------------------- datasets --

def upload_dataset(filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{BACKEND_BASE_URL}/api/datasets/upload", files=files, headers=_headers(), timeout=TIMEOUT)
    return _handle(resp)


def list_datasets() -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/datasets", headers=_headers(), timeout=TIMEOUT))


def get_dataset(dataset_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}", headers=_headers(), timeout=TIMEOUT))


def list_notes(dataset_id: int, kind: str | None = None) -> list[dict]:
    params = {"kind": kind} if kind else None
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}/notes",
                                params=params, headers=_headers(), timeout=TIMEOUT))


def create_note(dataset_id: int, kind: str, text: str) -> dict:
    return _handle(requests.post(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}/notes",
                                 json={"kind": kind, "text": text}, headers=_headers(), timeout=TIMEOUT))


def delete_note(dataset_id: int, note_id: int) -> None:
    resp = requests.delete(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}/notes/{note_id}",
                           headers=_headers(), timeout=TIMEOUT)
    if not resp.ok:
        _handle(resp)


def list_questions() -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions", headers=_headers(), timeout=TIMEOUT))


def ask_question(dataset_id: int, question: str, idempotency_key: str | None = None) -> dict:
    """Queue an analysis. Pass an idempotency key so a double click or a retried
    request returns the first submission instead of starting a second run."""
    headers = _headers()
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/questions",
        json={"dataset_id": dataset_id, "question": question},
        headers=headers, timeout=TIMEOUT,
    )
    return _handle(resp)


def get_status(question_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/status", headers=_headers(), timeout=TIMEOUT))


def get_dashboard(question_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/dashboard", headers=_headers(), timeout=TIMEOUT))


def get_audit_trail(question_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/audit-trail", headers=_headers(), timeout=TIMEOUT))


def get_critic_reviews(question_id: int) -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/critic-reviews", headers=_headers(), timeout=TIMEOUT))


def inspect_element(dashboard_id: int, element_id: str) -> dict:
    resp = requests.get(
        f"{BACKEND_BASE_URL}/api/dashboards/{dashboard_id}/elements/{element_id}/inspect",
        headers=_headers(), timeout=TIMEOUT,
    )
    return _handle(resp)


def replace_dataset_data(dataset_id: int, filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}/versions", files=files, headers=_headers(), timeout=TIMEOUT
    )
    return _handle(resp)


# ---------------------------------------------------------------- scheduled --

def list_scheduled() -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/scheduled-analyses", headers=_headers(), timeout=TIMEOUT))


def create_scheduled(dataset_id: int, question: str, interval: str = "daily",
                     change_threshold_pct: float = 10.0) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/scheduled-analyses",
        json={"dataset_id": dataset_id, "question": question, "interval": interval,
              "change_threshold_pct": change_threshold_pct},
        headers=_headers(), timeout=TIMEOUT,
    )
    return _handle(resp)


def update_scheduled(scheduled_id: int, **fields) -> dict:
    resp = requests.patch(
        f"{BACKEND_BASE_URL}/api/scheduled-analyses/{scheduled_id}", json=fields,
        headers=_headers(), timeout=TIMEOUT,
    )
    return _handle(resp)


def delete_scheduled(scheduled_id: int) -> dict:
    resp = requests.delete(
        f"{BACKEND_BASE_URL}/api/scheduled-analyses/{scheduled_id}", headers=_headers(), timeout=TIMEOUT,
    )
    return _handle(resp)


# ------------------------------------------------------------- share links --

def create_share(question_id: int, expires_in_days: int = 7) -> dict:
    return _handle(requests.post(f"{BACKEND_BASE_URL}/api/questions/{question_id}/share",
                                 json={"expires_in_days": expires_in_days}, headers=_headers(), timeout=TIMEOUT))


def list_shares(question_id: int) -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/shares",
                                headers=_headers(), timeout=TIMEOUT))


def revoke_share(share_id: int) -> None:
    resp = requests.delete(f"{BACKEND_BASE_URL}/api/shares/{share_id}", headers=_headers(), timeout=TIMEOUT)
    if not resp.ok:
        _handle(resp)


def get_public_dashboard(token: str) -> dict:
    """No auth headers on purpose: this is what an anonymous viewer of a share link sees."""
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/public/shared/{token}", timeout=TIMEOUT))


# ---------------------------------------------------------- auto-insights --

def get_insights(dataset_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}/insights",
                                headers=_headers(), timeout=TIMEOUT))


def generate_insights(dataset_id: int) -> dict:
    # Generous timeout: one LLM call, which may wait out a rate-limit backoff.
    return _handle(requests.post(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}/insights",
                                 headers=_headers(), timeout=120))


# ------------------------------------------------------------------ alerts --

def list_alerts(unread_only: bool = False, limit: int = 50) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/alerts",
                                params={"unread_only": unread_only, "limit": limit},
                                headers=_headers(), timeout=TIMEOUT))


def mark_alert_read(alert_id: int) -> None:
    resp = requests.post(f"{BACKEND_BASE_URL}/api/alerts/{alert_id}/read", headers=_headers(), timeout=TIMEOUT)
    if not resp.ok:
        _handle(resp)


def mark_all_alerts_read() -> None:
    resp = requests.post(f"{BACKEND_BASE_URL}/api/alerts/read-all", headers=_headers(), timeout=TIMEOUT)
    if not resp.ok:
        _handle(resp)


# -------------------------------------------------------------------- chat --

def chat_dashboard(question_id: int, message: str, history: list[dict]) -> dict:
    # Two LLM calls (answer + independent check) -- allow for slow providers.
    return _handle(requests.post(f"{BACKEND_BASE_URL}/api/questions/{question_id}/chat",
                                 json={"message": message, "history": history},
                                 headers=_headers(), timeout=120))


# ------------------------------------------------------------ correlations --

def get_correlations(dataset_id: int, min_abs: float = 0.5) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}/correlations",
                                params={"min_abs": min_abs}, headers=_headers(), timeout=TIMEOUT))


# -------------------------------------------------------------- chart studio --

def save_chart_view(dashboard_id: int, chart_key: str, chart_type: str | None, style: dict) -> dict:
    resp = requests.put(
        f"{BACKEND_BASE_URL}/api/dashboards/{dashboard_id}/charts/{chart_key}/view",
        json={"chart_type": chart_type, "style": style},
        headers=_headers(), timeout=TIMEOUT,
    )
    return _handle(resp)


def reset_chart_view(dashboard_id: int, chart_key: str) -> dict:
    resp = requests.delete(
        f"{BACKEND_BASE_URL}/api/dashboards/{dashboard_id}/charts/{chart_key}/view",
        headers=_headers(), timeout=TIMEOUT,
    )
    return _handle(resp)


# ================================================================== platform ==
# Small helpers so the functions below stay one line each.

def _get(path: str, timeout: int = TIMEOUT, **params) -> Any:
    clean = {k: v for k, v in params.items() if v is not None}
    return _handle(requests.get(f"{BACKEND_BASE_URL}{path}", params=clean or None, headers=_headers(), timeout=timeout))


def _post(path: str, body: dict | None = None, timeout: int = TIMEOUT, extra_headers: dict | None = None) -> Any:
    resp = requests.post(f"{BACKEND_BASE_URL}{path}", json=body, headers={**_headers(), **(extra_headers or {})},
                         timeout=timeout)
    if resp.status_code == 204:
        return None
    return _handle(resp)


def _put(path: str, body: dict) -> Any:
    return _handle(requests.put(f"{BACKEND_BASE_URL}{path}", json=body, headers=_headers(), timeout=TIMEOUT))


def _delete(path: str) -> None:
    resp = requests.delete(f"{BACKEND_BASE_URL}{path}", headers=_headers(), timeout=TIMEOUT)
    if not resp.ok:
        _handle(resp)


# -------------------------------------------------------------------- jobs --

def list_jobs(state: str | None = None, limit: int = 50) -> dict:
    return _get("/api/jobs", state=state, limit=limit)


def job_metrics() -> dict:
    return _get("/api/jobs/metrics")


def cancel_question(question_id: int) -> dict:
    return _post(f"/api/questions/{question_id}/cancel")


def retry_job(job_id: int) -> dict:
    return _post(f"/api/jobs/{job_id}/retry")


def get_trace(question_id: int) -> dict:
    return _get(f"/api/questions/{question_id}/trace")


# ------------------------------------------------ evidence + reproducibility --

def get_evidence(question_id: int) -> dict:
    return _get(f"/api/questions/{question_id}/evidence")


def get_manifest(question_id: int) -> dict:
    return _get(f"/api/questions/{question_id}/manifest")


def rerun_question(question_id: int, dataset_version_id: int | None = None) -> dict:
    return _post(f"/api/questions/{question_id}/rerun", {"dataset_version_id": dataset_version_id})


def compare_questions(question_id: int, other_id: int) -> dict:
    return _get(f"/api/questions/{question_id}/compare/{other_id}")


def list_dataset_versions(dataset_id: int) -> list[dict]:
    return _get(f"/api/datasets/{dataset_id}/versions")


def compare_dataset_versions(dataset_id: int, a: int, b: int) -> dict:
    return _get(f"/api/datasets/{dataset_id}/versions/compare", a=a, b=b)


# ----------------------------------------------------------- investigations --

def create_investigation(payload: dict) -> dict:
    return _post("/api/investigations", payload)


def list_investigations(limit: int = 50) -> dict:
    return _get("/api/investigations", limit=limit)


def get_investigation(investigation_id: int) -> dict:
    return _get(f"/api/investigations/{investigation_id}")


# ------------------------------------------------------------- data quality --

def get_quality(dataset_id: int) -> dict:
    return _get(f"/api/datasets/{dataset_id}/quality")


def run_quality(dataset_id: int) -> dict:
    return _post(f"/api/datasets/{dataset_id}/quality/run", timeout=120)


def set_quality_config(dataset_id: int, baseline_version_id: int | None, thresholds: dict) -> dict:
    return _put(f"/api/datasets/{dataset_id}/quality/config",
                {"baseline_version_id": baseline_version_id, "thresholds": thresholds})


def update_incident(incident_id: int, action: str) -> dict:
    return _post(f"/api/quality/incidents/{incident_id}/{action}")


# ------------------------------------------------------------------ alerts --

def get_alert(alert_id: int) -> dict:
    return _get(f"/api/alerts/{alert_id}")


def update_alert(alert_id: int, action: str) -> dict:
    return _post(f"/api/alerts/{alert_id}/{action}")


# ----------------------------------------------------------------- copilot --

def create_copilot_session(dataset_id: int, title: str = "") -> dict:
    return _post("/api/copilot/sessions", {"dataset_id": dataset_id, "title": title})


def list_copilot_sessions() -> list[dict]:
    return _get("/api/copilot/sessions")


def get_copilot_session(session_id: int) -> dict:
    return _get(f"/api/copilot/sessions/{session_id}")


def send_copilot_message(session_id: int, message: str) -> dict:
    return _post(f"/api/copilot/sessions/{session_id}/messages", {"message": message}, timeout=120)


def delete_copilot_session(session_id: int) -> None:
    _delete(f"/api/copilot/sessions/{session_id}")


# --------------------------------------------------------------- scenarios --

def scenario_models() -> dict:
    return _get("/api/scenarios/models")


def scenario_baseline(dataset_id: int, model: str, mapping: dict) -> dict:
    return _post("/api/scenarios/baseline", {"dataset_id": dataset_id, "model": model, "mapping": mapping})


def evaluate_scenarios(payload: dict) -> dict:
    return _post("/api/scenarios/evaluate", payload)


def save_scenario(payload: dict) -> dict:
    return _post("/api/scenarios", payload)


def list_scenarios() -> dict:
    return _get("/api/scenarios")


def compare_scenarios(ids: list[int]) -> dict:
    return _get("/api/scenarios/compare", ids=",".join(str(i) for i in ids))


def delete_scenario(scenario_id: int) -> None:
    _delete(f"/api/scenarios/{scenario_id}")


# ---------------------------------------------------------- semantic layer --

def list_metrics() -> list[dict]:
    return _get("/api/semantic/metrics")


def get_metric(metric_id: int) -> dict:
    return _get(f"/api/semantic/metrics/{metric_id}")


def create_metric(payload: dict) -> dict:
    return _post("/api/semantic/metrics", payload)


def add_metric_version(metric_id: int, payload: dict) -> dict:
    return _post(f"/api/semantic/metrics/{metric_id}/versions", payload)


def approve_metric_version(metric_id: int, version: int) -> dict:
    return _post(f"/api/semantic/metrics/{metric_id}/versions/{version}/approve")


def deprecate_metric(metric_id: int) -> dict:
    return _post(f"/api/semantic/metrics/{metric_id}/deprecate")


def validate_formula(formula: str, dataset_id: int | None) -> dict:
    return _post("/api/semantic/validate", {"formula": formula, "dataset_id": dataset_id})


def list_terms() -> list[dict]:
    return _get("/api/semantic/terms")


def create_term(payload: dict) -> dict:
    return _post("/api/semantic/terms", payload)


def semantic_graph() -> dict:
    return _get("/api/semantic/graph")


def semantic_search(query: str) -> dict:
    return _get("/api/semantic/search", q=query)


# ------------------------------------------------------------------- evals --

def list_eval_runs() -> list[dict]:
    return _get("/api/evals/runs")


def get_eval_run(run_id: int) -> dict:
    return _get(f"/api/evals/runs/{run_id}")


def compare_eval_runs(base: int, new: int) -> dict:
    return _get("/api/evals/compare", base=base, new=new)


def eval_suite() -> dict:
    return _get("/api/evals/suite")
