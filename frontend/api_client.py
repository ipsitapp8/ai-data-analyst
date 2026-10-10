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


def ask_question(dataset_id: int, question: str) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/questions",
        json={"dataset_id": dataset_id, "question": question},
        headers=_headers(), timeout=TIMEOUT,
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
