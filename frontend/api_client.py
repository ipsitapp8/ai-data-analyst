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
