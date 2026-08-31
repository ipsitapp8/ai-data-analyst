"""Thin requests-based client for the FastAPI backend."""
from __future__ import annotations

from typing import Any

import requests

from config import BACKEND_BASE_URL

TIMEOUT = 30


class ApiError(Exception):
    pass


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


def upload_dataset(filename: str, file_bytes: bytes) -> dict:
    files = {"file": (filename, file_bytes, "text/csv")}
    resp = requests.post(f"{BACKEND_BASE_URL}/api/datasets/upload", files=files, timeout=TIMEOUT)
    return _handle(resp)


def list_datasets() -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/datasets", timeout=TIMEOUT))


def get_dataset(dataset_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/datasets/{dataset_id}", timeout=TIMEOUT))


def list_questions() -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions", timeout=TIMEOUT))


def ask_question(dataset_id: int, question: str) -> dict:
    resp = requests.post(
        f"{BACKEND_BASE_URL}/api/questions",
        json={"dataset_id": dataset_id, "question": question},
        timeout=TIMEOUT,
    )
    return _handle(resp)


def get_status(question_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/status", timeout=TIMEOUT))


def get_dashboard(question_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/dashboard", timeout=TIMEOUT))


def get_audit_trail(question_id: int) -> dict:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/audit-trail", timeout=TIMEOUT))


def get_critic_reviews(question_id: int) -> list[dict]:
    return _handle(requests.get(f"{BACKEND_BASE_URL}/api/questions/{question_id}/critic-reviews", timeout=TIMEOUT))
