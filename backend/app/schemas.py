"""Pydantic response/request models for the FastAPI layer."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel


class DatasetProfile(BaseModel):
    id: int
    filename: str
    row_count: int
    col_count: int
    profile: dict[str, Any]
    uploaded_at: dt.datetime

    class Config:
        from_attributes = True


class QuestionCreate(BaseModel):
    dataset_id: int
    question: str


class QuestionCreated(BaseModel):
    question_id: int
    status: str


class StatusResponse(BaseModel):
    question_id: int
    question_text: str = ""
    status: str
    current_stage: str
    stage_detail: str
    steps: list[dict[str, Any]] = []
    retry_count: int = 0
    error: str | None = None


class DashboardResponse(BaseModel):
    question_id: int
    verified: bool
    verification_summary: str
    kpis: list[dict[str, Any]]
    charts: list[dict[str, Any]]
    narrative: str
    created_at: dt.datetime | None = None


class AuditEntry(BaseModel):
    id: int
    element_label: str
    element_type: str
    reasoning: str
    code: str | None = None
    stdout: str | None = None
    result: dict[str, Any] | None = None
    critic_verdict: str | None = None
    critic_summary: str | None = None
    created_at: dt.datetime


class AuditTrailResponse(BaseModel):
    question_id: int
    question_text: str
    plan: list[dict[str, Any]]
    entries: list[AuditEntry]
