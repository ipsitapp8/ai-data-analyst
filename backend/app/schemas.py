"""Pydantic response/request models for the FastAPI layer."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SignupRequest(BaseModel):
    email: str
    password: str
    display_name: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    id: int
    email: str
    display_name: str

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class CommunityCreate(BaseModel):
    name: str


class TeamCreate(BaseModel):
    name: str


class InviteRequest(BaseModel):
    email: str


class TeamOut(BaseModel):
    id: int
    community_id: int
    name: str
    role: str  # the caller's own role on this team

    class Config:
        from_attributes = True


class CommunityOut(BaseModel):
    id: int
    name: str
    teams: list[TeamOut] = []

    class Config:
        from_attributes = True


class MemberOut(BaseModel):
    id: int
    user_id: int | None
    email: str
    display_name: str | None
    role: str
    status: str


class InviteResponse(BaseModel):
    member: MemberOut
    email_sent: bool


class InspectResponse(BaseModel):
    code: str | None = None
    formula_explanation: str
    data_slice: dict[str, Any]
    # 1-based line in the uploaded CSV (header = line 1) for each data_slice
    # row; None where the row isn't a verbatim CSV row (e.g. an aggregate).
    csv_lines: list[int | None] = []
    csv_filename: str | None = None


class ChartViewIn(BaseModel):
    chart_type: str | None = Field(None, max_length=40, pattern=r"^[a-z0-9_]+$")
    style: dict[str, Any] = {}

    @field_validator("style")
    @classmethod
    def _style_is_small(cls, v: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(v, default=str)) > 4000:
            raise ValueError("style is too large")
        return v


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
    id: int
    question_id: int
    verified: bool
    verification_summary: str
    kpis: list[dict[str, Any]]
    charts: list[dict[str, Any]]
    narrative: str
    view_overrides: dict[str, Any] = {}
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
