"""Pydantic response/request models for the FastAPI layer."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel


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
    resent: bool = False


class BulkInviteRequest(BaseModel):
    emails: list[str]


class BulkInviteResult(BaseModel):
    email: str
    ok: bool
    message: str
    email_sent: bool = False


class BulkInviteResponse(BaseModel):
    results: list[BulkInviteResult]


class InspectResponse(BaseModel):
    code: str | None = None
    formula_explanation: str
    data_slice: dict[str, Any]
    # Fourth section: set only when this element's audit_trail row points at a
    # rejected Critic review.
    flagged: bool = False
    critic_reasoning: str | None = None
    critic_issues: list[str] = []


class DatasetProfile(BaseModel):
    id: int
    filename: str
    row_count: int
    col_count: int
    profile: dict[str, Any]
    uploaded_at: dt.datetime
    version: int = 1
    version_count: int = 1

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


class CriticRejection(BaseModel):
    summary: str
    issues: list[str] = []
    created_at: dt.datetime | None = None


class DashboardResponse(BaseModel):
    id: int
    question_id: int
    verified: bool
    verdict_state: str = "VERIFIED"
    flagged_count: int = 0
    rejections: list[CriticRejection] = []
    narrative_flagged: bool = False
    narrative_element_id: str | None = None
    dataset_version: int | None = None
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
    # The sandbox contract asks for a JSON object but doesn't enforce one --
    # step code legitimately prints a list too (e.g. one record per group),
    # so this must accept whatever JSON-serializable shape actually comes back.
    result: dict[str, Any] | list[Any] | None = None
    critic_verdict: str | None = None
    critic_summary: str | None = None
    created_at: dt.datetime


class AuditTrailResponse(BaseModel):
    question_id: int
    question_text: str
    plan: list[dict[str, Any]]
    entries: list[AuditEntry]


class ScheduledCreate(BaseModel):
    dataset_id: int
    question: str
    interval: str = "daily"  # daily|weekly
    change_threshold_pct: float = 10.0


class ScheduledUpdate(BaseModel):
    is_active: bool | None = None
    interval: str | None = None
    change_threshold_pct: float | None = None


class ScheduledOut(BaseModel):
    id: int
    dataset_id: int
    question_text: str
    interval: str
    change_threshold_pct: float
    is_active: bool
    created_at: dt.datetime | None = None
    last_run_at: dt.datetime | None = None
    next_run_at: dt.datetime | None = None
    last_dashboard_id: int | None = None
    last_question_id: int | None = None
    last_verdict_state: str | None = None
    last_trend: str | None = None
    last_change_summary: str = ""
