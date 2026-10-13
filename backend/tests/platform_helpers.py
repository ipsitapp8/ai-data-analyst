"""Shared helpers for the platform test modules."""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

from app import jobs, runtime
from app.database import SessionLocal
from app.evals.scripted import ScriptedProvider
from app.models import AnalysisJob, Dashboard, Dataset, Question

FIXTURES = Path(__file__).resolve().parents[1] / "evals" / "fixtures"
SALES_CSV = (FIXTURES / "sales.csv").read_bytes()
DRIFTED_CSV = (FIXTURES / "sales_drifted.csv").read_bytes()


def signup(client, email: str | None = None) -> tuple[str, str]:
    email = email or f"{uuid.uuid4().hex[:12]}@test.example"
    r = client.post("/api/auth/signup", json={"email": email, "password": "password123", "display_name": "T"})
    assert r.status_code == 200, r.text
    return r.json()["access_token"], email


def make_team(client, token: str, name: str = "T") -> dict:
    h = {"Authorization": f"Bearer {token}"}
    cid = client.post("/api/communities", json={"name": name}, headers=h).json()["id"]
    tid = client.post(f"/api/communities/{cid}/teams", json={"name": name}, headers=h).json()["id"]
    return {**h, "X-Team-Id": str(tid)}


def workspace(client, csv: bytes = SALES_CSV, filename: str = "sales.csv") -> dict:
    """A fresh user, team and uploaded dataset."""
    token, email = signup(client)
    headers = make_team(client, token)
    r = client.post("/api/datasets/upload", files={"file": (filename, csv, "text/csv")}, headers=headers)
    assert r.status_code == 200, r.text
    return {"h": headers, "team": int(headers["X-Team-Id"]), "ds": r.json()["id"], "email": email, "token": token}


def add_member(client, ws: dict, role: str = "member") -> dict:
    """A second user who is an active member of ws's team. Returns their headers."""
    from app.models import TeamMember, User

    token, email = signup(client)
    db = SessionLocal()
    try:
        user = db.query(User).filter_by(email=email).first()
        db.add(TeamMember(team_id=ws["team"], user_id=user.id, role=role, status="active"))
        db.commit()
    finally:
        db.close()
    return {"Authorization": f"Bearer {token}", "X-Team-Id": str(ws["team"])}


def ask(client, ws: dict, question: str, **headers) -> dict:
    r = client.post("/api/questions", json={"dataset_id": ws["ds"], "question": question},
                    headers={**ws["h"], **headers})
    assert r.status_code == 200, r.text
    return r.json()


def run_job(job_id: int, script: dict | None = None, runner=None) -> str | None:
    """Run one specific queued job on this thread (never 'the next job': other
    tests leave queued jobs behind in the shared database)."""
    if script is None:
        return jobs.run_inline(job_id, runner)
    with runtime.provider_override(ScriptedProvider(script)):
        return jobs.run_inline(job_id, runner)


def ask_and_run(client, ws: dict, question: str, script: dict | None = None) -> dict:
    created = ask(client, ws, question)
    state = run_job(created["job_id"], script or {})
    return {**created, "state": state}


def job_row(job_id: int) -> AnalysisJob:
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, job_id)
        db.expunge(job)
        return job
    finally:
        db.close()


def question_row(question_id: int) -> Question:
    db = SessionLocal()
    try:
        q = db.get(Question, question_id)
        db.expunge(q)
        return q
    finally:
        db.close()


def seed_dashboard(ws: dict, kpis: list[dict], *, verified: bool = True, verdict_state: str | None = None,
                   created_at: dt.datetime | None = None, **question_kw) -> tuple[int, int]:
    """A finished question + dashboard written directly (no pipeline)."""
    db = SessionLocal()
    try:
        ds = db.get(Dataset, ws["ds"])
        from app.dataset_versions import latest_version

        version = latest_version(db, ds)
        q = Question(team_id=ws["team"], dataset_id=ws["ds"], text="Revenue?", status="verified",
                     dataset_version_id=version.id if version else None, **question_kw)
        if created_at is not None:
            q.created_at = created_at
        db.add(q)
        db.commit()
        d = Dashboard(team_id=ws["team"], question_id=q.id, kpis_json=kpis, charts_json=[], narrative="",
                      verified=verified, verdict_state=verdict_state)
        db.add(d)
        db.commit()
        return q.id, d.id
    finally:
        db.close()
